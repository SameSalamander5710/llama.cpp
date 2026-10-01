// Copies of tensors between two devices of the same backend, for example two Vulkan GPUs.
//
// Checks that the data arrives intact for blocking and asynchronous copies, for tensors and views at any offset of
// a buffer, for data a backend is still computing, and for remote weights that the scheduler prefetches.
//
// The backend can report how its copies travelled. With
//   --expect-direct   every byte must have gone straight from one device to the other
//   --expect-shared   every byte must have gone through memory that both devices import, without a copy on the CPU
//   --expect-host     every byte must have gone through host staging buffers
// the test fails when that is not the case. Without any it only reports the numbers.
//
// With --bench it only measures the copy that a weight prefetch makes, and reports the speed and the CPU time it costs.
//
// What the Vulkan backend allows is chosen with GGML_VK_DIRECT_COPY, and the size of the shared staging chunks with
// GGML_VK_COPY_CHUNK_MB, so that the sizes in this test can be put on and across their boundaries.
//
// The test is skipped when fewer than two devices of the backend exist. The backend defaults to Vulkan and can be
// changed with --backend NAME.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <ctime>
#endif

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            exit(1); \
        } \
    } while (0)

struct test_device {
    ggml_backend_dev_t         dev;
    ggml_backend_ptr           backend;
    ggml_backend_buffer_type_t buft;
};

typedef void (*copy_stats_fn)(size_t * direct_bytes, size_t * shared_bytes, size_t * host_bytes);

// the copy that does not block and does not need the backends to be idle, see ggml-backend-impl.h
typedef bool (*peer_copy_async_fn)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst);
typedef void (*peer_copy_sync_fn)(ggml_backend_t backend_dst);

static copy_stats_fn copy_stats = nullptr;

static std::vector<float> make_data(size_t n, uint32_t seed) {
    std::vector<float> data(n);
    uint32_t x = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < n; i++) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        data[i] = (float) (x & 0xffff) / 32768.0f - 1.0f;
    }
    return data;
}

static ggml_context_ptr make_context(size_t n_tensors) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(64, false) + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    return ggml_context_ptr(ggml_init(params));
}

// a buffer of a device that holds weights, which is how a model is loaded
static ggml_backend_buffer_ptr alloc_weights(ggml_context * ctx, test_device & d) {
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx, d.buft));
    CHECK(buf);
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return buf;
}

static void copy_tensor(test_device & a, test_device & b, ggml_tensor * src, ggml_tensor * dst, bool async) {
    if (async) {
        ggml_backend_tensor_copy_async(a.backend.get(), b.backend.get(), src, dst);
        ggml_backend_synchronize(b.backend.get());
    } else {
        ggml_backend_tensor_copy(src, dst);
    }
}

// a view of n elements at element offset off of a tensor
static ggml_tensor * view_at(ggml_context * ctx, ggml_tensor * t, size_t off, size_t n) {
    return ggml_view_1d(ctx, t, (int64_t) n, off * sizeof(float));
}

// Copies a range of a tensor of a into another range of a tensor of b, and checks that the range arrived and that
// nothing around it was touched.
static void test_range_copy(test_device & a, test_device & b, size_t n_total, size_t src_off, size_t dst_off, size_t n, bool async) {
    auto ctx_a = make_context(4);
    auto ctx_b = make_context(4);
    ggml_tensor * big_a = ggml_new_tensor_1d(ctx_a.get(), GGML_TYPE_F32, (int64_t) n_total);
    ggml_tensor * big_b = ggml_new_tensor_1d(ctx_b.get(), GGML_TYPE_F32, (int64_t) n_total);
    ggml_tensor * src   = view_at(ctx_a.get(), big_a, src_off, n);
    ggml_tensor * dst   = view_at(ctx_b.get(), big_b, dst_off, n);

    auto buf_a = alloc_weights(ctx_a.get(), a);
    auto buf_b = alloc_weights(ctx_b.get(), b);

    const std::vector<float> data = make_data(n_total, 1);
    ggml_backend_tensor_set(big_a, data.data(), 0, n_total * sizeof(float));
    ggml_backend_buffer_clear(buf_b.get(), 0);

    copy_tensor(a, b, src, dst, async);

    std::vector<float> out(n_total);
    ggml_backend_tensor_get(big_b, out.data(), 0, n_total * sizeof(float));
    for (size_t i = 0; i < n_total; i++) {
        const bool inside = i >= dst_off && i < dst_off + n;
        const float expected = inside ? data[src_off + (i - dst_off)] : 0.0f;
        if (out[i] != expected) {
            fprintf(stderr, "mismatch at %zu of %zu (copy %zu elements from %zu to %zu, %s): %f != %f\n",
                    i, n_total, n, src_off, dst_off, async ? "async" : "sync", out[i], expected);
            exit(1);
        }
    }
}

static void test_ranges(test_device & a, test_device & b, bool async) {
    // whole tensors, a single element, odd sizes and offsets, and more than the 32 MiB the host path moves at once
    test_range_copy(a, b, 1,        0,     0,     1,        async);
    test_range_copy(a, b, 1000,     0,     0,     1000,     async);
    test_range_copy(a, b, 100000,   12345, 777,   70001,    async);
    test_range_copy(a, b, 262144,   3,     131,   200000,   async);
    test_range_copy(a, b, 20000000, 1001,  5003,  19000000, async);
    // exactly one and two MiB, and one element more or less, which are chunk boundaries for the 1 MiB chunks of the shared staging
    for (size_t n : { 262144 - 1, 262144, 262144 + 1, 524288 - 1, 524288, 524288 + 1, 786432 }) {
        test_range_copy(a, b, n + 64, 0, 0, n, async);
        test_range_copy(a, b, n + 64, 17, 31, n, async);
    }
}

// many weights back to back, then a single wait, like a prefetch does
static void test_many_async(test_device & a, test_device & b) {
    const int    n_tensors = 16;
    const size_t n         = 2*1024*1024 + 17;

    auto ctx_a = make_context(n_tensors);
    auto ctx_b = make_context(n_tensors);
    std::vector<ggml_tensor *> src(n_tensors), dst(n_tensors);
    for (int i = 0; i < n_tensors; i++) {
        src[i] = ggml_new_tensor_1d(ctx_a.get(), GGML_TYPE_F32, (int64_t) n);
        dst[i] = ggml_new_tensor_1d(ctx_b.get(), GGML_TYPE_F32, (int64_t) n);
    }
    auto buf_a = alloc_weights(ctx_a.get(), a);
    auto buf_b = alloc_weights(ctx_b.get(), b);

    for (int i = 0; i < n_tensors; i++) {
        const std::vector<float> data = make_data(n, 100 + i);
        ggml_backend_tensor_set(src[i], data.data(), 0, n * sizeof(float));
    }
    const auto t_start = std::chrono::steady_clock::now();
    for (int i = 0; i < n_tensors; i++) {
        ggml_backend_tensor_copy_async(a.backend.get(), b.backend.get(), src[i], dst[i]);
    }
    ggml_backend_synchronize(b.backend.get());
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    printf("  %.1f MiB in %.1f ms, %.2f GiB/s\n", n_tensors * n * sizeof(float) / 1048576.0, sec * 1e3,
           n_tensors * n * sizeof(float) / sec / 1073741824.0);

    for (int i = 0; i < n_tensors; i++) {
        const std::vector<float> data = make_data(n, 100 + i);
        std::vector<float> out(n);
        ggml_backend_tensor_get(dst[i], out.data(), 0, n * sizeof(float));
        CHECK(memcmp(out.data(), data.data(), n * sizeof(float)) == 0);
    }
}

// CPU time of all the threads of the process, in seconds
static double process_cpu_seconds() {
#if defined(_WIN32)
    FILETIME creation, exit_time, kernel, user;
    GetProcessTimes(GetCurrentProcess(), &creation, &exit_time, &kernel, &user);
    const auto to_s = [](const FILETIME & t) { return (double) (((uint64_t) t.dwHighDateTime << 32) | t.dwLowDateTime) * 1e-7; };
    return to_s(kernel) + to_s(user);
#else
    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
#endif
}

// The copy a weight prefetch makes, many weights one after the other. Reports the speed and what it costs the CPU,
// for comparing the paths on a machine with two real devices, for example with GGML_VK_DIRECT_COPY=0, 1 and 2.
static void bench_peer_copy(test_device & a, test_device & b, ggml_backend_reg_t reg, const char * label) {
    peer_copy_async_fn copy_async = (peer_copy_async_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_peer_copy_async");
    peer_copy_sync_fn  copy_sync  = (peer_copy_sync_fn)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_peer_copy_synchronize");
    if (copy_async == nullptr || copy_sync == nullptr) {
        printf("  the backend has no asynchronous peer copy\n");
        return;
    }

    const int    n_tensors = 6;
    const size_t n         = 8*1024*1024; // 32 MiB each
    auto ctx_a = make_context(n_tensors);
    auto ctx_b = make_context(n_tensors);
    std::vector<ggml_tensor *> src(n_tensors), dst(n_tensors);
    for (int i = 0; i < n_tensors; i++) {
        src[i] = ggml_new_tensor_1d(ctx_a.get(), GGML_TYPE_F32, (int64_t) n);
        dst[i] = ggml_new_tensor_1d(ctx_b.get(), GGML_TYPE_F32, (int64_t) n);
    }
    auto buf_a = alloc_weights(ctx_a.get(), a);
    auto buf_b = alloc_weights(ctx_b.get(), b);
    const std::vector<float> data = make_data(n, 5);
    for (int i = 0; i < n_tensors; i++) {
        ggml_backend_tensor_set(src[i], data.data(), 0, n * sizeof(float));
    }

    auto pass = [&]() {
        for (int i = 0; i < n_tensors; i++) {
            CHECK(copy_async(a.backend.get(), b.backend.get(), src[i], dst[i]));
        }
        copy_sync(b.backend.get());
        ggml_backend_synchronize(b.backend.get());
    };

    pass(); // finds out what the two devices can do, and allocates what the copies need

    const int n_pass = 5;
    const double cpu_start = process_cpu_seconds();
    const auto   t_start   = std::chrono::steady_clock::now();
    for (int i = 0; i < n_pass; i++) {
        pass();
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    const double cpu = process_cpu_seconds() - cpu_start;
    const double gib = (double) n_pass * n_tensors * n * sizeof(float) / 1073741824.0;

    std::vector<float> out(n);
    ggml_backend_tensor_get(dst[n_tensors - 1], out.data(), 0, n * sizeof(float));
    CHECK(memcmp(out.data(), data.data(), n * sizeof(float)) == 0);

    printf("%s: %.2f GiB in %.3f s = %.2f GiB/s, %.3f CPU s per GiB\n", label, gib, sec, gib / sec, cpu / gib);
}

// The scheduler does not wait for the source backend when the copy is asynchronous, so a copy of data that the
// source backend is still computing has to see the finished data.
static void test_copy_after_compute(test_device & a, test_device & b) {
    const size_t n = 4*1024*1024;

    for (int iter = 0; iter < 8; iter++) {
        auto ctx_a = make_context(8);
        auto ctx_b = make_context(2);

        ggml_tensor * x = ggml_new_tensor_1d(ctx_a.get(), GGML_TYPE_F32, (int64_t) n);
        ggml_tensor * y = x;
        const float scale = 1.5f;
        for (int i = 0; i < 3; i++) {
            y = ggml_scale(ctx_a.get(), y, scale);
        }
        ggml_set_input(x);
        ggml_set_output(y);

        ggml_cgraph * gf = ggml_new_graph_custom(ctx_a.get(), 64, false);
        ggml_build_forward_expand(gf, y);

        ggml_gallocr_ptr galloc(ggml_gallocr_new(a.buft));
        CHECK(ggml_gallocr_alloc_graph(galloc.get(), gf));

        ggml_tensor * y_dst = ggml_new_tensor_1d(ctx_b.get(), GGML_TYPE_F32, (int64_t) n);
        ggml_backend_buffer_ptr buf_b(ggml_backend_alloc_ctx_tensors_from_buft(ctx_b.get(), b.buft));
        CHECK(buf_b);

        const std::vector<float> data = make_data(n, 7 + iter);
        ggml_backend_tensor_set(x, data.data(), 0, n * sizeof(float));

        CHECK(ggml_backend_graph_compute_async(a.backend.get(), gf) == GGML_STATUS_SUCCESS);
        // no synchronization of a here
        ggml_backend_tensor_copy_async(a.backend.get(), b.backend.get(), y, y_dst);
        ggml_backend_synchronize(b.backend.get());

        std::vector<float> out(n);
        ggml_backend_tensor_get(y_dst, out.data(), 0, n * sizeof(float));
        const float k = scale * scale * scale;
        for (size_t i = 0; i < n; i++) {
            if (std::fabs(out[i] - data[i] * k) > 1e-5f) {
                fprintf(stderr, "stale or wrong data at %zu (iteration %d): %f != %f\n", i, iter, out[i], data[i] * k);
                exit(1);
            }
        }
    }
}

// The first copy between two devices also finds out which path they can use, and the copies run beside the compute of
// the destination device. Neither may disturb that compute, nor be disturbed by it. This has to run before any other
// copy between the two devices, so that it is the one that finds out.
static void test_first_copy_during_compute(test_device & a, test_device & b, ggml_backend_reg_t reg) {
    peer_copy_async_fn copy_async = (peer_copy_async_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_peer_copy_async");
    peer_copy_sync_fn  copy_sync  = (peer_copy_sync_fn)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_peer_copy_synchronize");
    if (copy_async == nullptr || copy_sync == nullptr) {
        printf("  the backend has no asynchronous peer copy, skipped\n");
        return;
    }

    const int64_t dim = 512, n_tok = 64;
    const int n_layers = 16;

    // compute on b
    auto ctx_c = make_context(n_layers * 3 + 4);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_c.get(), GGML_TYPE_F32, dim, dim);
    ggml_tensor * x = ggml_new_tensor_2d(ctx_c.get(), GGML_TYPE_F32, dim, n_tok);
    ggml_set_input(x);
    ggml_tensor * cur = x;
    for (int i = 0; i < n_layers; i++) {
        cur = ggml_scale(ctx_c.get(), ggml_mul_mat(ctx_c.get(), w, cur), 1.0f / std::sqrt((float) dim));
    }
    ggml_set_output(cur);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx_c.get(), 64, false);
    ggml_build_forward_expand(gf, cur);
    ggml_backend_buffer_ptr buf_w(ggml_backend_alloc_ctx_tensors_from_buft(ctx_c.get(), b.buft));
    CHECK(buf_w);

    // what is copied from a into b meanwhile, several tensors one after the other
    const int n_copy = 6;
    const size_t n = 3*1024*1024 + 5;
    auto ctx_a = make_context(n_copy);
    auto ctx_b = make_context(n_copy);
    std::vector<ggml_tensor *> src(n_copy), dst(n_copy);
    for (int i = 0; i < n_copy; i++) {
        src[i] = ggml_new_tensor_1d(ctx_a.get(), GGML_TYPE_F32, (int64_t) n);
        dst[i] = ggml_new_tensor_1d(ctx_b.get(), GGML_TYPE_F32, (int64_t) n);
    }
    auto buf_a = alloc_weights(ctx_a.get(), a);
    auto buf_b = alloc_weights(ctx_b.get(), b);
    for (int i = 0; i < n_copy; i++) {
        const std::vector<float> data = make_data(n, 300 + i);
        ggml_backend_tensor_set(src[i], data.data(), 0, n * sizeof(float));
    }

    const std::vector<float> wdata = make_data((size_t) (dim * dim), 77);
    const std::vector<float> xdata = make_data((size_t) (dim * n_tok), 78);
    ggml_backend_tensor_set(w, wdata.data(), 0, wdata.size() * sizeof(float));
    ggml_backend_tensor_set(x, xdata.data(), 0, xdata.size() * sizeof(float));

    ggml_gallocr_ptr galloc(ggml_gallocr_new(b.buft));
    CHECK(ggml_gallocr_alloc_graph(galloc.get(), gf));
    ggml_backend_tensor_set(x, xdata.data(), 0, xdata.size() * sizeof(float));

    CHECK(ggml_backend_graph_compute_async(b.backend.get(), gf) == GGML_STATUS_SUCCESS);
    // b is computing now
    for (int i = 0; i < n_copy; i++) {
        CHECK(copy_async(a.backend.get(), b.backend.get(), src[i], dst[i]));
    }
    copy_sync(b.backend.get());
    ggml_backend_synchronize(b.backend.get());

    for (int i = 0; i < n_copy; i++) {
        const std::vector<float> data = make_data(n, 300 + i);
        std::vector<float> out(n);
        ggml_backend_tensor_get(dst[i], out.data(), 0, n * sizeof(float));
        CHECK(memcmp(out.data(), data.data(), n * sizeof(float)) == 0);
    }

    std::vector<float> out((size_t) (dim * n_tok));
    ggml_backend_tensor_get(cur, out.data(), 0, out.size() * sizeof(float));
    std::vector<float> ref = xdata;
    for (int i = 0; i < n_layers; i++) {
        std::vector<float> next(ref.size());
        for (int64_t t = 0; t < n_tok; t++) {
            for (int64_t o = 0; o < dim; o++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < dim; k++) {
                    sum += wdata[o*dim + k] * ref[t*dim + k];
                }
                next[t*dim + o] = sum / std::sqrt((float) dim);
            }
        }
        ref = next;
    }
    for (size_t i = 0; i < out.size(); i++) {
        if (std::fabs(out[i] - ref[i]) > 1e-2f * (1.0f + std::fabs(ref[i]))) {
            fprintf(stderr, "the compute beside the copy is wrong at %zu: %f != %f\n", i, out[i], ref[i]);
            exit(1);
        }
    }
}

// a chain of layers whose weights are all in the memory of another device. With prefetch the scheduler runs the layers
// on the device that owns the activations and streams each weight over, without it the layers run where the weights are.
static void test_sched_remote_weights(test_device & dev_w, test_device & dev_l, bool prefetch) {
    const int n_layers = 6;
    const int64_t dim  = 96;
    const int64_t n_tokens = 8;

    ggml_backend_ptr backend_cpu(ggml_backend_cpu_init());
    CHECK(backend_cpu);

    auto ctx_w = make_context(n_layers);
    auto ctx_x = make_context(1);
    auto ctx   = make_context(n_layers * 3 + 4);

    std::vector<ggml_tensor *> w(n_layers);
    for (int i = 0; i < n_layers; i++) {
        w[i] = ggml_new_tensor_2d(ctx_w.get(), GGML_TYPE_F32, dim, dim);
    }
    ggml_tensor * x = ggml_new_tensor_2d(ctx_x.get(), GGML_TYPE_F32, dim, n_tokens);
    ggml_set_input(x);

    auto buf_w = alloc_weights(ctx_w.get(), dev_w);
    ggml_backend_buffer_ptr buf_x(ggml_backend_alloc_ctx_tensors_from_buft(ctx_x.get(), dev_l.buft));
    CHECK(buf_x);

    ggml_tensor * cur = x;
    for (int i = 0; i < n_layers; i++) {
        cur = ggml_mul_mat(ctx.get(), w[i], cur);
        cur = ggml_scale(ctx.get(), cur, 1.0f / std::sqrt((float) dim));
        cur = ggml_relu(ctx.get(), cur);
    }
    ggml_set_output(cur);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(gf, cur);

    ggml_backend_t             backends[3] = { dev_w.backend.get(), dev_l.backend.get(), backend_cpu.get() };
    ggml_backend_buffer_type_t bufts[3]    = { dev_w.buft, dev_l.buft, ggml_backend_cpu_buffer_type() };
    ggml_backend_sched_ptr     sched(ggml_backend_sched_new(backends, bufts, 3, 64, false, true));
    ggml_backend_sched_set_prefetch(sched.get(), prefetch);
    CHECK(ggml_backend_sched_alloc_graph(sched.get(), gf));

    std::vector<std::vector<float>> wdata(n_layers);
    for (int i = 0; i < n_layers; i++) {
        wdata[i] = make_data((size_t) (dim * dim), 50 + i);
        ggml_backend_tensor_set(w[i], wdata[i].data(), 0, wdata[i].size() * sizeof(float));
    }

    // computed twice, the second pass reuses the staging buffers
    for (int pass = 0; pass < 2; pass++) {
        const std::vector<float> xdata = make_data((size_t) (dim * n_tokens), 900 + pass);
        ggml_backend_tensor_set(x, xdata.data(), 0, xdata.size() * sizeof(float));

        CHECK(ggml_backend_sched_graph_compute(sched.get(), gf) == GGML_STATUS_SUCCESS);

        std::vector<float> out((size_t) (dim * n_tokens));
        ggml_backend_tensor_get(cur, out.data(), 0, out.size() * sizeof(float));

        // reference on the host
        std::vector<float> ref = xdata;
        for (int i = 0; i < n_layers; i++) {
            std::vector<float> next((size_t) (dim * n_tokens));
            for (int64_t t = 0; t < n_tokens; t++) {
                for (int64_t o = 0; o < dim; o++) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < dim; k++) {
                        sum += wdata[i][o*dim + k] * ref[t*dim + k];
                    }
                    next[t*dim + o] = std::max(0.0f, sum / std::sqrt((float) dim));
                }
            }
            ref = next;
        }
        for (size_t i = 0; i < out.size(); i++) {
            if (std::fabs(out[i] - ref[i]) > 1e-3f) {
                fprintf(stderr, "sched %s pass %d: mismatch at %zu: %f != %f\n", prefetch ? "prefetch" : "no prefetch", pass, i, out[i], ref[i]);
                exit(1);
            }
        }
    }
}

int main(int argc, char ** argv) {
    std::string backend_name = "Vulkan";
    bool expect_direct = false;
    bool expect_shared = false;
    bool expect_host   = false;
    bool bench         = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--expect-direct") == 0) {
            expect_direct = true;
        } else if (strcmp(argv[i], "--expect-shared") == 0) {
            expect_shared = true;
        } else if (strcmp(argv[i], "--expect-host") == 0) {
            expect_host = true;
        } else if (strcmp(argv[i], "--bench") == 0) {
            bench = true;
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend_name = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--backend NAME] [--bench] [--expect-direct | --expect-shared | --expect-host]\n", argv[0]);
            return 1;
        }
    }

    ggml_backend_load_all();

    std::vector<test_device> devices;
    ggml_backend_reg_t reg = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (backend_name == ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev))) {
            reg = ggml_backend_dev_backend_reg(dev);
            test_device d;
            d.dev     = dev;
            d.backend = ggml_backend_ptr(ggml_backend_dev_init(dev, nullptr));
            d.buft    = ggml_backend_dev_buffer_type(dev);
            CHECK(d.backend);
            devices.push_back(std::move(d));
        }
    }
    if (devices.size() < 2) {
        printf("skipped: %zu %s device(s), two are needed\n", devices.size(), backend_name.c_str());
        return 0;
    }
    printf("copying between %s and %s\n", ggml_backend_dev_name(devices[0].dev), ggml_backend_dev_name(devices[1].dev));

    copy_stats = (copy_stats_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_get_copy_stats");

    test_device & d0 = devices[0];
    test_device & d1 = devices[1];

    if (bench) {
        bench_peer_copy(d0, d1, reg, "peer copy 0 -> 1");
        bench_peer_copy(d1, d0, reg, "peer copy 1 -> 0");
        if (copy_stats) {
            size_t direct = 0, shared = 0, host = 0;
            copy_stats(&direct, &shared, &host);
            printf("path taken: %zu MiB direct, %zu MiB shared staging, %zu MiB host staging\n", direct >> 20, shared >> 20, host >> 20);
        }
        return 0;
    }

    printf("first copy beside compute 0 -> 1\n"); test_first_copy_during_compute(d0, d1, reg);
    printf("first copy beside compute 1 -> 0\n"); test_first_copy_during_compute(d1, d0, reg);

    // blocking copies first, in both directions. Between two devices, the first copy is also where the backend finds out
    // whether they can share memory, which the asynchronous copies after that rely on.
    printf("sync 0 -> 1\n"); test_ranges(d0, d1, false);
    printf("sync 1 -> 0\n"); test_ranges(d1, d0, false);

    size_t direct_before = 0, shared_before = 0, host_before = 0;
    if (copy_stats) {
        copy_stats(&direct_before, &shared_before, &host_before);
    }

    printf("async 0 -> 1\n"); test_ranges(d0, d1, true);
    printf("async 1 -> 0\n"); test_ranges(d1, d0, true);
    printf("many async 0 -> 1\n"); test_many_async(d0, d1);
    printf("many async 1 -> 0\n"); test_many_async(d1, d0);
    printf("copy after compute 0 -> 1\n"); test_copy_after_compute(d0, d1);
    printf("copy after compute 1 -> 0\n"); test_copy_after_compute(d1, d0);

    printf("scheduler, weights on 0, layers on 1, no prefetch\n"); test_sched_remote_weights(d0, d1, false);
    printf("scheduler, weights on 0, layers on 1, prefetch\n");    test_sched_remote_weights(d0, d1, true);
    printf("scheduler, weights on 1, layers on 0, prefetch\n");    test_sched_remote_weights(d1, d0, true);

    if (copy_stats) {
        size_t direct = 0, shared = 0, host = 0;
        copy_stats(&direct, &shared, &host);
        printf("bytes copied between devices: %zu direct, %zu through shared staging, %zu through host staging\n", direct, shared, host);
        printf("  after the blocking copies: %zu direct, %zu shared, %zu host\n", direct - direct_before, shared - shared_before, host - host_before);
        if (expect_direct) {
            CHECK(direct > 0 && shared == 0 && host == 0);
        }
        if (expect_shared) {
            CHECK(shared > 0 && direct == 0 && host == 0);
        }
        if (expect_host) {
            CHECK(host > 0 && direct == 0 && shared == 0);
        }
    } else if (expect_direct || expect_shared || expect_host) {
        fprintf(stderr, "the backend does not report how it copies\n");
        return 1;
    }

    printf("OK\n");
    return 0;
}
