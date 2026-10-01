// Tests the asynchronous copy between the memory of two Vulkan devices, and the weight prefetch that streams a weight from one device into the other one with it.
// Needs two Vulkan devices, and is skipped otherwise. A single physical device cannot be used twice, but a software driver can be listed twice to test on a machine without two GPUs, for example with VK_DRIVER_FILES=a.json:b.json and GGML_VK_VISIBLE_DEVICES=0,1.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct vulkan_devices {
    ggml_backend_dev_t dev[2] = { nullptr, nullptr };
    int n = 0;
};

static vulkan_devices find_vulkan_devices() {
    vulkan_devices r;
    for (size_t i = 0; i < ggml_backend_dev_count() && r.n < 2; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (strcmp(ggml_backend_reg_name(reg), "Vulkan") == 0) {
            r.dev[r.n++] = dev;
        }
    }
    return r;
}

struct peer_fns {
    ggml_backend_peer_copy_async_t       copy_async = nullptr;
    ggml_backend_peer_copy_synchronize_t copy_sync  = nullptr;
};

static peer_fns get_peer_fns(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    peer_fns f;
    f.copy_async = (ggml_backend_peer_copy_async_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_peer_copy_async");
    f.copy_sync  = (ggml_backend_peer_copy_synchronize_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_peer_copy_synchronize");
    return f;
}

static ggml_context * new_ctx(size_t n_tensors) {
    ggml_init_params params = { n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(256, false) + 4096, nullptr, true };
    return ggml_init(params);
}

// copies n_bytes between two views at the given byte offsets of two big tensors on different devices
static void test_peer_copy(const vulkan_devices & d, size_t n_bytes, size_t src_off, size_t dst_off) {
    ggml_backend_t be_src = ggml_backend_dev_init(d.dev[1], nullptr);
    ggml_backend_t be_dst = ggml_backend_dev_init(d.dev[0], nullptr);
    GGML_ASSERT(be_src && be_dst);
    const peer_fns f = get_peer_fns(d.dev[0]);
    GGML_ASSERT(f.copy_async && f.copy_sync);

    const size_t pad = 4096;
    const size_t n_src = (src_off + n_bytes + pad) / sizeof(float);
    const size_t n_dst = (dst_off + n_bytes + pad) / sizeof(float);

    ggml_context * ctx_src = new_ctx(4);
    ggml_context * ctx_dst = new_ctx(4);
    ggml_tensor * big_src = ggml_new_tensor_1d(ctx_src, GGML_TYPE_F32, n_src);
    ggml_tensor * big_dst = ggml_new_tensor_1d(ctx_dst, GGML_TYPE_F32, n_dst);
    // views of F32 tensors need a multiple of 4 bytes, so keep everything aligned to that
    GGML_ASSERT(n_bytes % 4 == 0 && src_off % 4 == 0 && dst_off % 4 == 0);
    ggml_tensor * src = ggml_view_1d(ctx_src, big_src, n_bytes/4, src_off);
    ggml_tensor * dst = ggml_view_1d(ctx_dst, big_dst, n_bytes/4, dst_off);

    ggml_backend_buffer_t buf_src = ggml_backend_alloc_ctx_tensors_from_buft(ctx_src, ggml_backend_dev_buffer_type(d.dev[1]));
    ggml_backend_buffer_t buf_dst = ggml_backend_alloc_ctx_tensors_from_buft(ctx_dst, ggml_backend_dev_buffer_type(d.dev[0]));
    GGML_ASSERT(buf_src && buf_dst);

    std::mt19937 rng(1234);
    std::vector<uint32_t> data(n_src);
    for (uint32_t & v : data) {
        v = rng();
    }
    ggml_backend_tensor_set(big_src, data.data(), 0, ggml_nbytes(big_src));
    std::vector<uint32_t> zeros(n_dst, 0);
    ggml_backend_tensor_set(big_dst, zeros.data(), 0, ggml_nbytes(big_dst));

    GGML_ASSERT(f.copy_async(be_src, be_dst, src, dst));
    f.copy_sync(be_dst);

    std::vector<uint32_t> out(n_dst);
    ggml_backend_tensor_get(big_dst, out.data(), 0, ggml_nbytes(big_dst));
    for (size_t i = 0; i < n_dst; i++) {
        const size_t byte = i * 4;
        const bool inside = byte >= dst_off && byte < dst_off + n_bytes;
        const uint32_t expected = inside ? data[(src_off + byte - dst_off) / 4] : 0;
        if (out[i] != expected) {
            fprintf(stderr, "mismatch at byte %zu of the destination (inside %d): got %08x, expected %08x\n", byte, inside, out[i], expected);
            GGML_ABORT("peer copy mismatch");
        }
    }

    ggml_backend_buffer_free(buf_src);
    ggml_backend_buffer_free(buf_dst);
    ggml_free(ctx_src);
    ggml_free(ctx_dst);
    ggml_backend_free(be_src);
    ggml_backend_free(be_dst);
}

// two tensors on the same device are not a job for the peer copy, and it must not take them
static void test_peer_copy_same_device(const vulkan_devices & d) {
    ggml_backend_t be = ggml_backend_dev_init(d.dev[0], nullptr);
    const peer_fns f = get_peer_fns(d.dev[0]);

    ggml_context * ctx = new_ctx(4);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(d.dev[0]));
    GGML_ASSERT(buf);
    GGML_ASSERT(!f.copy_async(be, be, a, b));
    f.copy_sync(be); // nothing in flight

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
}

// several copies in a row into one backend, then one synchronize
static void test_peer_copy_many(const vulkan_devices & d) {
    ggml_backend_t be_src = ggml_backend_dev_init(d.dev[1], nullptr);
    ggml_backend_t be_dst = ggml_backend_dev_init(d.dev[0], nullptr);
    const peer_fns f = get_peer_fns(d.dev[0]);

    const int n = 8;
    ggml_context * ctx_src = new_ctx(n);
    ggml_context * ctx_dst = new_ctx(n);
    std::vector<ggml_tensor *> src(n), dst(n);
    for (int i = 0; i < n; i++) {
        const int64_t ne = 1000 + 30011 * i; // sizes that do not line up with anything
        src[i] = ggml_new_tensor_1d(ctx_src, GGML_TYPE_F32, ne);
        dst[i] = ggml_new_tensor_1d(ctx_dst, GGML_TYPE_F32, ne);
    }
    ggml_backend_buffer_t buf_src = ggml_backend_alloc_ctx_tensors_from_buft(ctx_src, ggml_backend_dev_buffer_type(d.dev[1]));
    ggml_backend_buffer_t buf_dst = ggml_backend_alloc_ctx_tensors_from_buft(ctx_dst, ggml_backend_dev_buffer_type(d.dev[0]));

    std::mt19937 rng(42);
    std::vector<std::vector<float>> ref(n);
    for (int i = 0; i < n; i++) {
        ref[i].resize(ggml_nelements(src[i]));
        for (float & v : ref[i]) {
            v = (float) (rng() % 1000) / 7.0f;
        }
        ggml_backend_tensor_set(src[i], ref[i].data(), 0, ggml_nbytes(src[i]));
    }
    for (int i = 0; i < n; i++) {
        GGML_ASSERT(f.copy_async(be_src, be_dst, src[i], dst[i]));
    }
    f.copy_sync(be_dst);
    for (int i = 0; i < n; i++) {
        std::vector<float> out(ref[i].size());
        ggml_backend_tensor_get(dst[i], out.data(), 0, ggml_nbytes(dst[i]));
        GGML_ASSERT(memcmp(out.data(), ref[i].data(), ggml_nbytes(dst[i])) == 0);
    }

    ggml_backend_buffer_free(buf_src);
    ggml_backend_buffer_free(buf_dst);
    ggml_free(ctx_src);
    ggml_free(ctx_dst);
    ggml_backend_free(be_src);
    ggml_backend_free(be_dst);
}

// ---- prefetch of weights from the other device through the scheduler

static int g_peer_copy_logs = 0;

static void count_peer_copies(enum ggml_log_level, const char * text, void *) {
    if (strstr(text, "peer copy of") != nullptr) {
        g_peer_copy_logs++;
    }
    // everything else is only shown on request
    if (getenv("GGML_TEST_VERBOSE") != nullptr) {
        fputs(text, stderr);
    }
}

struct ffn_weights {
    ggml_tensor * attn;
    ggml_tensor * norm;
    ggml_tensor * gate;
    ggml_tensor * up;
    ggml_tensor * down;
};

static constexpr int64_t D_MODEL = 64;
static constexpr int64_t D_FF    = 192;
static constexpr int64_t N_TOK   = 64; // the Vulkan backend offloads ops with at least 32 columns

static ggml_tensor * build_layers(ggml_context * ctx, ggml_tensor * x, const std::vector<ffn_weights> & layers) {
    ggml_tensor * cur = x;
    for (const ffn_weights & l : layers) {
        ggml_tensor * a = ggml_mul_mat(ctx, l.attn, cur);
        ggml_tensor * r = ggml_add(ctx, a, cur);
        ggml_tensor * n = ggml_mul(ctx, ggml_rms_norm(ctx, r, 1e-5f), l.norm);
        ggml_tensor * g = ggml_mul_mat(ctx, l.gate, n);
        ggml_tensor * u = ggml_mul_mat(ctx, l.up, n);
        ggml_tensor * h = ggml_mul(ctx, ggml_silu(ctx, g), u);
        ggml_tensor * dn = ggml_mul_mat(ctx, l.down, h);
        cur = ggml_add(ctx, dn, r);
    }
    return cur;
}

static void new_layers(ggml_context * ctx_layer, ggml_context * ctx_ffn, std::vector<ffn_weights> & layers, int n_layers, bool remote_norm) {
    for (int il = 0; il < n_layers; il++) {
        ffn_weights l;
        l.attn = ggml_new_tensor_2d(ctx_layer, GGML_TYPE_F32, D_MODEL, D_MODEL);
        l.norm = ggml_new_tensor_1d(remote_norm ? ctx_ffn : ctx_layer, GGML_TYPE_F32, D_MODEL);
        l.gate = ggml_new_tensor_2d(ctx_ffn, GGML_TYPE_F32, D_MODEL, D_FF);
        l.up   = ggml_new_tensor_2d(ctx_ffn, GGML_TYPE_F32, D_MODEL, D_FF);
        l.down = ggml_new_tensor_2d(ctx_ffn, GGML_TYPE_F32, D_FF, D_MODEL);
        layers.push_back(l);
    }
}

static void fill_weights(const std::vector<ffn_weights> & layers, std::vector<std::vector<float>> & store, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (const ffn_weights & l : layers) {
        for (ggml_tensor * t : { l.attn, l.norm, l.gate, l.up, l.down }) {
            std::vector<float> v(ggml_nelements(t));
            for (float & x : v) {
                x = t == l.norm ? 1.0f + 0.1f * dist(rng) : dist(rng);
            }
            store.push_back(std::move(v));
        }
    }
}

static void upload_weights(const std::vector<ffn_weights> & layers, const std::vector<std::vector<float>> & store) {
    size_t i = 0;
    for (const ffn_weights & l : layers) {
        for (ggml_tensor * t : { l.attn, l.norm, l.gate, l.up, l.down }) {
            ggml_backend_tensor_set(t, store[i++].data(), 0, ggml_nbytes(t));
        }
    }
}

// reference on the CPU backend only
static std::vector<float> reference_output(const std::vector<std::vector<float>> & store, const std::vector<float> & input, int n_layers, bool remote_norm) {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu);

    ggml_context * ctx_w = new_ctx(5 * n_layers + 2);
    ggml_context * ctx   = new_ctx(64 * n_layers);
    std::vector<ffn_weights> layers;
    new_layers(ctx_w, ctx_w, layers, n_layers, remote_norm);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, cpu);
    upload_weights(layers, store);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, N_TOK);
    ggml_set_input(x);
    ggml_tensor * out = build_layers(ctx, x, layers);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    GGML_ASSERT(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    ggml_free(ctx_w);
    ggml_backend_free(cpu);
    return result;
}

// the layers run on Vulkan0, and the FFN weights are in the memory of Vulkan1
static void test_prefetch_from_other_device(const vulkan_devices & d, bool remote_norm, int n_layers, bool use_peer) {
    ggml_backend_t be0 = ggml_backend_dev_init(d.dev[0], nullptr);
    ggml_backend_t be1 = ggml_backend_dev_init(d.dev[1], nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(be0 && be1 && cpu);

    ggml_context * ctx_layer = new_ctx(2 * n_layers + 2);
    ggml_context * ctx_ffn   = new_ctx(4 * n_layers + 2);
    std::vector<ffn_weights> layers;
    new_layers(ctx_layer, ctx_ffn, layers, n_layers, remote_norm);
    ggml_backend_buffer_t buf_layer = ggml_backend_alloc_ctx_tensors(ctx_layer, be0);
    ggml_backend_buffer_t buf_ffn   = ggml_backend_alloc_ctx_tensors(ctx_ffn, be1);
    GGML_ASSERT(buf_layer && buf_ffn);
    ggml_backend_buffer_set_usage(buf_layer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(buf_ffn,   GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<std::vector<float>> store;
    fill_weights(layers, store, 7);
    upload_weights(layers, store);

    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> input(D_MODEL * N_TOK);
    for (float & v : input) {
        v = dist(rng);
    }
    const std::vector<float> expected = reference_output(store, input, n_layers, remote_norm);

    set_env("GGML_SCHED_PREFETCH_NO_PEER", use_peer ? nullptr : "1");
    ggml_backend_t             backends[3] = { be0, be1, cpu };
    ggml_backend_buffer_type_t bufts[3]    = { ggml_backend_get_default_buffer_type(be0), ggml_backend_get_default_buffer_type(be1), ggml_backend_get_default_buffer_type(cpu) };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 3, 256 * n_layers, false, true);
    set_env("GGML_SCHED_PREFETCH_NO_PEER", nullptr);

    g_peer_copy_logs = 0;
    for (int run = 0; run < 3; run++) {
        // the graph is built again for every run, like a decode does
        ggml_context * ctx = new_ctx(64 * n_layers);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, N_TOK);
        ggml_set_input(x);
        ggml_tensor * out = build_layers(ctx, x, layers);
        ggml_set_output(out);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256 * n_layers, false);
        ggml_build_forward_expand(gf, out);

        ggml_backend_sched_reset(sched);
        // a reset turns the prefetch off, as it does for llama.cpp
        ggml_backend_sched_set_prefetch(sched, true);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, gf));

        // every op of the FFN runs on the layer device, one split, weights read from their staged copies
        GGML_ASSERT(ggml_backend_sched_get_n_splits(sched) == 1);
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched, node) == be0);
            if (node->op == GGML_OP_MUL_MAT && strstr(node->src[0]->name, "#pf") == nullptr) {
                // only the attention matmuls read their weight in place
                bool is_attn = false;
                for (const ffn_weights & l : layers) {
                    is_attn = is_attn || node->src[0] == l.attn;
                }
                GGML_ASSERT(is_attn);
            }
        }

        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched);

        std::vector<float> result(ggml_nelements(out));
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        double max_err = 0.0;
        for (size_t i = 0; i < result.size(); i++) {
            max_err = std::max(max_err, (double) std::fabs(result[i] - expected[i]));
        }
        if (!(max_err < 1e-2)) {
            fprintf(stderr, "run %d: max error %g\n", run, max_err);
            GGML_ABORT("prefetch result differs from the reference");
        }
        ggml_free(ctx);
    }

    if (use_peer) {
        GGML_ASSERT(g_peer_copy_logs > 0);
    }
#ifndef _WIN32
    // On Windows an environment variable set here is not seen by a library that has its own C runtime, so the blocking path cannot be forced there. The result is checked against the reference either way
    if (!use_peer) {
        GGML_ASSERT(g_peer_copy_logs == 0);
    }
#endif

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf_layer);
    ggml_backend_buffer_free(buf_ffn);
    ggml_free(ctx_layer);
    ggml_free(ctx_ffn);
    ggml_backend_free(be0);
    ggml_backend_free(be1);
    ggml_backend_free(cpu);
}

int main() {
    ggml_backend_load_all();

    const vulkan_devices d = find_vulkan_devices();
    if (d.n < 2) {
        printf("SKIPPED: needs two Vulkan devices, found %d\n", d.n);
        return 0;
    }
    static vulkan_devices dev;
    dev = d;

    ggml_log_set(count_peer_copies, nullptr);

    // the mapped path where the memory is host visible, and the staged path, chosen by the test
    for (const char * force_staging : { (const char *) nullptr, "1" }) {
        set_env("GGML_VK_PEER_COPY_FORCE_STAGING", force_staging);
        printf("peer copy, %s\n", force_staging ? "staged" : "default");

        const size_t MiB = 1u << 20;
        // smaller than one chunk, an odd size, exactly a chunk, a few chunks and a tail, with offsets
        const size_t sizes[][3] = {
            { 4,                   0,    0 },
            { 4096 + 12,           8,    20 },
            { 16 * MiB,            0,    0 },
            { 16 * MiB,            256,  4 },
            { 40 * MiB + 28,       0,    0 },
            { 40 * MiB + 28,       1024, 36 },
            { 70 * MiB,            12,   0 },
        };
        for (const auto & s : sizes) {
            printf("  %zu bytes, offsets %zu/%zu ", s[0], s[1], s[2]);
            fflush(stdout);
            test_peer_copy(dev, s[0], s[1], s[2]);
            printf("PASSED\n");
        }
        printf("  many ");
        fflush(stdout);
        test_peer_copy_many(dev);
        printf("PASSED\n");
    }
    set_env("GGML_VK_PEER_COPY_FORCE_STAGING", nullptr);

    printf("same device ");
    test_peer_copy_same_device(dev);
    printf("PASSED\n");

    for (const char * force_staging : { (const char *) nullptr, "1" }) {
        set_env("GGML_VK_PEER_COPY_FORCE_STAGING", force_staging);
        for (bool use_peer : { true, false }) {
            for (bool remote_norm : { false, true }) {
                printf("prefetch from the other device, %s, norm %s, %s ", force_staging ? "staged" : "default",
                        remote_norm ? "remote" : "local", use_peer ? "async copy" : "blocking copy");
                fflush(stdout);
                test_prefetch_from_other_device(dev, remote_norm, 2, use_peer);
                printf("PASSED\n");
            }
        }
    }
    set_env("GGML_VK_PEER_COPY_FORCE_STAGING", nullptr);

    return 0;
}
