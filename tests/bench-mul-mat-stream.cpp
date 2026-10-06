// bench-mul-mat-stream: decode-shaped mat-vec benchmark that cannot be served from the GPU cache
//
// Why not test-backend-ops perf: it repeats the same MUL_MAT node on the same weights, and in perf mode
// it reports FLOPS and not bandwidth. A 73 MB FFN tensor (17408 x 5120 at Q6_K) fits in the 96 MB
// Infinity Cache of an RX 6700 XT, so back to back runs read it from cache and report a bandwidth the
// card cannot sustain in a real decode, where every layer streams different weights from GDDR6.
//
// This tool measures three things for every type and column count:
//   pair  up (K -> M) then down (M -> K), for `layers` layers that all own their weights, repeated `reps`
//         times in one graph. Each op consumes the output of the previous one, so the backend cannot
//         overlap them: this is the closest to a decode step (barrier between every op).
//   up    only the up/gate shape, M x K with K = hidden size, as independent ops that all read the same x
//   down  only the down shape, K x M with K = FFN width, as independent ops that all read the same x
// up and down are streaming numbers: they show how each shape does on its own, which a chain cannot,
// because the output of an up op has the wrong size to feed another up op. Independent ops may overlap
// at their tails, so compare up with down, and a configuration with another one, not up or down with pair.
// The weights read by one op were last touched `layers - 1` layers earlier, so they are cold as long as
// the weights of one shape are a few times the cache size.
//
// Examples:
//   bench-mul-mat-stream --list
//   bench-mul-mat-stream -d 6700 -t q6_K,q4_0,q8_0 -n 1,2,3,4,8 --peak-gbps 384 --calib
//   bench-mul-mat-stream -d 6700 -t q6_K -n 1,3 --check
//
// Per-op GPU times of the same run: set GGML_VK_PERF_LOGGER=1 (the Vulkan backend prints GB/s per op).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct bench_params {
    std::string device     = "";          // index or substring of the name/description, empty: first GPU
    std::vector<ggml_type> types = { GGML_TYPE_Q6_K };
    std::vector<int>       n_cols = { 1, 2, 3, 4 };
    int64_t m              = 17408;       // FFN width
    int64_t k              = 5120;        // hidden size
    int     layers         = 0;           // 0: choose from min_mb
    int     reps           = 4;           // times the layer sequence is repeated in one graph
    double  min_mb         = 800.0;       // minimum total weight size per type
    double  ic_mb          = 96.0;        // cache size, only used for the warning
    double  peak_gbps      = 0.0;         // memory bandwidth to report percentages against
    double  time_ms        = 1500.0;      // time to run each measurement for
    bool    calib          = false;
    bool    check          = false;
    bool    csv            = false;
    bool    list           = false;
};

static std::string lower(std::string s) {
    for (char & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t end = s.find(sep, pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        if (end > pos) {
            out.push_back(s.substr(pos, end - pos));
        }
        pos = end + 1;
    }
    return out;
}

static bool parse_type(const std::string & name, ggml_type & out) {
    const std::string want = lower(name);
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const ggml_type t = (ggml_type) i;
        const char * tn = ggml_type_name(t);
        if (tn != nullptr && lower(tn) == want && ggml_get_type_traits(t)->blck_size > 0) {
            out = t;
            return true;
        }
    }
    return false;
}

static void usage(const char * argv0) {
    printf("usage: %s [options]\n", argv0);
    printf("  --list                 list the devices and exit\n");
    printf("  -d, --device X         device index, or part of its name such as 6700 (default: first GPU), 'cpu' for the CPU\n");
    printf("  -t, --types a,b,c      weight types (default q6_K), e.g. q6_K,q4_0,q8_0\n");
    printf("  -n, --cols a,b,c       number of activation columns (default 1,2,3,4), 3 is MTP with 2 drafts\n");
    printf("  -m N / -k N            FFN width and hidden size (default 17408 / 5120)\n");
    printf("  --layers N             layers with their own weights (default: enough for --min-mb)\n");
    printf("  --reps N               repeats of the layer sequence per graph (default 4)\n");
    printf("  --min-mb X             minimum total weight size per type in MB (default 800)\n");
    printf("  --ic-mb X              last level cache size in MB, for the cache warning (default 96)\n");
    printf("  --peak-gbps X          memory bandwidth in GB/s to report percentages against (6700 XT: 384)\n");
    printf("  --time-ms X            run time of each measurement (default 1500)\n");
    printf("  --calib                also measure a read-only and a copy kernel as a bandwidth reference\n");
    printf("  --check                compare the device result with the CPU backend for every type and column count\n");
    printf("  --csv                  machine readable output\n");
}

static bool parse_args(int argc, char ** argv, bench_params & p) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", what);
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            usage(argv[0]);
            exit(0);
        } else if (a == "--list") {
            p.list = true;
        } else if (a == "-d" || a == "--device") {
            p.device = next("--device");
        } else if (a == "-t" || a == "--types") {
            p.types.clear();
            for (const auto & s : split(next("--types"), ',')) {
                ggml_type t;
                if (!parse_type(s, t)) {
                    fprintf(stderr, "unknown type '%s'\n", s.c_str());
                    return false;
                }
                p.types.push_back(t);
            }
        } else if (a == "-n" || a == "--cols") {
            p.n_cols.clear();
            for (const auto & s : split(next("--cols"), ',')) {
                const int n = atoi(s.c_str());
                if (n < 1) {
                    fprintf(stderr, "bad column count '%s'\n", s.c_str());
                    return false;
                }
                p.n_cols.push_back(n);
            }
        } else if (a == "-m") {
            p.m = atoll(next("-m"));
        } else if (a == "-k") {
            p.k = atoll(next("-k"));
        } else if (a == "--layers") {
            p.layers = atoi(next("--layers"));
        } else if (a == "--reps") {
            p.reps = std::max(1, atoi(next("--reps")));
        } else if (a == "--min-mb") {
            p.min_mb = atof(next("--min-mb"));
        } else if (a == "--ic-mb") {
            p.ic_mb = atof(next("--ic-mb"));
        } else if (a == "--peak-gbps") {
            p.peak_gbps = atof(next("--peak-gbps"));
        } else if (a == "--time-ms") {
            p.time_ms = atof(next("--time-ms"));
        } else if (a == "--calib") {
            p.calib = true;
        } else if (a == "--check") {
            p.check = true;
        } else if (a == "--csv") {
            p.csv = true;
        } else {
            fprintf(stderr, "unknown argument '%s'\n", a.c_str());
            usage(argv[0]);
            return false;
        }
    }
    if (p.m <= 0 || p.k <= 0 || p.types.empty() || p.n_cols.empty()) {
        return false;
    }
    return true;
}

static const char * dev_type_str(ggml_backend_dev_t dev) {
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "iGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accel";
        default:                             return "?";
    }
}

static void list_devices(FILE * out = stdout) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);
        fprintf(out, "%zu: %s (%s) [%s] %zu MiB free of %zu MiB\n", i, ggml_backend_dev_name(dev),
                ggml_backend_dev_description(dev), dev_type_str(dev), free_b >> 20, total_b >> 20);
    }
}

static ggml_backend_dev_t select_device(const std::string & sel) {
    const size_t n = ggml_backend_dev_count();
    if (sel.empty()) {
        for (size_t i = 0; i < n; i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                return dev;
            }
        }
        for (size_t i = 0; i < n; i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                return dev;
            }
        }
        return n > 0 ? ggml_backend_dev_get(0) : nullptr;
    }
    if (lower(sel) == "cpu") {
        return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    // a number below the device count is an index, anything else is looked up in the names, so that
    // "-d 6700" finds an RX 6700 XT and "-d 0" is the first device
    char * end = nullptr;
    const long idx = strtol(sel.c_str(), &end, 10);
    if (end != sel.c_str() && *end == '\0' && idx >= 0 && (size_t) idx < n) {
        return ggml_backend_dev_get((size_t) idx);
    }
    const std::string want = lower(sel);
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const std::string name = lower(std::string(ggml_backend_dev_name(dev)) + " " + ggml_backend_dev_description(dev));
        if (name.find(want) != std::string::npos) {
            return dev;
        }
    }
    return nullptr;
}

// ---- data -----------------------------------------------------------------------------------------------

// rows of weights with unit rms outputs, quantized once. The bytes are then repeated over the whole tensor: rows
// are independent, and quantizing hundreds of MB on one core would take minutes.
static bool make_weight_chunk(ggml_type type, int64_t k_in, int64_t n_rows, std::mt19937 & rng, std::vector<uint8_t> & out) {
    std::vector<float> f((size_t) (k_in * n_rows));
    const float a = std::sqrt(3.0f / (float) k_in);
    std::uniform_real_distribution<float> dist(-a, a);
    for (float & v : f) {
        v = dist(rng);
    }
    out.assign(ggml_row_size(type, k_in) * (size_t) n_rows, 0);
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(out.data(), f.data(), out.size());
            return true;
        case GGML_TYPE_F16:
            ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) out.data(), (int64_t) f.size());
            return true;
        case GGML_TYPE_BF16:
            ggml_fp32_to_bf16_row(f.data(), (ggml_bf16_t *) out.data(), (int64_t) f.size());
            return true;
        default:
            break;
    }
    if (!ggml_is_quantized(type)) {
        return false;
    }
    if (ggml_quantize_requires_imatrix(type)) {
        fprintf(stderr, "type %s needs an importance matrix to quantize, not supported here\n", ggml_type_name(type));
        return false;
    }
    ggml_quantize_init(type);
    const size_t written = ggml_quantize_chunk(type, f.data(), out.data(), 0, n_rows, k_in, nullptr);
    return written == out.size();
}

static void fill_weights(ggml_tensor * t, const std::vector<uint8_t> & chunk) {
    const size_t total = ggml_nbytes(t);
    size_t off = 0;
    while (off < total) {
        const size_t n = std::min(chunk.size(), total - off);
        ggml_backend_tensor_set(t, chunk.data(), off, n);
        off += n;
    }
}

static std::vector<float> make_activations(int64_t k, int n, std::mt19937 & rng) {
    std::vector<float> x((size_t) (k * n));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float & v : x) {
        v = dist(rng);
    }
    return x;
}

// ---- timing ---------------------------------------------------------------------------------------------

struct timing {
    double median_us = 0.0;
    double min_us    = 0.0;
    int    iters     = 0;
};

static bool time_graph(ggml_backend_t backend, ggml_cgraph * gf, double run_ms, timing & out) {
    for (int i = 0; i < 3; i++) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            return false;
        }
    }
    std::vector<double> t;
    double total_us = 0.0;
    while (total_us < run_ms * 1000.0 || t.size() < 5) {
        const int64_t t0 = ggml_time_us();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            return false;
        }
        const double dt = (double) (ggml_time_us() - t0);
        t.push_back(dt);
        total_us += dt;
        if (t.size() > 100000) {
            break;
        }
    }
    std::sort(t.begin(), t.end());
    out.median_us = t[t.size() / 2];
    out.min_us    = t.front();
    out.iters     = (int) t.size();
    return true;
}

// ---- the FFN shapes -------------------------------------------------------------------------------------

struct shape_result {
    bool   ok = false;
    double us_per_op = 0.0;   // median time of one op (one pair for the pair chain)
    size_t bytes_per_op = 0;  // weight bytes one op streams
    double gbps() const { return us_per_op > 0.0 ? (double) bytes_per_op / (us_per_op * 1e3) : 0.0; }
};

struct bench_result {
    bool         ok = false;
    int          layers = 0;
    size_t       up_bytes = 0, down_bytes = 0;
    shape_result pair, up, down;
};

enum graph_kind { KIND_PAIR, KIND_UP, KIND_DOWN };

// builds, runs and times one graph over the weights up[] / down[]
static shape_result run_graph(ggml_backend_t backend, const bench_params & p, graph_kind kind, int n, int layers,
        const std::vector<ggml_tensor *> & up, const std::vector<ggml_tensor *> & down, std::mt19937 & rng) {
    shape_result res;
    const int n_ops = (kind == KIND_PAIR ? 2 : 1) * layers * p.reps;
    ggml_init_params gparams = { ggml_tensor_overhead() * (size_t) (2 * n_ops + 16) + ggml_graph_overhead_custom((size_t) n_ops + 16, false), nullptr, true };
    ggml_context * ctx = ggml_init(gparams);
    const int64_t k_in = kind == KIND_DOWN ? p.m : p.k;
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_in, n);
    ggml_set_input(x);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, (size_t) n_ops + 16, false);
    if (kind == KIND_PAIR) {
        ggml_tensor * cur = x;
        for (int r = 0; r < p.reps; r++) {
            for (int l = 0; l < layers; l++) {
                cur = ggml_mul_mat(ctx, up[l], cur);
                cur = ggml_mul_mat(ctx, down[l], cur);
            }
        }
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
    } else {
        // independent ops. Every result is marked as an output so the allocator gives each its own buffer,
        // otherwise they would share one and the backend would have to order the writes
        for (int r = 0; r < p.reps; r++) {
            for (int l = 0; l < layers; l++) {
                ggml_tensor * o = ggml_mul_mat(ctx, kind == KIND_UP ? up[l] : down[l], x);
                ggml_set_output(o);
                ggml_build_forward_expand(gf, o);
            }
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fprintf(stderr, "failed to allocate the graph tensors\n");
        ggml_free(ctx);
        return res;
    }
    const std::vector<float> xdata = make_activations(k_in, n, rng);
    ggml_backend_tensor_set(x, xdata.data(), 0, xdata.size() * sizeof(float));

    timing t;
    if (time_graph(backend, gf, p.time_ms, t)) {
        res.ok = true;
        res.us_per_op = t.median_us / ((double) layers * p.reps);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

static bench_result run_bench(ggml_backend_t backend, const bench_params & p, ggml_type type, int n, std::mt19937 & rng,
        std::vector<uint8_t> & chunk_up, std::vector<uint8_t> & chunk_down, bool verbose_tensors) {
    bench_result res;

    const size_t row_up   = ggml_row_size(type, p.k); // up:   [k, m], m rows
    const size_t row_down = ggml_row_size(type, p.m); // down: [m, k], k rows
    res.up_bytes   = row_up   * (size_t) p.m;
    res.down_bytes = row_down * (size_t) p.k;
    const size_t pair_bytes = res.up_bytes + res.down_bytes;
    int layers = p.layers;
    if (layers <= 0) {
        layers = std::max(2, (int) std::ceil(p.min_mb * 1024.0 * 1024.0 / (double) pair_bytes));
    }
    res.layers = layers;

    // the up and down measurements stream one shape each, that is what has to be several times the cache
    const double shape_mb = (double) std::min(res.up_bytes, res.down_bytes) * layers / (1024.0 * 1024.0);
    if (verbose_tensors && shape_mb < 3.0 * p.ic_mb) {
        fprintf(stderr, "warning: %.0f MB of weights per shape is less than 3x the %.0f MB cache, "
                        "results may be inflated by cache hits (raise --min-mb or --layers)\n", shape_mb, p.ic_mb);
    }

    // weights
    ggml_init_params wparams = { ggml_tensor_overhead() * (size_t) (2 * layers + 8), nullptr, true };
    ggml_context * ctx_w = ggml_init(wparams);
    std::vector<ggml_tensor *> up(layers), down(layers);
    for (int l = 0; l < layers; l++) {
        up[l]   = ggml_new_tensor_2d(ctx_w, type, p.k, p.m);
        down[l] = ggml_new_tensor_2d(ctx_w, type, p.m, p.k);
        ggml_format_name(up[l],   "up.%d",   l);
        ggml_format_name(down[l], "down.%d", l);
    }
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (buf_w == nullptr) {
        fprintf(stderr, "failed to allocate %.0f MB of weights\n", (double) pair_bytes * layers / (1024.0 * 1024.0));
        ggml_free(ctx_w);
        return res;
    }
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int l = 0; l < layers; l++) {
        fill_weights(up[l],   chunk_up);
        fill_weights(down[l], chunk_down);
    }

    res.up   = run_graph(backend, p, KIND_UP,   n, layers, up, down, rng);
    res.down = run_graph(backend, p, KIND_DOWN, n, layers, up, down, rng);
    res.pair = run_graph(backend, p, KIND_PAIR, n, layers, up, down, rng);
    res.up.bytes_per_op   = res.up_bytes;
    res.down.bytes_per_op = res.down_bytes;
    res.pair.bytes_per_op = pair_bytes;
    res.ok = res.up.ok && res.down.ok && res.pair.ok;

    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    return res;
}

// ---- correctness against the CPU backend ---------------------------------------------------------------

static double nmse(const std::vector<float> & a, const std::vector<float> & ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) ref[i];
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
    }
    return den > 0.0 ? num / den : num;
}

// one mat-vec on `backend` and on the CPU backend with the same weights and activations
static bool check_one(ggml_backend_t backend, ggml_backend_t cpu, ggml_type type, int64_t k_in, int64_t m_out, int n,
        const std::vector<uint8_t> & chunk, std::mt19937 & rng, double & err) {
    auto run = [&](ggml_backend_t be, const std::vector<float> & xin, std::vector<float> & yout) -> bool {
        ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, type, k_in, m_out);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_in, n);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        bool ok = buf != nullptr;
        if (ok) {
            fill_weights(w, chunk);
            ggml_backend_tensor_set(x, xin.data(), 0, xin.size() * sizeof(float));
            ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
            if (ok) {
                yout.resize((size_t) (m_out * n));
                ggml_backend_tensor_get(y, yout.data(), 0, yout.size() * sizeof(float));
            }
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        return ok;
    };
    const std::vector<float> xin = make_activations(k_in, n, rng);
    std::vector<float> y_dev, y_cpu;
    if (!run(backend, xin, y_dev) || !run(cpu, xin, y_cpu)) {
        return false;
    }
    err = nmse(y_dev, y_cpu);
    return true;
}

// ---- bandwidth reference ----------------------------------------------------------------------------------

static void run_calibration(ggml_backend_t backend, const bench_params & p, std::mt19937 & rng) {
    // 6 tensors of 128 MB: 768 MB, well beyond the cache
    const int64_t ne0 = 8192, ne1 = 4096;
    const int R = 6;
    const double tensor_mb = (double) (ne0 * ne1 * (int64_t) sizeof(float)) / 1e6;

    for (int mode = 0; mode < 2; mode++) {
        const bool copy = mode == 1;
        ggml_init_params params = { ggml_tensor_overhead() * (size_t) (4 * R + 8) + ggml_graph_overhead_custom(64, false), nullptr, true };
        ggml_context * ctx = ggml_init(params);
        std::vector<ggml_tensor *> src(R), dst(R);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
        for (int i = 0; i < R; i++) {
            src[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
            if (copy) {
                dst[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, src[i], dst[i]));
            } else {
                ggml_build_forward_expand(gf, ggml_sum_rows(ctx, src[i]));
            }
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == nullptr) {
            fprintf(stderr, "calibration: failed to allocate %.0f MB\n", tensor_mb * R * (copy ? 2 : 1));
            ggml_free(ctx);
            continue;
        }
        const std::vector<float> data = make_activations(ne0, 1, rng);
        std::vector<float> row_data((size_t) (ne0 * ne1));
        for (int64_t r = 0; r < ne1; r++) {
            memcpy(row_data.data() + (size_t) (r * ne0), data.data(), (size_t) ne0 * sizeof(float));
        }
        for (int i = 0; i < R; i++) {
            ggml_backend_tensor_set(src[i], row_data.data(), 0, row_data.size() * sizeof(float));
        }
        timing t;
        if (time_graph(backend, gf, p.time_ms, t)) {
            const double bytes = tensor_mb * 1e6 * R * (copy ? 2.0 : 1.0); // a copy reads and writes
            const double gbps  = bytes / (t.median_us * 1e3);
            if (p.csv) {
                printf("calib,%s,,,,%.1f,%.1f,%.1f\n", copy ? "copy_read_write" : "sum_rows_read", t.median_us, gbps,
                       p.peak_gbps > 0 ? 100.0 * gbps / p.peak_gbps : 0.0);
            } else {
                printf("  %-22s %8.1f GB/s", copy ? "copy (read + write)" : "sum_rows (read only)", gbps);
                if (p.peak_gbps > 0) {
                    printf("  (%.0f%% of %.0f)", 100.0 * gbps / p.peak_gbps, p.peak_gbps);
                }
                printf("\n");
            }
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    bench_params p;
    if (!parse_args(argc, argv, p)) {
        usage(argv[0]);
        return 1;
    }
    if (p.list) {
        list_devices();
        return 0;
    }

    ggml_backend_dev_t dev = select_device(p.device);
    if (dev == nullptr) {
        fprintf(stderr, "device '%s' not found, devices:\n", p.device.c_str());
        list_devices(stderr);
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "failed to initialize %s\n", ggml_backend_dev_name(dev));
        return 1;
    }
    const bool is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_t cpu = nullptr;
    if (p.check && is_cpu) {
        fprintf(stderr, "note: --check compares a device with the CPU backend, nothing to compare with the CPU itself\n");
    }
    if (p.check && !is_cpu) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (cpu == nullptr) {
            fprintf(stderr, "--check needs the CPU backend\n");
            return 1;
        }
    }

    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    if (p.csv) {
        printf("kind,type,n,layers,weights_mb,us_per_op,gb_per_s,pct_peak\n");
    } else {
        printf("device: %s (%s), %zu MiB free\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), free_b >> 20);
        printf("shape : up %" PRId64 " x %" PRId64 " (K=%" PRId64 "), down %" PRId64 " x %" PRId64 " (K=%" PRId64 "), x%d reps per graph\n",
               p.m, p.k, p.k, p.k, p.m, p.m, p.reps);
        printf("GB/s  : weight bytes per second, 1 GB = 1e9 bytes, median of the graph runs\n");
        printf("up/down: each shape alone, independent ops (streaming). pair: up then down, dependent ops (decode-like)\n\n");
    }

    std::mt19937 rng(1234);
    int failures = 0;

    for (ggml_type type : p.types) {
        std::vector<uint8_t> chunk_up, chunk_down;
        const int64_t chunk_rows = 256;
        if (!make_weight_chunk(type, p.k, chunk_rows, rng, chunk_up) ||
            !make_weight_chunk(type, p.m, chunk_rows, rng, chunk_down)) {
            fprintf(stderr, "skipping type %s\n", ggml_type_name(type));
            continue;
        }
        if ((p.k % ggml_blck_size(type)) != 0 || (p.m % ggml_blck_size(type)) != 0) {
            fprintf(stderr, "skipping type %s, the shape is not a multiple of its block size %" PRId64 "\n",
                    ggml_type_name(type), (int64_t) ggml_blck_size(type));
            continue;
        }

        if (!p.csv) {
            printf("%s  (%.3f bits per weight)\n", ggml_type_name(type), 8.0 * ggml_type_size(type) / ggml_blck_size(type));
            printf("  %2s %6s %6s | %8s %7s %5s | %8s %7s %5s | %9s %7s %5s\n", "n", "layers", "MB",
                   "up us", "GB/s", "%pk", "down us", "GB/s", "%pk", "pair us", "GB/s", "%pk");
        }
        bool first = true;
        for (int n : p.n_cols) {
            const bench_result r = run_bench(backend, p, type, n, rng, chunk_up, chunk_down, first);
            first = false;
            if (!r.ok) {
                fprintf(stderr, "  n=%d failed\n", n);
                continue;
            }
            const double mb = (double) (r.up_bytes + r.down_bytes) * r.layers / 1e6;
            auto pct = [&](const shape_result & s) { return p.peak_gbps > 0 ? 100.0 * s.gbps() / p.peak_gbps : 0.0; };
            if (p.csv) {
                const struct { const char * kind; const shape_result * s; } rows[] = {
                    { "up", &r.up }, { "down", &r.down }, { "pair", &r.pair } };
                for (const auto & row : rows) {
                    printf("%s,%s,%d,%d,%.0f,%.1f,%.1f,%.1f\n", row.kind, ggml_type_name(type), n, r.layers, mb,
                           row.s->us_per_op, row.s->gbps(), pct(*row.s));
                }
            } else {
                printf("  %2d %6d %6.0f | %8.1f %7.1f %4.0f%% | %8.1f %7.1f %4.0f%% | %9.1f %7.1f %4.0f%%\n", n, r.layers, mb,
                       r.up.us_per_op, r.up.gbps(), pct(r.up), r.down.us_per_op, r.down.gbps(), pct(r.down),
                       r.pair.us_per_op, r.pair.gbps(), pct(r.pair));
            }
            fflush(stdout);

            if (cpu != nullptr) {
                double e_up = 0.0, e_down = 0.0;
                const bool ok_up   = check_one(backend, cpu, type, p.k, p.m, n, chunk_up,   rng, e_up);
                const bool ok_down = check_one(backend, cpu, type, p.m, p.k, n, chunk_down, rng, e_down);
                const double limit = 5e-4; // the threshold test-backend-ops uses for MUL_MAT
                const bool pass = ok_up && ok_down && e_up <= limit && e_down <= limit;
                if (!pass) {
                    failures++;
                }
                printf("     check vs CPU: up nmse %.2e, down nmse %.2e  %s\n", e_up, e_down, pass ? "OK" : "FAIL");
            }
        }
        if (!p.csv) {
            printf("\n");
        }
    }

    if (p.calib) {
        if (!p.csv) {
            printf("bandwidth reference (768 MB working set, lower bounds of what the card can stream):\n");
        }
        run_calibration(backend, p, rng);
    }

    ggml_backend_free(backend);
    if (cpu != nullptr) {
        ggml_backend_free(cpu);
    }
    return failures == 0 ? 0 : 2;
}
