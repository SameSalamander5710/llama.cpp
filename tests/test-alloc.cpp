#include "ggml-alloc.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

//
// dummy backend with configurable max_buffer_size, tracks allocations

uint8_t * const alloc_base = (uint8_t *) 16;

struct dummy_backend_context {
    size_t max_buffer_size = 64;
    size_t alignment       = 8;
    bool   is_host         = true;
    enum ggml_backend_dev_type type = GGML_BACKEND_DEVICE_TYPE_CPU;
    int    allocs_until_fail = 0; // the Nth buffer allocation from now fails, 0 = never

    ggml_backend_buffer_i              buffer_interface;
    ggml_backend_device                device;
    ggml_backend                       backend;
    std::vector<ggml_backend_buffer_t> buffers;

    size_t allocated_total() const {
        size_t n = 0;
        for (ggml_backend_buffer_t buf : buffers) {
            n += ggml_backend_buffer_get_size(buf);
        }
        return n;
    }
};

// ggml_backend_buffer_type interface

static const char * dummy_backend_buffer_type_get_name(ggml_backend_buffer_type_t) {
    return "dummy_buffer_type";
}

static ggml_backend_buffer_t dummy_backend_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    dummy_backend_context * ctx    = (dummy_backend_context *) buft->context;
    if (ctx->allocs_until_fail > 0 && --ctx->allocs_until_fail == 0) {
        return nullptr;
    }
    ggml_backend_buffer_t & buffer = ctx->buffers.emplace_back();
    buffer                         = ggml_backend_buffer_init(buft, ctx->buffer_interface, ctx, size);
    return buffer;
}

static size_t dummy_backend_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->alignment;
}

static size_t dummy_backend_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->max_buffer_size;
}

static bool dummy_backend_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->is_host;
}

// ggml_backend_buffer interface

static void dummy_backend_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    dummy_backend_context * ctx = (dummy_backend_context *) buffer->context;

    auto i = std::find(ctx->buffers.begin(), ctx->buffers.end(), buffer);
    GGML_ASSERT(i != ctx->buffers.end());
    ctx->buffers.erase(i);
}

static void * dummy_backend_buffer_get_base(ggml_backend_buffer_t) {
    return alloc_base;
}

static ggml_status dummy_backend_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor *) {
    return GGML_STATUS_SUCCESS;
}

static void dummy_backend_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor *, uint8_t, size_t, size_t) {}

static void dummy_backend_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t) {}

static void dummy_backend_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor *, void *, size_t, size_t) {}

static void dummy_backend_buffer_clear(ggml_backend_buffer_t, uint8_t) {}

// ggml_backend_device interface

static enum ggml_backend_dev_type dummy_backend_device_get_type(ggml_backend_dev_t device) {
    return ((dummy_backend_context *) device->context)->type;
}

static bool dummy_backend_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return true;
}

static bool dummy_backend_device_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return device->context == buft->context;
}

// mirror the I/O-heavy ops the accelerator backends offload, so MUL_MAT nodes
// leave the host weights they read on the host backend and end up on device
static bool dummy_backend_device_offload_op(ggml_backend_dev_t, const ggml_tensor * op) {
    return op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID ||
           op->op == GGML_OP_GET_ROWS || op->op == GGML_OP_OUT_PROD;
}

// ggml_backend interface

static const char * dummy_backend_get_name(ggml_backend_t) {
    return "dummy_backend";
}

// dummy_backend

struct dummy_backend {
    std::unique_ptr<dummy_backend_context> context;
    ggml_backend_buffer_type               buffer_type;
};

static dummy_backend dummy_backend_init(size_t max_buffer_size, size_t alignment = 8) {
    dummy_backend b{};
    b.context                  = std::make_unique<dummy_backend_context>();
    b.context->alignment       = alignment;
    b.context->max_buffer_size = max_buffer_size;

    b.context->buffer_interface.free_buffer   = dummy_backend_buffer_free_buffer;
    b.context->buffer_interface.get_base      = dummy_backend_buffer_get_base;
    b.context->buffer_interface.init_tensor   = dummy_backend_buffer_init_tensor;
    b.context->buffer_interface.memset_tensor = dummy_backend_buffer_memset_tensor;
    b.context->buffer_interface.set_tensor    = dummy_backend_buffer_set_tensor;
    b.context->buffer_interface.get_tensor    = dummy_backend_buffer_get_tensor;
    b.context->buffer_interface.clear         = dummy_backend_buffer_clear;

    b.context->device.context             = b.context.get();
    b.context->device.iface.get_type      = dummy_backend_device_get_type;
    b.context->device.iface.supports_op   = dummy_backend_device_supports_op;
    b.context->device.iface.supports_buft = dummy_backend_device_supports_buft;
    b.context->device.iface.offload_op    = dummy_backend_device_offload_op;

    b.context->backend.context        = b.context.get();
    b.context->backend.device         = &b.context->device;
    b.context->backend.iface.get_name = dummy_backend_get_name;

    b.buffer_type.device              = &b.context->device;
    b.buffer_type.context             = b.context.get();
    b.buffer_type.iface.get_name      = dummy_backend_buffer_type_get_name;
    b.buffer_type.iface.alloc_buffer  = dummy_backend_buffer_type_alloc_buffer;
    b.buffer_type.iface.get_alignment = dummy_backend_buffer_type_get_alignment;
    b.buffer_type.iface.get_max_size  = dummy_backend_buffer_type_get_max_size;
    b.buffer_type.iface.is_host       = dummy_backend_buffer_type_is_host;
    return b;
}

//
// test utilities

struct test_context_with_graph {
    ggml_context *   ctx;
    ggml_cgraph *    graph;
    ggml_context_ptr ctx_ptr;
};

static test_context_with_graph make_context() {
    ggml_init_params params{};
    params.mem_size = 48 * ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;

    ggml_context *   ctx     = ggml_init(params);
    ggml_context_ptr ctx_ptr = ggml_context_ptr(ctx);
    ggml_cgraph *    graph   = ggml_new_graph(ctx);
    return { ctx, graph, std::move(ctx_ptr) };
}

static ggml_tensor * make_input_1d(ggml_context * ctx, int64_t n_elements) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_set_input(t);
    return t;
}

static ggml_tensor * make_input_with_size(ggml_context * ctx, size_t size_bytes) {
    GGML_ASSERT(size_bytes % 4 == 0);
    return make_input_1d(ctx, size_bytes / 4);
}

static void assign_names(ggml_context * ctx, const char * prefix = "x") {
    int i = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        ggml_format_name(t, "%s%d", prefix, i++);
    }
}

static int get_leaf_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_leafs; ++i) {
        if (strncmp(graph->leafs[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "leaf not found: %s\n", tensor_name);
    return -1;
}

static int get_node_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (strncmp(graph->nodes[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "node not found: %s", tensor_name);
    return -1;
}

static ggml_gallocr_ptr allocate_graph(ggml_cgraph * graph, ggml_tensor * out, ggml_backend_buffer_type_t buft) {
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_ptr galloc = ggml_gallocr_ptr(ggml_gallocr_new(buft));
    bool             result = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(result);
    return galloc;
}

//
// correctness checks for result allocations

static void check_all_allocated(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        GGML_ASSERT(t->buffer != nullptr);
        GGML_ASSERT(t->data != nullptr);
    }
}

static void check_max_size(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        auto   buft     = ggml_backend_buffer_get_type(t->buffer);
        size_t max_size = ggml_backend_buft_get_max_size(buft);
        size_t offset   = (char *) t->data - (char *) ggml_backend_buffer_get_base(t->buffer);
        GGML_ASSERT(t->data >= ggml_backend_buffer_get_base(t->buffer));
        GGML_ASSERT((size_t) offset + ggml_nbytes(t) <= max_size);
    }
}

static bool can_reuse_memory(ggml_cgraph * graph, int current_i, ggml_tensor * current, ggml_tensor * other) {
    if (other->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    // Check if `other` is still "alive", ie. an input to any node after the `current` op
    for (int i = current_i; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (t == current && ggml_op_can_inplace(t->op)) {
                continue;
            }
            if (t->src[s] == other) {
                return false;
            }
            if (t->src[s] && t->src[s]->view_src == other) {
                return false;
            }
        }
    }
    return true;
}

static bool memory_overlap(ggml_tensor * a, ggml_tensor * b) {
    if (a->buffer != b->buffer) {
        return false;
    }
    int64_t a0 = (int64_t) a->data;
    int64_t a1 = a0 + ggml_nbytes(a);
    int64_t b0 = (int64_t) b->data;
    int64_t b1 = b0 + ggml_nbytes(b);
    return a1 > b0 && b1 > a0;
}

static ggml_tensor * get_view_source(ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

static void check_no_overlap(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        for (int j = 0; j < i; ++j) {
            ggml_tensor * t = ggml_graph_node(graph, i);
            ggml_tensor * o = ggml_graph_node(graph, j);
            GGML_ASSERT(t != o);

            if (get_view_source(t) == get_view_source(o)) {
                continue;
            }
            if (memory_overlap(t, o)) {
                GGML_ASSERT(can_reuse_memory(graph, i, t, o));
            }
        }
    }
}

//
// test cases

// Scenario where the first backend buffer is completely exhausted and there are further
// tensors which require a second buffer
static void test_max_size_too_many_tensors() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[7];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_mul(ctx, x[0], x[1]);
    x[4] = ggml_add(ctx, x[1], x[2]);
    x[5] = ggml_add(ctx, x[3], x[0]);
    x[6] = ggml_add(ctx, x[4], x[5]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[6], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 16 + 16);
}

// Scenario where there is some space left in the first buffer, but not enough to accommodate
// a larger tensor, so a second buffer is required
static void test_max_size_tensor_too_large() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);    // chunk 0, [0 , 16)
    x[1] = make_input_with_size(ctx, 8);     // chunk 0, [16, 24)
    x[2] = ggml_concat(ctx, x[0], x[1], 0);  // chunk 1, [0 , 24)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 24);
}

// Scenario where a single tensor exceeds the max buffer size - in this case the allocator
// should try to create a bigger buffer anyway, and wait for the backend to throw an error.
// Backends may report an artificially lower max size in some cases for compatibility reasons.
static void test_tensor_larger_than_max_size() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[1], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() == 24);
}

// This test assumes a max of 16 buffer chunks, and tries to allocate tensors that would
// require more. Expectation is that the last buffer should grow to fit everything,
// leaving it to the backend to error out if it can't allocate that much.
static void test_not_enough_chunks() {
    const int max_chunks = 16;
    const int max_size   = 8;

    dummy_backend backend      = dummy_backend_init(max_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[max_chunks + 1];
    for (int i = 0; i < max_chunks + 1; ++i) {
        x[i] = make_input_with_size(ctx, max_size);
    }
    ggml_tensor * acc = x[0];
    for (int i = 0; i < max_chunks; ++i) {
        acc = ggml_add(ctx, acc, x[i + 1]);
    }
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, acc, &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() > max_chunks * max_size);
}

// Fill up leftover unallocated space of a chunk after allocating a large tensor that
// requires a new chunk.
static void test_fill_leftover_space() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = ggml_pad(ctx, x[0], 2, 0, 0, 0);
    x[3] = ggml_mean(ctx, x[1]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[3], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 12 + 16);
}

// Check that views don't require any extra memory
static void test_view_inplace() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[6];
    x[0] = make_input_1d(ctx, 4);                // chunk 0, [0, 16)
    x[1] = ggml_reshape_2d(ctx, x[0], 2, 2);     // view of x0
    x[2] = ggml_permute(ctx, x[1], 1, 0, 2, 3);  // view of x0
    x[3] = ggml_view_1d(ctx, x[2], 2, 4);        // view of x0
    x[4] = make_input_1d(ctx, 2);                // chunk 0, [16, 24)
    x[5] = ggml_add(ctx, x[3], x[4]);            // reuse (inplace add)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[5], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 24);
}

static void test_reuse_and_free() {
    dummy_backend backend      = dummy_backend_init(40);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_add(ctx, x[1], x[2]);        // reuse, free x2
    x[4] = ggml_pad(ctx, x[0], 2, 0, 0, 0);  // alloc new buffer, free x0
    x[5] = ggml_scale(ctx, x[4], 2.0f);      // alloc from free block
    x[6] = ggml_add(ctx, x[4], x[5]);        // reuse, free x5
    x[7] = ggml_view_1d(ctx, x[6], 2, 8);    // view
    x[8] = ggml_add(ctx, x[3], x[7]);        // reuse
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 40 + 32 + 32);
}

static void test_merge_free_block(size_t max_buffer_size) {
    dummy_backend backend      = dummy_backend_init(max_buffer_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = make_input_with_size(ctx, 16);
    x[3] = ggml_mean(ctx, x[0]);
    x[4] = ggml_mean(ctx, x[1]);
    x[5] = ggml_pad(ctx, x[2], 2, 0, 0, 0);
    x[6] = ggml_add(ctx, x[3], x[4]);
    x[7] = ggml_pad(ctx, x[6], 5, 0, 0, 0);
    x[8] = ggml_add(ctx, x[5], x[7]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 32 + 24);
}

// Check that previously allocated but freed memory is preferred over allocating
// additional memory, even if the remaining space in a chunk would match tensor size better
static void test_prefer_already_allocated_memory() {
    dummy_backend backend      = dummy_backend_init(32, /*align*/ 4);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 24);  // [24b][8b unused]
    x[1] = ggml_mean(ctx, x[0]);           // [24b free][4b][4b unused]
    x[2] = ggml_mean(ctx, x[1]);           // should be allocated in the 24b block
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() <= 28);
}

// test for allocating on multiple devices with some tensors in the graph
// allocated externally (not by gallocr).
static void test_multiple_buffer_types() {
    dummy_backend backend_a = dummy_backend_init(32);
    dummy_backend backend_b = dummy_backend_init(SIZE_MAX);

    auto [ctx_a, _a, ctx_a_ptr] = make_context();
    auto [ctx_b, _b, ctx_b_ptr] = make_context();
    auto [ctx, graph, ctx_ptr]  = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx_a, 16);
    a[1] = make_input_with_size(ctx_a, 16);
    assign_names(ctx_a, "a");

    ggml_tensor * b[2];
    b[0] = make_input_with_size(ctx_b, 24);
    b[1] = make_input_with_size(ctx_b, 4);
    assign_names(ctx_b, "b");

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_mul(ctx, x[0], a[0]);
    x[2] = ggml_pad(ctx, x[1], 2, 0, 0, 0);
    x[3] = ggml_mul(ctx, x[2], b[0]);
    x[4] = ggml_mean(ctx, x[3]);
    x[5] = ggml_add(ctx, x[4], b[1]);
    x[6] = ggml_pad(ctx, x[5], 3, 0, 0, 0);
    x[7] = ggml_add(ctx, x[6], a[1]);
    x[8] = ggml_scale(ctx, x[7], 2.0f);
    assign_names(ctx, "x");

    ggml_backend_buffer_ptr    buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx_a, &backend_a.buffer_type));
    ggml_backend_buffer_ptr    buf_b(ggml_backend_alloc_ctx_tensors_from_buft(ctx_b, &backend_b.buffer_type));
    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };

    // assign buffer types manually to avoid extra complexity from backend scheduler
    ggml_set_output(x[8]);
    ggml_build_forward_expand(graph, x[8]);

    GGML_ASSERT(graph->n_leafs == 5);
    int leaf_buffer_ids[5];
    leaf_buffer_ids[get_leaf_id(graph, "a0")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "a1")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "b0")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "b1")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "x0")] = 0;

    GGML_ASSERT(graph->n_nodes == 8);
    int node_buffer_ids[8];
    node_buffer_ids[get_node_id(graph, "x1")] = 0;
    node_buffer_ids[get_node_id(graph, "x2")] = 0;
    node_buffer_ids[get_node_id(graph, "x3")] = 1;
    node_buffer_ids[get_node_id(graph, "x4")] = 1;
    node_buffer_ids[get_node_id(graph, "x5")] = 1;
    node_buffer_ids[get_node_id(graph, "x6")] = 1;
    node_buffer_ids[get_node_id(graph, "x7")] = 0;
    node_buffer_ids[get_node_id(graph, "x8")] = 0;

    ggml_gallocr_ptr galloc(ggml_gallocr_new_n(bufts, 2));
    ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    ggml_gallocr_alloc_graph(galloc.get(), graph);

    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend_a.context->allocated_total() <= 32 + 32 + 24);
    GGML_ASSERT(backend_b.context->allocated_total() <= 32 + 24);
}

static void test_buffer_size_zero() {
    dummy_backend backend_a    = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_b    = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);

    ggml_set_output(x[1]);
    ggml_build_forward_expand(graph, x[1]);

    int leaf_buffer_ids[1] = { 0 };
    int node_buffer_ids[1] = { 0 };

    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };
    ggml_gallocr_ptr           galloc   = ggml_gallocr_ptr(ggml_gallocr_new_n(bufts, 2));
    bool                       res1     = ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    bool                       res2     = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(res1 && res2);

    check_all_allocated(graph);
    GGML_ASSERT(backend_a.context->allocated_total() == 16);
    GGML_ASSERT(backend_b.context->allocated_total() == 0);
}

// Test re-using gallocr for a different graph. The new graph has the same
// total size, but one of the chunks is larger, so reallocation is required.
static void test_reallocation() {
    dummy_backend    backend = dummy_backend_init(32, /*align*/ 4);
    ggml_gallocr_ptr galloc;
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[4];
        x[0] = make_input_with_size(ctx, 24);
        x[1] = make_input_with_size(ctx, 16);
        x[2] = ggml_view_1d(ctx, x[0], 4, 0);
        x[3] = ggml_add(ctx, x[2], x[1]);
        assign_names(ctx);

        galloc = allocate_graph(graph, x[3], &backend.buffer_type);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[3];
        x[0] = make_input_with_size(ctx, 20);
        x[1] = make_input_with_size(ctx, 20);
        x[2] = ggml_add(ctx, x[0], x[1]);
        assign_names(ctx);
        ggml_set_output(x[2]);
        ggml_build_forward_expand(graph, x[2]);

        bool result = ggml_gallocr_alloc_graph(galloc.get(), graph);
        GGML_ASSERT(result);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
}

static void test_backend_graph_optimize(ggml_backend_t, ggml_cgraph * graph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(graph->n_nodes == 3);
    params->add_alloc_dep(params->user_data, graph->nodes[0], graph->nodes[2]);
}

static bool graph_reuses_allocation(bool add_alloc_dep) {
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    x[2] = ggml_scale(ctx, x[1], 2.0f);
    x[3] = ggml_scale(ctx, x[2], 2.0f);

    ggml_set_output(x[3]);
    ggml_build_forward_expand(graph, x[3]);

    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    if (add_alloc_dep) {
        backend.context->backend.iface.graph_optimize = test_backend_graph_optimize;
    }

    ggml_backend_t             backend_ptr = &backend.context->backend;
    ggml_backend_buffer_type_t buft        = &backend.buffer_type;
    ggml_backend_sched_ptr     sched(ggml_backend_sched_new(&backend_ptr, &buft, 1, 8, false, true));
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

    return x[1]->data == x[2]->data;
}

static void test_graph_optimize_alloc_dep() {
    GGML_ASSERT(graph_reuses_allocation(false));
    GGML_ASSERT(!graph_reuses_allocation(true));
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : ""); // an empty value removes the variable
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// Host-resident weights read by two device splits must be staged into two
// non-overlapping slots of the device prefetch buffer. The device backend has a
// non-host buffer type, the host backend owns the weights in a buffer pinned by
// the device (see test_buffer_is_pinned) — the generic candidate does not stage
// weights the consuming device cannot DMA without a bounce. MUL_MAT ops are
// offloaded to the device backend, so the weights must be staged across the
// backends; the host-resident intermediate between the two muls forces the
// graph into a device/host/device split sequence.
static void test_prefetch_staging_slots() {
    dummy_backend backend_device = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_host   = dummy_backend_init(SIZE_MAX);
    backend_device.context->is_host = false;
    backend_device.context->type    = GGML_BACKEND_DEVICE_TYPE_GPU;

    // the generic candidate only stages weights pinned by the consuming device (see
    // test_buffer_is_pinned); claim the host buffer type for backend_device the same way
    // ggml_backend_vk_host_buffer_type_for_device pins Vulkan_Host to its owning device
    ggml_backend_buffer_type pinned_buft = backend_host.buffer_type;
    pinned_buft.device = &backend_device.context->device;

    auto [ctx_x, _x, ctx_x_ptr] = make_context();
    auto [ctx_w, _w, ctx_w_ptr] = make_context();
    auto [ctx_h, _h, ctx_h_ptr] = make_context();
    auto [ctx, graph, ctx_ptr]  = make_context();

    // weights are plain tensors (no INPUT flag on purpose) in a buffer pinned by backend_device
    ggml_tensor * w0 = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, 4, 4);
    ggml_tensor * w1 = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, 4, 4);
    ggml_format_name(w0, "w0");
    ggml_format_name(w1, "w1");

    // activation on the device buffer, plain (unpinned) host intermediate
    ggml_tensor * x0 = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, 4, 1);
    ggml_tensor * h1 = ggml_new_tensor_2d(ctx_h, GGML_TYPE_F32, 4, 1);
    ggml_set_input(x0);
    ggml_set_input(h1);
    ggml_format_name(x0, "x0");
    ggml_format_name(h1, "h1");

    ggml_backend_buffer_ptr buf_x(ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &backend_device.buffer_type));
    ggml_backend_buffer_ptr buf_w(ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &pinned_buft));
    ggml_backend_buffer_ptr buf_h(ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, &backend_host.buffer_type));
    GGML_ASSERT(buf_x && buf_w && buf_h);
    ggml_backend_buffer_set_usage(buf_w.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // interleaved device/host/device splits, each device split reads one weight
    ggml_tensor * m0 = ggml_mul_mat(ctx, w0, x0);  // device split, stages w0
    ggml_tensor * g0 = ggml_add(ctx, w0, h1);      // host split
    ggml_tensor * m1 = ggml_mul_mat(ctx, w1, x0);  // device split, stages w1
    ggml_format_name(m0, "m0");
    ggml_format_name(g0, "g0");
    ggml_format_name(m1, "m1");

    ggml_set_output(m1);
    ggml_build_forward_expand(graph, m0);
    ggml_build_forward_expand(graph, g0);
    ggml_build_forward_expand(graph, m1);
    GGML_ASSERT(graph->n_nodes == 3);
    GGML_ASSERT(graph->n_leafs == 4);

    ggml_backend_t             backend_ptr[2] = { &backend_device.context->backend, &backend_host.context->backend };
    ggml_backend_buffer_type_t bufts[2]       = { &backend_device.buffer_type, &backend_host.buffer_type };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backend_ptr, bufts, 2, 8, false, true));
    ggml_backend_sched_set_prefetch(sched.get(), true);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

    // the graph must read the staged copies out of the device prefetch buffer, not the host weights
    GGML_ASSERT(m0->src[0] != w0);
    GGML_ASSERT(m1->src[0] != w1);
    GGML_ASSERT(ggml_backend_buffer_get_type(m0->src[0]->buffer) == &backend_device.buffer_type);
    GGML_ASSERT(ggml_backend_buffer_get_type(m1->src[0]->buffer) == &backend_device.buffer_type);

    // the two slots must not overlap; with a single weight per split the second
    // copy is exactly one slot_size away from the first, and reusing the same
    // slot would alias the two copies at the same address
    const uint8_t * a0 = (const uint8_t *) m0->src[0]->data;
    const uint8_t * b0 = (const uint8_t *) m1->src[0]->data;
    const size_t an = ggml_nbytes(m0->src[0]);
    const size_t bn = ggml_nbytes(m1->src[0]);
    GGML_ASSERT(a0 != b0);
    GGML_ASSERT(b0 >= a0 + an || a0 >= b0 + bn);
}

// Chained weight-offloading nodes on the same device backend with no
// intervening host-backend node — the shape that real FFN blocks produce
// under full GPU offload (the activation between ffn_up and ffn_down runs
// on the device too). Fusion is bounded by the staging slot in bytes, not by
// node count, so a chain of weights that fits in one slot becomes a single
// split and all of its DMAs hide under the previous split's compute.
static void test_prefetch_bounded_fusion() {
    const int n_weights = 20; // 20 x 64 B, far below the default slot limit

    dummy_backend backend_device = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_host   = dummy_backend_init(SIZE_MAX);
    backend_device.context->is_host = false;
    backend_device.context->type    = GGML_BACKEND_DEVICE_TYPE_GPU;

    // pin the weight buffer type to backend_device; the generic candidate requires it
    // (see test_buffer_is_pinned / test_prefetch_staging_slots)
    ggml_backend_buffer_type pinned_buft = backend_host.buffer_type;
    pinned_buft.device = &backend_device.context->device;

    auto [ctx_w, _w, ctx_w_ptr] = make_context();
    auto [ctx_x, _x, ctx_x_ptr] = make_context();
    auto [ctx,   graph, ctx_ptr] = make_context();

    // weights are plain tensors (no INPUT flag) in a buffer pinned by backend_device
    std::vector<ggml_tensor *> weights(n_weights);
    for (int i = 0; i < n_weights; i++) {
        weights[i] = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, 4, 4);
        ggml_format_name(weights[i], "w%d", i);
    }

    // activation on the device buffer
    ggml_tensor * x = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, 4, 1);
    ggml_set_input(x);
    ggml_format_name(x, "x");

    ggml_backend_buffer_ptr buf_x(ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &backend_device.buffer_type));
    ggml_backend_buffer_ptr buf_w(ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &pinned_buft));
    GGML_ASSERT(buf_x && buf_w);
    ggml_backend_buffer_set_usage(buf_w.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // chain: m0 = mul_mat(w0, x), m1 = mul_mat(w1, m0), ...
    // every MUL_MAT is offloaded to the device backend; the weights are on
    // the host backend and staged via prefetch; no host-backend node between
    // consecutive mul_mats — the discriminating case that the existing
    // test_prefetch_staging_slots does not exercise
    std::vector<ggml_tensor *> muls(n_weights);
    ggml_tensor * prev = x;
    for (int i = 0; i < n_weights; i++) {
        muls[i] = ggml_mul_mat(ctx, weights[i], prev);
        ggml_format_name(muls[i], "m%d", i);
        prev = muls[i];
    }
    ggml_set_output(prev);
    ggml_build_forward_expand(graph, prev);

    // sanity: the graph is a pure chain of weight-offloading nodes with no
    // host-backend intermediate that would force splits on its own
    GGML_ASSERT(graph->n_nodes == n_weights);
    for (int i = 0; i < n_weights; i++) {
        GGML_ASSERT(graph->nodes[i]->op == GGML_OP_MUL_MAT);
    }

    ggml_backend_t             backend_ptr[2] = { &backend_device.context->backend, &backend_host.context->backend };
    ggml_backend_buffer_type_t bufts[2]       = { &backend_device.buffer_type, &backend_host.buffer_type };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backend_ptr, bufts, 2, n_weights * 2, false, true));
    ggml_backend_sched_set_prefetch(sched.get(), true);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

    // the whole chain fits in one staging slot, so it fuses into a single split
    const int n_splits = ggml_backend_sched_get_n_splits(sched.get());
    GGML_ASSERT(n_splits == 1);

    // every weight must be staged into the device prefetch buffer
    std::vector<const uint8_t *> staged(n_weights);
    const size_t n = ggml_nbytes(muls[0]->src[0]);
    for (int i = 0; i < n_weights; i++) {
        GGML_ASSERT(muls[i]->src[0] != weights[i]);
        GGML_ASSERT(ggml_backend_buffer_get_type(muls[i]->src[0]->buffer) == &backend_device.buffer_type);
        staged[i] = (const uint8_t *) muls[i]->src[0]->data;
    }

    // one split means one slot, so the staged copies must not alias each other
    for (int i = 0; i < n_weights; i++) {
        for (int j = i + 1; j < n_weights; j++) {
            GGML_ASSERT(staged[i] + n <= staged[j] || staged[j] + n <= staged[i]);
        }
    }
}

// The staging slot is the only bound left on fusion: a weight that does not fit in
// what the split has staged so far must start a new split instead of overflowing
// the slot. 384 KiB weights give 2 per 1 MiB slot, so 8 weights must split in four.
static void test_prefetch_byte_budget_split() {
    const int n_weights   = 8;
    const int n_per_split = 2;

    dummy_backend backend_device = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_host   = dummy_backend_init(SIZE_MAX);
    backend_device.context->is_host = false;
    backend_device.context->type    = GGML_BACKEND_DEVICE_TYPE_GPU;

    // pin the weight buffer type to backend_device, as in test_prefetch_bounded_fusion
    ggml_backend_buffer_type pinned_buft = backend_host.buffer_type;
    pinned_buft.device = &backend_device.context->device;

    auto [ctx_w, _w, ctx_w_ptr] = make_context();
    auto [ctx_x, _x, ctx_x_ptr] = make_context();
    auto [ctx,   graph, ctx_ptr] = make_context();

    // F32 [4, 4, 6144] is 384 KiB, an exact multiple of the 8 B alignment. ne0 and ne1
    // stay 4 so the weights match the [4, 1, 6144] activation
    std::vector<ggml_tensor *> weights(n_weights);
    for (int i = 0; i < n_weights; i++) {
        weights[i] = ggml_new_tensor_3d(ctx_w, GGML_TYPE_F32, 4, 4, 6144);
        ggml_format_name(weights[i], "w%d", i);
    }

    ggml_tensor * x = ggml_new_tensor_3d(ctx_x, GGML_TYPE_F32, 4, 1, 6144);
    ggml_set_input(x);
    ggml_format_name(x, "x");

    ggml_backend_buffer_ptr buf_x(ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &backend_device.buffer_type));
    ggml_backend_buffer_ptr buf_w(ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &pinned_buft));
    GGML_ASSERT(buf_x && buf_w);
    ggml_backend_buffer_set_usage(buf_w.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<ggml_tensor *> muls(n_weights);
    for (int i = 0; i < n_weights; i++) {
        muls[i] = ggml_mul_mat(ctx, weights[i], x);
        ggml_format_name(muls[i], "m%d", i);
        ggml_set_output(muls[i]);
        ggml_build_forward_expand(graph, muls[i]);
    }
    GGML_ASSERT(graph->n_nodes == n_weights);

    ggml_backend_t             backend_ptr[2] = { &backend_device.context->backend, &backend_host.context->backend };
    ggml_backend_buffer_type_t bufts[2]       = { &backend_device.buffer_type, &backend_host.buffer_type };
    set_env("GGML_SCHED_PREFETCH_MAX_SLOT_MB", "1");
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backend_ptr, bufts, 2, 64, false, true));
    set_env("GGML_SCHED_PREFETCH_MAX_SLOT_MB", nullptr);
    ggml_backend_sched_set_prefetch(sched.get(), true);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

    // the byte budget, not the node count, is what breaks the row of nodes into pairs
    GGML_ASSERT(ggml_backend_sched_get_n_splits(sched.get()) == n_weights / n_per_split);

    // every weight still reaches the device through a staged copy
    std::vector<const uint8_t *> staged(n_weights);
    const size_t n = ggml_nbytes(muls[0]->src[0]);
    for (int i = 0; i < n_weights; i++) {
        GGML_ASSERT(muls[i]->src[0] != weights[i]);
        GGML_ASSERT(ggml_backend_buffer_get_type(muls[i]->src[0]->buffer) == &backend_device.buffer_type);
        staged[i] = (const uint8_t *) muls[i]->src[0]->data;
    }

    // the two copies of one split share a slot and must not overlap; the copies two
    // splits apart land on the same address again, that is the two-slot reuse
    for (int i = 0; i + 1 < n_weights; i++) {
        GGML_ASSERT(staged[i] + n <= staged[i + 1] || staged[i + 1] + n <= staged[i]);
    }
}

// A host buffer is pinned for a device when the device's own host buffer type owns it. The dummy host
// backend's memory stands in for pinned memory once its buffer type is claimed by the device.
static void test_buffer_is_pinned() {
    dummy_backend backend_device = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_host   = dummy_backend_init(SIZE_MAX);
    backend_device.context->is_host = false;
    backend_device.context->type    = GGML_BACKEND_DEVICE_TYPE_GPU;

    ggml_backend_buffer_type pinned_buft = backend_host.buffer_type;
    pinned_buft.device = &backend_device.context->device;

    ggml_backend_dev_t dev_device = &backend_device.context->device;
    ggml_backend_dev_t dev_host   = &backend_host.context->device;

    ggml_backend_buffer_ptr buf_pinned(ggml_backend_buft_alloc_buffer(&pinned_buft, 64));
    ggml_backend_buffer_ptr buf_plain (ggml_backend_buft_alloc_buffer(&backend_host.buffer_type, 64));
    ggml_backend_buffer_ptr buf_device(ggml_backend_buft_alloc_buffer(&backend_device.buffer_type, 64));
    GGML_ASSERT(buf_pinned && buf_plain && buf_device);

    GGML_ASSERT( ggml_backend_buffer_is_pinned(buf_pinned.get(), dev_device));
    GGML_ASSERT(!ggml_backend_buffer_is_pinned(buf_pinned.get(), dev_host));   // the CPU has nothing to pin for
    GGML_ASSERT(!ggml_backend_buffer_is_pinned(buf_plain.get(),  dev_device)); // plain CPU memory
    GGML_ASSERT(!ggml_backend_buffer_is_pinned(buf_device.get(), dev_device)); // not host memory
    GGML_ASSERT(!ggml_backend_buffer_is_pinned(nullptr, dev_device));
}

// MoE prefill with the expert tensors in host memory: three consecutive MUL_MAT_ID nodes (gate, up, down)
// offloaded to the device. Experts pinned by the device are staged in full through the prefetch slots when
// the batch routes to nearly every expert, everything else takes the stock copy path.
struct moe_prefetch_graph {
    static constexpr int n_nodes  = 3;
    static constexpr int n_expert = 8;
    static constexpr int n_used   = 2;

    dummy_backend backend_device = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_host   = dummy_backend_init(SIZE_MAX);
    ggml_backend_buffer_type pinned_buft = {};

    test_context_with_graph w   = make_context();
    test_context_with_graph act = make_context();
    test_context_with_graph g   = make_context();

    ggml_backend_buffer_ptr buf_w;
    ggml_backend_buffer_ptr buf_act;
    ggml_tensor * weights[n_nodes];
    ggml_tensor * x;
    ggml_tensor * ids;
    ggml_tensor * mm[n_nodes];

    moe_prefetch_graph(bool pinned, int n_tokens) {
        backend_device.context->is_host = false;
        backend_device.context->type    = GGML_BACKEND_DEVICE_TYPE_GPU;
        pinned_buft        = backend_host.buffer_type;
        pinned_buft.device = &backend_device.context->device;

        for (int i = 0; i < n_nodes; i++) {
            weights[i] = ggml_new_tensor_3d(w.ctx, GGML_TYPE_F32, 4, 4, n_expert);
            ggml_format_name(weights[i], "exps%d", i);
        }
        x   = ggml_new_tensor_3d(act.ctx, GGML_TYPE_F32, 4, 1, n_tokens);
        ids = ggml_new_tensor_2d(act.ctx, GGML_TYPE_I32, n_used, n_tokens);
        ggml_set_input(x);
        ggml_set_input(ids);
        ggml_format_name(x, "x");
        ggml_format_name(ids, "ids");

        buf_w.reset(ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, pinned ? &pinned_buft : &backend_host.buffer_type));
        buf_act.reset(ggml_backend_alloc_ctx_tensors_from_buft(act.ctx, &backend_device.buffer_type));
        GGML_ASSERT(buf_w && buf_act);
        ggml_backend_buffer_set_usage(buf_w.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        build_graph(g, mm);
    }

    // a graph over the same weights; the scheduler rewrites the nodes of a graph in place, so
    // evaluating the same situation twice needs a fresh one
    void build_graph(test_context_with_graph & gg, ggml_tensor * (&out)[n_nodes]) const {
        ggml_tensor * cur = x;
        for (int i = 0; i < n_nodes; i++) {
            cur = out[i] = ggml_mul_mat_id(gg.ctx, weights[i], cur, ids);
            ggml_format_name(cur, "mm%d", i);
        }
        ggml_set_output(cur);
        ggml_build_forward_expand(gg.graph, cur);
        GGML_ASSERT(gg.graph->n_nodes == n_nodes);
    }

    ggml_backend_sched_ptr make_sched() {
        ggml_backend_t             backends[2] = { &backend_device.context->backend, &backend_host.context->backend };
        ggml_backend_buffer_type_t bufts[2]    = { &backend_device.buffer_type, &backend_host.buffer_type };
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 64, false, true));
        ggml_backend_sched_set_prefetch(sched.get(), true);
        return sched;
    }

    // every expert tensor is read through a device copy, staged or not
    void check_on_device() const {
        for (int i = 0; i < n_nodes; i++) {
            GGML_ASSERT(mm[i]->src[0] != weights[i]);
            GGML_ASSERT(ggml_backend_buffer_get_type(mm[i]->src[0]->buffer) == &backend_device.buffer_type);
        }
    }
    static bool staged(const ggml_tensor * mmid) {
        const char * name = mmid->src[0]->name;
        const size_t n = strlen(name);
        return n >= 3 && strcmp(name + n - 3, "#pf") == 0;
    }
    bool staged(int i) const {
        return staged(mm[i]);
    }
};

static void test_prefetch_moe_pinned_experts() {
    moe_prefetch_graph m(/*pinned=*/true, /*n_tokens=*/16); // 32 routes for 8 experts
    ggml_backend_sched_ptr sched = m.make_sched();
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), m.g.graph));
    // the three experts fit in one slot, so they fuse into a single split
    GGML_ASSERT(ggml_backend_sched_get_n_splits(sched.get()) == 1);

    m.check_on_device();
    for (int i = 0; i < m.n_nodes; i++) {
        GGML_ASSERT(m.staged(i));
    }

    // all three land in the same slot at distinct offsets, so they must not overlap
    const uint8_t * p[m.n_nodes];
    for (int i = 0; i < m.n_nodes; i++) {
        p[i] = (const uint8_t *) m.mm[i]->src[0]->data;
    }
    const size_t n = ggml_nbytes(m.mm[0]->src[0]);
    for (int i = 0; i < m.n_nodes; i++) {
        for (int j = i + 1; j < m.n_nodes; j++) {
            GGML_ASSERT(p[i] + n <= p[j] || p[j] + n <= p[i]);
        }
    }
}

// experts in plain CPU memory are left to the selective copy of the stock path
static void test_prefetch_moe_unpinned_experts() {
    moe_prefetch_graph m(/*pinned=*/false, /*n_tokens=*/16);
    ggml_backend_sched_ptr sched = m.make_sched();
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), m.g.graph));

    m.check_on_device();
    for (int i = 0; i < m.n_nodes; i++) {
        GGML_ASSERT(!m.staged(i));
    }
}

// below two routes per expert the batch leaves experts unused, so the selective copy still saves traffic
static void test_prefetch_moe_small_batch() {
    moe_prefetch_graph m(/*pinned=*/true, /*n_tokens=*/4); // 8 routes for 8 experts
    ggml_backend_sched_ptr sched = m.make_sched();
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), m.g.graph));

    m.check_on_device();
    for (int i = 0; i < m.n_nodes; i++) {
        GGML_ASSERT(!m.staged(i));
    }
}

// staging that exceeds the slot limit must leave a valid graph: the weights fall back to the stock copies
static void test_prefetch_moe_slot_limit() {
    moe_prefetch_graph m(/*pinned=*/true, /*n_tokens=*/16);
    set_env("GGML_SCHED_PREFETCH_MAX_SLOT_MB", "0");
    ggml_backend_sched_ptr sched = m.make_sched();
    set_env("GGML_SCHED_PREFETCH_MAX_SLOT_MB", nullptr);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), m.g.graph));

    m.check_on_device();
    for (int i = 0; i < m.n_nodes; i++) {
        GGML_ASSERT(!m.staged(i));
    }
}

// a staging buffer that cannot be allocated must not leave the graph pointing at host weights
static void test_prefetch_moe_alloc_failure() {
    moe_prefetch_graph m(/*pinned=*/true, /*n_tokens=*/16);
    ggml_backend_sched_ptr sched = m.make_sched();
    m.backend_device.context->allocs_until_fail = 1; // the staging buffer is the first device allocation
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), m.g.graph));

    m.check_on_device();
    for (int i = 0; i < m.n_nodes; i++) {
        GGML_ASSERT(!m.staged(i));
    }
}

// a staging request that was refused is not retried by the next graph, so the outcome and the report
// do not depend on how many times the graph happens to be rebuilt
static void test_prefetch_moe_refusal_is_sticky() {
    moe_prefetch_graph m(/*pinned=*/true, /*n_tokens=*/16);
    ggml_backend_sched_ptr sched = m.make_sched();
    m.backend_device.context->allocs_until_fail = 1;
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), m.g.graph));
    GGML_ASSERT(!m.staged(0));

    ggml_backend_sched_reset(sched.get());
    ggml_backend_sched_set_prefetch(sched.get(), true);
    test_context_with_graph g2 = make_context();
    ggml_tensor * mm2[m.n_nodes];
    m.build_graph(g2, mm2);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), g2.graph)); // allocations succeed again now
    for (int i = 0; i < m.n_nodes; i++) {
        GGML_ASSERT(!m.staged(mm2[i]));
    }
}

static void run(const char * name, void (*f)()) {
    printf("%s ", name);
    fflush(stdout);
    f();
    printf("PASSED\n");
}

int main() {
    run("test_max_size_too_many_tensors", test_max_size_too_many_tensors);
    run("test_max_size_tensor_too_large", test_max_size_tensor_too_large);
    run("test_tensor_larger_than_max_size", test_tensor_larger_than_max_size);
    run("test_not_enough_chunks", test_not_enough_chunks);
    run("test_fill_leftover_space", test_fill_leftover_space);
    run("test_view_inplace", test_view_inplace);
    run("test_reuse_and_free", test_reuse_and_free);
    run("test_merge_free_block(32)", []() { test_merge_free_block(32); });
    run("test_merge_free_block(SIZE_MAX)", []() { test_merge_free_block(SIZE_MAX); });
    run("test_prefer_already_allocated_memory", test_prefer_already_allocated_memory);
    run("test_multiple_buffer_types", test_multiple_buffer_types);
    run("test_buffer_size_zero", test_buffer_size_zero);
    run("test_reallocation", test_reallocation);
    run("test_graph_optimize_alloc_dep", test_graph_optimize_alloc_dep);
    run("test_prefetch_staging_slots", test_prefetch_staging_slots);
    run("test_prefetch_bounded_fusion", test_prefetch_bounded_fusion);
    run("test_prefetch_byte_budget_split", test_prefetch_byte_budget_split);
    run("test_buffer_is_pinned", test_buffer_is_pinned);
    run("test_prefetch_moe_pinned_experts", test_prefetch_moe_pinned_experts);
    run("test_prefetch_moe_unpinned_experts", test_prefetch_moe_unpinned_experts);
    run("test_prefetch_moe_small_batch", test_prefetch_moe_small_batch);
    run("test_prefetch_moe_slot_limit", test_prefetch_moe_slot_limit);
    run("test_prefetch_moe_alloc_failure", test_prefetch_moe_alloc_failure);
    run("test_prefetch_moe_refusal_is_sticky", test_prefetch_moe_refusal_is_sticky);
    return 0;
}
