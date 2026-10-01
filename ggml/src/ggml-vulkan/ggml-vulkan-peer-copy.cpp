#include "ggml-vulkan-common.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Copies between the memory of two Vulkan devices.
//
// A pair of devices uses the first of these paths that works for it, in this order:
//   1. direct. The device buffers are exportable and the destination device imports the memory of the source buffer and reads it with a copy command of its own. The bytes never touch host memory. This needs memory that can be shared between the two drivers, which on Linux is a dma-buf and on Windows an NT handle. The Vulkan specification only promises an NT handle to work within one physical device, so between two different GPUs it depends on the drivers.
//   2. shared staging. Both devices import the same pinned host allocation (VK_EXT_external_memory_host). The source device writes a chunk into it and the destination device reads it, with the chunks overlapping. The bytes pass through system memory as DMA in both directions, with no copy on the CPU.
//   3. host staging, the original path, that is the rest of this comment:
//        source device -> host staging -> host staging -> destination device
//
// Each path of a pair is verified with real data the first time it is used, because a driver that advertises memory sharing may still not make it work between two particular devices. A pair that fails moves on to the next. Verification, the copies of 1 and 2, and the blocking copy between devices all use command pools, fences and a queue submit of their own like the worker below does, so none of them has to wait for the backends to be idle.
//
// GGML_VK_DIRECT_COPY chooses what is allowed (see ggml-vulkan-buffers.cpp), GGML_VK_COPY_CHUNK_MB the size of the chunks of shared staging, 16 by default. GGML_VK_PEER_COPY_FORCE_STAGING makes everything use path 3.
//
// The blocking copy (ggml_vk_buffer_copy) does this with a command buffer and a fence per hop and cannot run while the backends have work in flight, because it recycles the command pools of the device queues. This file does the same hops on a worker thread of its own, so the caller can launch compute on the destination device while the copy streams in.
// The worker touches nothing a backend uses. It has its own command pool, command buffers, fences and staging buffers per device, and the only state it shares with the backends is the queue submit handle, which carries its own lock. It never frees or resets anything that is in flight.
// The copy is cut into chunks and the hops are pipelined with two staging slots per device, so the read of the next chunk overlaps the write of the current one.
// The caller owns the ordering around the copy:
//   - the source must not change until the copy has finished (it is made for weights)
//   - the destination must not be in use by the GPU when the copy is enqueued, the worker starts writing right away
//   - ggml_vk_peer_copy_synchronize() returns when every copy enqueued so far has landed, and whatever the destination device submits afterwards sees the data

static constexpr size_t vk_peer_chunk_size = 16u << 20;

namespace {

struct vk_peer_job {
    vk_buffer src;
    size_t    src_offset;
    vk_buffer dst;
    size_t    dst_offset;
    size_t    size;
};

bool vk_peer_host_accessible(const vk_buffer & buf) {
    // GGML_VK_PEER_COPY_FORCE_STAGING makes every buffer take the staged path, to test that path on devices where all memory is host visible
    if (getenv("GGML_VK_PEER_COPY_FORCE_STAGING") != nullptr) {
        return false;
    }
    // HostCached is required. Without it the memory is write-combined over the PCIe aperture and a host memcpy of it runs at a fraction of the bus speed, so such a buffer is staged instead.
    const vk::MemoryPropertyFlags need = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;
    return buf->ptr != nullptr && (buf->memory_property_flags & need) == need;
}

// what the worker owns on one device: a command pool, two command buffers with a fence each, and two host staging buffers. A slot is free once its fence has been waited for.
struct vk_peer_side {
    explicit vk_peer_side(const vk_device & dev) : device(dev) {
        handle = device->transfer_queue->handle;

        vk::CommandPoolCreateInfo pool_info(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, device->transfer_queue->queue_family_index);
        pool = device->device.createCommandPool(pool_info);

        vk::CommandBufferAllocateInfo alloc_info(pool, vk::CommandBufferLevel::ePrimary, 2);
        const std::vector<vk::CommandBuffer> bufs = device->device.allocateCommandBuffers(alloc_info);
        for (int i = 0; i < 2; i++) {
            cmd[i]   = bufs[i];
            fence[i] = device->device.createFence({});
        }
    }

    ~vk_peer_side() {
        try {
            wait_all();
        } catch (...) {
        }
        for (int i = 0; i < 2; i++) {
            ggml_vk_destroy_buffer(stage[i]);
            device->device.destroyFence(fence[i]);
        }
        // frees the command buffers with it
        device->device.destroyCommandPool(pool);
    }

    vk_peer_side(const vk_peer_side &) = delete;
    vk_peer_side & operator=(const vk_peer_side &) = delete;

    void wait(int slot) {
        if (!pending[slot]) {
            return;
        }
        VK_CHECK(device->device.waitForFences({ fence[slot] }, true, UINT64_MAX), "vk_peer_side wait", device);
        device->device.resetFences({ fence[slot] });
        pending[slot] = false;
    }

    void wait_all() {
        wait(0);
        wait(1);
    }

    // staging buffers of at least `size` bytes
    void ensure_stage(size_t size) {
        if (stage[0] != nullptr && stage[0]->size >= size) {
            return;
        }
        wait_all();
        for (int i = 0; i < 2; i++) {
            ggml_vk_destroy_buffer(stage[i]);
            stage[i] = ggml_vk_create_buffer_check(device, size,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        }
    }

    // copies between two buffers of this device on the slot, and returns without waiting
    void submit_copy(int slot, const vk::Buffer & from, size_t from_offset, const vk::Buffer & to, size_t to_offset, size_t size) {
        wait(slot);

        cmd[slot].reset();
        cmd[slot].begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        cmd[slot].copyBuffer(from, to, vk::BufferCopy(from_offset, to_offset, size));
        cmd[slot].end();

        vk::SubmitInfo submit_info{};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd[slot];
        handle->submit(submit_info, fence[slot]);
        pending[slot] = true;
    }

    vk_device                        device;
    std::shared_ptr<vk_queue_handle> handle;
    vk::CommandPool                  pool;
    vk::CommandBuffer                cmd[2];
    vk::Fence                        fence[2];
    bool                             pending[2] = { false, false };
    vk_buffer                        stage[2];
};

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Pairs of devices
// ---------------------------------------------------------------------------------------------------------------------

static std::atomic<size_t> vk_copy_direct_bytes {0};
static std::atomic<size_t> vk_copy_shared_bytes {0};
static std::atomic<size_t> vk_copy_host_bytes   {0};

void ggml_vk_copy_stats(size_t * direct_bytes, size_t * shared_bytes, size_t * host_bytes) {
    *direct_bytes = vk_copy_direct_bytes.load();
    *shared_bytes = vk_copy_shared_bytes.load();
    *host_bytes   = vk_copy_host_bytes.load();
}

void ggml_vk_peer_count_host_bytes(size_t size) {
    vk_copy_host_bytes.fetch_add(size);
}

void ggml_vk_peer_count_direct_bytes(size_t size) {
    vk_copy_direct_bytes.fetch_add(size);
}

static void * vk_pinned_alloc(size_t size) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void * ptr = nullptr;
    return posix_memalign(&ptr, 65536, size) == 0 ? ptr : nullptr;
#endif
}

static void vk_pinned_free(void * ptr) {
    if (ptr == nullptr) {
        return;
    }
#if defined(_WIN32)
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    free(ptr);
#endif
}

// What the copies from the device src into the device dst use. One copy at a time uses a pair.
struct vk_peer_pair {
    vk_peer_pair(const vk_device & src_, const vk_device & dst_) : src(src_), dst(dst_) {}

    ~vk_peer_pair() {
        // the resources that use the allocation go before it
        src_side.reset();
        dst_side.reset();
        ggml_vk_destroy_buffer(src_view);
        ggml_vk_destroy_buffer(dst_view);
        vk_pinned_free(host);
    }

    vk_peer_pair(const vk_peer_pair &) = delete;
    vk_peer_pair & operator=(const vk_peer_pair &) = delete;

    vk_device src;
    vk_device dst;

    std::mutex mutex;

    std::atomic<int> direct_state { 0 }; // 0 = not verified yet, 1 = usable, -1 = not usable
    int              shared_state = 0;

    // command pool, fences and queue submit of each device, for the copies of the two paths
    std::unique_ptr<vk_peer_side> src_side;
    std::unique_ptr<vk_peer_side> dst_side;

    // shared staging: a pinned allocation of two slots, and how each device sees it
    void *    host      = nullptr;
    size_t    slot_size = 0;
    vk_buffer src_view;
    vk_buffer dst_view;
};

static std::shared_ptr<vk_peer_pair> vk_peer_pair_get(vk_device & src, vk_device & dst, bool create) {
    std::lock_guard<std::mutex> lock(dst->peer_mutex);
    auto it = dst->peer_pairs.find(src.get());
    if (it != dst->peer_pairs.end()) {
        return it->second;
    }
    if (!create) {
        return nullptr;
    }
    std::shared_ptr<vk_peer_pair> pair = std::make_shared<vk_peer_pair>(src, dst);
    dst->peer_pairs[src.get()] = pair;
    return pair;
}

static vk_peer_side & vk_peer_pair_side(vk_peer_pair & p, bool of_src) {
    std::unique_ptr<vk_peer_side> & side = of_src ? p.src_side : p.dst_side;
    if (!side) {
        side = std::make_unique<vk_peer_side>(of_src ? p.src : p.dst);
    }
    return *side;
}

// ---- direct

// the destination device reads the memory of the source buffer, in one copy
static bool vk_peer_run_direct(vk_peer_pair & p, vk_buffer & src, size_t src_offset, vk_buffer & dst, size_t dst_offset, size_t size) {
    vk_buffer imp = ggml_vk_buffer_get_import(src, p.dst);
    if (!imp) {
        return false;
    }
    vk_peer_side & side = vk_peer_pair_side(p, false);
    side.submit_copy(0, imp->buffer, src_offset, dst->buffer, dst_offset, size);
    side.wait(0);
    return true;
}

// ---- shared staging

static bool vk_peer_shared_create(vk_peer_pair & p) {
    if (!p.src->external_memory_host || !p.dst->external_memory_host) {
        return false;
    }

    size_t chunk_mb = 16;
    if (const char * env = getenv("GGML_VK_COPY_CHUNK_MB")) {
        chunk_mb = std::min<size_t>(std::max<size_t>(strtoull(env, nullptr, 10), 1), 256);
    }
    const size_t align = std::max<size_t>(std::max<size_t>(p.src->min_imported_host_pointer_alignment, p.dst->min_imported_host_pointer_alignment), 1);
    p.slot_size = (chunk_mb * 1024 * 1024 + align - 1) / align * align;

    p.host = vk_pinned_alloc(2 * p.slot_size);
    if (p.host == nullptr || (reinterpret_cast<uintptr_t>(p.host) & (align - 1))) {
        return false;
    }

    p.src_view = ggml_vk_buffer_from_host_ptr(p.src, p.host, 2 * p.slot_size);
    p.dst_view = ggml_vk_buffer_from_host_ptr(p.dst, p.host, 2 * p.slot_size);
    if (!p.src_view || !p.dst_view) {
        return false;
    }

    vk_peer_pair_side(p, true);
    vk_peer_pair_side(p, false);
    return true;
}

// Moves size bytes through the shared staging memory. While the destination device drains chunk k from one slot,
// the source device already fills the other slot with chunk k + 1. That slot was last read for chunk k - 1, so that
// copy has to be done first.
static void vk_peer_run_shared(vk_peer_pair & p, vk_buffer & src, size_t src_offset, vk_buffer & dst, size_t dst_offset, size_t size) {
    vk_peer_side & src_side = *p.src_side;
    vk_peer_side & dst_side = *p.dst_side;

    const size_t slot    = p.slot_size;
    const size_t n_chunk = (size + slot - 1) / slot;
    auto chunk_len = [&](size_t k) { return std::min(slot, size - k*slot); };

    // buffer of the source device -> slot of the staging memory
    auto read_chunk = [&](size_t k) {
        src_side.submit_copy((int) (k & 1), src->buffer, src_offset + k*slot, p.src_view->buffer, (k & 1)*slot, chunk_len(k));
    };

    read_chunk(0);
    for (size_t k = 0; k < n_chunk; k++) {
        const int s = (int) (k & 1);
        src_side.wait(s);
        // slot of the staging memory -> buffer of the destination device
        dst_side.submit_copy(s, p.dst_view->buffer, s*slot, dst->buffer, dst_offset + k*slot, chunk_len(k));
        if (k + 1 < n_chunk) {
            dst_side.wait(s ^ 1);
            read_chunk(k + 1);
        }
    }
    dst_side.wait_all();
    src_side.wait_all();
}

// ---- verification

static uint8_t vk_peer_pattern(size_t i) {
    return (uint8_t) (i * 131 + (i >> 12) * 7 + 7);
}

// Fills a buffer of src with a pattern, has run copy a part of it into a buffer of dst, and checks the part and that
// nothing around it was touched. It uses command pools and staging of its own, so it can run beside compute.
template <typename Run>
static bool vk_peer_verify(vk_device & src, vk_device & dst, size_t len, bool src_exportable, Run run) {
    constexpr size_t src_off = 4096 + 5;
    constexpr size_t dst_off = 256 + 3;
    const size_t n = len + 2*4096;

    bool ok = false;
    try {
        vk_buffer s = ggml_vk_create_buffer_device(src, n, src_exportable);
        vk_buffer d = ggml_vk_create_buffer_device(dst, n, false);

        if (!src_exportable || s->export_handle_type != vk::ExternalMemoryHandleTypeFlagBits{}) {
            vk_peer_side side_s(src);
            vk_peer_side side_d(dst);
            side_s.ensure_stage(n);
            side_d.ensure_stage(n);

            uint8_t * pattern = (uint8_t *) side_s.stage[0]->ptr;
            for (size_t i = 0; i < n; i++) {
                pattern[i] = vk_peer_pattern(i);
            }
            side_s.submit_copy(0, side_s.stage[0]->buffer, 0, s->buffer, 0, n);
            side_s.wait(0);
            memset(side_d.stage[0]->ptr, 0, n);
            side_d.submit_copy(0, side_d.stage[0]->buffer, 0, d->buffer, 0, n);
            side_d.wait(0);

            if (run(s, src_off, d, dst_off, len)) {
                side_d.submit_copy(1, d->buffer, 0, side_d.stage[1]->buffer, 0, n);
                side_d.wait(1);
                const uint8_t * out = (const uint8_t *) side_d.stage[1]->ptr;
                ok = true;
                for (size_t i = 0; i < n && ok; i++) {
                    const bool inside = i >= dst_off && i < dst_off + len;
                    ok = out[i] == (inside ? vk_peer_pattern(src_off + (i - dst_off)) : 0);
                }
            }
        }
        ggml_vk_destroy_buffer(s);
        ggml_vk_destroy_buffer(d);
    } catch (const vk::SystemError & e) {
        VK_LOG_DEBUG("vk_peer_verify: " << e.what());
        ok = false;
    }
    return ok;
}

static bool vk_peer_verify_direct(vk_peer_pair & p) {
    const vk::ExternalMemoryHandleTypeFlagBits type = p.src->export_handle_type;
    if (type == vk::ExternalMemoryHandleTypeFlagBits{} ||
        !ggml_vk_can_import_from(p.dst->physical_device, p.dst->id_props, p.src, type, p.src->export_dedicated)) {
        return false;
    }
    const bool ok = vk_peer_verify(p.src, p.dst, 64*1024, true, [&](vk_buffer & s, size_t so, vk_buffer & d, size_t doff, size_t len) {
        return vk_peer_run_direct(p, s, so, d, doff, len);
    });
    if (ok) {
        GGML_LOG_INFO("ggml_vulkan: direct copy %s -> %s enabled (%s)\n", p.src->name.c_str(), p.dst->name.c_str(), ggml_vk_handle_type_name(type));
    } else {
        VK_LOG_DEBUG("vk_peer_verify_direct: " << p.src->name << " -> " << p.dst->name << " is not usable");
    }
    return ok;
}

static bool vk_peer_verify_shared(vk_peer_pair & p) {
    bool ok = false;
    try {
        ok = vk_peer_shared_create(p);
    } catch (const vk::SystemError & e) {
        VK_LOG_DEBUG("vk_peer_shared_create: " << e.what());
    }
    if (ok) {
        // more than one chunk, and a size that does not divide into chunks
        ok = vk_peer_verify(p.src, p.dst, p.slot_size + 12345, false, [&](vk_buffer & s, size_t so, vk_buffer & d, size_t doff, size_t len) {
            vk_peer_run_shared(p, s, so, d, doff, len);
            return true;
        });
    }
    if (ok) {
        GGML_LOG_INFO("ggml_vulkan: shared staging copy %s -> %s enabled (%zu MiB chunks)\n", p.src->name.c_str(), p.dst->name.c_str(), p.slot_size / (1024*1024));
    } else {
        VK_LOG_DEBUG("vk_peer_verify_shared: " << p.src->name << " -> " << p.dst->name << " is not usable");
    }
    return ok;
}

// ---- entry points

static bool vk_peer_force_staging() {
    return getenv("GGML_VK_PEER_COPY_FORCE_STAGING") != nullptr;
}

vk_peer_path ggml_vk_peer_copy_try(vk_buffer & src, size_t src_offset, vk_buffer & dst, size_t dst_offset, size_t size) {
    if (src->device == dst->device || size == 0) {
        return VK_PEER_PATH_NONE;
    }
    const int mode = ggml_vk_copy_mode();
    if (mode == 0 || vk_peer_force_staging()) {
        return VK_PEER_PATH_NONE;
    }

    std::shared_ptr<vk_peer_pair> pair = vk_peer_pair_get(src->device, dst->device, true);
    std::lock_guard<std::mutex> lock(pair->mutex);

    if (mode != 2 && src->export_handle_type != vk::ExternalMemoryHandleTypeFlagBits{} && pair->direct_state.load() >= 0) {
        if (pair->direct_state.load() == 0) {
            pair->direct_state = vk_peer_verify_direct(*pair) ? 1 : -1;
        }
        if (pair->direct_state.load() > 0 && vk_peer_run_direct(*pair, src, src_offset, dst, dst_offset, size)) {
            vk_copy_direct_bytes.fetch_add(size);
            return VK_PEER_PATH_DIRECT;
        }
    }

    // Shared staging replaces a copy through DMA, the CPU and DMA. When both buffers are host mapped, the copy on the
    // CPU alone is less work, so that is left to the caller unless shared staging was asked for.
    const bool both_mapped = vk_peer_host_accessible(src) && vk_peer_host_accessible(dst);

    if (pair->shared_state >= 0 && (mode == 2 || !both_mapped)) {
        if (pair->shared_state == 0) {
            pair->shared_state = vk_peer_verify_shared(*pair) ? 1 : -1;
        }
        if (pair->shared_state > 0) {
            vk_peer_run_shared(*pair, src, src_offset, dst, dst_offset, size);
            vk_copy_shared_bytes.fetch_add(size);
            return VK_PEER_PATH_SHARED;
        }
    }

    return VK_PEER_PATH_NONE;
}

bool ggml_vk_peer_direct_ready(vk_buffer & src, vk_device & dst) {
    if (src->export_handle_type == vk::ExternalMemoryHandleTypeFlagBits{} || vk_peer_force_staging()) {
        return false;
    }
    std::shared_ptr<vk_peer_pair> pair = vk_peer_pair_get(src->device, dst, false);
    return pair != nullptr && pair->direct_state.load() > 0;
}


struct vk_peer_copy_worker {
    vk_peer_copy_worker() {
        thread = std::thread([this] { run(); });
    }

    ~vk_peer_copy_worker() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv_job.notify_all();
        thread.join();
        // the thread is gone, so the resources are not in use
        sides.clear();
    }

    void enqueue(vk_peer_job && job) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            jobs.push_back(std::move(job));
        }
        cv_job.notify_one();
    }

    void synchronize() {
        std::string failure;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv_idle.wait(lock, [this] { return jobs.empty() && !busy; });
            failure.swap(error);
        }
        if (!failure.empty()) {
            GGML_ABORT("ggml_vulkan: peer copy failed: %s\n", failure.c_str());
        }
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            cv_job.wait(lock, [this] { return stop || !jobs.empty(); });
            if (jobs.empty()) {
                // stop was requested and everything enqueued before it has been done
                return;
            }
            vk_peer_job job = std::move(jobs.front());
            jobs.pop_front();
            busy = true;
            lock.unlock();

            std::string failure;
            try {
                process(job);
            } catch (const std::exception & e) {
                failure = e.what();
            } catch (...) {
                failure = "unknown error";
            }
            // release the buffers of the job here, not under the lock
            job = vk_peer_job{};

            lock.lock();
            if (!failure.empty() && error.empty()) {
                error = failure;
            }
            busy = false;
            cv_idle.notify_all();
        }
    }

    vk_peer_side & side(const vk_device & device) {
        std::unique_ptr<vk_peer_side> & s = sides[device.get()];
        if (!s) {
            s = std::make_unique<vk_peer_side>(device);
        }
        return *s;
    }

    void process(const vk_peer_job & job) {
        if (job.size == 0) {
            return;
        }

        // without a copy on the CPU, if the two devices can do that
        vk_buffer src_buf = job.src;
        vk_buffer dst_buf = job.dst;
        const vk_peer_path path = ggml_vk_peer_copy_try(src_buf, job.src_offset, dst_buf, job.dst_offset, job.size);
        if (path != VK_PEER_PATH_NONE) {
            GGML_LOG_DEBUG("ggml_vulkan: peer copy of %zu bytes, %s\n", job.size, path == VK_PEER_PATH_DIRECT ? "direct" : "shared staging");
            return;
        }
        ggml_vk_peer_count_host_bytes(job.size);

        const bool src_direct = vk_peer_host_accessible(job.src);
        const bool dst_direct = vk_peer_host_accessible(job.dst);

        GGML_LOG_DEBUG("ggml_vulkan: peer copy of %zu bytes, source %s, destination %s\n",
                job.size, src_direct ? "mapped" : "staged", dst_direct ? "mapped" : "staged");

        if (src_direct && dst_direct) {
            memcpy((uint8_t *) job.dst->ptr + job.dst_offset, (const uint8_t *) job.src->ptr + job.src_offset, job.size);
            return;
        }

        const size_t chunk   = std::min(vk_peer_chunk_size, job.size);
        const size_t n_chunk = (job.size + chunk - 1) / chunk;

        vk_peer_side * src_side = nullptr;
        vk_peer_side * dst_side = nullptr;
        // staging buffers are only ever grown, so round small jobs up to not reallocate for every size
        const size_t stage_size = std::max(chunk, (size_t) 4 << 20);
        if (!src_direct) {
            src_side = &side(job.src->device);
            src_side->ensure_stage(stage_size);
        }
        if (!dst_direct) {
            dst_side = &side(job.dst->device);
            dst_side->ensure_stage(stage_size);
        }

        auto chunk_len = [&](size_t c) { return std::min(chunk, job.size - c*chunk); };

        // hop 1 of a chunk: the source device writes it to its staging slot
        auto read_chunk = [&](size_t c) {
            src_side->submit_copy((int) (c & 1), job.src->buffer, job.src_offset + c*chunk, src_side->stage[c & 1]->buffer, 0, chunk_len(c));
        };

        if (!src_direct) {
            read_chunk(0);
        }

        for (size_t c = 0; c < n_chunk; c++) {
            const int    slot = (int) (c & 1);
            const size_t len  = chunk_len(c);

            const void * host = nullptr;
            if (src_direct) {
                host = (const uint8_t *) job.src->ptr + job.src_offset + c*chunk;
            } else {
                src_side->wait(slot);
                host = src_side->stage[slot]->ptr;
                // the other slot was consumed by the previous chunk, so the next read can start while this one moves on
                if (c + 1 < n_chunk) {
                    read_chunk(c + 1);
                }
            }

            if (dst_direct) {
                memcpy((uint8_t *) job.dst->ptr + job.dst_offset + c*chunk, host, len);
            } else {
                // the write of chunk c - 2 used this slot
                dst_side->wait(slot);
                memcpy(dst_side->stage[slot]->ptr, host, len);
                dst_side->submit_copy(slot, dst_side->stage[slot]->buffer, 0, job.dst->buffer, job.dst_offset + c*chunk, len);
            }
        }

        if (dst_side != nullptr) {
            dst_side->wait_all();
        }
        if (src_side != nullptr) {
            src_side->wait_all();
        }
    }

    std::mutex              mutex;
    std::condition_variable cv_job;
    std::condition_variable cv_idle;
    std::deque<vk_peer_job> jobs;
    bool                    busy = false;
    bool                    stop = false;
    std::string             error;

    // keyed by device, created by the worker thread on first use and destroyed after it has joined
    std::map<vk_device_struct *, std::unique_ptr<vk_peer_side>> sides;

    std::thread thread;
};

bool ggml_vk_peer_copy_async(ggml_backend_vk_context * ctx, vk_buffer & src, size_t src_offset, vk_buffer & dst, size_t dst_offset, size_t size) {
    if (src->device == dst->device) {
        // the same device has a copy of its own
        return false;
    }
    if (ctx->peer_copy == nullptr) {
        ctx->peer_copy = new vk_peer_copy_worker();
    }
    ctx->peer_copy->enqueue({ src, src_offset, dst, dst_offset, size });
    return true;
}

void ggml_vk_peer_copy_synchronize(ggml_backend_vk_context * ctx) {
    if (ctx->peer_copy != nullptr) {
        ctx->peer_copy->synchronize();
    }
}

void ggml_vk_peer_copy_destroy(ggml_backend_vk_context * ctx) {
    delete ctx->peer_copy;
    ctx->peer_copy = nullptr;
}
