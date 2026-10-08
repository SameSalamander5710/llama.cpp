#include "ggml-vulkan-common.h"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <thread>

// Asynchronous copy between the memory of two Vulkan devices. A buffer of one device cannot be read by another, so the bytes go through host memory:
//     source device -> host staging -> host staging -> destination device
// ggml_vk_buffer_copy does the same hops with a command buffer and a fence per hop, but it recycles the command pools of the device queues, so it cannot run while the backends have work in flight.
// This file does the hops on a worker thread of its own, so the caller can launch compute on the destination device while the copy streams in.
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
