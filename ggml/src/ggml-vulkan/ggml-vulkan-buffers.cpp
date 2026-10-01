#include "ggml-vulkan-common.h"

#include <cstdlib>
#include <mutex>
#if !defined(_WIN32)
#include <unistd.h>
#endif

ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_vk_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_vk_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static std::vector<uint32_t> ggml_vk_find_memory_properties(const vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    std::vector<uint32_t> indices;

    for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        if ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    return indices;
}

static vk_buffer ggml_vk_create_buffer(vk_device& device, size_t size, const std::initializer_list<vk::MemoryPropertyFlags> & req_flags_list,
                                       void *import_ptr = nullptr, vk::ExternalMemoryHandleTypeFlagBits export_handle_type = {}) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    if (size > device->max_buffer_size) {
        throw vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    if (size == 0) {
        buf->size = 0;
        return buf;
    }

    vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryAllocateFlags mem_flags {};
    if (device->buffer_device_address) {
        usage_flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
        mem_flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;
    }

    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        usage_flags,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    // memory that another device can import, see ggml_vk_import_buffer
    const bool exporting = !import_ptr && export_handle_type != vk::ExternalMemoryHandleTypeFlagBits{};

    vk::ExternalMemoryBufferCreateInfo external_memory_bci;
    if (import_ptr) {
        external_memory_bci.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
        buffer_create_info.setPNext(&external_memory_bci);
    } else if (exporting) {
        external_memory_bci.handleTypes = export_handle_type;
        buffer_create_info.setPNext(&external_memory_bci);
    }

    buf->buffer = device->device.createBuffer(buffer_create_info);

    vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    const vk::MemoryPriorityAllocateInfoEXT mem_priority_info { 1.0f };

    vk::MemoryAllocateFlagsInfo mem_flags_info { mem_flags };

    if (device->memory_priority) {
        mem_flags_info.setPNext(&mem_priority_info);
    }

    vk::ExportMemoryAllocateInfo export_info { export_handle_type };
    vk::MemoryDedicatedAllocateInfo dedicated_info { {}, buf->buffer };
    const void * alloc_pnext = &mem_flags_info;
    if (exporting) {
        export_info.setPNext(&mem_flags_info);
        alloc_pnext = &export_info;
        if (device->export_dedicated) {
            dedicated_info.setPNext(&export_info);
            alloc_pnext = &dedicated_info;
        }
    }

    if (import_ptr) {
        vk::MemoryHostPointerPropertiesEXT host_pointer_props;
        try {
            host_pointer_props = device->device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, import_ptr);
        } catch (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroyBuffer(buf->buffer);
            return {};
        }
        vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

        uint32_t memory_type_idx;
        vk::MemoryPropertyFlags property_flags = *req_flags_list.begin();
        for (memory_type_idx = 0; memory_type_idx < 32; ++memory_type_idx) {
            if (!(host_pointer_props.memoryTypeBits & (1u << memory_type_idx))) {
                continue;
            }
            if (!(mem_req.memoryTypeBits & (1u << memory_type_idx))) {
                continue;
            }

            vk::MemoryType memory_type = mem_props.memoryTypes[memory_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            if ((memory_type.propertyFlags & property_flags) == property_flags) {
                property_flags = memory_type.propertyFlags;
                break;
            }
        }
        if (memory_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroyBuffer(buf->buffer);
            return {};
        }

        buf->memory_property_flags = mem_props.memoryTypes[memory_type_idx].propertyFlags;
        try {
            vk::ImportMemoryHostPointerInfoEXT import_info;
            import_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
            import_info.pHostPointer = import_ptr;
            import_info.setPNext(&mem_flags_info);
            buf->device_memory = device->device.allocateMemory({ size, memory_type_idx, &import_info });
        } catch (const vk::SystemError& e) {
        }
    } else {
        for (auto it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            const auto & req_flags = *it;

            const std::vector<uint32_t> memory_type_indices = ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags);

            if (memory_type_indices.empty()) {
                continue;
            }

            bool done = false;

            for (auto mtype_it = memory_type_indices.begin(); mtype_it != memory_type_indices.end(); mtype_it++) {
                try {
                    buf->device_memory = device->device.allocateMemory({ mem_req.size, *mtype_it, alloc_pnext });
                    buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;
                    buf->memory_type_index = *mtype_it;
                    buf->alloc_size = mem_req.size;
                    buf->export_handle_type = exporting ? export_handle_type : vk::ExternalMemoryHandleTypeFlagBits{};
                    done = true;
                    break;
                } catch (const vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    if (it + 1 == req_flags_list.end() && mtype_it + 1 == memory_type_indices.end()) {
                        device->device.destroyBuffer(buf->buffer);
                        throw e;
                    }
                }
            }

            if (done) {
                break;
            }
        }
    }

    if (!buf->device_memory) {
        device->device.destroyBuffer(buf->buffer);
        throw vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    buf->ptr = nullptr;

    if (import_ptr) {
        buf->ptr = import_ptr;
    } else {
        if (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
        }
    }

    device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);

    buf->device = device;
    buf->size = size;

    if (device->buffer_device_address) {
        const vk::BufferDeviceAddressInfo addressInfo(buf->buffer);
        buf->bda_addr = device->device.getBufferAddress(addressInfo);
    }

    device->memory_logger->log_allocation(buf, size);

    return buf;
}

vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags) {
    try {
        return ggml_vk_create_buffer(device, size, {req_flags, fallback_flags});
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }
}

static vk_buffer ggml_vk_create_buffer_device_impl(vk_device& device, size_t size, vk::ExternalMemoryHandleTypeFlagBits export_handle_type) {
    vk_buffer buf;
    try {
        if (device->prefer_host_memory) {
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal}, nullptr, export_handle_type);
        } else if (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent}, nullptr, export_handle_type);
        } else if (device->disable_host_visible_vidmem) {
            if (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent}, nullptr, export_handle_type);
            } else {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal}, nullptr, export_handle_type);
            }
        } else {
            // use rebar if available, otherwise fallback to device only visible memory
            if (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent}, nullptr, export_handle_type);
            } else {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal}, nullptr, export_handle_type);
            }
        }
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }

    return buf;
}

vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size, bool exportable) {
    if (exportable && device->export_handle_type != vk::ExternalMemoryHandleTypeFlagBits{}) {
        try {
            return ggml_vk_create_buffer_device_impl(device, size, device->export_handle_type);
        } catch (const vk::SystemError& e) {
            // the driver may refuse exportable memory, or be out of memory. Fall back to a plain buffer,
            // which only loses the direct device to device copies for this buffer
            VK_LOG_DEBUG("ggml_vk_create_buffer_device: exportable allocation failed (" << e.what() << "), retrying plain");
        }
    }
    return ggml_vk_create_buffer_device_impl(device, size, {});
}

void ggml_vk_destroy_buffer(vk_buffer& buf) {
    if (buf == nullptr) {
        return;
    }

    if (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }

    buf.reset();
}

void * ggml_vk_host_malloc(vk_device& device, size_t size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");
    vk_buffer buf = ggml_vk_create_buffer(device, size,
        {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});

    if(!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n",
            size/1024.0/1024.0);
        device->device.freeMemory(buf->device_memory);
        device->device.destroyBuffer(buf->buffer);
        return nullptr;
    }

    std::lock_guard<std::shared_mutex> guard(device->pinned_memory_mutex);
    device->pinned_memory.push_back(std::make_tuple(buf->ptr, size, buf));

    return buf->ptr;
}

void ggml_vk_host_free(vk_device& device, void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");
    std::lock_guard<std::shared_mutex> guard(device->pinned_memory_mutex);

    vk_buffer buf;
    size_t index;
    for (size_t i = 0; i < device->pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(device->pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
        if (ptr >= addr && ptr < endr) {
            buf = std::get<2>(device->pinned_memory[i]);
            index = i;
            break;
        }
    }
    if (buf == nullptr) {
        fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
        return;
    }

    ggml_vk_destroy_buffer(buf);

    device->pinned_memory.erase(device->pinned_memory.begin() + index);
}

void ggml_vk_host_get(const vk_device& device, const void * ptr, vk_buffer& buf, size_t& buf_offset) {
    std::shared_lock<std::shared_mutex> guard(device->pinned_memory_mutex);
    buf = nullptr;
    buf_offset = 0;
    for (size_t i = 0; i < device->pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(device->pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
        if (ptr >= addr && ptr < endr) {
            buf = std::get<2>(device->pinned_memory[i]);
            buf_offset = ((const uint8_t *)ptr) - addr;
            break;
        }
    }
}

void ggml_vk_ensure_sync_staging_buffer(vk_device& device, size_t size) {
    if (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(device->sync_staging);
        device->sync_staging = ggml_vk_create_buffer_check(device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

void ggml_vk_ensure_sync_staging_buffer(ggml_backend_vk_context * ctx, size_t size) {
    if (ctx->sync_staging == nullptr || ctx->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_buffer_check(ctx->device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

static void ggml_vk_buffer_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_buffer& dst, size_t offset, const ggml_tensor * tensor, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(ctx->device, tensor->data, buf, buf_offset);

    const uint64_t ne0 = tensor->ne[0];
    const uint64_t ne1 = tensor->ne[1];
    const uint64_t ne2 = tensor->ne[2];
    const uint64_t ne3 = tensor->ne[3];
    const uint64_t nb0 = tensor->nb[0];
    const uint64_t nb1 = tensor->nb[1];
    const uint64_t nb2 = tensor->nb[2];
    const uint64_t nb3 = tensor->nb[3];
    const ggml_type type = tensor->type;
    const uint64_t ts = ggml_type_size(type);
    const uint64_t bs = ggml_blck_size(type);

    const uint64_t dstnb0 = ts;
    const uint64_t dstnb1 = dstnb0*(ne0/bs);
    const uint64_t dstnb2 = dstnb1*ne1;
    const uint64_t dstnb3 = dstnb2*ne2;

    const uint64_t ne = ggml_nelements(tensor);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices;

        for (uint64_t i3 = 0; i3 < ne3; i3++) {
            for (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                if (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_offset + i3*nb3 + i2*nb2, offset + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } else {
                    for (uint64_t i1 = 0; i1 < ne1; i1++) {
                        if (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_offset + i3*nb3 + i2*nb2 + i1*nb1, offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } else {
                            const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                            const uint64_t d_off = offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            for (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        return;
    }

    if (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_buffer& staging = ctx->device->sync_staging;
    const uint64_t copy_size = ts*ne/bs;
    ggml_vk_ensure_sync_staging_buffer(ctx->device, copy_size);
    VkBufferCopy buf_copy{ 0, offset, copy_size };

    ggml_vk_sync_buffers(ctx, subctx);
    vkCmdCopyBuffer(subctx->s->buffer->buf, (VkBuffer)staging->buffer, (VkBuffer)dst->buffer, 1, &buf_copy);

    for (uint64_t i3 = 0; i3 < ne3; i3++) {
        for (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            if (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2, dstnb2, &subctx->in_memcpys);
            } else {
                for (uint64_t i1 = 0; i1 < ne1; i1++) {
                    if (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->in_memcpys);
                    } else {
                        const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                        const uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        for (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (const uint8_t *) tensor->data + s_off + i0*nb0, dstnb0, &subctx->in_memcpys);
                        }
                    }
                }
            }
        }
    }
}

bool ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices(1);
        if (width == spitch && width == dpitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } else {
            slices.resize(height);
            for (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * dpitch;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        return true;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        // copy was not handled caller needs to fall back
        return false;
    }

    // Staging buffer required
    const size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(dst->device, staging_size);

    vk_buffer& staging_buffer = dst->device->sync_staging;

    std::vector<vk::BufferCopy> slices(1);
    if (width == dpitch) {
        slices[0].srcOffset = 0;
        slices[0].dstOffset = offset;
        slices[0].size = staging_size;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = i * width;
            slices[i].dstOffset = offset + i * dpitch;
            slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(staging_buffer->buffer, dst->buffer, slices);

    if (width == spitch) {
        deferred_memcpy((uint8_t *)staging_buffer->ptr, src, staging_size, &subctx->in_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_buffer->ptr + i * width, (const uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
    return true;
}

bool ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t size, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    return ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

void ggml_vk_buffer_write_2d(vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        if (width == spitch && width == dpitch) {
            memcpy((uint8_t *)dst->ptr + offset, src, width * height);
        } else {
            for (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *)dst->ptr + offset + i * dpitch, (const uint8_t *) src + i * spitch, width);
            }
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        bool ret = ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        for (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        for (auto& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_queue_command_pools_cleanup(dst->device);
    }
}

void ggml_vk_buffer_write(vk_buffer& dst, size_t offset, const void * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_buffer_write_2d(dst, offset, src, size, size, size, 1);
}

bool ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
    GGML_ASSERT(width > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != nullptr);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_offset);

    std::vector<vk::BufferCopy> slices(1);
    if (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(src->buffer, buf->buffer, slices);

        return true;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        // copy was not handled caller needs to fall back
        return false;
    }

    // Fall back to staging buffer
    const size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(src->device, staging_size);

    vk_buffer& staging_buffer = src->device->sync_staging;

    std::vector<vk::BufferCopy> staging_slices(1);
    if (width == spitch) {
        staging_slices[0].srcOffset = offset;
        staging_slices[0].dstOffset = 0;
        staging_slices[0].size = staging_size;
    } else {
        staging_slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            staging_slices[i].srcOffset = offset + i * spitch;
            staging_slices[i].dstOffset = i * width;
            staging_slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(src->buffer, staging_buffer->buffer, staging_slices);

    if (width == dpitch) {
        deferred_memcpy(dst, staging_buffer->ptr, staging_size, &subctx->out_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (const uint8_t *) staging_buffer->ptr + i * width, width, &subctx->out_memcpys);
        }
    }
    return true;
}

static bool ggml_vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t size, bool sync_staging = false) {
    return ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

void ggml_vk_buffer_read_2d(vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->buffer << ", " << offset << ", " << width << ", " << height << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    if(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->buffer->buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            {},
            { { vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eHostRead } },
            {}, {});
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        if (width == spitch && width == dpitch) {
            memcpy(dst, (const uint8_t *) src->ptr + offset, width * height);
        } else {
            for (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *) dst + i * dpitch, (const uint8_t *) src->ptr + offset + i * spitch, width);
            }
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        bool ret = ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        for (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
    }
}

void ggml_vk_buffer_read(vk_buffer& src, size_t offset, void * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->buffer << ", " << offset << ", " << size << ")");
    ggml_vk_buffer_read_2d(src, offset, dst, size, size, size, 1);
}

void ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    VkBufferCopy bc{ src_offset, dst_offset, size };

    vkCmdCopyBuffer(ctx->s->buffer->buf, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &bc);
}

// ---------------------------------------------------------------------------------------------------------------------
// Exporting device buffers, and importing them into another device
//
// A device buffer is allocated with exportable memory when more than one device is in use and some other device can
// import it. The device that wants the data imports the memory of the source buffer and pulls from it with a copy
// command on its own queue, so the data never lands in host memory. How the copies between two devices are chosen,
// verified and run is in ggml-vulkan-peer-copy.cpp.
//
// GGML_VK_DIRECT_COPY selects what is allowed:
//   0        host staging only, like before
//   1        direct, then shared staging, then host staging (default)
//   2        shared staging, then host staging
//   3        like 1, and also tries opaque handles between different physical devices. The Vulkan specification only
//            guarantees those to work within one physical device, but some drivers accept them. Verified like the rest.
// ---------------------------------------------------------------------------------------------------------------------

int ggml_vk_copy_mode() {
    static const int mode = []() {
        const char * env = getenv("GGML_VK_DIRECT_COPY");
        const int value = env ? atoi(env) : 1;
        return value >= 0 && value <= 3 ? value : 1;
    }();
    return mode;
}

// The opaque handle of the platform, the one that exists on every driver that can share memory at all
#if defined(_WIN32)
static constexpr vk::ExternalMemoryHandleTypeFlagBits vk_opaque_handle_type = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;
#else
static constexpr vk::ExternalMemoryHandleTypeFlagBits vk_opaque_handle_type = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
#endif

const char * ggml_vk_handle_type_name(vk::ExternalMemoryHandleTypeFlagBits type) {
    switch (type) {
        case vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT:   return "DMA_BUF";
        case vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32: return "OPAQUE_WIN32";
        default:                                                 return "OPAQUE_FD";
    }
}

// A native handle of the memory of a buffer, a file descriptor on Linux and an NT handle on Windows.
// A file descriptor belongs to the driver once the memory is imported. An NT handle never does, importing it makes
// the driver take its own reference, and the handle has to be closed here either way.
class vk_native_handle {
public:
#if defined(_WIN32)
    typedef HANDLE type;
    static type none() { return nullptr; }
#else
    typedef int type;
    static type none() { return -1; }
#endif

    vk_native_handle() = default;
    explicit vk_native_handle(type h) : handle(h) {}
    vk_native_handle(const vk_native_handle &) = delete;
    vk_native_handle & operator=(const vk_native_handle &) = delete;
    ~vk_native_handle() {
        if (handle != none()) {
#if defined(_WIN32)
            CloseHandle(handle);
#else
            close(handle);
#endif
        }
    }

    type get() const { return handle; }

    // the import succeeded
    void imported() {
#if !defined(_WIN32)
        handle = none();
#endif
    }

private:
    type handle = none();
};

static vk_native_handle ggml_vk_export_native_handle(vk_device & device, vk::DeviceMemory memory, vk::ExternalMemoryHandleTypeFlagBits type) {
#if defined(_WIN32)
    return vk_native_handle(device->device.getMemoryWin32HandleKHR({ memory, type }));
#else
    return vk_native_handle(device->device.getMemoryFdKHR({ memory, type }));
#endif
}

static vk::DeviceMemory ggml_vk_import_native_handle(vk_device & device, vk::DeviceSize size, uint32_t memory_type_index,
                                                     vk::ExternalMemoryHandleTypeFlagBits type, const vk_native_handle & handle,
                                                     const vk::MemoryDedicatedAllocateInfo * dedicated) {
#if defined(_WIN32)
    vk::ImportMemoryWin32HandleInfoKHR import_info { type, handle.get() };
#else
    vk::ImportMemoryFdInfoKHR import_info { type, handle.get() };
#endif
    if (dedicated) {
        import_info.setPNext(dedicated);
    }
    return device->device.allocateMemory({ size, memory_type_index, &import_info });
}

static vk::ExternalMemoryProperties ggml_vk_external_buffer_props(vk::PhysicalDevice pd, vk::BufferUsageFlags usage, vk::ExternalMemoryHandleTypeFlagBits type) {
    const vk::PhysicalDeviceExternalBufferInfo info { {}, usage, type };
    return pd.getExternalBufferProperties(info).externalMemoryProperties;
}

bool ggml_vk_same_physical_device(const vk::PhysicalDeviceIDProperties & a, const vk::PhysicalDeviceIDProperties & b) {
    return std::equal(std::begin(a.deviceUUID), std::end(a.deviceUUID), std::begin(b.deviceUUID)) &&
           std::equal(std::begin(a.driverUUID), std::end(a.driverUUID), std::begin(b.driverUUID));
}

static bool ggml_vk_has_extension(vk::PhysicalDevice pd, const char * name) {
    for (const auto & ext : pd.enumerateDeviceExtensionProperties()) {
        if (strcmp(ext.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

bool ggml_vk_can_import_from(vk::PhysicalDevice pd, const vk::PhysicalDeviceIDProperties & pd_id, const vk_device & exporter,
                             vk::ExternalMemoryHandleTypeFlagBits type, bool exporter_dedicated) {
#if defined(_WIN32)
    if (!ggml_vk_has_extension(pd, "VK_KHR_external_memory_win32")) {
        return false;
    }
#else
    if (!ggml_vk_has_extension(pd, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)) {
        return false;
    }
    if (type == vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT && !ggml_vk_has_extension(pd, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME)) {
        return false;
    }
#endif
    // an opaque handle is only guaranteed to work within one physical device
    if (type == vk_opaque_handle_type && ggml_vk_copy_mode() != 3 && !ggml_vk_same_physical_device(exporter->id_props, pd_id)) {
        return false;
    }
    const vk::ExternalMemoryProperties props = ggml_vk_external_buffer_props(pd, vk::BufferUsageFlagBits::eTransferSrc, type);
    if (!(props.externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eImportable)) {
        return false;
    }
    // memory exported as a dedicated allocation has to be imported as one too, which is what the importer does.
    // But memory that is not dedicated cannot be imported by a device that only imports dedicated memory
    if ((props.externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eDedicatedOnly) && !exporter_dedicated) {
        return false;
    }
    return true;
}

// Picks the handle type the buffers of a device are exported with, so that one of the other devices in use can
// import them. Leaves device->export_handle_type empty when there is none, which keeps allocations as they were.
void ggml_vk_init_direct_copy(vk_device& device, const std::vector<vk::PhysicalDevice> & peers) {
    device->export_handle_type = {};
    device->export_dedicated = false;

    const int mode = ggml_vk_copy_mode();
    if (mode == 0 || mode == 2) {
        return;
    }
#if defined(_WIN32)
    if (!device->external_memory_win32) {
        return;
    }
#else
    if (!device->external_memory_fd) {
        return;
    }
#endif

    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    if (device->buffer_device_address) {
        usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }

    // dma-buf is the handle meant for sharing memory between different devices, an opaque handle only
    // works between instances of the same device
    std::vector<vk::ExternalMemoryHandleTypeFlagBits> types;
#if !defined(_WIN32)
    if (device->external_memory_dma_buf) {
        types.push_back(vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT);
    }
#endif
    types.push_back(vk_opaque_handle_type);

    for (const auto type : types) {
        const vk::ExternalMemoryProperties exp = ggml_vk_external_buffer_props(device->physical_device, usage, type);
        if (!(exp.externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eExportable)) {
            continue;
        }
        const bool dedicated = (bool)(exp.externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eDedicatedOnly);

        for (const auto & peer : peers) {
            if (peer == device->physical_device) {
                continue;
            }
            const auto peer_props = peer.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties>();
            if (ggml_vk_can_import_from(peer, peer_props.get<vk::PhysicalDeviceIDProperties>(), device, type, dedicated)) {
                device->export_handle_type = type;
                device->export_dedicated = dedicated;
                VK_LOG_DEBUG("ggml_vk_init_direct_copy(" << device->name << "): export as " << ggml_vk_handle_type_name(type));
                return;
            }
        }
    }
}

// Imports the memory of an exportable buffer of another device as a buffer that can only be copied from.
static vk_buffer ggml_vk_import_buffer(vk_device& device, vk_buffer& src) {
    const vk::ExternalMemoryHandleTypeFlagBits type = src->export_handle_type;
    GGML_ASSERT(type != vk::ExternalMemoryHandleTypeFlagBits{});

    vk::Buffer buffer = VK_NULL_HANDLE;
    try {
        vk_native_handle handle = ggml_vk_export_native_handle(src->device, src->device_memory, type);

        vk::ExternalMemoryBufferCreateInfo external_bci { type };
        vk::BufferCreateInfo bci { vk::BufferCreateFlags(), src->size, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive, 0, nullptr };
        bci.setPNext(&external_bci);
        buffer = device->device.createBuffer(bci);

        const vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buffer);
        uint32_t type_bits = mem_req.memoryTypeBits;
        uint32_t memory_type_index = 0;
        const vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

#if !defined(_WIN32)
        if (type == vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT) {
            type_bits &= device->device.getMemoryFdPropertiesKHR(type, handle.get()).memoryTypeBits;
            if (type_bits == 0) {
                throw vk::OutOfDeviceMemoryError("no memory type can import the dma-buf");
            }
            // prefer device local memory, that is where an exported device buffer lives
            bool found = false;
            for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
                if (!(type_bits & (1u << i))) {
                    continue;
                }
                if (!found || (mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                    memory_type_index = i;
                    found = true;
                    if (mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) {
                        break;
                    }
                }
            }
        } else
#endif
        {
            // an opaque handle is imported with the memory type it was exported with, which has the same index
            // on the same physical device and has to be found by its properties on another one
            if (ggml_vk_same_physical_device(src->device->id_props, device->id_props)) {
                memory_type_index = src->memory_type_index;
            } else {
                bool found = false;
                for (uint32_t i = 0; i < mem_props.memoryTypeCount && !found; i++) {
                    if ((type_bits & (1u << i)) && mem_props.memoryTypes[i].propertyFlags == src->memory_property_flags) {
                        memory_type_index = i;
                        found = true;
                    }
                }
                if (!found) {
                    throw vk::OutOfDeviceMemoryError("no memory type matches the exported memory");
                }
            }
            if (!(type_bits & (1u << memory_type_index))) {
                throw vk::OutOfDeviceMemoryError("the exported memory type cannot hold the imported buffer");
            }
        }
        if (mem_req.size > src->alloc_size) {
            throw vk::OutOfDeviceMemoryError("the imported buffer needs more memory than the exported one");
        }

        vk::MemoryDedicatedAllocateInfo dedicated_info { {}, buffer };

        vk_buffer buf = std::make_shared<vk_buffer_struct>();
        buf->device_memory = ggml_vk_import_native_handle(device, src->alloc_size, memory_type_index, type, handle,
                                                          src->device->export_dedicated ? &dedicated_info : nullptr);
        handle.imported();

        // buf owns the memory and the buffer from here on, and releases them when it is destroyed
        buf->buffer = buffer;
        buffer = VK_NULL_HANDLE;
        buf->memory_property_flags = mem_props.memoryTypes[memory_type_index].propertyFlags;
        buf->ptr = nullptr;
        buf->alloc_size = src->alloc_size;
        buf->memory_type_index = memory_type_index;
        buf->device = device;
        buf->size = src->size;
        device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);
        return buf;
    } catch (const vk::SystemError & e) {
        GGML_LOG_WARN("ggml_vulkan: importing a buffer of %s into %s failed (%s)\n", src->device->name.c_str(), device->name.c_str(), e.what());
        if (buffer) {
            device->device.destroyBuffer(buffer);
        }
        return nullptr;
    }
}

// the view of src on device, imported on first use
vk_buffer ggml_vk_buffer_get_import(vk_buffer& src, vk_device& device) {
    std::lock_guard<std::mutex> guard(src->imports_mutex);
    for (auto & imp : src->imports) {
        if (imp->device == device) {
            return imp;
        }
    }
    vk_buffer imp = ggml_vk_import_buffer(device, src);
    if (imp) {
        src->imports.push_back(imp);
    }
    return imp;
}

// whether a copy out of src into the device dst can be recorded without blocking
bool ggml_vk_buffer_copy_direct_ready(vk_buffer& src, vk_device& dst) {
    return ggml_vk_peer_direct_ready(src, dst);
}

// Records a copy from a buffer of another device into ctx, a context of the device of dst. Only succeeds once a
// copy between the two devices has verified the direct path, and does not wait for anything the source device is
// still working on.
bool ggml_vk_buffer_copy_direct_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    if (!ggml_vk_buffer_copy_direct_ready(src, dst->device)) {
        return false;
    }
    vk_buffer imp = ggml_vk_buffer_get_import(src, dst->device);
    if (!imp) {
        return false;
    }
    ggml_vk_buffer_copy_async(ctx, dst, dst_offset, imp, src_offset, size);
    ggml_vk_peer_count_direct_bytes(size);
    return true;
}

void ggml_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    if (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);
    } else {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        if (size == 0) {
            return;
        }

        // without a copy on the CPU, when the two devices can do that
        if (ggml_vk_peer_copy_try(src, src_offset, dst, dst_offset, size) != VK_PEER_PATH_NONE) {
            return;
        }

        ggml_vk_peer_count_host_bytes(size);

        // Copy device to device through host staging, in chunks so the staging buffer of both devices stays small no matter how large the tensor is
        constexpr size_t chunk_size = 32*1024*1024;

        for (size_t off = 0; off < size; off += chunk_size) {
            const size_t chunk = std::min(chunk_size, size - off);
            ggml_vk_ensure_sync_staging_buffer(src->device, chunk);

            // Copy to src staging buffer
            ggml_vk_buffer_copy(src->device->sync_staging, 0, src, src_offset + off, chunk);
            // Copy to dst buffer
            ggml_vk_buffer_write(dst, dst_offset + off, src->device->sync_staging->ptr, chunk);
        }
    }
}

void ggml_vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

    if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + offset, c, size, &ctx->memsets);
        return;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
}

void ggml_vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + offset, c, size);
        return;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
    ggml_vk_ctx_end(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_queue_command_pools_cleanup(dst->device);
}

ggml_backend_buffer_i ggml_backend_vk_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vk_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_buffer_clear,
    /* .reset           = */ NULL,
};

vk_buffer ggml_vk_buffer_from_host_ptr(vk_device & device, void * ptr, size_t size) {
    if (!device->external_memory_host) {
        return {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    if (uptr & (device->min_imported_host_pointer_alignment - 1)) {
        return {};
    }
    if (size & (device->min_imported_host_pointer_alignment - 1)) {
        return {};
    }

    const vk::MemoryPropertyFlags property_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;

    vk_buffer buf {};
    try {
        buf = ggml_vk_create_buffer(device, size, { property_flags }, ptr);
    } catch (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    return buf;
}

