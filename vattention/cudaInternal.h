#define CHECK_CUDA(x)                                                                 \
    do                                                                                \
    {                                                                                 \
        CUresult res = x;                                                             \
        if (res != CUDA_SUCCESS)                                                      \
        {                                                                             \
            const char *errStr = NULL;                                                \
            (void)cuGetErrorString(res, &errStr);                                     \
            std::cerr << __FILE__ << ':' << __LINE__ << ' ' << #x                     \
                      << "failed (" << (unsigned)res << "): " << errStr << std::endl; \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

u64 do_cuda_default_init(int device, u64 page_size)
{
    u64 phys_granularity;
    CHECK_CUDA(cuInit(0));
    CHECK_CUDA(cuCtxGetCurrent(&ctx));
    if (ctx == NULL)
    {
        std::cerr << "[vAttention] No CUDA context found.";
        std::cerr << " Please initialize PyTorch before configuring vAttention." << std::endl;
        exit(1);
    }
    
    // Clear and initialize memory allocation properties
    memset(&prop, 0, sizeof(prop));
    memset(&accessDesc, 0, sizeof(accessDesc));
    
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    
    // Initialize access descriptor with proper device context
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = device;
    
    CHECK_CUDA(cuMemGetAllocationGranularity(&phys_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    
    // Validate alignment requirements
    if (phys_granularity != page_size) {
        std::cerr << "[vAttention] Error: Physical granularity mismatch. Expected: " 
                  << page_size << ", Got: " << phys_granularity << std::endl;
        exit(1);
    }
    
    return phys_granularity;
}

u64 do_cuda_init(int device, u64 page_size)
{
    if (is_uvm_backend(page_size))
        return do_cuda_uvm_init(device, page_size);

    return do_cuda_default_init(device, page_size);
}

u64 reserve_cuda_pages(u64 num_layers, u64 free_memory, u64 page_size)
{
    Log log;
    u64 num_phys_blocks = get_num_phys_blocks(num_layers, free_memory, page_size);
    log.log("Reserving " + std::to_string(num_phys_blocks) + " pages of size " + std::to_string(page_size) + " ...");

    while (cuda_pages.size() < num_phys_blocks)
    {
        CUmemGenericAllocationHandle cuda_page;
        CHECK_CUDA(cuMemCreate(&cuda_page, page_size, &prop, 0));
        cuda_pages.push_back(cuda_page);
    }

    return cuda_pages.size();
}

/* This function must be called only after do_cuda_init */
u64 reserve_gpu_pages(u64 num_layers, u64 free_memory, u64 page_size)
{
    if (is_uvm_backend(page_size))
        return reserve_uvm_pages(num_layers, free_memory, page_size);

    return reserve_cuda_pages(num_layers, free_memory, page_size);
}

inline void map_cuda_pages(int reqId,
                        int layer_idx,
                        u64 req_offset,
                        CUdeviceptr kcache_ptr,
                        CUdeviceptr vcache_ptr,
                        CUPage k_page,
                        CUPage v_page) {
    // Lock to prevent race conditions in multi-threaded memory mapping
    std::lock_guard<std::mutex> lock(memory_mapping_mutex);
    
    // Memory mapping with race condition protection
    
    // Validate alignment - req_offset must be page_size aligned for cuMemMap
    if (req_offset % page_size != 0) {
        std::cerr << "[vAttention] Error: req_offset " << req_offset 
                  << " is not aligned to page_size " << page_size << std::endl;
        exit(1);
    }
    
    // Validate memory handles are valid before mapping
    if (k_page == 0 || v_page == 0) {
        std::cerr << "[vAttention] Error: Invalid memory handle (k_page=" 
                  << k_page << ", v_page=" << v_page << ")" << std::endl;
        exit(1);
    }
    
    // Validate virtual address ranges
    if (kcache_ptr == 0 || vcache_ptr == 0) {
        std::cerr << "[vAttention] Error: Invalid virtual address pointers" << std::endl;
        exit(1);
    }
    
    // Validate access descriptor is properly initialized
    if (accessDesc.location.type != CU_MEM_LOCATION_TYPE_DEVICE || 
        accessDesc.flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE) {
        std::cerr << "[vAttention] Error: Access descriptor not properly initialized" << std::endl;
        exit(1);
    }
    
    // Check if this mapping already exists to prevent double mapping
    auto mapping_key = std::make_tuple(reqId, req_offset, layer_idx);
    if (cuda_pagemap.find(mapping_key) != cuda_pagemap.end()) {
        return; // Already mapped, skip
    }
    
    // Check if virtual address range is already mapped
    CUresult map_result_k = cuMemMap(kcache_ptr + req_offset, page_size, 0, k_page, 0);
    if (map_result_k != CUDA_SUCCESS) {
        const char *errStr = NULL;
        cuGetErrorString(map_result_k, &errStr);
        std::cerr << "[vAttention] cuMemMap failed for k_page: " << errStr 
                  << " (addr=" << std::hex << (kcache_ptr + req_offset) << std::dec << ")" << std::endl;
        exit(1);
    }
    
    CUresult map_result_v = cuMemMap(vcache_ptr + req_offset, page_size, 0, v_page, 0);
    if (map_result_v != CUDA_SUCCESS) {
        const char *errStr = NULL;
        cuGetErrorString(map_result_v, &errStr);
        std::cerr << "[vAttention] cuMemMap failed for v_page: " << errStr 
                  << " (addr=" << std::hex << (vcache_ptr + req_offset) << std::dec << ")" << std::endl;
        exit(1);
    }
    CHECK_CUDA(cuMemSetAccess(kcache_ptr + req_offset, page_size, &accessDesc, 1));
    CHECK_CUDA(cuMemSetAccess(vcache_ptr + req_offset, page_size, &accessDesc, 1));
    cuda_pagemap[std::make_tuple(reqId, req_offset, layer_idx)] = std::make_pair(k_page, v_page);
}

void do_cuda_kvcache_cleanup() {
    // First unmap all individual pages from the pagemap
    for (auto& mapping : cuda_pagemap) {
        auto [reqId, req_offset, layer_idx] = mapping.first;
        auto [k_page, v_page] = mapping.second;
        
        // Find the corresponding virtual addresses
        CUdeviceptr kcache_ptr = 0, vcache_ptr = 0;
        if (layer_idx < k_tensors.size()) {
            kcache_ptr = reinterpret_cast<CUdeviceptr>(k_tensors[layer_idx].data_ptr());
            vcache_ptr = reinterpret_cast<CUdeviceptr>(v_tensors[layer_idx].data_ptr());
            
            // Safely unmap if addresses are valid
            if (kcache_ptr != 0 && vcache_ptr != 0) {
                CUresult result_k = cuMemUnmap(kcache_ptr + req_offset, page_size);
                CUresult result_v = cuMemUnmap(vcache_ptr + req_offset, page_size);
                // Don't exit on unmap failure during cleanup - just log
                if (result_k != CUDA_SUCCESS || result_v != CUDA_SUCCESS) {
                    std::cerr << "[vAttention] Warning: Failed to unmap some pages during cleanup" << std::endl;
                }
            }
        }
    }
    cuda_pagemap.clear();
    
    // Then release virtual address spaces
    for (int i = 0; i < k_tensors.size(); i++) {
        CUdeviceptr k_ptr = reinterpret_cast<CUdeviceptr>(k_tensors[i].data_ptr());
        CUdeviceptr v_ptr = reinterpret_cast<CUdeviceptr>(v_tensors[i].data_ptr());
        
        if (k_ptr != 0) {
            CUresult result = cuMemAddressFree(k_ptr, virt_buff_size);
            if (result != CUDA_SUCCESS) {
                std::cerr << "[vAttention] Warning: Failed to free k virtual address space" << std::endl;
            }
        }
        
        if (v_ptr != 0) {
            CUresult result = cuMemAddressFree(v_ptr, virt_buff_size);
            if (result != CUDA_SUCCESS) {
                std::cerr << "[vAttention] Warning: Failed to free v virtual address space" << std::endl;
            }
        }
    }

    // Finally release physical page handles
    for(int i = 0; i < cuda_pages.size(); i++) {
        CUresult result = cuMemRelease(cuda_pages[i]);
        if (result != CUDA_SUCCESS) {
            const char *errStr = NULL;
            cuGetErrorString(result, &errStr);
            std::cerr << "[vAttention] Warning: Failed to release page handle " << i 
                      << ": " << errStr << std::endl;
        }
    }
}
