#include "C3/DCUCompiledKernel.h"
#include "C3/C3KernelRegistry.h"
#include "C3/GCVMBridge.h"
#include "CtorchError.h"

#include <iostream>
#include <cstring>
#include <sstream>

#ifdef WITH_DCU
    #include <hsa/hsa.h>
    #include <hsa/hsa_ext_amd.h>
#endif

namespace ct {
namespace c3 {

// ======================= 构造 / 析构 =======================

DCUCompiledKernel::DCUCompiledKernel(std::string code_object,
                                       std::string kernel_name,
                                       const Graph& graph,
                                       int device)
    : code_object_(std::move(code_object))
    , kernel_name_(std::move(kernel_name))
    , device_(device)
{
    cache_key_ = "dcu_" + graph.toString() + "_" + std::to_string(device);
#ifdef WITH_DCU
    if (!loadHSAModule()) {
        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
            "DCUCompiledKernel: failed to load HSA module, execute() will fail");
    }
#endif
}

DCUCompiledKernel::~DCUCompiledKernel() {
#ifdef WITH_DCU
    // 1. Free device memory (must free first)
    if (d_output_buffer_) { 
        hsa_memory_free(d_output_buffer_); 
        d_output_buffer_ = nullptr; 
    }
    if (d_input_buffers_) {
        for (size_t i = 0; i < 2; ++i) { 
            if (d_input_buffers_[i]) hsa_memory_free(d_input_buffers_[i]); 
        }
        delete[] d_input_buffers_; d_input_buffers_ = nullptr;
    }
    
    // 2. Destroy HSA queue (must destroy before executable)
    if (hsa_queue_) { 
        hsa_queue_destroy(hsa_queue_); 
        hsa_queue_ = nullptr; 
    }
    
    // 3. Destroy executable and code object
    if (hsa_executable_.handle) { 
        hsa_executable_destroy(hsa_executable_); 
        hsa_executable_ = {0};
    }
    if (hsa_code_object_.handle) { 
        hsa_code_object_destroy(hsa_code_object_); 
        hsa_code_object_ = {0};
    }
    
    // 4. Do NOT call hsa_shut_down() — GCVM owns the process-wide HSA lifecycle.
    // Calling hsa_shut_down() will cause subsequent runs/components to segfault.
#endif
}

// ======================= loadHSAModule =======================

bool DCUCompiledKernel::loadHSAModule() {
#ifndef WITH_DCU
    return false;
#else
    if (code_object_.empty()) return false;

    // 1. Init HSA runtime
    hsa_status_t status = hsa_init();
    if (status != HSA_STATUS_SUCCESS) {
        // HSA might already be initialized by GCVM, which returns 4104 (REINITIALIZED)
        if (status != 4104) {
            std::cerr << "[DCU] hsa_init failed: " << status << std::endl;
            return false;
        }
    }
    hsa_initialized_ = false;  // we don't own HSA lifecycle, GCVM does

    // 2. Find GPU agent
    auto find_gpu = [](hsa_agent_t agent, void* data) -> hsa_status_t {
        hsa_device_type_t type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type == HSA_DEVICE_TYPE_GPU) {
            *(hsa_agent_t*)data = agent;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    };
    status = hsa_iterate_agents(find_gpu, &gpu_agent_);
    if (gpu_agent_.handle == 0) {
        std::cerr << "[DCU] No GPU agent found" << std::endl;
        return false;
    }

    // 3. Find kernarg region and global region
    struct RegionInfo {
        hsa_region_t kernarg;
        hsa_region_t global_fine;
        hsa_region_t global_coarse;
    };
    RegionInfo ri = {{0}, {0}, {0}};
    auto enum_regions = [](hsa_region_t region, void* data) -> hsa_status_t {
        auto* ri = (RegionInfo*)data;
        hsa_region_segment_t segment;
        hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
        if (segment == HSA_REGION_SEGMENT_GLOBAL) {
            hsa_region_global_flag_t flags;
            hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
            size_t size;
            hsa_region_get_info(region, HSA_REGION_INFO_SIZE, &size);
            std::cerr << "  [DCU] Region handle=" << region.handle << " flags=0x" << std::hex << flags << std::dec << " size=" << size << std::endl;
            if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
                ri->kernarg = region;
            }
            if (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
                if (ri->global_fine.handle == 0) ri->global_fine = region;
            }
            if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
                if (ri->global_coarse.handle == 0) ri->global_coarse = region;
            }
        }
        return HSA_STATUS_SUCCESS;
    };
    status = hsa_agent_iterate_regions(gpu_agent_, enum_regions, &ri);
    kernarg_region_ = ri.kernarg;
    global_region_ = ri.global_fine;
    if (global_region_.handle == 0) global_region_ = ri.global_coarse;

    if (kernarg_region_.handle == 0) {
        std::cerr << "  [DCU] No dedicated kernarg region, using global" << std::endl;
        kernarg_region_ = global_region_;
    }

    // [DCU Fix] Find actual GPU coarse-grained memory pool for high-performance VRAM allocation
    auto find_pool = [](hsa_amd_memory_pool_t p, void* data) -> hsa_status_t {
        hsa_amd_memory_pool_global_flag_t flags;
        hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
            *(hsa_amd_memory_pool_t*)data = p;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    };
    hsa_pool_.handle = 0;
    hsa_amd_agent_iterate_memory_pools(gpu_agent_, find_pool, &hsa_pool_);
    if (hsa_pool_.handle != 0) {
        std::cerr << "[DCU] Found AMD GPU coarse-grained memory pool: " << hsa_pool_.handle << std::endl;
    } else {
        std::cerr << "[DCU] Warning: AMD coarse-grained pool not found, fallback to standard allocation" << std::endl;
    }

    // 4. Deserialize code object
    status = hsa_code_object_deserialize(code_object_.data(), code_object_.size(), "", &hsa_code_object_);
    if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "[DCU] hsa_code_object_deserialize failed: " << status << std::endl;
        return false;
    }

    // 5. Create executable and load code object
    status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, "", &hsa_executable_);
    if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "[DCU] hsa_executable_create failed: " << status << std::endl;
        return false;
    }

    status = hsa_executable_load_code_object(hsa_executable_, gpu_agent_, hsa_code_object_, "");
    if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "[DCU] hsa_executable_load_code_object failed: " << status << std::endl;
        return false;
    }

    // 6. Freeze
    status = hsa_executable_freeze(hsa_executable_, nullptr);
    if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "[DCU] hsa_executable_freeze failed: " << status << std::endl;
        return false;
    }

    // 7. Get kernel symbol (try name, then name.kd)
    hsa_executable_symbol_t kernel_symbol;
    std::string kd_name = kernel_name_ + ".kd";
    status = hsa_executable_get_symbol_by_name(hsa_executable_, kernel_name_.c_str(), &gpu_agent_, &kernel_symbol);
    if (status != HSA_STATUS_SUCCESS) {
        status = hsa_executable_get_symbol_by_name(hsa_executable_, kd_name.c_str(), &gpu_agent_, &kernel_symbol);
        if (status != HSA_STATUS_SUCCESS) {
            std::cerr << "[DCU] kernel symbol not found: " << kernel_name_ << std::endl;
            return false;
        }
    }

    // 8. Get kernel handle and kernarg size
    status = hsa_executable_symbol_get_info(kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_handle_);
    if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "[DCU] failed to get kernel object" << std::endl;
        return false;
    }
    hsa_executable_symbol_get_info(kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_size_);

    // 9. Create queue
    status = hsa_queue_create(gpu_agent_, 64, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0, &hsa_queue_);
    if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "[DCU] hsa_queue_create failed: " << status << std::endl;
        return false;
    }

    std::cerr << "[DCU] HSA module loaded OK, kernel_handle=0x" << std::hex << kernel_handle_
              << " kernarg_size=" << std::dec << kernarg_size_ << std::endl;
    return true;
#endif
}

// ======================= allocateDeviceMemory =======================

bool DCUCompiledKernel::allocateDeviceMemory(size_t output_bytes) {
#ifndef WITH_DCU
    (void)output_bytes; return false;
#else
    if (d_output_buffer_ && d_output_bytes_ >= output_bytes) return true;
    if (d_output_buffer_) { hsa_memory_free(d_output_buffer_); d_output_buffer_ = nullptr; }
    d_output_bytes_ = output_bytes;
    hsa_status_t st;
    if (hsa_pool_.handle != 0) {
        st = hsa_amd_memory_pool_allocate(hsa_pool_, output_bytes, 0, &d_output_buffer_);
    } else {
        st = hsa_memory_allocate(global_region_, output_bytes, &d_output_buffer_);
    }
    if (st != HSA_STATUS_SUCCESS) {
        std::cerr << "[DCU] hsa_memory_allocate(output) failed: " << st << std::endl;
        return false;
    }
    return true;
#endif
}

// ======================= copyInputsToDevice =======================

bool DCUCompiledKernel::copyInputsToDevice(const std::vector<Tensor>& inputs) {
#ifndef WITH_DCU
    (void)inputs; return false;
#else
    if (!d_input_buffers_) d_input_buffers_ = new void*[inputs.size()]();
    for (size_t i = 0; i < inputs.size(); ++i) {
        size_t bytes = inputs[i].numel() * sizeof(float);
        if (d_input_buffers_[i]) hsa_memory_free(d_input_buffers_[i]);
        hsa_status_t st;
        if (hsa_pool_.handle != 0) {
            st = hsa_amd_memory_pool_allocate(hsa_pool_, bytes, 0, &d_input_buffers_[i]);
        } else {
            st = hsa_memory_allocate(global_region_, bytes, &d_input_buffers_[i]);
        }
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "[DCU] hsa_memory_allocate(input " << i << ") failed: " << st << std::endl;
            return false;
        }
        st = hsa_memory_copy(d_input_buffers_[i], (void*)inputs[i].data_read<float>(), bytes);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "[DCU] hsa_memory_copy(input " << i << ") failed: " << st << std::endl;
            return false;
        }
    }
    return true;
#endif
}

// ======================= copyOutputToHost =======================

Tensor DCUCompiledKernel::copyOutputToHost(size_t numel, const std::vector<size_t>& shape) {
    Tensor out(ShapeTag{}, shape);
#ifdef WITH_DCU
    if (d_output_buffer_) {
        hsa_status_t st = hsa_memory_copy(out.data_write<float>(), d_output_buffer_, numel * sizeof(float));
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "[DCU] hsa_memory_copy(D2H) failed: " << st << std::endl;
        }
    }
#endif
    return out;
}

// ======================= launchKernel =======================

bool DCUCompiledKernel::launchKernel(const std::vector<Tensor>& inputs) {
#ifndef WITH_DCU
    (void)inputs; return false;
#else
    if (!hsa_queue_ || kernel_handle_ == 0) return false;

    // Allocate kernarg
    void* kernarg = nullptr;
    hsa_status_t st = hsa_memory_allocate(kernarg_region_, kernarg_size_, &kernarg);
    if (st != HSA_STATUS_SUCCESS) {
        std::cerr << "[DCU] kernarg alloc failed: " << st << std::endl;
        return false;
    }

    // Set up kernarg: c3_kernel(ptr a, ptr b, ptr out, i64 n, i64 M, i64 K, i64 N)
    size_t n = inputs[0].numel();
    long n_val = (long)n, M = 0, K = 0, Nval = 0;
    memset(kernarg, 0, kernarg_size_);
    if (inputs.size() >= 2) {
        memcpy(kernarg, &d_input_buffers_[0], sizeof(void*));
        memcpy((char*)kernarg + 8, &d_input_buffers_[1], sizeof(void*));
    }
    memcpy((char*)kernarg + 16, &d_output_buffer_, sizeof(void*));
    memcpy((char*)kernarg + 24, &n_val, sizeof(long));
    memcpy((char*)kernarg + 32, &M, sizeof(long));
    memcpy((char*)kernarg + 40, &K, sizeof(long));
    memcpy((char*)kernarg + 48, &Nval, sizeof(long));

    // Dispatch
    uint64_t packet_id = hsa_queue_add_write_index_relaxed(hsa_queue_, 1);
    hsa_kernel_dispatch_packet_t* packet =
        (hsa_kernel_dispatch_packet_t*)hsa_queue_->base_address + (packet_id % hsa_queue_->size);

    packet->workgroup_size_x = 64;
    packet->workgroup_size_y = 1;
    packet->workgroup_size_z = 1;
    packet->grid_size_x = n;
    packet->grid_size_y = 1;
    packet->grid_size_z = 1;
    packet->kernel_object = kernel_handle_;
    packet->kernarg_address = kernarg;
    packet->private_segment_size = 0;
    packet->group_segment_size = 0;

    uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                      (1 << HSA_PACKET_HEADER_BARRIER);
    uint16_t setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    __atomic_store_n((uint16_t*)(&packet->header), header, __ATOMIC_RELEASE);
    __atomic_store_n((uint16_t*)(&packet->setup), setup, __ATOMIC_RELEASE);

    hsa_signal_store_relaxed(hsa_queue_->doorbell_signal, packet_id);

    // Wait for completion
    hsa_signal_wait_acquire(hsa_queue_->doorbell_signal, HSA_SIGNAL_CONDITION_LT,
                            packet_id + 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

    hsa_memory_free(kernarg);
    return true;
#endif
}

// ======================= execute =======================

std::vector<Tensor> DCUCompiledKernel::execute(const std::vector<Tensor>& inputs) {
    std::vector<Tensor> outputs;
#ifndef WITH_DCU
    (void)inputs;
    return outputs;
#else
    if (!hsa_queue_ || kernel_handle_ == 0) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
            "DCUCompiledKernel::execute: HSA module not loaded");
        return outputs;
    }

    if (!copyInputsToDevice(inputs)) return outputs;

    size_t output_numel = inputs[0].numel();
    std::vector<size_t> output_shape = inputs[0].shape();
    if (!allocateDeviceMemory(output_numel * sizeof(float))) return outputs;

    if (!launchKernel(inputs)) return outputs;

    // HSA doorbell signal wait already done in launchKernel
    outputs.push_back(copyOutputToHost(output_numel, output_shape));
    return outputs;
#endif
}

// ======================= installIntoRegistry =======================

bool DCUCompiledKernel::installIntoRegistry(op op_type, const KernelShapeInfo& shapes) {
    (void)op_type; (void)shapes;
    return false;
}

}  // namespace c3
}  // namespace ct
