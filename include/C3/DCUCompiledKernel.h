#ifndef CTORCH_C3_DCU_COMPILED_KERNEL_H
#define CTORCH_C3_DCU_COMPILED_KERNEL_H

#include "C3/C3Engine.h"
#include "C3/GCVMBridge.h"
#include "Tensor.h"

#include <memory>
#include <string>
#include <vector>

#ifdef WITH_DCU
    #include <hip/hip_runtime.h>
    #include <hsa/hsa.h>
    #include <hsa/hsa_ext_amd.h>
#endif

namespace ct {
namespace c3 {

class DCUCompiledKernel : public CompiledKernel {
public:
    DCUCompiledKernel(std::string code_object,
                      std::string kernel_name,
                      const Graph& graph,
                      int device = 0);
    ~DCUCompiledKernel() override;

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override;

    [[nodiscard]] const std::string& cacheKey() const override { return cache_key_; }
    [[nodiscard]] DeviceType targetDevice() const override { return DeviceType::kDCU; }
    [[nodiscard]] size_t workspaceBytes() const override { return workspace_bytes_; }

    bool installIntoRegistry(op op_type, const KernelShapeInfo& shapes) override;

private:
    std::string code_object_;
    std::string kernel_name_;
    std::string cache_key_;
    int device_;
    size_t workspace_bytes_ = 0;

#ifdef WITH_DCU
    // HSA state
    hsa_executable_t hsa_executable_ = {0};
    hsa_code_object_t hsa_code_object_ = {0};
    uint64_t kernel_handle_ = 0;
    uint32_t kernarg_size_ = 0;
    hsa_agent_t gpu_agent_ = {0};
    hsa_region_t kernarg_region_ = {0};
    hsa_region_t global_region_ = {0};
    hsa_amd_memory_pool_t hsa_pool_ = {0};
    hsa_queue_t* hsa_queue_ = nullptr;
    bool hsa_initialized_ = false;

    // Device memory (allocated using hsa_amd_memory_pool_allocate VRAM)
    void** d_input_buffers_ = nullptr;
    void* d_output_buffer_ = nullptr;
    size_t d_output_bytes_ = 0;
#endif

    bool loadHSAModule();
    bool allocateDeviceMemory(size_t output_bytes);
    bool copyInputsToDevice(const std::vector<Tensor>& inputs);
    Tensor copyOutputToHost(size_t numel, const std::vector<size_t>& shape);
    bool launchKernel(const std::vector<Tensor>& inputs);
};

}  // namespace c3
}  // namespace ct

#endif
