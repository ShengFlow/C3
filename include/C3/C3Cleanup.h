/**
 * @file C3Cleanup.h
 * @generation SHARED 跨代退出清理
 * @brief C3 退出清理公共 helper
 * @details 统一 C3 端到端测试/程序的退出清理序列，确保所有 CompiledKernel / LLVM
 *          module 在静态析构前释放，避免退出时 recursive_mutex / removeModule 崩溃。
 *          所有 C3 测试（test_c3_mnist_train、test_region_fusion_auto 等）在 main
 *          返回前都应调用 ct::c3::shutdownAll()。
 * @date 2026/8/7
 */

#ifndef CTORCH_C3_C3CLEANUP_H
#define CTORCH_C3_C3CLEANUP_H

#include "C3Engine.h"
#include "C3BackwardCapture.h"
#include "C3HotPathManager.h"
#include "RegionFusion.h"

namespace ct {
namespace c3 {

/**
 * @brief 统一的 C3 退出清理序列
 * @details 清理顺序固定为：
 *          1. C3HotPathManager::shutdown()    —— 停止热路径检测/后台编译
 *          2. C3Engine::shutdown()            —— 等待所有后台编译完成并回收线程
 *          3. C3Engine::clearCache()          —— 清空内存缓存
 *          4. RegionFusionRegistry::clear()   —— 释放区域融合注册表
 *          5. C3KernelRegistry::uninstallAll()—— 卸载所有注入的 kernel
 *          6. C3Engine::drainFlatOutPool()    —— 释放 MIMO flat 输出缓冲池(§4.93)
 *
 *          顺序不可随意调整：必须先停止后台任务并清空缓存，再释放各注册表，
 *          确保所有 CompiledKernel / LLVM module 在静态析构前释放。
 */
inline void shutdownAll() {
    C3HotPathManager::instance().shutdown();
    // 反向捕获器仍可能有 detached 编译任务；必须在 Engine/LLVM 资源释放前等待。
    C3BackwardCapture::getInstance().shutdown();
    C3Engine::getInstance().shutdown();
    C3Engine::getInstance().clearCache();
    RegionFusionRegistry::getInstance().clear();
    C3KernelRegistry::getInstance().uninstallAll();
    // 6. [§4.93 A1] 释放 MIMO flat 输出缓冲池中已归还的 buffer(进程级常驻占用)。
    //    放在最后: 前面各步释放 kernel/注册表时可能触发 Tensor 析构将 buffer 归还入池,
    //    须在其后再 drain 才能清干净。池结构本身不析构, 故此后若有 Tensor 析构仍安全。
    C3Engine::drainFlatOutPool();
}

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_C3CLEANUP_H