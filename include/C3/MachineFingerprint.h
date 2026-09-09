/**
 * @file MachineFingerprint.h
 * @brief C3 部署机器指纹（deploy-time 校准产物, O(1) 运行时读取）
 * @date 2026-09-07 (docs/C3_DEPLOY_AUTOTUNE_DESIGN.md 落地骨架)
 *
 * @details 部署时由 `c3ctl calibrate` 一次性测试并写盘; 运行时(Engine/编译路径)启动时
 *          load 一次进内存, 之后以 O(1) 常数时间查询——不每次解析文件。
 *          首版只含 cost model 需要的关键: 有效内存带宽 + 单次 launch 税(等价字节)。
 *          缺失文件/字段 → 用保守默认常量, 绝不抛异常影响训练。
 */

#ifndef CTORCH_C3_MACHINE_FINGERPRINT_H
#define CTORCH_C3_MACHINE_FINGERPRINT_H

#include <cstdint>
#include <mutex>
#include <string>

namespace ct {
namespace c3 {

/// 机器指纹内容（一次 load 后为纯数据, 供 O(1) 读）
struct FingerprintData {
    std::string machine_label;   ///< CPU 型号+核数等(轻量自述, 供人读)
    std::string calibrated_at;   ///< 校准时间(ISO)
    double bandwidth_gbps = 0.0; ///< 有效内存带宽 GB/s
    double launch_us = 0.0;      ///< 单次 kernel launch 固定耗时 us
    uint64_t launch_unit_bytes = 0; ///< launch 税等价字节 = launch_us * bandwidth_bytes/us
    std::string method_version;  ///< 校准方法版本(数据可复现性)
};

/**
 * @class MachineFingerprint
 * @brief 线程安全单例; load 一次后 getter 均 O(1)。
 */
class MachineFingerprint {
public:
    static MachineFingerprint& instance();

    /// 若尚未加载, 尝试从默认路径加载(env C3_FINGERPRINT 优先, 否则 ./c3.fingerprint)。
    /// 进程内仅尝试一次(成败都缓存), 不强制: 无文件则保持默认常量, 返回 false。
    bool loadDefault();

    /// 从显式路径加载; 成功返回 true 并填入 data_。
    bool load(const std::string& path);

    /// 当前是否已成功加载过一份指纹。
    bool loaded() const;

    const FingerprintData& data() const;

    /// O(1): launch 税等价字节(未校准回退 400KB 保守默认)
    uint64_t launchUnitBytes() const;
    /// O(1): 单次 launch 固定耗时 us(未校准回退 2.0)
    double launchUs() const;
    /// O(1): 有效内存带宽 GB/s(未校准回退 200.0)
    double bandwidthGbps() const;

    /// 写入一份指纹到 path(供 c3ctl calibrate 调用)。
    static bool save(const std::string& path, const FingerprintData& d);

    /// 运行时默认配置文件名
    static const char* kDefaultPath;

private:
    MachineFingerprint() = default;
    mutable std::mutex mutex_;
    FingerprintData data_;
    bool loaded_ = false;
    bool default_attempted_ = false;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_MACHINE_FINGERPRINT_H
