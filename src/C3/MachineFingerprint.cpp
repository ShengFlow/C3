/**
 * @file MachineFingerprint.cpp
 * @brief C3 部署机器指纹实现（deploy-time 校准产物, O(1) 运行时读取）
 * @date 2026-09-07
 */

#include "C3/MachineFingerprint.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace ct {
namespace c3 {

const char* MachineFingerprint::kDefaultPath = "c3.fingerprint";

namespace {
constexpr uint64_t kDefaultLaunchUnitBytes = 400ull * 1024; // ~2us @ 200GB/s
constexpr double kDefaultLaunchUs = 2.0;
constexpr double kDefaultBandwidthGbps = 200.0;
}

MachineFingerprint& MachineFingerprint::instance() {
    static MachineFingerprint s;
    return s;
}

bool MachineFingerprint::loadDefault() {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (default_attempted_) return loaded_;
        default_attempted_ = true; // 进程内只尝试一次(成败都缓存), 避免运行时反复 open 缺失文件
    }
    const char* env = std::getenv("C3_FINGERPRINT");
    std::string path = env ? std::string(env) : std::string(kDefaultPath);
    return load(path); // load 内部置 loaded_ = true(成功时)
}

bool MachineFingerprint::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    FingerprintData d;
    std::string line;
    auto num = [&](const std::string& key, double* out, std::string* raw=nullptr) {
        // helper not used; keep simple below
    };
    (void)num;
    while (std::getline(f, line)) {
        // 去首尾空白
        size_t b = line.find_first_not_of(" \t\r");
        if (b == std::string::npos) continue;
        size_t e = line.find_last_not_of(" \t\r");
        std::string l = line.substr(b, e - b + 1);
        if (l.empty() || l[0] == '#') continue;
        size_t eq = l.find('=');
        if (eq == std::string::npos) continue;
        std::string key = l.substr(0, eq);
        std::string val = l.substr(eq + 1);
        // 去 key 两侧空白(行内可能有 "machine_label = v")
        size_t kb = key.find_first_not_of(" \t");
        if (kb != std::string::npos) key = key.substr(kb);
        size_t ke = key.find_last_not_of(" \t\r");
        if (ke != std::string::npos) key = key.substr(0, ke + 1);
        size_t vb = val.find_first_not_of(" \t");
        if (vb != std::string::npos) val = val.substr(vb);
        size_t ve = val.find_last_not_of(" \t\r");
        if (ve != std::string::npos) val = val.substr(0, ve + 1);
        if (key == "machine_label") d.machine_label = val;
        else if (key == "calibrated_at") d.calibrated_at = val;
        else if (key == "method_version") d.method_version = val;
        else if (key == "bandwidth_gbps") d.bandwidth_gbps = std::atof(val.c_str());
        else if (key == "launch_us") d.launch_us = std::atof(val.c_str());
        else if (key == "launch_unit_bytes") d.launch_unit_bytes = (uint64_t)std::atoll(val.c_str());
    }
    if (d.launch_unit_bytes == 0 && d.launch_us > 0 && d.bandwidth_gbps > 0) {
        d.launch_unit_bytes = (uint64_t)(d.launch_us * d.bandwidth_gbps * 1000.0);
    }
    {
        std::lock_guard<std::mutex> lk(mutex_);
        data_ = d;
        loaded_ = true;
    }
    return true;
}

bool MachineFingerprint::loaded() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return loaded_;
}

const FingerprintData& MachineFingerprint::data() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_;
}

uint64_t MachineFingerprint::launchUnitBytes() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return loaded_ && data_.launch_unit_bytes > 0 ? data_.launch_unit_bytes
                                                  : kDefaultLaunchUnitBytes;
}

double MachineFingerprint::launchUs() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return loaded_ && data_.launch_us > 0 ? data_.launch_us : kDefaultLaunchUs;
}

double MachineFingerprint::bandwidthGbps() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return loaded_ && data_.bandwidth_gbps > 0 ? data_.bandwidth_gbps : kDefaultBandwidthGbps;
}

bool MachineFingerprint::save(const std::string& path, const FingerprintData& d) {
    std::ofstream f(path);
    if (!f) return false;
    f << "# C3 machine fingerprint (deploy-time calibration via c3ctl calibrate)\n";
    f << "# runtime loads once, then reads O(1). Format key = value\n";
    f << "machine_label = " << d.machine_label << "\n";
    f << "calibrated_at = " << d.calibrated_at << "\n";
    f << "method_version = " << d.method_version << "\n";
    f << "bandwidth_gbps = " << d.bandwidth_gbps << "\n";
    f << "launch_us = " << d.launch_us << "\n";
    f << "launch_unit_bytes = " << d.launch_unit_bytes << "\n";
    return f.good();
}

} // namespace c3
} // namespace ct
