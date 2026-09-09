/**
 * @file ForwardCapture.h
 * @brief C3 forward 整图捕获（只读快照, off-path）
 * @date 2026-09-07 (通用图融合: docs/C3_UNIVERSAL_FUSION_DESIGN.md L1 后半前置)
 *
 * @details 补通用图融合缺失的 forward 输入: 一次前向的完整 C3 Graph。
 * 实现为纯只读快照——autograd 前向已维护 ::Node DAG, 这里从根输出 Tensor 沿
 * getUpStreamNodes 上游遍历, 翻译成 c3::Graph, 供 FusionPlanner 判定。
 *
 * 语义:
 *  - 不触发任何执行/编译, 不触碰 dispatch / in_autograd 短路(红线外)。
 *  - ::Node::getUpStreamNodes()[i] 与 getInputs()[i] 索引对齐。
 *  - 输入 i: 上游是计算 op 节点 → 递归(图内中间量);
 *            上游是 GradAccumulator(requires_grad leaf/参数) → 外部叶(去重);
 *            上游是 nullptr(非 grad 常量) → 外部叶(每次出现新建, 非共享不影响连通性)。
 *  - 不支持的前向 op → 报错列出类型(捕获层暂不静默吞掉, 便于补齐)。
 */

#ifndef CTORCH_C3_FORWARD_CAPTURE_H
#define CTORCH_C3_FORWARD_CAPTURE_H

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Graph.h"
#include "Tensor.h"

namespace ct {
namespace c3 {

/// 一次 forward 捕获结果
struct ForwardCaptureResult {
    Graph graph;                              ///< 捕获到的 C3 forward Graph
    std::unordered_map<const void*, size_t> node_to_c3; ///< autograd op Node* -> c3 节点 id
    size_t root_c3_id = SIZE_MAX;             ///< 根输出对应的 c3 节点 id
    bool ok = false;
    std::string error;
    size_t external_leaf_count = 0;           ///< 去重后的外部叶输入数
    size_t unsupported_node_count = 0;        ///< 遇不支持前向 op 的节点数(置 ok=false)
};

/**
 * @class ForwardCapture
 * @brief 从根输出 Tensor 上游构建 forward 的 c3::Graph(只读)。
 */
class ForwardCapture {
public:
    /// 捕获以 root 为输出的整条上游 forward 子图
    static ForwardCaptureResult capture(const Tensor& root);

private:
    ForwardCapture() = default;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_FORWARD_CAPTURE_H
