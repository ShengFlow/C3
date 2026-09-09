/**
 * @file ForwardCapture.cpp
 * @brief C3 forward 整图捕获实现（只读, off-path）
 * @date 2026-09-07
 */

#include "C3/ForwardCapture.h"

#include <functional>
#include <optional>
#include <string>
#include <typeinfo>
#include <unordered_set>

#include "AutoGrad/Node.h"

namespace ct {
namespace c3 {

namespace {

const ::Node* gradAccumulatorIsLeaf(const ::Node* n) {
    // n 为 GradAccumulator ⇒ requires_grad 叶(参数/输入), 是捕获的外部边界
    return n ? (std::string(typeid(*n).name()).find("GradAccumulator") != std::string::npos ? n : nullptr)
             : nullptr;
}

std::vector<size_t> shapeOf(const Tensor& t) { return t.shape(); }

/// 由 autograd op 类型(RTTI)构建 c3 节点 op。仅识别支持的类别; 未知返回 nullopt。
std::optional<NodeVariant> buildOpVariant(const ::Node* node) {
    const auto& inputs = node->getInputs();
    auto inD = [&](size_t i) {
        return TensorDesc::fromShape(i < inputs.size() ? shapeOf(inputs[i])
                                                       : std::vector<size_t>{});
    };
    const std::string nm = std::string(typeid(*node).name());
    if (nm.find("MatMulNode") != std::string::npos)   return NodeVariant{MatMulNode{inD(0), inD(1)}};
    if (nm.find("AddNode") != std::string::npos)      return NodeVariant{AddNode{inD(0), inD(1)}};
    if (nm.find("SubNode") != std::string::npos)      return NodeVariant{SubNode{inD(0), inD(1)}};
    if (nm.find("MulNode") != std::string::npos)      return NodeVariant{MulNode{inD(0), inD(1)}};
    if (nm.find("DivNode") != std::string::npos)      return NodeVariant{DivNode{inD(0), inD(1)}};
    if (nm.find("NegNode") != std::string::npos)      return NodeVariant{NegNode{inD(0)}};
    if (nm.find("ReLUNode") != std::string::npos)     return NodeVariant{ReLUNode{inD(0)}};
    if (nm.find("SigmoidNode") != std::string::npos)  return NodeVariant{SigmoidNode{inD(0)}};
    if (nm.find("TanhNode") != std::string::npos)     return NodeVariant{TanhNode{inD(0)}};
    if (nm.find("ExpNode") != std::string::npos)      return NodeVariant{ExpNode{inD(0)}};
    if (nm.find("LogNode") != std::string::npos)      return NodeVariant{LogNode{inD(0)}};
    if (nm.find("SoftmaxNode") != std::string::npos)  return NodeVariant{SoftmaxNode{inD(0), /*axis*/1}};
    if (nm.find("CrossEntropyNode") != std::string::npos)
        return NodeVariant{CrossEntropyNode{inD(0), inD(1)}};
    return std::nullopt;
}

} // namespace

ForwardCaptureResult ForwardCapture::capture(const Tensor& root) {
    ForwardCaptureResult res;
    std::shared_ptr<::Node> root_node = root.getRelatedNode();
    if (!root_node) {
        res.ok = false;
        res.error = "root tensor has no related autograd node (forward graph empty / not requires_grad)";
        return res;
    }
    if (gradAccumulatorIsLeaf(root_node.get())) {
        res.ok = false;
        res.error = "root is a leaf (requires_grad input), no forward computation above it";
        return res;
    }

    // 后序 DFS: 先处理上游计算 op, 再处理当前 → topo 序(子先父后)
    std::vector<const ::Node*> topo;
    std::unordered_set<const ::Node*> done;
    std::function<void(const ::Node*)> dfs = [&](const ::Node* n) {
        if (!n || done.count(n)) return;
        done.insert(n);
        const auto& ups = n->getUpStreamNodes();
        for (const auto& up : ups) {
            if (!up) continue;
            const ::Node* u = up.get();
            if (gradAccumulatorIsLeaf(u)) continue; // 叶边界, 不向下
            dfs(u);
        }
        topo.push_back(n);
    };
    dfs(root_node.get());

    Graph& g = res.graph;
    std::unordered_map<const void*, size_t> leaf_to_c3; // GradAccumulator* -> c3 input id
    auto leafId = [&](const ::Node* acc, const TensorDesc& d) -> size_t {
        auto it = leaf_to_c3.find(acc);
        if (it != leaf_to_c3.end()) return it->second;
        size_t id = g.addInput(d);
        leaf_to_c3[acc] = id;
        res.external_leaf_count++;
        return id;
    };

    bool first_op = true;
    for (const ::Node* node : topo) {
        auto variant = buildOpVariant(node);
        if (!variant) {
            res.unsupported_node_count++;
            res.ok = false;
            res.error = "unsupported forward node type: " + std::string(typeid(*node).name());
            return res;
        }
        const auto& ups = node->getUpStreamNodes();
        const auto& inputs = node->getInputs();
        std::vector<size_t> c3_ins;
        c3_ins.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            TensorDesc in_desc = TensorDesc::fromShape(shapeOf(inputs[i]));
            if (i < ups.size() && ups[i]) {
                const ::Node* u = ups[i].get();
                if (gradAccumulatorIsLeaf(u)) {
                    c3_ins.push_back(leafId(u, in_desc));
                    continue;
                }
                // 计算中间量: 上游 op 应已在本图(处理于更早的 topo 位置)
                auto it = res.node_to_c3.find(u);
                if (it == res.node_to_c3.end()) {
                    res.ok = false;
                    res.error = "internal: computed upstream not yet translated";
                    return res;
                }
                c3_ins.push_back(it->second);
            } else {
                // 非 grad 常量: 每次出现新建外部叶
                size_t id = g.addInput(in_desc);
                res.external_leaf_count++;
                c3_ins.push_back(id);
            }
        }
        size_t out_id = g.addNode(*variant, c3_ins,
                                  TensorDesc::fromShape(node->getResultShape()));
        res.node_to_c3[node] = out_id;
        if (first_op && node == root_node.get()) { first_op = false; }
    }

    auto root_it = res.node_to_c3.find(root_node.get());
    if (root_it == res.node_to_c3.end()) {
        res.ok = false;
        res.error = "root node not translated";
        return res;
    }
    res.root_c3_id = root_it->second;
    g.markOutput(res.root_c3_id);
    res.ok = true;
    return res;
}

} // namespace c3
} // namespace ct
