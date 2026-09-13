/**
 * @file C3Error.h
 * @brief C3 层统一错误出口(§4.114)
 * @details 此前 c3 各处直接 `throw std::runtime_error(msg)`, 完全绕过框架的错误日志
 *          (无平台/类型/错误码, 不进 log 流)。本头提供两个语义化出口, 把
 *          「用哪个 ErrorPlatform / ErrorType」的判断收敛到一处实现。
 *
 *          **关键约束(勿误改)**: `CtorchError::throwException` 内部就是
 *          `throw std::runtime_error(msg)`(见 include/CtorchError.h), 差别仅在于它先调用
 *          `log(...)` 记录。因此本改动**不改变抛出类型**, 所有既有
 *          `catch (const std::exception&)` / `catch (const std::runtime_error&)`
 *          语义与 `what()` 文本均逐字不变 —— 变的是可观测性, 不是异常契约。
 *
 *          设计: 本头只依赖 <string>, 不引入宿主头; 宿主头(含枚举定义)只在
 *          C3Error.cpp 里出现, 避免把 CtorchError.h 拉进 c3 的公共头依赖链。
 */
#ifndef CTORCH_C3_ERROR_H
#define CTORCH_C3_ERROR_H

#include <string>

namespace ct {
namespace c3 {

/**
 * @brief 编译/IR 级失败出口: 图结构非法、MLIR 校验失败、lowering 失败、算子不支持等
 * @note 记录为 (ErrorPlatform::kGENERAL, ErrorType::UNKNOWN) —— c3 的编译错误在抛出点
 *       与具体设备无关, 故平台取通用; ErrorType 枚举无「编译」类目(且枚举属公共 ABI 面,
 *       不宜扩展), 故取 UNKNOWN。
 */
[[noreturn]] void throwCompileError(const std::string& msg);

/**
 * @brief 引擎/内核级失败出口: ExecutionEngine 创建失败、kernel 符号查不到、invokePacked 失败等
 * @note 记录为 (ErrorPlatform::kGENERAL, ErrorType::KERNEL_LAUNCH)。
 */
[[noreturn]] void throwExecError(const std::string& msg);

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_ERROR_H
