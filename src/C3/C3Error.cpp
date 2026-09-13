/**
 * @file C3Error.cpp
 * @brief C3 层统一错误出口的实现(§4.114) —— 唯一允许出现宿主错误头的 c3 文件
 */
#include "C3/C3Error.h"

#include "CtorchError.h"  // 宿主: ErrorPlatform / ErrorType / CtorchError::throwException

namespace ct {
namespace c3 {
namespace {

/// 转发到宿主错误出口。`CtorchError::throwException` 恒抛(内部即 log + throw),
/// 但其声明**未标注 [[noreturn]]**, 故此处显式补 `__builtin_unreachable()`:
/// 既让编译器接受外层函数的 [[noreturn]] 契约, 也不改变语义(该点确实不可达)。
[[noreturn]] void logAndThrow(ErrorType type, const std::string& msg) {
    CtorchError::throwException(ErrorPlatform::kGENERAL, type, msg);
    __builtin_unreachable();
}

} // namespace

void throwCompileError(const std::string& msg) {
    logAndThrow(ErrorType::UNKNOWN, msg);
}

void throwExecError(const std::string& msg) {
    logAndThrow(ErrorType::KERNEL_LAUNCH, msg);
}

} // namespace c3
} // namespace ct
