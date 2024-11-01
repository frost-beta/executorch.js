#ifndef SRC_EVALUE_H_
#define SRC_EVALUE_H_

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <kizunapi.h>

namespace er = executorch::runtime;

namespace ki {

template<>
struct Type<er::Tag> {
  static constexpr const char* name = "Tag";
  static inline napi_status ToNode(napi_env env,
                                   er::Tag value,
                                   napi_value* result) {
    return ConvertToNode(env, static_cast<uint32_t>(value), result);
  }
};

template<typename T>
struct Type<er::ArrayRef<T>> {
  static constexpr const char* name = "Array";
  static inline napi_status ToNode(napi_env env,
                                   const er::ArrayRef<T>& arr,
                                   napi_value* result) {
    return VectorLikeToNode(env, arr, result);
  }
};

template<typename T>
struct Type<er::Span<T>> {
  static constexpr const char* name = "Array";
  static inline napi_status ToNode(napi_env env,
                                   const er::Span<T>& span,
                                   napi_value* result) {
    return VectorLikeToNode(env, span, result);
  }
};

template<>
struct Type<er::EValue> {
  static napi_status ToNode(napi_env env,
                            const er::EValue& value,
                            napi_value* result);
  static std::optional<er::EValue> FromNode(napi_env env,
                                            napi_value value);
};

}  // namespace ki

#endif  // SRC_EVALUE_H_
