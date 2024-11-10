#ifndef SRC_EVALUE_H_
#define SRC_EVALUE_H_

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <kizunapi.h>

namespace ea = executorch::aten;
namespace er = executorch::runtime;

namespace etjs {

napi_value CreateTagEnum(napi_env env);

}  // namespace etjs

namespace ki {

template<typename T>
struct Type<ea::optional<T>> {
  static constexpr const char* name = Type<T>::name;
  static napi_status ToNode(napi_env env,
                            const ea::optional<T>& value,
                            napi_value* result) {
    if (!value)
      return napi_get_undefined(env, result);
    return ConvertToNode(env, value.value(), result);
  }
  static std::optional<ea::optional<T>> FromNode(napi_env env,
                                                 napi_value value) {
    napi_valuetype type;
    if (napi_typeof(env, value, &type) != napi_ok)
      return std::nullopt;
    if (type == napi_undefined || type == napi_null)
      return ea::optional<T>();
    return Type<T>::FromNode(env, value);
  }
};

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
};

}  // namespace ki

#endif  // SRC_EVALUE_H_
