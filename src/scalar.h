#ifndef SRC_SCALAR_H_
#define SRC_SCALAR_H_

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <kizunapi.h>

namespace ea = executorch::aten;

namespace ki {

template<>
struct Type<ea::ScalarType> {
  static constexpr const char* name = "ScalarType";
  static napi_status ToNode(napi_env env,
                            ea::ScalarType value,
                            napi_value* result);
  static std::optional<ea::ScalarType> FromNode(napi_env env,
                                                napi_value value);
};

template<>
struct Type<ea::Scalar> : public AllowPassByValue<ea::Scalar> {
  static constexpr const char* name = "Scalar";
  static constexpr bool allow_function_call = true;
  static void Define(napi_env env, napi_value, napi_value prototype);
  static ea::Scalar* Constructor(Arguments* args);
};

}  // namespace ki

#endif  // SRC_SCALAR_H_
