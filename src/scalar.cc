#include "src/scalar.h"

namespace ki {

// static
napi_status Type<ea::ScalarType>::ToNode(napi_env env,
                                         ea::ScalarType type,
                                         napi_value* result) {
  return ConvertToNode(env, static_cast<int>(type), result);
}

// static
std::optional<ea::ScalarType> Type<ea::ScalarType>::FromNode(napi_env env,
                                                             napi_value value) {
  return std::nullopt;
}

// static
void Type<ea::Scalar>::Define(napi_env, napi_value, napi_value) {
}

// static
ea::Scalar* Type<ea::Scalar>::Constructor(Arguments* args) {
  if (auto b = args->TryGetNext<bool>(); b)
    return new ea::Scalar(b.value());
  if (auto d = args->TryGetNext<double>(); d)
    return new ea::Scalar(d.value());
  args->ThrowError("Boolean or Number");
  return nullptr;
}

}  // namespace ki
