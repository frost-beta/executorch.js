#include "src/scalar.h"

namespace etjs {

napi_value CreateScalarTypeEnum(napi_env env) {
  napi_value obj = ki::CreateObject(env);
#define DEFINE_ENUM(unused, name) \
  ki::Set(env, obj, \
          #name, static_cast<int>(ea::ScalarType::name), \
          static_cast<int>(ea::ScalarType::name), #name);
  ET_FORALL_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
  return obj;
}

}  // namespace etjs

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
  auto i = FromNodeTo<int>(env, value);
  if (!i)
    return std::nullopt;
  if (i.value() < 0 || i.value() >= static_cast<int>(ea::ScalarType::Undefined))
    return std::nullopt;
  return static_cast<ea::ScalarType>(i.value());
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
