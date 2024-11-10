#include "src/evalue.h"
#include "src/scalar.h"
#include "src/tensor.h"

namespace etjs {

napi_value CreateTagEnum(napi_env env) {
  napi_value obj = ki::CreateObject(env);
#define DEFINE_ENUM(name) \
  ki::Set(env, obj, \
          #name, static_cast<int>(er::Tag::name), \
          static_cast<int>(er::Tag::name), #name);
  EXECUTORCH_FORALL_TAGS(DEFINE_ENUM)
#undef DEFINE_ENUM
  return obj;
}

}  // namespace etjs

namespace ki {

// static
napi_status Type<er::EValue>::ToNode(napi_env env,
                                     const er::EValue& evalue,
                                     napi_value* result) {
  switch (evalue.tag) {
    case er::Tag::None:
      return napi_get_null(env, result);
    case er::Tag::Tensor:
      return ConvertToNode(env, evalue.toTensor(), result);
    case er::Tag::String:
      return ConvertToNode(env, evalue.toString().data(), result);
    case er::Tag::Bool:
      return ConvertToNode(env, evalue.toBool(), result);
    case er::Tag::Double:
      return ConvertToNode(env, evalue.toDouble(), result);
    case er::Tag::Int:
      return ConvertToNode(env, evalue.toInt(), result);
    case er::Tag::ListBool:
      return ConvertToNode(env, evalue.toBoolList(), result);
    case er::Tag::ListDouble:
      return ConvertToNode(env, evalue.toDoubleList(), result);
    case er::Tag::ListInt:
      return ConvertToNode(env, evalue.toIntList(), result);
    case er::Tag::ListTensor:
      return ConvertToNode(env, evalue.toTensorList(), result);
    case er::Tag::ListOptionalTensor:
      return ConvertToNode(env, evalue.toListOptionalTensor(), result);
    default:
      return napi_generic_failure;
  }
}

}  // namespace ki
