#include "src/evalue.h"
#include "src/scalar.h"
#include "src/tensor.h"

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
    default:
      return napi_generic_failure;
  }
}

// static
std::optional<er::EValue> Type<er::EValue>::FromNode(napi_env env,
                                                     napi_value value) {
  napi_valuetype type;
  if (napi_typeof(env, value, &type) != napi_ok)
    return std::nullopt;
  switch (type) {
    case napi_null:
      return er::EValue();
    case napi_number:
      return er::EValue(FromNodeTo<double>(env, value).value());
    case napi_boolean:
      return er::EValue(FromNodeTo<bool>(env, value).value());
    case napi_object:
      if (auto t = FromNodeTo<etjs::Tensor*>(env, value); t)
        return er::EValue(ea::Tensor(t.value()->impl()));
      if (auto s = FromNodeTo<ea::Scalar*>(env, value); s)
        return er::EValue(*s.value());
      // EValue does not store the array, skip it for now.
    case napi_string:
      // EValue does not store the string, skip it for now.
    default:
      return std::nullopt;
  }
}

}  // namespace ki
