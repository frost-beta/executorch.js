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
