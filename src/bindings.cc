#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/runtime.h>
#include <kizunapi.h>

#include "src/buffer.h"
#include "src/error.h"

namespace ee = executorch::extension;
namespace er = executorch::runtime;

namespace ki {

template<>
struct Type<etjs::UnmanagedBuffer> {
  static constexpr const char* name = "UnmanagedBuffer";
  static std::optional<etjs::UnmanagedBuffer> FromNode(napi_env env,
                                                       napi_value value) {
    void* data;
    size_t size;
    if (napi_get_buffer_info(env, value, &data, &size) != napi_ok)
      return std::nullopt;
    // We are assuming the Buffer is kept alive in JS.
    return etjs::UnmanagedBuffer{data, size};
  }
};

template<>
struct Type<er::Error> {
  static constexpr const char* name = "Error";
  static napi_status ToNode(napi_env env, er::Error value, napi_value* result) {
    if (value == er::Error::Ok)
      return napi_get_undefined(env, result);
    return napi_create_error(env,
                             ToNodeValue(env, etjs::ErrorCodeToString(value)),
                             ToNodeValue(env, etjs::ErrorCodeToMessage(value)),
                             result);
  }
};

template<>
struct Type<er::EventTracer*> {
  static constexpr const char* name = "EventTracer";
  static std::optional<er::EventTracer*> FromNode(napi_env env,
                                                  napi_value value) {
    // Always pass nullptr for now.
    return static_cast<er::EventTracer*>(nullptr);
  }
};

template<>
struct Type<er::Program::Verification> {
  static constexpr const char* name = "Verification";
  static std::optional<er::Program::Verification> FromNode(napi_env env,
                                                           napi_value value) {
    auto str = FromNodeTo<std::string>(env, value);
    if (!str)
      return std::nullopt;
    if (*str == "minimal")
      return er::Program::Verification::Minimal;
    if (*str == "internal-consistency")
      return er::Program::Verification::InternalConsistency;
    return std::nullopt;
  }
};

template<typename T>
struct Type<er::Result<T>> {
  static constexpr const char* name = Type<T>::name;
  static napi_status ToNode(napi_env env,
                            const er::Result<T>& value,
                            napi_value* result) {
    if (!value.ok())
      return ConvertToNode(env, value.error(), result);
    return ConvertToNode(env, value.get(), result);
  }
};

template<>
struct Type<ee::Module> {
  static constexpr const char* name = "Module";
  static void Define(napi_env env, napi_value, napi_value prototype) {
    Set(env, prototype,
        "load", &ee::Module::load,
        "isLoaded", &ee::Module::is_loaded,
        "methodNames", &ee::Module::method_names,
        "loadMethod", &ee::Module::load_method,
        "isMethodLoaded", &ee::Module::is_method_loaded);
  }
  static inline ee::Module* Constructor(Arguments* args) {
    if (auto s = args->TryGetNext<std::string>(); s) {
      return new ee::Module(s.value());
    }
    if (auto u = args->GetNext<etjs::UnmanagedBuffer>(); u) {
      return new ee::Module(std::make_unique<ee::BufferDataLoader>(u->data,
                                                                   u->size));
    }
    args->ThrowError("String or Buffer");
    return nullptr;
  }
  static inline void Destructor(ee::Module* mod) {
    delete mod;
  }
};

}  // namespace ki

namespace {

napi_value Init(napi_env env, napi_value exports) {
  er::runtime_init();
  ki::Set(env, exports,
          "Module", ki::Class<ee::Module>());
  return exports;
}

}  // namespace

NAPI_MODULE(executorch, Init)
