#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/runtime.h>

#include "src/evalue.h"
#include "src/error.h"
#include "src/scalar.h"
#include "src/tensor.h"

namespace ee = executorch::extension;
namespace er = executorch::runtime;

namespace ki {

// ======================== Intermediate types =========================

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

// ============================= Exports ===============================

template<>
struct Type<er::TensorInfo> {
  static constexpr const char* name = "TensorInfo";
  static napi_status ToNode(napi_env env,
                            const er::TensorInfo& value,
                            napi_value* result) {
    *result = CreateObject(env);
    Set(env, *result,
        "sizes", value.sizes(),
        "dimOrder", value.dim_order(),
        "scalarType", value.scalar_type(),
        "isMemoryPlanned", value.is_memory_planned(),
        "nbytes", value.nbytes());
    return napi_ok;
  }
};

template<>
struct Type<er::MethodMeta> : public AllowPassByValue<er::MethodMeta> {
  static constexpr const char* name = "MethodMeta";
  static void Define(napi_env env, napi_value, napi_value prototype) {
    Set(env, prototype,
        "name", &er::MethodMeta::name,
        "numInputs", &er::MethodMeta::num_inputs,
        "inputTag", &er::MethodMeta::input_tag,
        "inputTensorMeta", &er::MethodMeta::input_tensor_meta,
        "numOutputs", &er::MethodMeta::num_outputs,
        "outputTag", &er::MethodMeta::output_tag,
        "outputTensorMeta", &er::MethodMeta::output_tensor_meta,
        "numMemoryPlannedBuffers", &er::MethodMeta::num_memory_planned_buffers,
        "memoryPlannedBufferSize", &er::MethodMeta::memory_planned_buffer_size);
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
        "isMethodLoaded", &ee::Module::is_method_loaded,
        "methodMeta", &ee::Module::method_meta,
        "execute", static_cast<er::Result<std::vector<er::EValue>>
                                   (ee::Module::*)(
                                        const std::string&,
                                        const std::vector<er::EValue>&)>(
                                             &ee::Module::execute));
  }
  static inline ee::Module* Constructor(Arguments* args) {
    if (auto s = args->TryGetNext<std::string>(); s) {
      return new ee::Module(s.value());
    }
    if (auto u = args->GetNext<etjs::Buffer>(); u) {
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
          "Module", ki::Class<ee::Module>(),
          "Scalar", ki::Class<ea::Scalar>(),
          "Tensor", ki::Class<etjs::Tensor>(),
          "ScalarType", etjs::CreateScalarTypeEnum(env));
  return exports;
}

}  // namespace

NAPI_MODULE(executorch, Init)
