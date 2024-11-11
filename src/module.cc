#include "src/module.h"

#include <format>

#include <executorch/extension/data_loader/buffer_data_loader.h>

#include "src/evalue.h"
#include "src/error.h"
#include "src/scalar.h"
#include "src/tensor.h"
#include "src/worker.h"

namespace {

// According to MethodMeta::input_tag/output_tag, these are the types we only
// need to support
using EValueVariant = std::variant<ea::Tensor, std::string, double, bool>;

std::variant<std::string, er::Result<std::vector<er::EValue>>>
ExecuteImpl(ee::Module* mod,
            const std::string& name,
            const std::vector<EValueVariant>& args) {
  auto meta = mod->method_meta(name);
  if (!meta.ok())
    return std::format("Method \"{}\" does not exist.", name);
  if (meta->num_inputs() != args.size())
    return std::format("Expect {} arg(s) but only got {}.",
                       meta->num_inputs(), args.size());
  std::vector<er::EValue> inputs;
  for (size_t i = 0; i < args.size(); ++i) {
    er::Tag tag = meta->input_tag(i).get();
    switch (tag) {
      case er::Tag::Tensor:
        if (auto* t = std::get_if<ea::Tensor>(&args[i]); t) {
          inputs.push_back(er::EValue(std::move(*t)));
          break;
        }
        return std::format("Argument {} should be Tensor.", i);
      case er::Tag::String:
        if (auto* s = std::get_if<std::string>(&args[i]); s) {
          inputs.push_back(er::EValue(s->c_str(), s->size()));
          break;
        }
        return std::format("Argument {} should be String.", i);
      case er::Tag::Int:
        if (auto* d = std::get_if<double>(&args[i]); d) {
          inputs.push_back(er::EValue(static_cast<int64_t>(*d)));
          break;
        }
        return std::format("Argument {} should be interger.", i);
      case er::Tag::Double:
        if (auto* d = std::get_if<double>(&args[i]); d) {
          inputs.push_back(er::EValue(*d));
          break;
        }
        return std::format("Argument {} should be number.", i);
      default:
        return std::format("Unexpected EValue tag {}.", static_cast<int>(tag));
    }
  }
  return mod->execute(name, inputs);
}

napi_value Execute(ee::Module* mod,
                   napi_env env,
                   std::string name,
                   std::vector<EValueVariant> args) {
  return etjs::RunInWorker<decltype(ExecuteImpl(mod, name, args))>(
      env,
      "execute",
      [mod, name = std::move(name), args = std::move(args)]() {
        return ExecuteImpl(mod, name, args);
      });
}

napi_value ExecuteSync(ee::Module* mod,
                       napi_env env,
                       const std::string& name,
                       const std::vector<EValueVariant>& args) {
  return ki::ToNodeValue(env, ExecuteImpl(mod, name, args));
}

napi_value Load(ee::Module* mod,
                napi_env env,
                er::Program::Verification verification) {
  return etjs::RunInWorker<decltype(mod->load())>(
      env,
      "load",
      [mod, verification]() {
        return mod->load(verification);
      });
}

}  // namespace

namespace ki {

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

// static
void Type<ee::Module>::Define(napi_env env, napi_value, napi_value prototype) {
  Set(env, prototype,
      "load", MemberFunction(&Load),
      "loadSync", &ee::Module::load,
      "isLoaded", &ee::Module::is_loaded,
      "methodNames", &ee::Module::method_names,
      "loadMethod", &ee::Module::load_method,
      "isMethodLoaded", &ee::Module::is_method_loaded,
      "methodMeta", &ee::Module::method_meta,
      "execute", MemberFunction(&Execute),
      "executeSync", MemberFunction(&ExecuteSync));
}

// static
ee::Module* Type<ee::Module>::Constructor(Arguments* args) {
  if (auto s = args->TryGetNext<std::string>(); s) {
    return new ee::Module(s.value(),
                          // Some linux envs do not support mlock.
                          ee::Module::LoadMode::MmapUseMlockIgnoreErrors);
  }
  if (auto u = args->GetNext<etjs::Buffer>(); u) {
    return new ee::Module(std::make_unique<ee::BufferDataLoader>(u->data,
                                                                 u->size));
  }
  args->ThrowError("String or Buffer");
  return nullptr;
}

// static
void Type<ee::Module>::Destructor(ee::Module* mod) {
  delete mod;
}

}  // namespace ki
