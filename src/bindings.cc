#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/runtime.h>

#include "src/evalue.h"
#include "src/module.h"
#include "src/scalar.h"
#include "src/tensor.h"

namespace er = executorch::runtime;

namespace {

napi_value Init(napi_env env, napi_value exports) {
  er::runtime_init();
  napi_value backends = ki::CreateObject(env);
  ki::Set(env, backends,
#if defined(ETJS_BACKEND_COREML)
          "coreml", true,
#else
          "coreml", false,
#endif
#if defined(ETJS_BACKEND_MPS)
          "mps", true,
#else
          "mps", false,
#endif
#if defined(ETJS_BACKEND_XNNPACK)
          "xnnpack", true,
#else
          "xnnpack", false,
#endif
          "cpu", true);
  ki::Set(env, exports,
          "Module", ki::Class<ee::Module>(),
          "Scalar", ki::Class<ea::Scalar>(),
          "Tensor", ki::Class<etjs::Tensor>(),
          "ScalarType", etjs::CreateScalarTypeEnum(env),
          "Tag", etjs::CreateTagEnum(env),
          "backends", backends,
          "elementSize", &er::elementSize);
  return exports;
}

}  // namespace

NAPI_MODULE(executorch, Init)
