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
  ki::Set(env, exports,
          "Module", ki::Class<ee::Module>(),
          "Scalar", ki::Class<ea::Scalar>(),
          "Tensor", ki::Class<etjs::Tensor>(),
          "ScalarType", etjs::CreateScalarTypeEnum(env),
          "elementSize", &er::elementSize);
  return exports;
}

}  // namespace

NAPI_MODULE(executorch, Init)
