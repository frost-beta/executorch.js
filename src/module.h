#ifndef SRC_MODULE_H_
#define SRC_MODULE_H_

#include <executorch/extension/module/module.h>
#include <kizunapi.h>

namespace ee = executorch::extension;

namespace ki {

template<>
struct Type<ee::Module> {
  static constexpr const char* name = "Module";
  static void Define(napi_env env, napi_value, napi_value prototype);
  static ee::Module* Constructor(Arguments* args);
  static void Destructor(ee::Module* mod);
};

}  // namespace ki

#endif  // SRC_MODULE_H_
