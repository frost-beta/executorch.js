#ifndef SRC_TENSOR_H_
#define SRC_TENSOR_H_

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <kizunapi.h>

namespace ea = executorch::aten;

namespace etjs {

class Tensor {
 public:
  Tensor(std::vector<uint8_t> data,
         std::vector<ea::SizesType> shape,
         ea::ScalarType dtype);
  ~Tensor();

  ea::TensorImpl* impl() const { return &impl_; }

 private:
  std::vector<uint8_t> data_;
  std::vector<ea::SizesType> shape_;
  std::vector<ea::DimOrderType> dim_order_;
  std::vector<ea::StridesType> strides_;
  ea::TensorImpl impl_;
};

}  // namespace etjs

namespace ki {

template<>
struct Type<etjs::Tensor> : public AllowPassByValue<etjs::Tensor> {
  static constexpr const char* name = "Tensor";
  static constexpr bool allow_function_call = true;
  static void Define(napi_env env, napi_value, napi_value prototype);
  static etjs::Tensor* Constructor(napi_env env,
                                   napi_value value,
                                   std::optional<ea::ScalarType> dtype);
};

}  // namespace ki

#endif  // SRC_TENSOR_H_
