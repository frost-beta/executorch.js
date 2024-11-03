#ifndef SRC_TENSOR_H_
#define SRC_TENSOR_H_

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <kizunapi.h>

namespace ea = executorch::aten;

namespace etjs {

// Intermediate type for representing typed buffer.
struct Buffer {
  void* data;
  size_t size;
};

// Provide storage for tensor data.
class Tensor {
 public:
  Tensor(std::vector<uint8_t> data,
         ea::ScalarType dtype,
         std::vector<ea::SizesType> shape,
         std::vector<ea::DimOrderType> dim_order = {},
         std::vector<ea::StridesType> strides = {});
  Tensor(Buffer data,
         ea::ScalarType dtype,
         std::vector<ea::SizesType> shape,
         std::vector<ea::DimOrderType> dim_order = {},
         std::vector<ea::StridesType> strides = {});
  ~Tensor();

  ea::TensorImpl* impl() { return &impl_; }

  const Buffer& data() const { return data_; }
  ea::ScalarType dtype() const { return impl_.dtype(); }
  const std::vector<ea::SizesType>& shape() const { return shape_; }
  const std::vector<ea::DimOrderType>& dim_order() const { return dim_order_; }
  const std::vector<ea::StridesType>& strides() const { return strides_; }
  size_t size() const { return impl_.numel(); }
  size_t nbytes() const { return impl_.nbytes(); }
  size_t itemsize() const { return impl_.element_size(); }

 private:
  Buffer data_;
  std::vector<ea::SizesType> shape_;
  std::vector<ea::DimOrderType> dim_order_;
  std::vector<ea::StridesType> strides_;
  ea::TensorImpl impl_;
  // Only used when this class manages its own data.
  std::vector<uint8_t> managed_data_;
};

}  // namespace etjs

namespace ki {

template<>
struct Type<ea::Tensor> {
  static constexpr const char* name = "Tensor";
  static napi_status ToNode(napi_env env,
                            const ea::Tensor& value,
                            napi_value* result);
};

template<>
struct Type<etjs::Buffer> {
  static constexpr const char* name = "Buffer";
  static napi_status ToNode(napi_env env,
                            const etjs::Buffer& value,
                            napi_value* result);
  static std::optional<etjs::Buffer> FromNode(napi_env env, napi_value value);
};

template<>
struct Type<etjs::Tensor> {
  static constexpr const char* name = "Tensor";
  static void Define(napi_env env, napi_value, napi_value prototype);
  static etjs::Tensor* Constructor(
      std::variant<etjs::Buffer, std::vector<double>> data,
      ea::ScalarType dtype,
      std::vector<ea::SizesType> shape);
  static void Destructor(etjs::Tensor* ptr);
};

}  // namespace ki

#endif  // SRC_TENSOR_H_
