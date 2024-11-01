#include "src/tensor.h"

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <numeric>

#include "src/scalar.h"

namespace er = executorch::runtime;

namespace etjs {

namespace {

std::vector<ea::DimOrderType> DefaultDimOrder(
    size_t dim,
    const std::vector<ea::StridesType>& strides = {}) {
  std::vector<ea::DimOrderType> dim_order(dim);
  std::iota(dim_order.begin(), dim_order.end(), 0);
  if (!strides.empty()) {
    std::sort(dim_order.begin(), dim_order.end(), [&](size_t a, size_t b) {
      return strides[a] > strides[b];
    });
  }
  return dim_order;
}

std::vector<ea::StridesType> DefaultStrides(
    const std::vector<ea::SizesType>& shape,
    const std::vector<ea::DimOrderType>& dim_order) {
  const auto dim = shape.size();
  std::vector<ea::StridesType> strides(dim);
  auto error = er::dim_order_to_stride(
      shape.data(), dim_order.data(), dim, strides.data());
  ET_CHECK_MSG(error == er::Error::Ok, "Failed to compute strides.");
  return strides;
}

}  // namespace

Tensor::Tensor(std::vector<uint8_t> data,
               std::vector<ea::SizesType> shape,
               ea::ScalarType dtype,
               std::vector<ea::DimOrderType> dim_order,
               std::vector<ea::StridesType> strides)
    : Tensor(Buffer{data.data(), data.size()}, std::move(shape), dtype,
             std::move(dim_order), std::move(strides)) {
  managed_data_ = std::move(data);
}

Tensor::Tensor(Buffer data,
               std::vector<ea::SizesType> shape,
               ea::ScalarType dtype,
               std::vector<ea::DimOrderType> dim_order,
               std::vector<ea::StridesType> strides)
    : data_(data),
      shape_(std::move(shape)),
      dim_order_(DefaultDimOrder(shape_.size(), strides)),
      strides_(DefaultStrides(shape_, dim_order_)),
      impl_(dtype, shape_.size(), shape_.data(), data_.data,
            dim_order_.data(), strides_.data(),
            shape_.empty() ? ea::TensorShapeDynamism::STATIC
                           : ea::TensorShapeDynamism::DYNAMIC_BOUND) {
  ET_CHECK_MSG(data_.size >= nbytes(), "Tensor size exceeds data size.");
}

Tensor::~Tensor() = default;

}  // namespace etjs

namespace ki {

// static
napi_status Type<ea::Tensor>::ToNode(napi_env env,
                                     const ea::Tensor& value,
                                     napi_value* result) {
  auto* data_ptr = static_cast<const uint8_t*>(value.const_data_ptr());
  auto* tensor = new etjs::Tensor(
      // The value data likely comes from inference output, which will get
      // invalided soon and we must copy it.
      std::vector<uint8_t>(data_ptr, data_ptr + value.nbytes()),
      std::vector<ea::SizesType>(value.sizes().begin(), value.sizes().end()),
      value.dtype(),
      std::vector<ea::DimOrderType>(value.dim_order().begin(),
                                    value.dim_order().end()),
      std::vector<ea::StridesType>(value.strides().begin(),
                                   value.strides().end()));
  return ConvertToNode(env, tensor, result);
}

// static
napi_status Type<etjs::Buffer>::ToNode(napi_env env,
                                       const etjs::Buffer& value,
                                       napi_value* result) {
  return napi_create_external_buffer(env, value.size, value.data, nullptr,
                                     nullptr, result);
}

// static
std::optional<etjs::Buffer> Type<etjs::Buffer>::FromNode(napi_env env,
                                                         napi_value value) {
  void* data;
  size_t size;
  if (napi_get_buffer_info(env, value, &data, &size) != napi_ok)
    return std::nullopt;
  // We are assuming the Buffer is kept alive in JS.
  return etjs::Buffer{data, size};
}

// static
void Type<etjs::Tensor>::Define(napi_env env,
                                napi_value constructor,
                                napi_value prototype) {
  DefineProperties(env, prototype,
                   Property("data", Getter(&etjs::Tensor::data)),
                   Property("shape", Getter(&etjs::Tensor::shape)),
                   Property("dimOrder", Getter(&etjs::Tensor::dim_order)),
                   Property("strides", Getter(&etjs::Tensor::strides)),
                   Property("size", Getter(&etjs::Tensor::size)),
                   Property("nbytes", Getter(&etjs::Tensor::nbytes)),
                   Property("itemsize", Getter(&etjs::Tensor::itemsize)),
                   Property("dtype", Getter(&etjs::Tensor::dtype)));
}

// static
etjs::Tensor* Type<etjs::Tensor>::Constructor(etjs::Buffer buffer,
                                              std::vector<ea::SizesType> shape,
                                              ea::ScalarType dtype) {
  return new etjs::Tensor(buffer, std::move(shape), dtype);;
}

// static
void Type<etjs::Tensor>::Destructor(etjs::Tensor* ptr) {
  // Memory is managed by TypeBridge<etjs::Tensor>::Finalize.
}

template<>
struct TypeBridge<etjs::Tensor> {
  static inline etjs::Tensor* Wrap(etjs::Tensor* ptr) {
    return ptr;
  }
  static inline void Finalize(etjs::Tensor* ptr) {
    delete ptr;
  }
};

}  // namespace ki
