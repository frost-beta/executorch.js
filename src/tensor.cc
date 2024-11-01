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
               ea::ScalarType dtype)
    : data_(std::move(data)),
      shape_(std::move(shape)),
      dim_order_(DefaultDimOrder(shape_.size())),
      strides_(DefaultStrides(shape_, dim_order_)),
      impl_(dtype, shape_.size(), shape_.data(), data_.data(),
            dim_order_.data(), strides_.data(),
            shape_.empty() ? ea::TensorShapeDynamism::STATIC
                           : ea::TensorShapeDynamism::DYNAMIC_BOUND) {}

Tensor::~Tensor() = default;

}  // namespace etjs

namespace ki {

// static
void Type<etjs::Tensor>::Define(napi_env, napi_value, napi_value) {
}

// static
etjs::Tensor* Type<etjs::Tensor>::Constructor(
    napi_env env,
    napi_value value,
    std::optional<ea::ScalarType> dtype) {
  return nullptr;
}

}  // namespace ki
