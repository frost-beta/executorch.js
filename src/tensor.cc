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
    const std::vector<ea::StridesType>& strides) {
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
               ea::ScalarType dtype,
               std::vector<ea::SizesType> shape,
               std::vector<ea::DimOrderType> dim_order,
               std::vector<ea::StridesType> strides)
    : Tensor(Buffer{data.data(), data.size()},
             dtype,
             std::move(shape),
             std::move(dim_order),
             std::move(strides)) {
  managed_data_ = std::move(data);
}

Tensor::Tensor(Buffer data,
               ea::ScalarType dtype,
               std::vector<ea::SizesType> shape,
               std::vector<ea::DimOrderType> dim_order,
               std::vector<ea::StridesType> strides)
    : data_(data),
      shape_(std::move(shape)),
      dim_order_(dim_order.empty() ? DefaultDimOrder(shape_.size(), strides)
                                   : std::move(dim_order)),
      strides_(strides.empty() ? DefaultStrides(shape_, dim_order_)
                               : std::move(strides)),
      impl_(dtype,
            shape_.size(),
            shape_.data(),
            data_.data,
            dim_order_.data(),
            strides_.data(),
            shape_.empty() ? ea::TensorShapeDynamism::STATIC
                           : ea::TensorShapeDynamism::DYNAMIC_BOUND) {
  ET_CHECK_MSG(data_.size >= nbytes(), "Tensor size exceeds data size.");
}

Tensor::~Tensor() = default;

}  // namespace etjs

namespace {

// Convert the element at index in tensor to JS value.
napi_value ElementToValue(etjs::Tensor* tensor, napi_env env, size_t index) {
  napi_value result = nullptr;
  ET_SWITCH_REALHBBF16_TYPES(tensor->dtype(), nullptr, "tovalue", CTYPE, [&] {
    result = ki::ToNodeValue(env, tensor->data<CTYPE>()[index]);
  });
  return result;
}

// Convert the tensor to array.
napi_value TensorToArray(etjs::Tensor* tensor,
                         napi_env env,
                         size_t index = 0,
                         size_t dim = 0) {
  napi_value ret;
  if (napi_create_array_with_length(env, tensor->shape()[dim], &ret) != napi_ok)
    return nullptr;
  for (size_t i = 0; i < tensor->shape()[dim]; ++i) {
    if (dim == tensor->ndim() - 1) {
      // The last dimension only has scalars.
      napi_set_element(env, ret, i, ElementToValue(tensor, env, index));
    } else {
      napi_set_element(env, ret, i, TensorToArray(tensor, env, index, dim + 1));
    }
    index += tensor->strides()[dim];
  }
  return ret;
}

// Convert the tensor to scalar.
napi_value Item(etjs::Tensor* tensor, napi_env env) {
  if (tensor->size() != 1) {
    ki::ThrowError(env, "item() can only be called on tensors of size 1.");
    return nullptr;
  }
  return ElementToValue(tensor, env, 0);
}

// Convert the tensor to scalar or nested array.
napi_value ToList(etjs::Tensor* tensor, napi_env env) {
  if (tensor->ndim() == 0)
    return Item(tensor, env);
  return TensorToArray(tensor, env);
}

}  // namespace

namespace ki {

template<>
struct Type<er::etensor::Half> {
  static constexpr const char* name = "Float16";
  static napi_status ToNode(napi_env env,
                            er::etensor::Half value,
                            napi_value* result) {
    return ConvertToNode(env, static_cast<float>(value), result);
  }
};

template<>
struct Type<er::etensor::BFloat16> {
  static constexpr const char* name = "BFloat16";
  static napi_status ToNode(napi_env env,
                            er::etensor::BFloat16 value,
                            napi_value* result) {
    return ConvertToNode(env, static_cast<float>(value), result);
  }
};

// static
napi_status Type<ea::Tensor>::ToNode(napi_env env,
                                     const ea::Tensor& value,
                                     napi_value* result) {
  auto* data_ptr = static_cast<const uint8_t*>(value.const_data_ptr());
  auto* tensor = new etjs::Tensor(
      // The value data likely comes from inference output, which will get
      // invalided soon and we must copy it.
      std::vector<uint8_t>(data_ptr, data_ptr + value.nbytes()),
      value.dtype(),
      std::vector<ea::SizesType>(value.sizes().begin(), value.sizes().end()),
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
                   Property("data", Getter(&etjs::Tensor::buffer)),
                   Property("dtype", Getter(&etjs::Tensor::dtype)),
                   Property("shape", Getter(&etjs::Tensor::shape)),
                   Property("dimOrder", Getter(&etjs::Tensor::dim_order)),
                   Property("strides", Getter(&etjs::Tensor::strides)),
                   Property("size", Getter(&etjs::Tensor::size)),
                   Property("nbytes", Getter(&etjs::Tensor::nbytes)),
                   Property("itemsize", Getter(&etjs::Tensor::itemsize)));
  Set(env, prototype,
      "item", MemberFunction(&Item),
      "tolist", MemberFunction(&ToList));
}

// static
etjs::Tensor* Type<etjs::Tensor>::Constructor(
    std::variant<etjs::Buffer, std::vector<double>> data,
    ea::ScalarType dtype,
    std::vector<ea::SizesType> shape,
    std::vector<ea::DimOrderType> dim_order,
    std::vector<ea::StridesType> strides) {
  // When a Buffer is passed, we assume the caller will keep it alive and we
  // just read its content.
  if (auto* b = std::get_if<etjs::Buffer>(&data); b) {
    return new etjs::Tensor(*b,
                            dtype,
                            std::move(shape),
                            std::move(dim_order),
                            std::move(strides));
  }
  // When an array of number is passed, cast elements to native type of dtype
  // and save them into a buffer.
  if (auto* v = std::get_if<std::vector<double>>(&data); v) {
    std::vector<uint8_t> casted_data(v->size() * er::elementSize(dtype));
    ET_SWITCH_REALHBBF16_TYPES(dtype, nullptr, "etjs::Tensor", CTYPE, [&] {
      std::transform(
          v->begin(),
          v->end(),
          reinterpret_cast<CTYPE*>(casted_data.data()),
          [](double element) { return static_cast<CTYPE>(element); });
    });
    return new etjs::Tensor(std::move(casted_data),
                            dtype,
                            std::move(shape),
                            std::move(dim_order),
                            std::move(strides));
  }
  return nullptr;
}

// static
void Type<etjs::Tensor>::Destructor(etjs::Tensor* ptr) {
  // Memory is managed by TypeBridge<etjs::Tensor>::Finalize.
}

// Allow passing pointers of etjs::Tensor to JS, the code assumes we never free
// the object in C++.
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
