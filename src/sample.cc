#include "src/sample.h"

#include <random>

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include "src/tensor.h"

namespace etjs {

namespace {

template<typename T>
struct ProbIndex {
  T prob;
  size_t index;
};

template<typename T>
size_t SampleArgMax(T* probs, size_t size) {
  size_t max_i = 0;
  T max_p = probs[0];
  for (size_t i = 1; i < size; i++) {
    if (probs[i] > max_p) {
      max_i = i;
      max_p = probs[i];
    }
  }
  return max_i;
}

template<typename T>
size_t SampleMult(const std::vector<T>& probs, float coin) {
  T cdf{};
  for (size_t i = 0; i < probs.size(); i++) {
    cdf += probs[i];
    if (coin < cdf)
      return i;
  }
  return probs.size() - 1;
}

template<typename T>
size_t SampleTopP(const std::vector<T>& probs, float top_p, float coin) {
  size_t n0 = 0;
  std::vector<ProbIndex<T>> probindex(probs.size());

  float cutoff = (1.0f - top_p) / (probs.size() - 1);
  for (size_t i = 0; i < probs.size(); i++) {
    if (probs[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probs[i];
      n0++;
    }
  }

  std::sort(probindex.begin(), probindex.end(),
            [](const auto& a, const auto& b) { return a.prob > b.prob; });

  T cumulative_prob = 0;
  int32_t last_idx = n0 - 1;
  for (size_t i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > top_p) {
      last_idx = i;
      break;
    }
  }

  T r = coin * cumulative_prob;
  T cdf = 0;
  for (size_t i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf)
      return probindex[i].index;
  }
  return probindex[last_idx].index;
}

template<typename T>
void Softmax(std::vector<T>& x) {
  T max_val = *std::max_element(x.begin(), x.end());

  T sum = 0;
  for (size_t i = 0; i < x.size(); i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }

  for (size_t i = 0; i < x.size(); i++) {
    x[i] = x[i] / sum;
  }
}

float RandomF32() {
  static std::mt19937 engine;
  static std::uniform_real_distribution<float> distribution(0.0, 1.0);
  return distribution(engine);
}

template<typename T>
size_t Sample(T* input, size_t size, float temperature, float top_p) {
  if (temperature == 0)
    return SampleArgMax(input, size);

  std::vector<T> logits(input, input + size);
  for (size_t i = 0; i < size; i++)
    logits[i] = logits[i] / temperature;

  Softmax(logits);

  float coin = RandomF32();
  if (top_p <= 0 || top_p >= 1)
    return SampleMult(logits, coin);
  else
    return SampleTopP(logits, top_p, coin);
}

}  // namespace

size_t Sample(Tensor* tensor, float temperature, float top_p) {
  ET_CHECK_MSG(tensor->size() > 0, "Tensor can not be empty");
  ET_CHECK_MSG(tensor->ndim() == 1 ||
               (tensor->ndim() == 2 && tensor->shape()[0] == 1),
               "Tensor's shape must be [N] or [1, N].");
  size_t ret = 0;
  ET_SWITCH_REALHBBF16_TYPES(tensor->dtype(), nullptr, "sample", CTYPE, [&] {
    ret = Sample(tensor->data<CTYPE>(), tensor->size(), temperature, top_p);
  });
  return ret;
}

}  // namespace etjs
