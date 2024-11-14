#ifndef SRC_SAMPLE_H_
#define SRC_SAMPLE_H_

#include <stddef.h>

namespace etjs {

class Tensor;

size_t Sample(Tensor* tensor, float temperature, float top_p);

}  // namespace etjs

#endif  // SRC_SAMPLE_H_
