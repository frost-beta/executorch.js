#ifndef SRC_BUFFER_H_
#define SRC_BUFFER_H_

namespace etjs {

// Intermediate type for representing typed buffer.
struct UnmanagedBuffer {
  const void* data;
  size_t size;
};

}  // namespace etjs

#endif  // SRC_BUFFER_H_
