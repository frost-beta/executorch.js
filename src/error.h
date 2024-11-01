#ifndef SRC_ERROR_H_
#define SRC_ERROR_H_

#include <executorch/runtime/core/error.h>

namespace etjs {

inline const char* ErrorCodeToString(executorch::runtime::Error value) {
  using executorch::runtime::Error;
  switch (value) {
    case Error::Ok:
      return "Ok";
    case Error::Internal:
      return "Internal";
    case Error::InvalidState:
      return "InvalidState";
    case Error::EndOfMethod:
      return "EndOfMethod";
    case Error::NotSupported:
      return "NotSupported";
    case Error::NotImplemented:
      return "NotImplemented";
    case Error::InvalidArgument:
      return "InvalidArgument";
    case Error::InvalidType:
      return "InvalidType";
    case Error::OperatorMissing:
      return "OperatorMissing";
    case Error::NotFound:
      return "NotFound";
    case Error::MemoryAllocationFailed:
      return "MemoryAllocationFailed";
    case Error::AccessFailed:
      return "AccessFailed";
    case Error::InvalidProgram:
      return "InvalidProgram";
    case Error::DelegateInvalidCompatibility:
      return "DelegateInvalidCompatibility";
    case Error::DelegateMemoryAllocationFailed:
      return "DelegateMemoryAllocationFailed";
    case Error::DelegateInvalidHandle:
      return "DelegateInvalidHandle";
    default:
      return "Unknown";
  }
}

inline const char* ErrorCodeToMessage(executorch::runtime::Error value) {
  using executorch::runtime::Error;
  switch (value) {
    case Error::Ok:
      return "";
    case Error::Internal:
      return "An internal error occurred";
    case Error::InvalidState:
      return "Executor is in an invalid state for a target operation";
    case Error::EndOfMethod:
      return "There are no more steps of execution to run";
    case Error::NotSupported:
      return "Operation is not supported in the current context";
    case Error::NotImplemented:
      return "Operation is not yet implemented";
    case Error::InvalidArgument:
      return "User provided an invalid argument";
    case Error::InvalidType:
      return "Object is an invalid type for the operation";
    case Error::OperatorMissing:
      return "Operator(s) missing in the operator registry";
    case Error::NotFound:
      return "Requested resource could not be found";
    case Error::MemoryAllocationFailed:
      return "Could not allocate the requested memory";
    case Error::AccessFailed:
      return "Could not access a resource";
    case Error::InvalidProgram:
      return "Error caused by the contents of a program";
    case Error::DelegateInvalidCompatibility:
      return "Backend receives an incompatible delegate version";
    case Error::DelegateMemoryAllocationFailed:
      return "Backend fails to allocate memory";
    case Error::DelegateInvalidHandle:
      return "The handle is invalid";
    default:
      return "Unknown error";
  }
}

}  // namespace etjs

#endif  // SRC_ERROR_H_
