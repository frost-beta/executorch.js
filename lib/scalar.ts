import bindings from '../build/Release/executorch.node';

export enum DType {
  UInt8    = bindings.ScalarType.Byte,
  Int8     = bindings.ScalarType.Char,
  Int16    = bindings.ScalarType.Short,
  Int32    = bindings.ScalarType.Int,
  Float16  = bindings.ScalarType.Half,
  Float32  = bindings.ScalarType.Float,
  Float64  = bindings.ScalarType.Double,
  Bool     = bindings.ScalarType.Bool,
  BFloat16 = bindings.ScalarType.BFloat16,
}
