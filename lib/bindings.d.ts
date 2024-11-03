declare module '*/build/Release/executorch.node' {
  class Module {
  }

  class Scalar {
  }

  class Tensor {
    constructor(data: Buffer | number[], dtype: number, shape: number[]);
    get data(): Buffer;
    get dtype(): number;
    get shape(): number[];
    get dimOrder(): number[];
    get strides(): number[];
    get size(): number;
    get nbytes(): number;
    get itemsize(): number;
  }

  enum ScalarType {
    Byte,
    Char,
    Short,
    Int,
    Long,
    Half,
    Float,
    Double,
    Bool,
    BFloat16,
  }

  function elementSize(dtype: number): number;
}
