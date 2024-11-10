type Nested<T> = Nested<T>[] | T;

declare module '*/build/Release/executorch.node' {
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

  enum Tag {
    None,
    Tensor,
    String,
    Double,
    Int,
    Bool,
    ListBool,
    ListDouble,
    ListInt,
    ListTensor,
    ListScalar,
    ListOptionalTensor,
  }

  interface TensorInfo {
    sizes: number[];
    dimOrder: number[];
    scalarType: ScalarType;
    isMemoryPlanned: boolean;
    nbytes: number;
  }

  interface MethodMeta {
    name(): string;
    numInputs(): number;
    inputTag(index: number): Tag | Error;
    inputTensorMeta(index: number): TensorInfo | Error;
    numOutputs(): number;
    outputTag(index: number): Tag | Error;
    outputTensorMeta(index: number): TensorInfo | Error;
    numMemoryPlannedBuffers(): number;
    memoryPlannedBufferSize(index: number): number | Error;
  }

  class Module {
    constructor(filePathOrBuffer: string | Uint8Array);
    load(verification: 'minimal' | 'internal-consistency'): Error;
    isLoaded(): boolean;
    methodNames(): string[];
    methodMeta(name: string): MethodMeta | Error;
  }

  class Tensor {
    constructor(data: Uint8Array | number[], dtype: number, shape: number[], dimOrder: number[], strides: number[]);
    item(): number | boolean;
    tolist(): Nested<number | boolean>;
    get data(): Uint8Array;
    get dtype(): number;
    get shape(): number[];
    get dimOrder(): number[];
    get strides(): number[];
    get size(): number;
    get nbytes(): number;
    get itemsize(): number;
  }

  function elementSize(dtype: number): number;
}
