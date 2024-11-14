type Nested<T> = Nested<T>[] | T;

export enum ScalarType {
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

export enum Tag {
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

export interface TensorInfo {
  sizes: number[];
  dimOrder: number[];
  scalarType: ScalarType;
  isMemoryPlanned: boolean;
  nbytes: number;
}

export interface MethodMeta {
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

export class Module {
  constructor(filePathOrBuffer: string | Uint8Array);
  load(verification: 'minimal' | 'internal-consistency'): Promise<undefined | Error>;
  loadSync(verification: 'minimal' | 'internal-consistency'): undefined | Error;
  isLoaded(): boolean;
  methodNames(): string[];
  methodMeta(name: string): MethodMeta | Error;
  execute(name: string, args: unknown[]): Promise<unknown[] | string | Error>;
  executeSync(name: string, args: unknown[]): unknown[] | string | Error;
}

export class Tensor {
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

export interface Backends {
  cpu: boolean;
  coreml: boolean;
  mps: boolean;
  xnnpack: boolean;
}

export const backends: Backends;

export const config: 'Debug' | 'Release';

export function elementSize(dtype: number): number;
export function sample(tensor: Tensor, temperature: number, topP: number): number;
