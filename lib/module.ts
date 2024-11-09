import bindings from '../build/Release/executorch.node';
import {DType} from './scalar.js';
import {Tensor} from './tensor.js';

/**
 * The supported types for conversions between C++ and JavaScript.
 */
export type EValue = Tensor | number | boolean | null;

/**
 * Enum representing EValue types.
 */
export enum EValueTag {
  None               = bindings.Tag.None,
  Tensor             = bindings.Tag.Tensor,
  String             = bindings.Tag.String,
  Double             = bindings.Tag.Double,
  Int                = bindings.Tag.Int,
  Bool               = bindings.Tag.Bool,
  ListBool           = bindings.Tag.ListBool,
  ListDouble         = bindings.Tag.ListDouble,
  ListInt            = bindings.Tag.ListInt,
  ListTensor         = bindings.Tag.ListTensor,
  ListScalar         = bindings.Tag.ListScalar,
  ListOptionalTensor = bindings.Tag.ListOptionalTensor,
}

/**
 * Detailed information about an EValue.
 */
export interface EValueInfo {
  tag: EValueTag;
  dtype?: DType;
  shape?: number[];
  dimOrder?: number[];
  nbytes?: number;
}

/**
 * Load exported edge PyTorch models.
 */
export class Module {
  /**
   * The methods of this class are dynamically loaded.
   */
  [key: string]: Function | undefined;

  // Internal binding to the executorch::extension::Module instance.
  readonly #mod: bindings.Module;

  /**
   * @param filePathOrBuffer - When a string is passed, it is treated as file
   * path and will be loaded with mmap. When a Uint8Array is passed, its content
   * is used as the model file.
   */
  constructor(filePathOrBuffer: string | Uint8Array) {
    this.#mod = new bindings.Module(filePathOrBuffer);
  }

  /**
   * Load the model.
   */
  load() {
    const error = this.#mod.load('minimal');
    if (error)
      throw error;
  }

  /**
   * Return if any model has been loaded.
   */
  isLoaded() {
    return this.#mod.isLoaded();
  }

  /**
   * Return information about the methods in the model.
   */
  getMethods() {
    const names = this.#mod.methodNames();
    return names.map((name) => {
      const meta = this.#mod.methodMeta(name);
      if (meta instanceof Error)
        throw meta;
      const inputs: EValueInfo[] = [];
      for (let i = 0; i < meta.numInputs(); ++i)
        inputs.push(parseEValueInfo(meta.inputTag(i), meta.inputTensorMeta(i)));
      const outputs: EValueInfo[] = [];
      for (let i = 0; i < meta.numOutputs(); ++i)
        outputs.push(parseEValueInfo(meta.outputTag(i), meta.outputTensorMeta(i)));
      return {name, inputs, outputs};
    });
  }
}

function parseEValueInfo(tag: bindings.Tag | Error,
                         info: bindings.TensorInfo | Error): EValueInfo {
  if (tag instanceof Error)
    throw tag;
  if (info instanceof Error)
    throw info;
  const result = {tag: tag as unknown as EValueTag};
  if (tag != bindings.Tag.Tensor)
    return result;
  return {
    ...result,
    dtype: info.scalarType as unknown as DType,
    shape: info.sizes,
    dimOrder: info.dimOrder,
    nbytes: info.nbytes,
  };
}
