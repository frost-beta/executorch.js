import bindings from '../bindings.js';
import {DType} from './scalar.js';
import {Tensor} from './tensor.js';

/**
 * The supported types for conversions between C++ and JavaScript.
 */
export type EValue = Tensor | string | number | boolean;

/**
 * Detailed information about an EValue.
 */
export interface EValueInfo {
  tag: bindings.Tag;
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
  [key: string]: Function;

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
  loadSync() {
    const error = this.#mod.loadSync('minimal');
    if (error)
      throw error;
    for (const name of this.getMethodNames()) {
      this[name] = async function(...args: EValue[]) {
        return executionResult(await this.#mod.execute(name, args));
      };
      this[name + 'Sync'] = function(...args: EValue[]) {
        return executionResult(this.#mod.executeSync(name, args));
      };
    }
  }

  /**
   * Return if any model has been loaded.
   */
  isLoaded() {
    return this.#mod.isLoaded();
  }

  /**
   * Return names of loaded model's methods.
   */
  getMethodNames() {
    return this.#mod.methodNames();
  }

  /**
   * Return information about the methods in the model.
   */
  getMethods() {
    return this.getMethodNames().map((name) => {
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

function executionResult(result: unknown[] | string | Error) {
  if (result instanceof Error)
    throw result;
  if (typeof result == 'string')
    throw new Error(result);
  if (result.length == 1)
    return result[0];
  else
    return result;
}

function parseEValueInfo(tag: bindings.Tag | Error,
                         info: bindings.TensorInfo | Error): EValueInfo {
  if (tag instanceof Error)
    throw tag;
  if (info instanceof Error)
    throw info;
  const result = {tag: tag as unknown as bindings.Tag};
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
