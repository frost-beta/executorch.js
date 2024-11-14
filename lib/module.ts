import bindings from '../bindings.js';
import {DType} from './common.js';
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
  async load() {
    const error = await this.#mod.load('minimal');
    if (error)
      throw error;
    this.#populateMethods();
  }

  /**
   * Load the model synchronously.
   */
  loadSync() {
    const error = this.#mod.loadSync('minimal');
    if (error)
      throw error;
    this.#populateMethods();
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
      for (let i = 0; i < meta.numInputs(); ++i) {
        const tag = meta.inputTag(i);
        if (tag instanceof Error)
          throw tag;
        if (tag == bindings.Tag.Tensor)
          inputs.push(parseEValueInfo(tag, meta.inputTensorMeta(i)));
        else
          inputs.push({tag});
      }
      const outputs: EValueInfo[] = [];
      for (let i = 0; i < meta.numOutputs(); ++i) {
        const tag = meta.outputTag(i);
        if (tag instanceof Error)
          throw tag;
        if (tag == bindings.Tag.Tensor)
          outputs.push(parseEValueInfo(tag, meta.outputTensorMeta(i)));
        else
          outputs.push({tag});
      }
      return {name, inputs, outputs};
    });
  }

  #populateMethods() {
    for (const name of this.getMethodNames()) {
      this[name] = async function(...args: EValue[]) {
        return executionResult(await this.#mod.execute(name, args));
      };
      this[name + 'Sync'] = function(...args: EValue[]) {
        return executionResult(this.#mod.executeSync(name, args));
      };
    }
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

function parseEValueInfo(tag: bindings.Tag,
                         info: bindings.TensorInfo | Error): EValueInfo {
  if (info instanceof Error)
    throw info;
  return {
    tag,
    dtype: info.scalarType as unknown as DType,
    shape: info.sizes,
    dimOrder: info.dimOrder,
    nbytes: info.nbytes,
  };
}
