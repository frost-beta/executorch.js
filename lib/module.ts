import * as bindings from '../build/Release/executorch.node';
import {Tensor} from './tensor.js';

/**
 * The supported types for conversions between C++ and JavaScript.
 */
export type EValue = Tensor | number | boolean | null;

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
      return {
        name,
      }
    });
  }
}
