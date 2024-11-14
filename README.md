# ExecuTorch.js

JavaScript bindings for [ExecuTorch](https://pytorch.org/executorch-overview),
a runtime for inferencing PyTorch models.

## Supported platforms

JavaScript runtimes:

* Node.js >= 22 (and compatible runtimes like Electron and Bun)
* <i>Plan on: WebAssembly, React Native</i>

Platforms:

* Linux x64
* macOS arm64
* macOS x64
  * [Some ops](https://github.com/pytorch/executorch/issues/6839) required
    for running LLMs are not supported yet.
* <i>Plan on: Windows, Linux arm64, iOS, Android</i>

Backends:

* [MPS](https://developer.apple.com/documentation/metalperformanceshaders) (macOS only)
* [XNNPACK](https://github.com/google/XNNPACK)
* <i>Plan on: Vulkan</i>

## Install

You can install ExecuTorch.js from npm:

```console
$ npm install executorch
```

The default `executorch` package includes support for all backends, for users
who want to reduce binary size, you can install packages with specific backeds:

|                             | CPU | MPS | Vulkan | XNNPACK |
|-----------------------------|-----|-----|--------|---------|
| executorch                  | ✔️   | ✔️   | ❌      | ✔️       |
| @executorch/runtime         | ✔️   | 🍏   | ❌      | 🐧       |
| @executorch/runtime-all     | ✔️   | ✔️   | ❌      | ✔️       |
| @executorch/runtime-cpu     | ✔️   | ❌   | ❌      | ❌       |
| @executorch/runtime-mps     | ✔️   | ✔️   | ❌      | ❌       |
| @executorch/runtime-xnnpack | ✔️   | ❌   | ❌      | ✔️       |

The `@executorch/runtime` package is a speical one that uses MPS backend on
macOS and XNNPACK backend for other platforms.

For debugging purpose each package also has a Debug version that can be enabled
by setting the `npm_config_debug` environment variable when installing:

```console
$ env npm_config_debug=true npm install @executorch/runtime-xnnpack
```

## Quick start

Download the mobilenet model:

```console
$ wget https://huggingface.co/frost-beta/mobilenet-v2-executorch-cpu/resolve/main/mv2.pte
```

Run following code with Node.js:

```typescript
import {Module, Tensor} from './dist/index.js';

// A tensor of shape [ 1, 3, 224, 224 ].
const input = Array.from({length: 1}, () =>
              Array.from({length: 3}, () =>
              Array.from({length: 224}, () =>
              Array.from({length: 224}, () => Math.random()))));

const mod = new Module('mv2.pte');
await mod.load();
const output = await mod.forward(new Tensor(input));
console.log(output.tolist());
```

## Examples

* [llama3-torch.js](https://github.com/frost-beta/llama3-torch.js) - A simple
  chat CLI for LLama 3.

## APIs

```typescript
/**
 * Load exported edge PyTorch models.
 */
export declare class Module {
    /**
     * The methods of this class are dynamically loaded.
     */
    [key: string]: Function;
    /**
     * @param filePathOrBuffer - When a string is passed, it is treated as file
     * path and will be loaded with mmap. When a Uint8Array is passed, its content
     * is used as the model file.
     */
    constructor(filePathOrBuffer: string | Uint8Array);
    /**
     * Load the model.
     *
     * @remarks
     *
     * After loading, the model's methods will be added to the instance
     * dynamically, with both async and async versions for each method, the sync
     * version will have a "Sync" suffix appended to its name.
     */
    load(): Promise<void>;
    /**
     * Load the model synchronously.
     */
    loadSync(): void;
    /**
     * Return if any model has been loaded.
     */
    isLoaded(): boolean;
    /**
     * Return names of loaded model's methods.
     */
    getMethodNames(): string[];
}

/**
 * Data type.
 */
export declare enum DType {
    Uint8,
    Int8,
    Int16,
    Int32,
    Float16,
    Float32,
    Float64,
    Bool,
    BFloat16
}

type Nested<T> = Nested<T>[] | T;

/**
 * A multi-dimensional matrix containing elements of a single data type.
 */
export declare class Tensor {
    /**
     * The tensor's data stored as JavaScript Uint8Array.
     */
    readonly data: Uint8Array;
    /**
     * Data-type of the tensor’s elements.
     */
    readonly dtype: DType;
    /**
     * Array of tensor dimensions.
     */
    readonly shape: number[];
    /**
     * @param input - A scalar, or a (nested) Array, or a Uint8Array buffer.
     * @param dtype - The data type of the elements.
     * @param options - Extra information of the tensor.
     * @param options.shape
     * @param options.dimOrder
     * @param options.strides
     */
    constructor(input: Nested<boolean | number> | Uint8Array,
                dtype?: DType,
                { shape, dimOrder, strides }?: { shape?: number[]; dimOrder?: number[]; strides?: number[]; });
    /**
     * Return the tensor as a scalar.
     */
    item(): number | boolean;
    /**
     * Return the tensor as a scalar or (nested) Array.
     */
    tolist(): Nested<number | boolean>;
    /**
     * Return a TypedArray view of tensor's data.
     */
    toTypedArray(): Int8Array | Uint8Array | Int16Array | Int32Array | Float32Array | Float64Array;
    /**
     * A permutation of the dimensions, from the outermost to the innermost one.
     */
    get dimOrder(): number[];
    /**
     * Array of indices to step in each dimension when traversing the tensor.
     */
    get strides(): number[];
    /**
     * Number of tensor dimensions.
     */
    get ndim(): number;
    /**
     * Number of elements in the tensor.
     */
    get size(): number;
    /**
     * Total bytes consumed by the elements of the tensor.
     */
    get nbytes(): number;
    /**
     * Length of one tensor element in bytes.
     */
    get itemsize(): number;
}

/**
 * Samples from the given tensor using a softmax over logits.
 */
export declare function sample(logits: Tensor,
                               {
                                 temperature = 1,
                                 topP = 1,
                               }?: { temperature?: number; topP?: number }): number;
```

## Development

Source code architecture:

* `src/` - C++ source code.
* `lib/` - TypeScript source code.
* `bindings.js`/`bindings.d.ts` - Glue code between C++ and TypeScript.
* `install.js` - Script that downloads compiled binaries when installing.
* `tests/` - Tests for TypeScript code.
* `build/` - Generated project files and binaries from C++ code.
* `dist/` - Generated JavaScript code from TypeScript code.
