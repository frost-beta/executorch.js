# ExecuTorch.js

JavaScript bindings for [ExecuTorch](https://pytorch.org/executorch-overview),
a runtime for inferencing PyTorch models.

## Supported platforms

JavaScript runtimes:

* Node.js >= 22 (and compatible runtimes like Electron and Bun)
* <i>WebAssembly is not supported yet but on the plan</i>

Platforms:

* macOS x64 and arm64
* Linux x64
* <i>Windows and Linux arm64 are not supported yet but on the plan</i>

Backends:

* [MPS](https://developer.apple.com/documentation/metalperformanceshaders) (macOS only)
* [XNNPACK](https://github.com/google/XNNPACK)
* <i>Vulkan is not supported yet but on the plan</i>

## Install

You can install ExecuTorch.js from npm:

```console
npm install executorch
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
env npm_config_debug=true npm install @executorch/runtime-xnnpack
```