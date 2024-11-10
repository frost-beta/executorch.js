# ExecuTorch.js

JavaScript bindings for [ExecuTorch](https://pytorch.org/executorch-overview),
a runtime for inferencing PyTorch models.

## Supported platforms

JavaScript runtimes:

* Node.js >= 22 (and compatible runtimes like Electron and Bun)
* WebAssembly is not supported yet but on the plan

Platforms:

* macOS x64 and arm64
* Linux x64
* Windows and Linux arm64 are not supported yet but on the plan

Backends:

* [XNNPACK](https://github.com/google/XNNPACK)
* [MPS](https://developer.apple.com/documentation/metalperformanceshaders) (macOS)
* [Vulkan](https://www.vulkan.org) is not supported yet but on the plan
