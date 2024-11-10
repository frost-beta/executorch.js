import bindings from '../build/Release/executorch.node';

export const backends = {
  cpu     : true,
  coreml  : bindings.backends.coreml,
  mps     : bindings.backends.mps,
  xnnpack : bindings.backends.xnnpack,
}
