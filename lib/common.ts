import bindings from '../bindings.js';
import type {Tensor} from './tensor.js';

/**
 * Data type.
 */
export enum DType {
  Uint8    = bindings.ScalarType.Byte,
  Int8     = bindings.ScalarType.Char,
  Int16    = bindings.ScalarType.Short,
  Int32    = bindings.ScalarType.Int,
  Int64    = bindings.ScalarType.Long,
  Float16  = bindings.ScalarType.Half,
  Float32  = bindings.ScalarType.Float,
  Float64  = bindings.ScalarType.Double,
  Bool     = bindings.ScalarType.Bool,
  BFloat16 = bindings.ScalarType.BFloat16,
}

/**
 * Samples from the given tensor using a softmax over logits.
 */
export function sample(logits: Tensor,
                       {
                         temperature = 1,
                         topP = 1,
                       }: {temperature?: number, topP?: number} = {}) {
  if (logits.size == 0)
    throw new Error('The logits must not be empty.');
  if (logits.ndim == 0 ||
      logits.ndim > 2 ||
      logits.ndim == 2 && logits.shape[0] != 1)
    throw new Error('The shape of logits must be [N] or [1, N].');
  return bindings.sample(logits.holder, temperature, topP);
}
