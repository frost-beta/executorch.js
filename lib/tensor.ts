import bindings from '../bindings.js';
import {DType} from './scalar.js';

type Nested<T> = Nested<T>[] | T;

/**
 * Optional options describing the tensor.
 */
export interface TensorOptions {
  shape?: number[];
  dimOrder?: number[];
  strides?: number[];
}

/**
 * A multi-dimensional matrix containing elements of a single data type.
 */
export class Tensor {
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

  // Internal binding to the executorch::aten::Tensor instance.
  private readonly holder: bindings.Tensor;

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
              {shape, dimOrder = [], strides = []}: TensorOptions = {}) {
    if (input instanceof Uint8Array) {
      // Initialized from serialized data.
      if (!dtype || !shape)
        throw new Error('Must provide dtype and shape when input is Uint8Array.');
      if (input.length / bindings.elementSize(dtype) < getSizeFromShape(shape))
        throw new Error('The input has not enough storage for passed shape.');
      this.dtype = dtype;
      this.shape = shape;
      this.data = input;
      this.holder = new bindings.Tensor(this.data, this.dtype, this.shape, dimOrder, strides);
    } else {
      // Create from JavaScript array or scalar.
      this.dtype = dtype ?? getInputDType(input);
      this.shape = shape ?? getInputShape(input);
      let flatData = Array.isArray(input) ? input.flat() : [ input ];
      if (typeof flatData[0] != 'number')
        flatData = flatData.map(f => Number(f));
      if (shape && flatData.length < getSizeFromShape(shape))
        throw new Error('The input has less data than set by passed shape.');
      this.holder = new bindings.Tensor(flatData as number[], this.dtype, this.shape, dimOrder, strides);
      // Get a view of internal buffer.
      this.data = this.holder.data;
      // Make sure the data is destroyed after holder.
      Object.defineProperty(this.data, 'holder', {enumerable: false, value: this.holder});
    }
  }

  /**
   * Return the tensor as a scalar.
   */
  item(): number | boolean {
    return this.holder.item();
  }

  /**
   * Return the tensor as a scalar or (nested) Array.
   */
  tolist(): Nested<number | boolean> {
    return this.holder.tolist();
  }

  /**
   * Return a TypedArray view of tensor's data.
   */
  toTypedArray() {
    const arrayType = getTypedArrayFromDType(this.dtype);
    return new arrayType(this.data.buffer);
  }

  /**
   * A permutation of the dimensions, from the outermost to the innermost one.
   */
  get dimOrder(): number[] {
    return this.holder.dimOrder;
  }

  /**
   * Array of indices to step in each dimension when traversing the tensor.
   */
  get strides(): number[] {
    return this.holder.strides;
  }

  /**
   * Number of tensor dimensions.
   */
  get ndim(): number {
    return this.shape.length;
  }

  /**
   * Number of elements in the tensor.
   */
  get size(): number {
    return this.holder.size;
  }

  /**
   * Total bytes consumed by the elements of the tensor.
   */
  get nbytes(): number {
    return this.holder.nbytes;
  }

  /**
   * Length of one tensor element in bytes.
   */
  get itemsize(): number {
    return this.holder.itemsize;
  }
}

function getSizeFromShape(shape: number[]) {
  return shape.length > 0 ? shape.reduce((a, b) => a * b) : 1;
}

function getInputDType(input: Nested<boolean | number>) {
  if (Array.isArray(input))
    return getInputDType(input[0]);
  switch (typeof input) {
    case 'boolean': return DType.Bool;
    case 'number': return DType.Float32;
    default: throw new TypeError(`Unsupported input: ${input}.`);
  }
}

function getInputShape(input: Nested<boolean | number>): number[] {
  if (!Array.isArray(input))
    return [];
  if (input.length == 0)
    throw new Error('Can not contain empty array in input.');
  const shape = [ input.length ];
  if (!Array.isArray(input[0]))
    return shape;
  const subShape = getInputShape(input[0]);
  if (subShape.length == 0)
    return shape;
  if (!input.every(a => Array.isArray(a) && a.length === subShape[0]))
    throw new Error('Sub-arrays should have the same length');
  return shape.concat(subShape);
}

function getTypedArrayFromDType(dtype: DType) {
  switch (dtype) {
    case DType.UInt8   : return Uint8Array;
    case DType.Int8    : return Int8Array;
    case DType.Int16   : return Int16Array;
    case DType.Int32   : return Int32Array;
    case DType.Float32 : return Float32Array;
    case DType.Float64 : return Float64Array;
    case DType.Bool    : return Uint8Array;
    default: throw new Error(`No matching TypedArray for DType.${DType[dtype]}.`);
  }
}
