import * as bindings from '../build/Release/executorch.node';
import {DType} from './scalar.js';

type Nested<T> = Nested<T>[] | T;

export interface TensorOptions {
  shape?: number[];
}

/**
 * A multi-dimensional matrix containing elements of a single data type.
 */
export class Tensor {
  data: Buffer;
  dtype: DType;
  shape: number[];

  #holder: bindings.Tensor;

  constructor(input: Nested<boolean | number> | Buffer,
              dtype?: DType,
              {shape}: TensorOptions = {}) {
    if (input instanceof Buffer) {
      // Initialized from serialized data.
      if (!dtype || !shape)
        throw new Error('Must provide dtype and shape when input is Buffer.');
      if (input.length * bindings.elementSize(dtype) < getShapeSize(shape))
        throw new Error('The input has not enough storage for passed shape.');
      this.dtype = dtype;
      this.shape = shape;
      this.data = input;
      this.#holder = new bindings.Tensor(this.data, this.dtype, this.shape);
    } else {
      // Create from JavaScript array or scalar.
      this.dtype = dtype ?? getInputDType(input);
      this.shape = shape ?? getInputShape(input);
      let flatData = Array.isArray(input) ? input.flat() : [ input ];
      if (typeof flatData[0] != 'number')
        flatData = flatData.map(f => Number(f));
      if (shape && flatData.length < getShapeSize(shape))
        throw new Error('The input has less data than set by passed shape.');
      this.#holder = new bindings.Tensor(flatData as number[], this.dtype, this.shape);
      // Get a view of internal buffer.
      this.data = this.#holder.data;
      // Make sure the data is destroyed after holder.
      Object.defineProperty(this.data, 'holder', {enumerable: false, value: this.#holder});
    }
  }

  toTypedArray() {
    const arrayType = getTypedArrayFromDType(this.dtype);
    return new arrayType(this.data.buffer);
  }

  get dimOrder(): number[] {
    return this.#holder.dimOrder;
  }

  get strides(): number[] {
    return this.#holder.strides;
  }

  get size(): number {
    return this.#holder.size;
  }

  get nbytes(): number {
    return this.#holder.nbytes;
  }

  get itemsize(): number {
    return this.#holder.itemsize;
  }
}

function getShapeSize(shape: number[]) {
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

function parseInput(input: Nested<boolean | number>, dtype: DType, shape: number[]) {
  const size = shape.length > 0 ? shape.reduce((a, b) => a * b) : 1;
  const elementSize = bindings.elementSize(dtype);
  const data = Buffer.alloc(size * elementSize);
  return data;
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
