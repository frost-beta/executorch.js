import {Tensor, DType} from '..';
import {assert} from 'chai';

describe('Tensor', () => {
  it('number scalar', () => {
    const tensor = new Tensor(8964);
    assert.deepEqual(Array.from(tensor.toTypedArray()), [ 8964 ]);
    assert.equal(tensor.dtype, DType.Float32);
    assert.deepEqual(tensor.shape, []);
    assert.deepEqual(tensor.dimOrder, []);
    assert.deepEqual(tensor.strides, []);
    assert.equal(tensor.size, 1);
    assert.equal(tensor.nbytes, 4);
    assert.equal(tensor.itemsize, 4);
  });

  it('boolean scalar', () => {
    const tensor = new Tensor(true);
    assert.deepEqual(Array.from(tensor.toTypedArray()), [ 1 ]);
    assert.equal(tensor.dtype, DType.Bool);
    assert.deepEqual(tensor.shape, []);
    assert.deepEqual(tensor.dimOrder, []);
    assert.deepEqual(tensor.strides, []);
    assert.equal(tensor.size, 1);
    assert.equal(tensor.nbytes, 1);
    assert.equal(tensor.itemsize, 1);
  });

  it('bfloat16 scalar', () => {
    const tensor = new Tensor(42, DType.BFloat16);
    assert.deepEqual(Array.from(tensor.data), [ 40, 66 ]);
    assert.equal(tensor.dtype, DType.BFloat16);
    assert.deepEqual(tensor.shape, []);
    assert.deepEqual(tensor.dimOrder, []);
    assert.deepEqual(tensor.strides, []);
    assert.equal(tensor.size, 1);
    assert.equal(tensor.nbytes, 2);
    assert.equal(tensor.itemsize, 2);
  });

  it('nested array', () => {
    const tensor = new Tensor([ [ 1, 2, 3 ] ]);
    assert.deepEqual(Array.from(tensor.toTypedArray()), [ 1, 2, 3 ]);
    assert.equal(tensor.dtype, DType.Float32);
    assert.deepEqual(tensor.shape, [ 1, 3 ]);
    assert.deepEqual(tensor.dimOrder, [ 0, 1 ]);
    assert.deepEqual(tensor.strides, [ 3, 1 ]);
    assert.equal(tensor.size, 3);
    assert.equal(tensor.nbytes, 12);
    assert.equal(tensor.itemsize, 4);
  });

  it('serialization', () => {
    const input = new Tensor([ 8, 9, 6, 4 ]);
    const output = new Tensor(input.data, input.dtype, {shape: input.shape});
    assert.deepEqual(input.toTypedArray(), output.toTypedArray());
  });
});
