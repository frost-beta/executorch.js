import fs from 'node:fs';
import {DType, Module, Tensor} from '..';
import {assert} from 'chai';

const fixtures = `${__dirname}/fixtures`;

describe('Module', () => {
  it('mmap file', () => {
    const mod = new Module(`${fixtures}/mv2.pte`);
    mod.loadSync();
    assert.deepEqual(mod.getMethodNames(), [ 'forward' ]);
  });

  it('read file', () => {
    const mod = new Module(fs.readFileSync(`${fixtures}/mv2.pte`));
    mod.loadSync();
    assert.deepEqual(mod.getMethodNames(), [ 'forward' ]);
  });

  it('cpu forward', function () {
    this.timeout(10 * 1000);
    const mod = new Module(`${fixtures}/mv2.pte`);
    mod.loadSync();
    const {shape} = mod.getMethods()[0].inputs[0];
    const input = new Tensor(Buffer.alloc(4 * getSizeFromShape(shape!), 1), DType.Float32, {shape});
    const output = mod.forward(input);
    assert.deepEqual(output.shape, [ 1, 1000 ]);
  });
});

function getSizeFromShape(shape: number[]) {
  return shape.length > 0 ? shape.reduce((a, b) => a * b) : 1;
}
