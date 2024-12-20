import fs from 'node:fs';
import {DType, Module, Tensor, backends, config} from '..';
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

  const models = {
    cpu: 'mv2.pte',
    mps: 'mv2_mps_float16.pte',
    xnnpack: 'mv2_xnnpack_fp32.pte',
  };
  for (const [ name, model ] of Object.entries(models)) {
    if (!backends[name as keyof typeof backends])
      continue;
    it(`${name} backend`, async function () {
      this.timeout((config == 'Debug' ? 20 : 10) * 1000);
      const mod = new Module(`${fixtures}/${model}`);
      await mod.load();
      const {shape} = mod.getMethods()[0].inputs[0];
      const input = new Tensor(Buffer.alloc(4 * getSizeFromShape(shape!)), DType.Float32, {shape});
      const output = await mod.forward(input);
      assert.deepEqual(output.shape, [ 1, 1000 ]);
    });
  }
});

function getSizeFromShape(shape: number[]) {
  return shape.length > 0 ? shape.reduce((a, b) => a * b) : 1;
}
