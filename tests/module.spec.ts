import fs from 'node:fs';
import {Module, Tensor} from '..';
import {assert} from 'chai';

const fixtures = `${__dirname}/fixtures`;

describe('Module', () => {
  it('mmap file', () => {
    const mod = new Module(`${fixtures}/mv2.pte`);
    mod.load();
    assert.deepEqual(mod.getMethods().map(m => m.name), [ 'forward' ]);
  });

  it('read file', () => {
    const mod = new Module(fs.readFileSync(`${fixtures}/mv2.pte`));
    mod.load();
    assert.deepEqual(mod.getMethods().map(m => m.name), [ 'forward' ]);
  });
});
