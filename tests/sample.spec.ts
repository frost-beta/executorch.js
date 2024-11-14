import {DType, Tensor, sample} from '..';
import {assert} from 'chai';

describe('Sample', () => {
  it('argmax', () => {
    const logits = Array.from({length: 128}, () => Math.random() * 0.9);
    logits[89] = 1;
    const index = sample(new Tensor(logits), {temperature: 0});
    assert.equal(index, 89);
  });

  it('argmax bfloat16', () => {
    const logits = Array.from({length: 128}, () => Math.random() * 0.9);
    logits[64] = 1;
    const index = sample(new Tensor(logits, DType.BFloat16), {temperature: 0});
    assert.equal(index, 64);
  });
});
