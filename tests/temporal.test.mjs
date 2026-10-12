import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const f=JSON.parse(readFileSync(new URL('../data/temporal_reference.json',import.meta.url),'utf8'));
test('temporal data is held out and all trace samples are finite',()=>{
  assert.equal(f.schema,'monday.temporal.v1');
  assert.equal(f.runs.length,8);
  assert.equal(f.examples.length,3);
  const p=f.protocol;
  assert.equal(p.train_episodes_per_seed,70);
  assert.equal(p.holdout_episodes_per_seed,24);
  assert.equal(new Set([...p.train_seeds,...p.holdout_seeds]).size,16);
  for(const ex of f.examples){for(const k of ['truth_mV','instant_mV','linear_history_mV','quadratic_history_mV']){assert.equal(ex[k].length,ex.t_ms.length);assert.ok(ex[k].every(Number.isFinite));}}
});
test('learned nonlinear history model improves over simpler baselines in exploratory gate',()=>{
  const m=f.mean_holdout_rmse_mV;
  assert.ok(m.instant>m.linear_history);
  assert.ok(m.linear_history>m.quadratic_history);
  assert.equal(f.wins.quadratic_beats_linear_history,8);
});
test('one soma snapshot is not a sufficient hidden-state coordinate',()=>{
  const x=f.matched_soma;
  assert.ok(x.preprobe_abs_error_mV<1e-7);
  assert.ok(x.max_probe_increment_difference_mV>1);
  assert.equal(x.curve.t_ms.length,x.curve.delta_A_mV.length);
  assert.equal(x.curve.delta_A_mV.length,x.curve.delta_B_mV.length);
});
