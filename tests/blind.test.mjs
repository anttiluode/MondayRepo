import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createBlindExperiment,fitOperator,chooseProbe,uncertainty,validateCableFixture,interpolateFixture,predictionMatrix,rmse} from '../blind.mjs';
import {CONTEXTS,PORTS,OUTPUTS} from '../model.mjs';
const fixture=JSON.parse(fs.readFileSync(new URL('../data/cable_reference.json',import.meta.url),'utf8'));
const near=(a,b,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b} within ${tol}`);

test('estimator only gets observations: no candidate matrices or sealed truth',()=>{
  const e=createBlindExperiment({seed:42});
  assert.equal(Object.hasOwn(e,'a'),false);
  assert.equal(Object.hasOwn(e,'b'),false);
  assert.equal(Object.hasOwn(e,'truth'),false);
  assert.equal(Object.hasOwn(e,'hidden'),false);
  assert.deepEqual(e.samples,[]);
  e.measure(0,0);
  assert.equal(e.samples.length,1);
  const copy=e.samples;copy[0].observed[0]=9999;
  assert.notEqual(e.samples[0].observed[0],9999);
});

test('design selector depends on probe contexts, not observed values',()=>{
  const data=[{context:0,port:0,observed:[1,2,3]}, {context:1,port:1,observed:[1,2,3]}];
  const swapped=data.map(v=>({...v,observed:[9999,-9999,42]}));
  assert.deepEqual(chooseProbe(fitOperator(data)),chooseProbe(fitOperator(swapped)));
});

test('rest-only observations leave context slope unmeasured',()=>{
  const e=createBlindExperiment({seed:123,sigma:0.05});
  for(let j=0;j<4;j++)e.measure(0,j);
  const fit=e.fit();
  assert.ok(uncertainty(fit,1,0)>100*uncertainty(fit,0,0));
  assert.notEqual(e.choose().context,0);
});

test('affine blind inversion learns withheld contexts on new synthetic specimen',()=>{
  const e=createBlindExperiment({seed:20261012,sigma:.12});
  for(let j=0;j<4;j++)e.measure(0,j);
  const rest=e.audit();
  for(let n=0;n<28;n++){const p=e.choose();e.measure(p.context,p.port)}
  const result=e.audit();
  const affine=result.find(r=>r.name==='Affine').rmse;
  const rank2=result.find(r=>r.name==='Rank 2').rmse;
  assert.ok(affine<0.10,`affine heldout RMSE ${affine}`);
  assert.ok(rank2<0.10);
  assert.ok(affine<rest.find(r=>r.name==='Affine').rmse/3);
  assert.ok(affine<result.find(r=>r.name==='Constant').rmse/3);
  assert.deepEqual(new Set(e.samples.map(v=>v.context)),new Set(CONTEXTS));
});

test('rank-two factorization is a projection of learned response, not hidden truth',()=>{
  const e=createBlindExperiment({seed:20261012,sigma:.12});
  for(let j=0;j<4;j++)for(let c of [-1,0,1])for(let k=0;k<5;k++)e.measure(c,j);
  const rank2=e.fit(2),H=[...rank2.H0.map(r=>r.slice()),...rank2.H1.map(r=>r.slice())];
  // For a shared 2-D output subspace, 3 output rows are rank <= 2:
  // every 3x3 minor of the 3x8 horizontal concatenation vanishes.
  for(let j=0;j<4;j++){
    const cols=[rank2.H0.map(r=>r[j]),rank2.H1.map(r=>r[j]),rank2.H0.map(r=>r[(j+1)%4])];
    const [a,b,c]=cols;
    const det=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]);
    near(det,0,1e-10);
  }
  const full=predictionMatrix(e.fit(),.3);
  assert.equal(full.length,OUTPUTS.length);assert.equal(full[0].length,PORTS.length);
});

test('Operaattori-style reference fixture is valid and has measurable context effects',()=>{
  assert.ok(validateCableFixture(fixture));
  assert.equal(fixture.not_biological_data,true);
  assert.equal(fixture.responses[0].length,4);
  const at0=interpolateFixture(fixture,0,0);
  fixture.responses[2][0].forEach((v,i)=>near(v,at0[i]));
  const atHalf=interpolateFixture(fixture,-.5,3);
  fixture.responses[1][3].forEach((v,i)=>near(v,atHalf[i]));
  assert.ok(Math.abs(interpolateFixture(fixture,-1,0)[0]-interpolateFixture(fixture,1,0)[0])>0.1);
});

test('blind inverse recovers cable simulator observable better than constant model',()=>{
  const e=createBlindExperiment({seed:20261012,sigma:.012,substrate:'cable',fixture});
  for(let j=0;j<4;j++)e.measure(0,j);
  for(let n=0;n<28;n++){const p=e.choose();e.measure(p.context,p.port)}
  const r=e.audit();
  const a=r.find(v=>v.name==='Affine').rmse;
  assert.ok(a<0.025,`cable heldout RMSE ${a}`);
  assert.ok(a<r.find(v=>v.name==='Constant').rmse/3);
  assert.ok(a<r.find(v=>v.name==='Rank 1').rmse);
});

test('fixed-rank inference never gets oracle matrices via an audit',()=>{
  const e=createBlindExperiment({seed:8888});
  const before=e.samples.length;
  const audit=e.audit();
  assert.equal(e.samples.length,before);
  assert.equal(audit.length,4);
  assert.deepEqual(Object.keys(e.fit()).sort(),['H0','H1','n','rank','sigma','variances'].sort());
});
