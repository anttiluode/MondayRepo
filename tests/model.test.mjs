import test from 'node:test';
import assert from 'node:assert/strict';
import {
  circuit,responseMatrix,responseColumn,maxResponseDifference,createExperiment,
  passiveCorrelations,posterior,CONTEXTS,PORTS,OUTPUTS
} from '../model.mjs';
const near=(a,b,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`Expected ${a} near ${b}`);
test('response dimensions are 3 outputs by 4 input ports',()=>{
  const H=responseMatrix(circuit('A'),0);
  assert.equal(H.length,OUTPUTS.length);
  H.forEach(row=>assert.equal(row.length,PORTS.length));
});
test('different physical response hypotheses agree at reference context',()=>{
  const a=responseMatrix(circuit('A'),0),b=responseMatrix(circuit('B','identifiable'),0);
  a.forEach((row,i)=>row.forEach((x,j)=>near(x,b[i][j])));
});
test('different hypotheses become distinguishable after nonzero context',()=>{
  const best=maxResponseDifference('identifiable');
  assert.ok(Math.abs(best.context)>0);
  assert.ok(best.score>0.1);
});
test('gauge twin has distinct internal coefficients but exactly equal responses',()=>{
  const a=circuit('A','gauge'),b=circuit('B','gauge');
  assert.notDeepEqual(a.K,b.K);
  assert.notDeepEqual(a.V,b.V);
  for(const c of [...CONTEXTS,-0.5,0.13,2]){
    const ha=responseMatrix(a,c), hb=responseMatrix(b,c);
    ha.forEach((row,i)=>row.forEach((x,j)=>near(x,hb[i][j],1e-12)));
  }
  near(maxResponseDifference('gauge').score,0,1e-12);
});
test('gauge twins cannot be identified by any sequence of allowed probes',()=>{
  const e=createExperiment({mode:'gauge',seed:41});
  for(let n=0;n<80;n++) e.pulse(CONTEXTS[n%3],n%4);
  near(e.posteriorA,0.5,1e-12);
});
test('informative interventions identify either distinguishable truth with fixed noise seed',()=>{
  const best=maxResponseDifference('identifiable');
  for(const truth of ['A','B']){
    const e=createExperiment({mode:'identifiable',seed:812,sigma:0.22,truth});
    for(let n=0;n<8;n++) e.pulse(best.context,best.port);
    assert.equal(e.posteriorA>0.5,truth==='A');
    assert.ok(truth==='A'?e.posteriorA>0.99:e.posteriorA<0.01);
  }
});
test('posterior remains finite and bounded for extreme evidence',()=>{
  near(posterior(0),0.5);
  assert.ok(posterior(1e6)<1 && posterior(-1e6)>0);
});
test('correlation heatmap is symmetric, unit diagonal and audio changes it',()=>{
  const raw=passiveCorrelations({audioGain:2.2});
  const cleaned=passiveCorrelations({audioGain:2.2,removed:true});
  raw.forEach((row,i)=>row.forEach((x,j)=>{
    near(x,raw[j][i]); if(i===j)near(x,1);
  }));
  assert.ok(raw.some((r,i)=>r.some((v,j)=>i!==j&&Math.abs(v-cleaned[i][j])>0.1)));
});
