/** Run deterministic descriptive comparison; no selection of winners in advance.
 * Equal budget for each policy; all evaluation contexts held out.
 */
import fs from 'node:fs';
import {createBlindExperiment} from '../blind.mjs';
import {makeRng} from '../model.mjs';
const cable=JSON.parse(fs.readFileSync(new URL('../data/cable_reference.json',import.meta.url)));
const budget=Number(process.argv[2]||12);
if(!Number.isInteger(budget)||budget<1||budget>80)throw Error('budget must be 1..80');
const runs=[];
for(const substrate of ['factorized','cable'])for(const policy of ['active','fixed','random']){
  const vals=[];
  for(let n=0;n<20;n++){
    const seed=12001+n*673;
    const e=createBlindExperiment({substrate,fixture:cable,seed,sigma:substrate==='factorized'?.12:.012});
    for(let j=0;j<4;j++)e.measure(0,j);
    const rng=makeRng(seed+812);
    for(let t=0;t<budget;t++){
      const p=policy==='active'?e.choose():policy==='fixed'?{context:t%2===0?-1:1,port:Math.floor(t/2)%4}:{context:[-1,0,1][Math.floor(rng()*3)],port:Math.floor(rng()*4)};
      e.measure(p.context,p.port);
    }
    const errors=e.audit();
    vals.push(Object.fromEntries(errors.map(x=>[x.name,x.rmse])));
  }
  const mean=k=>Number((vals.reduce((a,v)=>a+v[k],0)/vals.length).toFixed(6));
  runs.push({substrate,policy,n_seeds:20,n_rest_pulses:4,n_additional:budget,holdout_contexts:[-0.5,0.5,1.5],mean_rmse:{constant:mean('Constant'),affine:mean('Affine'),rank2:mean('Rank 2'),rank1:mean('Rank 1')}});
}
const report={scope:'Synthetic trained-observer comparison; cable uses prerecorded deterministic responses. No neural inference claims.',methods:'Bayesian ridge response operator with two context coefficients per output/port; fixed and random have same extra pulse budget.',seed_protocol:'12001 + 673*i, 20 seeds per condition',results:runs};
console.log(JSON.stringify(report,null,2));
