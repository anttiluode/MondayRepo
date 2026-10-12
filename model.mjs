/**
 * MondayRepo experimental core. Deterministic, synthetic, fully browser compatible.
 * Three observed outputs, four controllable input ports, two latent coordinates.
 * This is a context-gated linear response operator, not biological data or
 * a literal transformer QKV implementation.
 */
export const OUTPUTS = ['Soma A', 'Soma B', 'Soma C'];
export const PORTS = ['Dendrite 1', 'Dendrite 2', 'Dendrite 3', 'Dendrite 4'];
export const CONTEXTS = [-1, 0, 1];
export const K = [[0.90,0.15],[0.10,1.00],[0.65,-0.85],[-0.80,0.65]];
export const V = [[1.00,0.20],[0.15,1.05],[-0.55,0.80]];
export const G0 = [[1.00,0.18],[-0.08,0.85]];
export const G1 = [[0.45,-0.40],[0.30,-0.33]];

export const transpose = a => a[0].map((_,i) => a.map(row=>row[i]));
export const multiply = (a,b) => a.map(row=>b[0].map((_,j)=>row.reduce((sum,v,k)=>sum+v*b[k][j],0)));
export const add = (a,b,scale=1) => a.map((row,i)=>row.map((v,j)=>v+scale*b[i][j]));
export const scale = (a,k) => a.map(row=>row.map(v=>v*k));
export const inverse2 = a => {
  const d=a[0][0]*a[1][1]-a[0][1]*a[1][0];
  if(Math.abs(d)<1e-12) throw new Error('Singular basis change');
  return [[a[1][1]/d,-a[0][1]/d],[-a[1][0]/d,a[0][0]/d]];
};
export const makeRng = seed => {
  let s=seed>>>0;
  return () => {
    s=(s+0x6D2B79F5)>>>0;
    let t=s;
    t=Math.imul(t^(t>>>15),t|1);
    t^=t+Math.imul(t^(t>>>7),t|61);
    return ((t^(t>>>14))>>>0)/4294967296;
  };
};
export function normal(rng){return Math.sqrt(-2*Math.log(Math.max(1e-12,rng())))*Math.cos(2*Math.PI*rng());}
const R = [[1.0,0.65],[-0.35,1.10]];
export function circuit(kind='A', mode='identifiable'){
  if(kind==='A') return {K,V,G0,G1,name:'Circuit A'};
  if(mode==='gauge') {
    const inv=inverse2(R);
    return {
      K:multiply(K,transpose(inv)),
      V:multiply(V,R),
      G0:multiply(multiply(inv,G0),R),
      G1:multiply(multiply(inv,G1),R),
      name:'Circuit B (changed latent basis)'
    };
  }
  return {K,V,G0,G1:scale(G1,-0.65),name:'Circuit B (different context response)'};
}
export function responseMatrix(model, context){
  return multiply(multiply(model.V,add(model.G0,model.G1,context)),transpose(model.K));
}
export function responseColumn(model,context,port){
  return responseMatrix(model,context).map(row=>row[port]);
}
export function maxResponseDifference(mode='identifiable',contexts=CONTEXTS){
  const A=circuit('A',mode),B=circuit('B',mode);
  let best={context:0,port:0,score:0};
  for(const context of contexts){
    for(let port=0;port<PORTS.length;port++){
      const a=responseColumn(A,context,port), b=responseColumn(B,context,port);
      const score=a.reduce((s,x,i)=>s+(x-b[i])**2,0);
      if(score>best.score) best={context,port,score};
    }
  }
  return best;
}
export function posterior(logOdds){return 1/(1+Math.exp(-Math.max(-24,Math.min(24,logOdds))));}
export function updateLogOdds(logOdds,observation,predA,predB,sigma){
  const sqA=observation.reduce((s,v,i)=>s+(v-predA[i])**2,0);
  const sqB=observation.reduce((s,v,i)=>s+(v-predB[i])**2,0);
  return Math.max(-24,Math.min(24,logOdds + (sqB-sqA)/(2*sigma*sigma)));
}
export function createExperiment({mode='identifiable',seed=20261012,sigma=0.22,truth=null}={}){
  const rng=makeRng(seed),truthName=truth??(rng()<0.5?'A':'B');
  const a=circuit('A',mode),b=circuit('B',mode);
  let logOdds=0;
  const history=[];
  function pulse(context,port){
    const pa=responseColumn(a,context,port), pb=responseColumn(b,context,port);
    const truthPred=truthName==='A'?pa:pb;
    const observed=truthPred.map(v=>v+sigma*normal(rng));
    logOdds=updateLogOdds(logOdds,observed,pa,pb,sigma);
    history.push({context,port,observed,predA:pa,predB:pb,pA:posterior(logOdds),score:pa.reduce((s,v,i)=>s+(v-pb[i])**2,0)});
    return history[history.length-1];
  }
  return {a,b,mode,sigma,seed,truthName,pulse,history,get posteriorA(){return posterior(logOdds)},get logOdds(){return logOdds}};
}
/** Correlations of observed continuous sensor signals, NOT an attention matrix.
 * audioGain injects a known common nuisance waveform; removed=true subtracts
 * its EXACT synthetic contribution, an oracle control (not real EEG cleaning).
 */
export function passiveCorrelations({audioGain=1.6,removed=false,seed=9901,n=480}={}){
  const rng=makeRng(seed), h=responseMatrix(circuit('A'),0.35), samples=[];
  for(let t=0;t<n;t++){
    const p=t/n;
    const a=Math.sin(2*Math.PI*7*p)+0.55*Math.sin(2*Math.PI*19*p+0.45)+0.16*normal(rng);
    const z1=Math.sin(2*Math.PI*3*p+0.3)+0.35*normal(rng);
    const z2=Math.cos(2*Math.PI*5*p)+0.35*normal(rng);
    const u=[z1+0.24*normal(rng),z2+0.24*normal(rng),0.55*z1-0.2*z2+0.25*normal(rng),-0.35*z1+0.4*z2+0.22*normal(rng)];
    const mix=[1.0,0.95,1.15];
    const row=h.map((weights,i)=>weights.reduce((s,w,j)=>s+w*u[j],0)+0.23*normal(rng)+(removed?0:audioGain*mix[i]*a));
    samples.push(row);
  }
  const columns=transpose(samples),means=columns.map(a=>a.reduce((s,v)=>s+v,0)/a.length);
  const centered=columns.map((a,i)=>a.map(v=>v-means[i]));
  const inner=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
  return centered.map(a=>centered.map(b=>inner(a,b)/Math.sqrt(inner(a,a)*inner(b,b))));
}
