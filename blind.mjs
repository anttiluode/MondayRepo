/** Blind operator reconstruction. No true parameters, candidate circuit list,
 * or oracle response matrix is passed to the estimator or its probe selector.
 * Probe contract: y_i = (H0[i][port] + c H1[i][port]) + N(0, sigma^2).
 */
import {CONTEXTS, PORTS, OUTPUTS, circuit, responseMatrix, makeRng, normal} from './model.mjs';

export const zeroMatrix = () => OUTPUTS.map(()=>PORTS.map(()=>0));
const regularizer = 0.25; // isotropic Gaussian prior, precision 0.25
export function inverse2x2(a,b,d){
  const det=a*d-b*b;
  if(!(det>1e-16)) throw new Error('Design not invertible');
  return [[d/det,-b/det],[-b/det,a/det]];
}
export function fitOperator(samples, sigma=0.12, rank=0){
  if(!(sigma>0))throw new Error('positive sigma required');
  const H0=zeroMatrix(),H1=zeroMatrix(),variances=[];
  const noisePrecision=1/(sigma*sigma);
  for(let j=0;j<PORTS.length;j++){
    let a=regularizer,b=0,d=regularizer;
    const rhs0=OUTPUTS.map(()=>0),rhs1=OUTPUTS.map(()=>0);
    for(const row of samples){
      if(row.port!==j)continue;
      if(!Number.isFinite(row.context)||row.observed?.length!==OUTPUTS.length||row.observed.some(v=>!Number.isFinite(v)))throw Error('Invalid measurement');
      const c=row.context;
      a+=noisePrecision;b+=noisePrecision*c;d+=noisePrecision*c*c;
      row.observed.forEach((v,i)=>{rhs0[i]+=noisePrecision*v;rhs1[i]+=noisePrecision*c*v;});
    }
    const cov=inverse2x2(a,b,d);
    variances.push(cov);
    for(let i=0;i<OUTPUTS.length;i++){
      H0[i][j]=cov[0][0]*rhs0[i]+cov[0][1]*rhs1[i];
      H1[i][j]=cov[1][0]*rhs0[i]+cov[1][1]*rhs1[i];
    }
  }
  if(rank===1 || rank===2){
    // Find the dominant shared output subspace of all intercept/slope columns.
    // Orthogonal iteration for small (3 x 3) PSD Gram matrix, with a robust
    // Jacobi diagonalization to avoid fragile eigenvector orientation.
    const gram=OUTPUTS.map((_,i)=>OUTPUTS.map((_,j)=>
      H0[i].reduce((s,v,p)=>s+v*H0[j][p]+H1[i][p]*H1[j][p],0)));
    const eigen=jacobiSymmetric(gram).slice(0,rank);
    for(const mat of [H0,H1]){
      const old=mat.map(row=>row.slice());
      for(let i=0;i<OUTPUTS.length;i++)for(let j=0;j<PORTS.length;j++){
        mat[i][j]=eigen.reduce((sum,e)=>sum+e.vec[i]*e.vec.reduce((t,w,k)=>t+w*old[k][j],0),0);
      }
    }
  }else if(rank!==0)throw Error('rank must be 0 (full), 1 or 2');
  return {H0,H1,variances,sigma,rank,n:samples.length};
}
function jacobiSymmetric(A){
  const n=A.length, a=A.map(x=>x.slice()),v=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>+(i===j)));
  for(let it=0;it<70;it++){
    let p=0,q=1,max=0;
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(Math.abs(a[i][j])>max){max=Math.abs(a[i][j]);p=i;q=j;}
    if(max<1e-12)break;
    const t=0.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]);
    const c=Math.cos(t),s=Math.sin(t); // R = [c s; -s c]
    for(let k=0;k<n;k++)if(k!==p&&k!==q){const kp=a[k][p],kq=a[k][q];a[k][p]=a[p][k]=c*kp-s*kq;a[k][q]=a[q][k]=s*kp+c*kq;}
    const pp=a[p][p],pq=a[p][q],qq=a[q][q];
    a[p][p]=c*c*pp-2*c*s*pq+s*s*qq;
    a[q][q]=s*s*pp+2*c*s*pq+c*c*qq;
    a[p][q]=a[q][p]=0;
    for(let k=0;k<n;k++){const vp=v[k][p],vq=v[k][q];v[k][p]=c*vp-s*vq;v[k][q]=s*vp+c*vq;}
  }
  return Array.from({length:n},(_,k)=>({value:a[k][k],vec:v.map(r=>r[k])})).sort((x,y)=>y.value-x.value);
}
export function predict(fit,c,port){return OUTPUTS.map((_,i)=>fit.H0[i][port]+c*fit.H1[i][port]);}
export function predictionMatrix(fit,c){return OUTPUTS.map((_,i)=>PORTS.map((_,j)=>fit.H0[i][j]+c*fit.H1[i][j]));}
export function uncertainty(fit,c,port){const v=fit.variances[port];return v[0][0]+2*c*v[0][1]+c*c*v[1][1];}
export function designScore(fit,c,port,targetContexts=[-1,0,1]){
  const V=fit.variances[port],den=fit.sigma**2+uncertainty(fit,c,port);
  if(den<=0)throw Error('bad variance');
  return targetContexts.reduce((s,t)=>{
    const a=V[0][0]+(c+t)*V[0][1]+c*t*V[1][1];
    return s+a*a/den;
  },0);
}
export function chooseProbe(fit,contexts=CONTEXTS){
  let best=null;
  for(const c of contexts)for(let j=0;j<PORTS.length;j++){
    const score=designScore(fit,c,j);
    if(best===null||score>best.score+1e-14)best={context:c,port:j,score};
  }
  return best;
}
export function rmse(fit,truth,contexts){
  let err=0,n=0;
  for(const c of contexts)for(let j=0;j<PORTS.length;j++){
    const p=predict(fit,c,j),r=truth(c,j);
    for(let i=0;i<p.length;i++){err+=(p[i]-r[i])**2;n++;}
  }
  return Math.sqrt(err/n);
}
export function createBlindExperiment({seed=20261012,sigma=0.12,substrate='factorized',fixture=null}={}){
  if(!(sigma>0))throw Error('noise must be positive');
  const rng=makeRng(seed);
  let hidden;
  if(substrate==='factorized'){
    // Seeded changes to the generative model, never passed to the inference.
    const base=circuit('A');
    const V=base.V.map(row=>row.map(v=>v*(0.85+0.30*rng())));
    const K=base.K.map(row=>row.map(v=>v*(0.85+0.30*rng())));
    const G0=base.G0,G1=base.G1.map(row=>row.map(v=>v*(0.65+0.7*rng())));
    const params={V,K,G0,G1};
    hidden=(c,j)=>responseMatrix(params,c).map(row=>row[j]);
  }else if(substrate==='cable'){
    if(!fixture||!validateCableFixture(fixture))throw Error('Valid cable reference fixture required');
    // Piecewise linear in context only for the *measurement fixture*; no
    // hidden cable coefficients are furnished to the learner.
    hidden=(c,j)=>interpolateFixture(fixture,c,j);
  }else throw Error('Unknown substrate');
  const samples=[];
  const fit=(rank=0)=>fitOperator(samples,sigma,rank);
  const measure=(context,port)=>{
    if(!CONTEXTS.includes(context)||!Number.isInteger(port)||port<0||port>=PORTS.length)throw Error('Outside declared training probe family');
    const observed=hidden(context,port).map(x=>x+sigma*normal(rng));
    const record={context,port,observed};samples.push(record);return {...record,observed:observed.slice()};
  };
  const audit=(contexts=[-0.5,0.5,1.5])=>{
    // Sealed scoring only. Never used by the fit or query selector.
    const models=[{name:'Constant',obj:fit(0),constant:true},{name:'Affine',obj:fit(0)},{name:'Rank 2',obj:fit(2)},{name:'Rank 1',obj:fit(1)}];
    return models.map(m=>{
      let total=0,num=0;
      for(const c of contexts)for(let j=0;j<PORTS.length;j++){
        const pred=predict(m.obj,m.constant?0:c,j),actual=hidden(c,j);
        for(let i=0;i<OUTPUTS.length;i++){total+=(pred[i]-actual[i])**2;num++;}
      }
      return {name:m.name,rmse:Math.sqrt(total/num)};
    });
  };
  return {substrate,seed,sigma,measure,fit,choose:()=>chooseProbe(fit()),audit,get samples(){return samples.map(r=>({...r,observed:r.observed.slice()}));}};
}
export function validateCableFixture(f){
  return f?.schema==='monday.cable.v1' && Array.isArray(f.contexts) && f.contexts.length>=2 && Array.isArray(f.responses) &&
    f.responses.length===f.contexts.length && f.responses.every(r=>r.length===PORTS.length && r.every(v=>Array.isArray(v)&&v.length===OUTPUTS.length&&v.every(Number.isFinite))) &&
    f.contexts.every(Number.isFinite)&&f.contexts.every((c,i)=>i===0||c>f.contexts[i-1]);
}
export function interpolateFixture(f,c,port){
  if(!validateCableFixture(f))throw Error('Bad fixture');
  const cs=f.contexts;
  let i=0;
  while(i<cs.length-2&&c>cs[i+1])i++;
  const t=(c-cs[i])/(cs[i+1]-cs[i]);
  return f.responses[i][port].map((x,k)=>x*(1-t)+f.responses[i+1][port][k]*t);
}
