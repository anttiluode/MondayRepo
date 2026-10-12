import {OUTPUTS,PORTS,CONTEXTS,createExperiment,maxResponseDifference,responseMatrix,passiveCorrelations,makeRng} from './model.mjs';
const $=id=>document.getElementById(id);
let mode='identifiable',context=0,port=0,sigma=0.22,seed=20261012,artifactGain=1.6,truthShown=false;
let experiment=createExperiment({mode,seed,sigma});
const getObs=()=>{
  const result=Array.from({length:OUTPUTS.length},()=>Array(PORTS.length).fill(null));
  // Show last observation for each context/port, not a pooled average of mismatched contexts.
  for(const row of experiment.history.filter(r=>r.context===context)){
    row.observed.forEach((v,i)=>result[i][row.port]=v);
  }
  return result;
};
function colored(v,max=1.3){
  if(v===null||!Number.isFinite(v)) return null;
  const frac=Math.min(1,Math.abs(v)/max);
  const base=[36,49,55],dest=v>=0?[111,184,159]:[186,120,108];
  const f=Math.pow(frac,0.73)*.89;
  return `rgb(${base.map((b,i)=>Math.round(b+(dest[i]-b)*f)).join(',')})`;
}
function matrix(el,m,{columns=PORTS,rows=OUTPUTS,max=1.3,markPort=null}={}){
  el.replaceChildren();
  const top=document.createElement('span');top.className='axis';el.append(top);
  columns.forEach((col,j)=>{
    const a=document.createElement('span');a.className='axis';a.textContent=columns===PORTS?`P${j+1}`:`S${j+1}`;a.title=col;el.append(a);
  });
  m.forEach((row,i)=>{
    const label=document.createElement('span');label.className='axis left';label.textContent=rows===OUTPUTS?`S${i+1}`:rows[i];el.append(label);
    row.forEach((v,j)=>{
      const cell=document.createElement('span');cell.className='cell'+(v===null?' empty':'')+(markPort===j&&v!==null?' observed':'');
      cell.textContent=v===null?'·':(v>=0?'+':'')+v.toFixed(2);
      cell.title=`${rows[i]}, ${columns[j]}: ${v===null?'not measured':v.toFixed(4)}`;
      const c=colored(v,max);if(c)cell.style.backgroundColor=c;
      el.append(cell);
    });
  });
}
function redrawButtons(){
  document.querySelectorAll('[data-mode]').forEach(b=>{const yes=b.dataset.mode===mode;b.classList.toggle('active',yes);b.setAttribute('aria-pressed',String(yes));});
  document.querySelectorAll('[data-context]').forEach(b=>{const yes=Number(b.dataset.context)===context;b.classList.toggle('active',yes);b.setAttribute('aria-pressed',String(yes));});
  document.querySelectorAll('[data-port]').forEach(b=>{const yes=Number(b.dataset.port)===port;b.classList.toggle('active',yes);b.setAttribute('aria-pressed',String(yes));});
}
function render(){
  redrawButtons();
  const p=experiment.posteriorA, n=experiment.history.length;
  $('posterior-number').innerHTML=`${(p*100).toFixed(1)}<span>%</span>`;
  $('pa-label').textContent=(p*100).toFixed(1)+'%';
  $('pb-label').textContent=((1-p)*100).toFixed(1)+'%';
  $('pulse-count').textContent=n+' PULSE'+(n===1?'':'S');
  $('meter-a').style.width=(p*100)+'%';
  $('meter-b').style.width=((1-p)*100)+'%';
  const verdict=$('verdict');verdict.className='verdict';
  if(mode==='gauge'){
    verdict.textContent='NOT IDENTIFIABLE';verdict.classList.add('impossible');
    $('result-caption').textContent='Exact observational equivalence: the two circuits differ in internal coordinates but produce identical responses for every allowed pulse. The posterior cannot move from 50/50.';
  }else if(p>0.95||p<0.05){
    verdict.textContent=`FAVORS ${p>0.95?'A':'B'}`;verdict.classList.add('identified');
    $('result-caption').textContent='The observations favor one response hypothesis under this finite, correctly specified two-candidate model. This does not identify unique internal coordinates.';
  }else{
    verdict.textContent='UNRESOLVED';
    $('result-caption').textContent=n?'Evidence is still inconclusive. At context 0 the rivals make exactly the same predictions; try changing the context.':'At context 0 the rivals make identical predictions. Select another context or ask the instrument for its most informative pulse.';
  }
  matrix($('map-a'),responseMatrix(experiment.a,context));
  matrix($('map-b'),responseMatrix(experiment.b,context));
  const observed=getObs();matrix($('map-measured'),observed,{markPort:port});
  const ledger=$('ledger');ledger.replaceChildren();
  if(n===0){const tr=document.createElement('tr');const td=document.createElement('td');td.colSpan=5;td.className='empty-ledger';td.textContent='No pulses recorded yet. The instrument is waiting.';tr.append(td);ledger.append(tr);}
  else experiment.history.slice(-7).reverse().forEach((obs,k)=>{
    const tr=document.createElement('tr');
    const cells=[`${n-k}`,obs.context>0?'+1':String(obs.context),`P${obs.port+1}`,obs.observed.map(x=>(x>=0?'+':'')+x.toFixed(2)).join(' / '),(obs.pA*100).toFixed(1)+'%'];
    cells.forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td)});ledger.append(tr);
  });
  $('reveal').textContent=truthShown?`Truth: ${experiment.truthName} · hide`:'Reveal hidden circuit';
}
function renderPassive(){
  const args={audioGain:artifactGain};
  matrix($('passive-raw'),passiveCorrelations(args),{columns:OUTPUTS,rows:OUTPUTS,max:1});
  matrix($('passive-clean'),passiveCorrelations({...args,removed:true}),{columns:OUTPUTS,rows:OUTPUTS,max:1});
  $('artifact-label').textContent=artifactGain.toFixed(2);
}
function newSpecimen(next=true){
  if(next)seed=(seed+7919)>>>0;
  truthShown=false;
  experiment=createExperiment({mode,seed,sigma});
  $('experiment-status').textContent=`New sealed specimen #${seed}. All prior evidence cleared.`;
  render();
}
function fire(c=context,j=port,label='chosen'){
  context=c;port=j;
  const row=experiment.pulse(context,port);
  $('experiment-status').textContent=`${label} pulse: context ${context>0?'+':''}${context}, port ${port+1}. Predicted A/B separation ${row.score.toFixed(3)} (squared output units).`;
  render();
}
document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>{
  mode=button.dataset.mode;context=0;port=0;newSpecimen(false);
}));
document.querySelectorAll('[data-context]').forEach(button=>button.addEventListener('click',()=>{context=Number(button.dataset.context);render()}));
document.querySelectorAll('[data-port]').forEach(button=>button.addEventListener('click',()=>{port=Number(button.dataset.port);render()}));
$('noise').addEventListener('input',e=>{$('noise-label').textContent=Number(e.target.value).toFixed(2)+' σ'});
$('noise').addEventListener('change',e=>{sigma=Number(e.target.value);newSpecimen(false)});
$('artifact').addEventListener('input',e=>{artifactGain=Number(e.target.value);renderPassive()});
$('pulse').addEventListener('click',()=>fire());
$('best').addEventListener('click',()=>{
  const best=maxResponseDifference(mode);
  if(best.score<1e-15){
    $('experiment-status').textContent='No discriminating pulse exists in this model. Try any context or input; its likelihood ratio remains exactly 1.';
    return;
  }
  fire(best.context,best.port,'Best discriminating');
});
$('random').addEventListener('click',()=>{
  const rng=makeRng(seed+experiment.history.length*107+4831);
  fire(CONTEXTS[Math.floor(rng()*CONTEXTS.length)],Math.floor(rng()*PORTS.length),'Random');
});
$('new-specimen').addEventListener('click',()=>newSpecimen(true));
$('reveal').addEventListener('click',()=>{
  truthShown=!truthShown;
  $('experiment-status').textContent=truthShown?`The sealed generator is circuit ${experiment.truthName}. This disclosure is not an inference.`:'Generator identity hidden again. Prior disclosure cannot be unseen.';
  render();
});
render();renderPassive();
