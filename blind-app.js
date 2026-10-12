import {OUTPUTS,PORTS,CONTEXTS,makeRng} from './model.mjs';
import {createBlindExperiment,predictionMatrix,uncertainty,validateCableFixture} from './blind.mjs';

const $=id=>document.getElementById(id);
let specimen,fixture=null,substrate='factorized',seed=20261012,sigma=.12,context=0,port=0;
let audit=null;
const namePort=j=>`P${j+1}`;
const paintedCell=(v)=>{
  if(v===null||!Number.isFinite(v))return null;
  const r=Math.min(1,Math.abs(v)/1.15)**.7*.87;
  const start=[36,49,55],end=v>=0?[111,184,159]:[186,120,108];
  return `rgb(${start.map((a,i)=>Math.round(a+(end[i]-a)*r)).join(',')})`;
};
function paint(el,arr){
  el.replaceChildren();
  const corner=document.createElement('span');corner.className='axis';el.append(corner);
  PORTS.forEach((n,j)=>{const a=document.createElement('span');a.className='axis';a.title=n;a.textContent=namePort(j);el.append(a)});
  arr.forEach((row,i)=>{
    const label=document.createElement('span');label.className='axis left';label.textContent=`S${i+1}`;el.append(label);
    row.forEach((v,j)=>{const d=document.createElement('span');d.className='cell'+(v===null?' empty':'');d.textContent=v===null?'·':`${v>=0?'+':''}${v.toFixed(2)}`;d.title=v===null?'Unmeasured':`Output ${i+1}, port ${j+1}: ${v.toFixed(5)}`;let c=paintedCell(v);if(c)d.style.backgroundColor=c;el.append(d)});
  });
}
function render(){
  document.querySelectorAll('[data-blind-substrate]').forEach(b=>{const active=b.dataset.blindSubstrate===substrate;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active))});
  document.querySelectorAll('[data-blind-context]').forEach(b=>{const active=Number(b.dataset.blindContext)===context;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active))});
  document.querySelectorAll('[data-blind-port]').forEach(b=>{const active=Number(b.dataset.blindPort)===port;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active))});
  const samples=specimen.samples,fit=specimen.fit();
  const solved=PORTS.filter((_,j)=>new Set(samples.filter(o=>o.port===j).map(o=>o.context)).size>=2).length;
  $('blind-count').textContent=samples.length+' MEASUREMENTS';
  $('blind-coverage').textContent=`${solved}/${PORTS.length} PORTS`;
  $('blind-uncertainty').textContent=Math.sqrt(uncertainty(fit,context,port)).toFixed(3);
  paint($('blind-map'),predictionMatrix(fit,context));
  const measured=OUTPUTS.map(()=>PORTS.map(()=>null));
  samples.filter(o=>o.context===context).forEach(o=>o.observed.forEach((v,i)=>measured[i][o.port]=v));
  paint($('blind-observed'),measured);
  if(audit)renderAudit();
}
function renderAudit(){
  const box=$('blind-audit-results');box.replaceChildren();
  if(!audit)return;
  const sorted=[...audit].sort((a,b)=>a.rmse-b.rmse);
  const table=document.createElement('table');
  const thead=document.createElement('thead'),tr=document.createElement('tr');
  ['Fitted model','Withheld RMSE ↓'].forEach(label=>{let th=document.createElement('th');th.textContent=label;tr.append(th)});thead.append(tr);table.append(thead);
  const body=document.createElement('tbody');
  sorted.forEach(v=>{const row=document.createElement('tr');let l=document.createElement('td');l.textContent=v.name;let score=document.createElement('td');score.textContent=v.rmse.toFixed(4);row.append(l,score);body.append(row)});table.append(body);box.append(table);
  const note=document.createElement('p');note.className='audit-note';note.textContent='True noise-free responses were consulted ONLY by this scoring panel. The training observations, fit, and uncertainty selector have no access to them. Smaller error is better; compare across fresh seeds before making any model-ranking claim.';box.append(note);
}
function newSpecimen(next=false){
  if(next)seed=(seed+7919)>>>0;
  specimen=createBlindExperiment({substrate,seed,sigma,fixture});
  audit=null;
  $('blind-audit-results').innerHTML='<p class="fine-print">Sealed scoring not requested for this specimen.</p>';
  $('blind-noise').value=String(sigma);$('blind-noise-label').textContent=sigma.toFixed(3)+' σ';
  $('blind-status').textContent=`Specimen #${seed} opened. ${substrate==='cable'?'Recorded cable response fixture':'Seeded latent circuit'}; no candidate parameters supplied to learner.`;
  render();
}
function measure(c=context,j=port,label='Chosen'){
  const o=specimen.measure(c,j);
  context=c;port=j;audit=null;
  $('blind-audit-results').innerHTML='<p class="fine-print">New evidence collected. Re-run the sealed audit to evaluate the updated fit.</p>';
  $('blind-status').textContent=`${label} pulse at context ${c>=0?'+':''}${c} and port ${j+1}. Observed ${o.observed.map(v=>v.toFixed(3)).join(' / ')}. Updated from this observation only.`;
  render();
}
$('blind-measure').addEventListener('click',()=>measure());
$('blind-best').addEventListener('click',()=>{const best=specimen.choose();measure(best.context,best.port,'Uncertainty-chosen')});
$('blind-random').addEventListener('click',()=>{const r=makeRng((seed+specimen.samples.length*4099+837)>>>0);measure(CONTEXTS[Math.floor(r()*CONTEXTS.length)],Math.floor(r()*PORTS.length),'Random')});
$('blind-rest').addEventListener('click',()=>{for(let j=0;j<PORTS.length;j++)measure(0,j,'Rest-sweep')});
$('blind-new').addEventListener('click',()=>newSpecimen(true));
const doAudit=()=>{audit=specimen.audit();renderAudit();$('blind-status').textContent=`Audited ${specimen.samples.length} measured pulses against withheld contexts. The audit made no new observations and did not update the model.`};
$('blind-audit').addEventListener('click',doAudit);
$('blind-audit-main').addEventListener('click',doAudit);
document.querySelectorAll('[data-blind-context]').forEach(b=>b.addEventListener('click',()=>{context=Number(b.dataset.blindContext);render()}));
document.querySelectorAll('[data-blind-port]').forEach(b=>b.addEventListener('click',()=>{port=Number(b.dataset.blindPort);render()}));
$('blind-noise').addEventListener('input',e=>$('blind-noise-label').textContent=Number(e.target.value).toFixed(3)+' σ');
$('blind-noise').addEventListener('change',e=>{sigma=Number(e.target.value);newSpecimen()});
document.querySelectorAll('[data-blind-substrate]').forEach(b=>b.addEventListener('click',()=>{
  if(b.dataset.blindSubstrate==='cable'&&!fixture){$('blind-status').textContent='Cable reference not loaded; cannot select this substrate.';return}
  substrate=b.dataset.blindSubstrate;context=0;port=0;sigma=substrate==='cable'?.012:.12;
  newSpecimen();
}));
newSpecimen();
fetch('./data/cable_reference.json').then(r=>{if(!r.ok)throw Error('Missing fixture');return r.json()}).then(f=>{if(!validateCableFixture(f))throw Error('Invalid cable reference schema');fixture=f;}).catch(e=>{document.querySelector('[data-blind-substrate="cable"]').title='Cable fixture unavailable: '+e.message;});
