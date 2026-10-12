/** Read-only renderer. The website loads upstream-generated, held-out
 * temporal traces. It does not learn from withheld test data in the browser.
 */
const $=id=>document.getElementById(id);
const num=(x,d=2)=>Number(x).toFixed(d);
const palette={truth:'#9fe5c9',instant:'#e0a579',linear_history:'#8da7ea',quadratic_history:'#f1d27c',A:'#97e4c7',B:'#dfa485'};
const safe=(s)=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function makeTraceSVG(times,series,options={}){
  const W=900,H=312,L=68,R=22,T=24,B=43;
  const points=series.flatMap(s=>s.values).filter(Number.isFinite);
  if(!times.length||!points.length)return '<p>No trace data.</p>';
  let y0=Math.min(0,...points),y1=Math.max(0,...points);
  const pad=Math.max(1e-7,(y1-y0)*.10);y0-=pad;y1+=pad;
  const x0=times[0],x1=times[times.length-1];
  const X=t=>L+(t-x0)/(x1-x0)*(W-L-R),Y=v=>H-B-(v-y0)/(y1-y0)*(H-T-B);
  const fmt=v=>num(v,1);
  let out=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${safe(options.label||'Response traces')}" xmlns="http://www.w3.org/2000/svg">`;
  for(let k=0;k<=4;k++){
    const y=H-B-k*(H-T-B)/4, v=y0+k*(y1-y0)/4;
    out+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="#52635c" stroke-opacity=".35"/><text x="${L-9}" y="${y+4}" fill="#a4b2ac" font-size="13" text-anchor="end">${fmt(v)}</text>`;
  }
  for(let k=0;k<=4;k++){
    const t=x0+k*(x1-x0)/4,x=X(t);
    out+=`<line x1="${x}" y1="${T}" x2="${x}" y2="${H-B}" stroke="#52635c" stroke-opacity=".2"/><text x="${x}" y="${H-B+23}" fill="#a4b2ac" font-size="13" text-anchor="middle">${fmt(t)}</text>`;
  }
  if(Number.isFinite(options.probe)){
    const x=X(options.probe);
    out+=`<line x1="${x}" y1="${T}" x2="${x}" y2="${H-B}" stroke="#f2cb8d" stroke-dasharray="6 5" stroke-width="1.6"/><text x="${Math.min(W-95,x+8)}" y="${T+18}" fill="#f2cb8d" font-size="12">PROBE</text>`;
  }
  for(const s of series){
    const d=s.values.map((v,i)=>`${i?'L':'M'}${X(times[i]).toFixed(2)},${Y(v).toFixed(2)}`).join(' ');
    out+=`<path d="${d}" fill="none" stroke="${palette[s.key]||'#fff'}" stroke-width="${s.key==='truth'?'3.5':'2.4'}" stroke-linejoin="round" stroke-linecap="round" opacity="${s.key==='instant'?'.83':'1'}"/>`;
  }
  out+=`<text x="${W/2}" y="${H-7}" fill="#b0c1ba" font-size="12" text-anchor="middle">Time (ms)</text>`;
  out+=`<text transform="translate(16 ${H/2}) rotate(-90)" fill="#b0c1ba" font-size="12" text-anchor="middle">${safe(options.unit||'Soma ΔV (mV)')}</text></svg>`;
  return out;
}
function legend(items){return '<div class="temporal-legend">'+items.map(({key,name})=>`<span><i style="background:${palette[key]}"></i>${safe(name)}</span>`).join('')+'</div>';}

async function init(){
  const warning=$('temporal-status');
  try{
    const response=await fetch('./data/temporal_reference.json',{cache:'no-store'});
    if(!response.ok)throw Error(`HTTP ${response.status}`);
    const data=await response.json();
    if(data.schema!=='monday.temporal.v1'||data.examples.length<3)throw Error('Wrong data fixture schema');
    const m=data.mean_holdout_rmse_mV;
    $('temporal-metrics').innerHTML=[['instant','CURRENT ONLY'],['linear_history','LINEAR HISTORY'],['quadratic_history','NONLINEAR HISTORY']].map(([key,label])=>`<div class="temporal-stat"><span>${label}</span><strong>${num(m[key])}<small> mV</small></strong></div>`).join('');
    $('temporal-wins').textContent=`On ${data.wins.n_runs} held-out seeds: linear history beat the instantaneous model ${data.wins.linear_history_beats_instant} times; quadratic history beat linear history ${data.wins.quadratic_beats_linear_history} times. These are exploratory model comparisons, not independent biological measurements.`;
    const choice=$('temporal-case');
    choice.innerHTML=data.examples.map((ex,i)=>`<option value="${i}">Unseen history ${i+1} · probe port ${ex.meta.probe_port+1}</option>`).join('');
    const switches=[...document.querySelectorAll('[data-temporal-line]')];
    const draw=()=>{
      const ex=data.examples[Number(choice.value)||0];
      const chosen=[{key:'truth',values:ex.truth_mV,name:'Actual soma'}];
      for(const cb of switches)if(cb.checked){const key=cb.dataset.temporalLine;chosen.push({key,values:ex[key+'_mV'],name:key});}
      $('temporal-plot').innerHTML=makeTraceSVG(ex.t_ms,chosen,{probe:ex.meta.probe_time_ms,label:'True and predicted soma voltage from held-out triple-history episode'});
      $('temporal-legend').innerHTML=legend(chosen.map(s=>({key:s.key,name:s.name.replaceAll('_',' ')})));
      $('temporal-description').textContent=`Three earlier input events at ports ${ex.meta.history_ports.map(x=>x+1).join(', ')}; withheld late probe at t=${num(ex.meta.probe_time_ms,2)} ms. The learner sees the entire stimulus program but cannot inspect dendritic voltages. Predictions are from training on separate episodes.`;
    };
    choice.addEventListener('change',draw);
    switches.forEach(cb=>cb.addEventListener('change',draw));draw();
    const pair=data.matched_soma;
    $('temporal-match-metrics').innerHTML=`The soma voltages at t=${num(pair.preprobe_at_ms,1)} ms differ by only <strong>${pair.preprobe_abs_error_mV.toExponential(2)} mV</strong>. Yet the two incremental responses to the same probe differ by up to <strong>${num(pair.max_probe_increment_difference_mV,3)} mV</strong>. The histories were deliberately amplitude-matched on that one scalar observation.`;
    const curves=pair.curve;
    $('temporal-pair-plot').innerHTML=makeTraceSVG(curves.t_ms,[{key:'A',values:curves.delta_A_mV},{key:'B',values:curves.delta_B_mV}],{probe:pair.probe.onset_ms,label:'Difference between two probe-response increments after matched one-time soma readings'});
    $('temporal-pair-legend').innerHTML=legend([{key:'A',name:'History A: proximal conditioning'},{key:'B',name:'History B: distal conditioning'}]);
    warning.textContent='Data: pinned Operaattori numerical runtime · independent train/test stimulus sequences · browser displays saved responses.';
  }catch(error){warning.textContent='Could not load the temporal reference: '+error.message;console.error(error);}
}
init();
