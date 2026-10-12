#!/usr/bin/env python3
"""MondayRepo temporal inverse gate. Run with the original Operaattori checkout.

This is a synthetic system-identification benchmark, NOT observed brain activity.
Forward source: Operaattori reduced.causal_graph_circuit.CausalGraphCircuit.
Learners observe known stimulus drive, soma voltage and no internal cable state.
Train/test split: independent RNGs; train 0-3 early events and common late
probe near 19 ms, test 3 early events and a late probe near 21 ms.
Models: current-only ridge, causal multi-timescale linear feature memory,
and the same feature memory with fixed quadratic cross terms.
No trained coefficients, gradients, neuron states or test traces enter probes.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

DT=.1
STEPS=320
PORT_NODES=(2,3,4,5)
SITE_NODES=list(PORT_NODES)+[6]
TAUS=(.3,1,3,8,20)
TRAIN_SEEDS=(1201,1202,1203,1204,1205,1206,1207,1208)
TEST_SEEDS=(2201,2202,2203,2204,2205,2206,2207,2208)
N_TRAIN=70
N_TEST=24
RIDGE_PENALTY=3.0


def graph():
    edges=[(0,1,.21),(1,2,.18),(2,3,.15),(3,4,.13),(4,5,.12),(1,6,.18)]
    leak=np.array([.035,.028,.025,.023,.023,.021,.025],dtype=float)
    G=np.diag(leak)
    for i,j,g in edges:
        G[i,i]+=g;G[j,j]+=g;G[i,j]-=g;G[j,i]-=g
    C=np.array([.10,.08,.07,.06,.06,.06,.06],dtype=float)
    return G,C


def pulse(t, at, decay=5.0, rise=.35):
    d=np.maximum(t-at,0.0)
    shape=np.where(t>at,np.exp(-d/decay)-np.exp(-d/rise),0.0)
    z=float(np.max(shape))
    return shape/z if z>0 else shape


def stimulus(rng,group):
    t=np.arange(STEPS,dtype=float)*DT
    ga=np.zeros((5,STEPS),dtype=float)
    if group=='train':
        n=int(rng.choice([0,1,2,3],p=[.12,.42,.34,.12]))
        ports=rng.choice(5,size=n,replace=True)
        times=np.sort(rng.uniform(2.0,14.5,size=n))
        test_at=19.0+rng.uniform(-.6,.6)
    elif group=='holdout':
        n=3
        ports=rng.choice([0,1,2,3],size=n,replace=False)
        times=np.array([4.5,9.0,14.3])+rng.uniform(-.4,.4,size=3)
        test_at=21.0+rng.uniform(-.8,.8)
    else:raise ValueError(group)
    for site,at in zip(ports,times):
        strength=rng.uniform(.20,.60)
        ga[site]+=strength*pulse(t,float(at),decay=rng.uniform(5,12))
    probe_port=int(rng.integers(4))
    ga[probe_port]+=rng.uniform(.09,.24)*pulse(t,test_at,decay=4.0,rise=.25)
    gn=2.8*ga
    return ga,gn,{'history_ports':[int(i) for i in ports],
                  'probe_port':probe_port,'probe_time_ms':float(test_at)}


def feature_views(ga):
    """No somatic responses or hidden voltages are used to make features."""
    u=ga.T
    state_banks=[]
    for tau in TAUS:
        alpha=np.exp(-DT/tau)
        z=np.zeros(u.shape[1]); hist=np.empty_like(u)
        for k,row in enumerate(u):
            z=alpha*z+(1-alpha)*row
            hist[k]=z
        state_banks.append(hist)
    bias=np.ones((len(u),1))
    instant=np.concatenate((bias,u),axis=1)
    linear=np.concatenate((instant,*state_banks),axis=1)
    slow=state_banks[2];medium=state_banks[1]
    products=np.array([slow[:,i]*slow[:,j] for i in range(5) for j in range(i,5)]).T
    same_site=np.array([slow[:,i]*medium[:,i] for i in range(5)]).T
    nonlinear=np.concatenate((linear,products,same_site),axis=1)
    return instant,linear,nonlinear


def sample_episodes(CausalGraphCircuit,n,seed,group):
    G,C=graph()
    neuron=CausalGraphCircuit.from_dense_passive_graph(G,C,SITE_NODES,soma_node=0,dt_ms=DT)
    rng=np.random.default_rng(seed)
    episodes=[]
    for k in range(n):
        ga,gn,meta=stimulus(rng,group)
        sol=neuron.run(ga,gn)
        if not sol.all_steps_converged or sol.max_site_consistency_mV>1e-7:
            raise RuntimeError('Forward nonlinear cable failed to converge')
        episodes.append((ga,sol.soma_depolarization_mV,meta))
    return episodes


def fit_predict(train,test):
    ys=np.concatenate([y for _,y,_ in train])
    outputs=[]
    for view in range(3):
        X=np.concatenate([feature_views(ga)[view] for ga,_,_ in train])
        mean=X.mean(axis=0);std=X.std(axis=0)
        std[std<1e-10]=1
        mean[0]=0.;std[0]=1.
        normalized=(X-mean)/std
        penalty=np.eye(X.shape[1])*RIDGE_PENALTY
        penalty[0,0]=0.
        coef=np.linalg.solve(normalized.T@normalized+penalty,normalized.T@ys)
        pred=[((feature_views(ga)[view]-mean)/std)@coef for ga,_,_ in test]
        outputs.append(pred)
    return outputs


def score(pred,test):
    truth=np.concatenate([y for _,y,_ in test])
    return [float(np.sqrt(np.mean((np.concatenate(p)-truth)**2))) for p in pred]


def matched_soma(CausalGraphCircuit):
    """Two upstream stimulation histories with matched one-time soma reading.
    Force the same later probe; subtract each history's unprobed baseline.
    No class labels or history are inferred from a single soma snapshot.
    """
    G,C=graph();neuron=CausalGraphCircuit.from_dense_passive_graph(G,C,SITE_NODES,0,dt_ms=DT)
    t=np.arange(STEPS,dtype=float)*DT
    conditioning=pulse(t,3.0,decay=12.0)
    test_pulse=pulse(t,19.,decay=4.,rise=.25)
    onset_idx=190
    def run(port,amplitude,probed):
        ga=np.zeros((5,STEPS),dtype=float)
        ga[port]+=amplitude*conditioning
        if probed:ga[2]+=.16*test_pulse
        result=neuron.run(ga,ga*2.8)
        if not result.all_steps_converged:raise RuntimeError('Nonconverged matched probe')
        return result.soma_depolarization_mV
    ampA=.08
    baselineA=run(0,ampA,False)
    target=float(baselineA[onset_idx])
    lo,hi=.001,1.5
    assert run(3,lo,False)[onset_idx]<target<run(3,hi,False)[onset_idx]
    for _ in range(42):
        mid=(hi+lo)/2
        if run(3,mid,False)[onset_idx]<target:lo=mid
        else:hi=mid
    ampB=(hi+lo)/2
    baselineB=run(3,ampB,False)
    responseA=run(0,ampA,True)
    responseB=run(3,ampB,True)
    deltaA=responseA-baselineA
    deltaB=responseB-baselineB
    return {
      'description':'one soma snapshot matched; different branch history; same subsequent probe',
      'preprobe_at_ms':19.0,'preprobe_soma_A_mV':float(baselineA[onset_idx]),
      'preprobe_soma_B_mV':float(baselineB[onset_idx]),
      'preprobe_abs_error_mV':float(abs(baselineA[onset_idx]-baselineB[onset_idx])),
      'conditioner_A':{'site_index':0,'g_ampa_peak_uS':ampA},
      'conditioner_B':{'site_index':3,'g_ampa_peak_uS':ampB},
      'probe':{'site_index':2,'g_ampa_peak_uS':.16,'onset_ms':19.0},
      'max_probe_increment_difference_mV':float(np.max(abs(deltaA-deltaB))),
      'probe_A_peak_mV':float(deltaA.max()),'probe_B_peak_mV':float(deltaB.max()),
      'curve':{'t_ms':[float(v) for v in t[::4]],
        'delta_A_mV':[float(v) for v in deltaA[::4]],
        'delta_B_mV':[float(v) for v in deltaB[::4]]}
    }


def run_gate(CausalGraphCircuit):
    trials=[];example=[]
    for k,(train_seed,test_seed) in enumerate(zip(TRAIN_SEEDS,TEST_SEEDS)):
        train=sample_episodes(CausalGraphCircuit,N_TRAIN,train_seed,'train')
        test=sample_episodes(CausalGraphCircuit,N_TEST,test_seed,'holdout')
        pred=fit_predict(train,test)
        errors=score(pred,test)
        trials.append({'run_index':k,'train_seed':train_seed,'test_seed':test_seed,
                       'rmse_mV':dict(zip(['instant','linear_history','quadratic_history'],errors))})
        if k==0:
            t=np.arange(STEPS)*DT
            for sample in [0,1,2]:
                ga,y,meta=test[sample]
                ex={'sample':sample,'meta':meta,'t_ms':[float(v) for v in t[::4]],
                    'truth_mV':[float(v) for v in y[::4]],
                    'input_total_uS':[float(v) for v in ga.sum(axis=0)[::4]]}
                for label,vals in zip(['instant','linear_history','quadratic_history'],pred):
                    ex[label+'_mV']=[float(v) for v in vals[sample][::4]]
                example.append(ex)
    names=['instant','linear_history','quadratic_history']
    means={name:float(np.mean([run['rmse_mV'][name] for run in trials])) for name in names}
    win_linear=int(sum(q['rmse_mV']['linear_history']<q['rmse_mV']['instant'] for q in trials))
    win_nonlinear=int(sum(q['rmse_mV']['quadratic_history']<q['rmse_mV']['linear_history'] for q in trials))
    return {
      'schema':'monday.temporal.v1','description':'Synthetic held-out soma-trace recovery from known stimulation and unobserved internal neuronal state',
      'scientific_boundary':'Stimulus programs known to learner; intraneuronal state hidden. Training uses observed soma traces. No anatomical/QKV recovery or real EEG.',
      'forward_runtime':'Original Operaattori reduced.causal_graph_circuit.CausalGraphCircuit',
      'forward_graph':'seven-compartment illustrative graph, soma index 0, five controllable sites',
      'protocol':{'dt_ms':DT,'n_steps':STEPS,'train_episodes_per_seed':N_TRAIN,'holdout_episodes_per_seed':N_TEST,
                  'train_description':'random 0-3 conditioners, probe around t=19ms',
                  'holdout_description':'three conditioners with withheld ordered times and probe around t=21ms',
                  'train_seeds':list(TRAIN_SEEDS),'holdout_seeds':list(TEST_SEEDS),
                  'ridge_penalty':RIDGE_PENALTY,'feature_time_constants_ms':list(TAUS),
                  'model_sizes':{'instant':6,'linear_history':31,'quadratic_history':51},
                  'observer_receives':'full programmed stimulus conductances and training soma traces only, no internal cable voltages'},
      'mean_holdout_rmse_mV':means,
      'wins':{'linear_history_beats_instant':win_linear,'quadratic_beats_linear_history':win_nonlinear,'n_runs':len(trials)},
      'runs':trials,'examples':example,'matched_soma':matched_soma(CausalGraphCircuit)
    }


def verify(a,b):
    # exact metadata and numerical comparisons, no output rewrite in check mode
    if a['schema']!=b['schema'] or a['protocol']!=b['protocol']:raise SystemExit('FAIL: incompatible protocol')
    for key in ['mean_holdout_rmse_mV','wins']:
        if key=='wins':
            if a[key]!=b[key]:raise SystemExit('FAIL: win counts differ')
        else:
            for sub in a[key]:
                if abs(a[key][sub]-b[key][sub])>1e-7:raise SystemExit('FAIL: model RMSE differs: '+sub)
    worst=0.0;count=0
    for x,y in zip(a['examples'],b['examples']):
        for name in ['truth_mV','instant_mV','linear_history_mV','quadratic_history_mV','input_total_uS']:
            diff=np.max(np.abs(np.asarray(x[name])-np.asarray(y[name])))
            worst=max(worst,float(diff));count+=len(x[name])
    for name in ['delta_A_mV','delta_B_mV']:
        d=np.max(np.abs(np.asarray(a['matched_soma']['curve'][name])-np.asarray(b['matched_soma']['curve'][name])))
        worst=max(worst,float(d));count+=len(a['matched_soma']['curve'][name])
    print(f'UPSTREAM TEMPORAL CHECK: {count} held-out/paired sampled trace values; max absolute delta={worst:.3g} mV')
    if worst>1e-7:raise SystemExit('CHECK FAILED: temporal traces differ')
    print('CHECK PASSED')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--operaattori',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('data/temporal_reference.json'))
    ap.add_argument('--check',type=Path)
    args=ap.parse_args()
    original=args.operaattori.resolve()
    if not (original/'reduced'/'causal_graph_circuit.py').is_file():
        ap.error('Upstream original source file missing from --operaattori directory')
    sys.path.insert(0,str(original))
    from reduced.causal_graph_circuit import CausalGraphCircuit
    gate=run_gate(CausalGraphCircuit)
    print(json.dumps({'mean_holdout_rmse_mV':gate['mean_holdout_rmse_mV'],
                      'wins':gate['wins'],
                      'matched_soma':{k:v for k,v in gate['matched_soma'].items() if k!='curve'}},indent=2))
    if args.check:
        fixture=json.loads(args.check.read_text())
        verify(gate,fixture)
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(gate,indent=2)+'\n')
        print('wrote',args.out)

if __name__=='__main__':main()
