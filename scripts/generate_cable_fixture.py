#!/usr/bin/env python3
"""Generate a small measured response fixture from Operaattori's actual cable runtime.

Requires a local checkout of https://github.com/anttiluode/Operaattori.
The original source is imported, not copied into MondayRepo.

Use:
  python scripts/generate_cable_fixture.py --operaattori ../Operaattori
  python scripts/generate_cable_fixture.py --operaattori ../Operaattori --check data/cable_reference.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

CONTEXTS = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
PORT_NODES = (2, 3, 4, 5)
CONTEXT_NODE = 6
DT = 0.1
NSTEPS = 310


def pulse(t, onset, decay, rise):
    delay = t - onset
    val = np.where(delay > 0, np.exp(-np.maximum(delay,0) / decay) - np.exp(-np.maximum(delay,0)/rise), 0.0)
    m = float(val.max())
    return val / m if m > 0 else val


def small_cable_graph():
    # Purely illustrative passive circuit (not a real reconstructed neuron).
    edges = [(0, 1, .21), (1, 2, .18), (2, 3, .15), (3, 4, .13), (4, 5, .12), (1, 6, .18)]
    leak = np.array([.035, .028, .025, .023, .023, .021, .025])
    G = np.diag(leak)
    for i, j, g in edges:
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g
    C = np.array([.10, .08, .07, .06, .06, .06, .06])
    return G, C


def generate(CausalGraphCircuit):
    G, C = small_cable_graph()
    sites = list(PORT_NODES) + [CONTEXT_NODE]
    neuron = CausalGraphCircuit.from_dense_passive_graph(G, C, sites, soma_node=0, dt_ms=DT)
    t = np.arange(NSTEPS) * DT
    conditioner = pulse(t, 2.0, decay=15, rise=.55)
    test = pulse(t, 9.0, decay=5.0, rise=.35)
    all_responses=[]
    convergence=[]

    def run(context, port):
        ga = np.zeros((len(sites), NSTEPS), dtype=float)
        gn = np.zeros_like(ga)
        # Stable nonnegative, context-dependent background drive at a separate
        # dendritic site. Context is varied *before* the test pulse.
        amp = max(0.0, .65 + .35*context)
        ga[-1] = (0.18*amp)*conditioner
        gn[-1] = (0.50*amp)*conditioner
        if port is not None:
            ga[port] = 0.0045*test
            gn[port] = 0.0230*test
        result = neuron.run(ga, gn)
        if not result.all_steps_converged or result.max_site_consistency_mV>1e-7:
            raise ArithmeticError(f"cable Newton nonconvergence context={context}, port={port}")
        convergence.append(result.max_newton_iterations)
        return result.soma_depolarization_mV

    for c in CONTEXTS:
        baseline = run(c, None)
        outputs=[]
        for p in range(len(PORT_NODES)):
            diff = run(c, p) - baseline
            # Three observables from *one* soma trace; not three neurons.
            # Peak is non-smooth and needs an explicit prediction test.
            out = [float(np.max(diff)), float(diff[125]), float(np.mean(diff[130:230]))]
            outputs.append(out)
        all_responses.append(outputs)
    return {
        'schema': 'monday.cable.v1',
        'provenance': 'Operaattori reduced.causal_graph_circuit.CausalGraphCircuit on illustrative 7-compartment passive graph',
        'not_biological_data': True,
        'model': 'passive cable + implicit local AMPA/NMDA under context conditioning',
        'contexts': CONTEXTS,
        'ports': list(PORT_NODES),
        'outputs': ['probe peak delta soma mV', 'probe delta at t=12.5ms mV', 'probe mean delta t=13..23ms mV'],
        'responses': all_responses,
        'protocol': {'dt_ms': DT, 'ntime': NSTEPS, 'soma_node': 0, 'context_node': CONTEXT_NODE, 'site_nodes': sites},
        'max_newton_iterations': max(convergence)
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--operaattori',required=True,type=Path,help='Path to checked-out Operaattori repository root')
    parser.add_argument('--out',type=Path,default=Path('data/cable_reference.json'))
    parser.add_argument('--check',type=Path,help='Compare generated numeric fixture with the checked-in reference instead of writing')
    a=parser.parse_args()
    repo=a.operaattori.resolve()
    if not (repo/'reduced'/'causal_graph_circuit.py').is_file():
        parser.error(f'{repo} does not contain reduced/causal_graph_circuit.py')
    sys.path.insert(0, str(repo))
    from reduced.causal_graph_circuit import CausalGraphCircuit
    f=generate(CausalGraphCircuit)
    if a.check:
        stored=json.loads(a.check.read_text())
        assert stored['contexts']==f['contexts'] and stored['schema']==f['schema']
        discrepancy=float(np.max(np.abs(np.asarray(stored['responses'])-np.asarray(f['responses']))))
        print(f'max absolute response difference: {discrepancy:.3g} mV')
        if discrepancy>1e-8:raise SystemExit('CHECK FAILED')
        print('CHECK PASSED')
    else:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(f,indent=2)+'\n')
        print(f'wrote {a.out}; max Newton iterations {f["max_newton_iterations"]}; values mV')

if __name__=='__main__':main()
