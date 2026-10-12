# Temporal history inference: validation ledger (2026-10-12)

**Status: PASS on the declared synthetic upstream-runtime gate; not an empirical biological result.**

## Question

Can a learner predict a **whole future soma trace** from the externally known conductance history when the actual dendritic states remain invisible? Can two different hidden dendritic histories give an identical one-time soma reading but respond differently to the same new stimulus?

## Upstream provenance

- [Original Operaattori CausalGraphCircuit](https://github.com/anttiluode/Operaattori/blob/f2398939dc2ef2ec829093953c20637c74e82cd0/reduced/causal_graph_circuit.py), commit `f2398939dc2ef2ec829093953c20637c74e82cd0`, Git source blob `433621a1a5a39a6ea3c2dcf47082c8e2a9085a47`.
- [Pinned upstream Actions run — first successful generation](https://github.com/anttiluode/MondayRepo/actions/runs/38210094937).
- [Runner](../experiments/temporal_gate.py), [data and predictions](../data/temporal_reference.json), [CI workflow](../.github/workflows/temporal.yml).
- The forward solver is imported from the real upstream checkout, *not* copied or approximated for production data. The browser plots saved results.

## Experimental setup

Seven-compartment **illustrative** passive cable graph, with a soma at compartment 0, four controllable dendritic ports and one additional conditioning site. The original nonlinear local AMPA/NMDA solver integrates 320 steps at 0.1 ms, with zero initial electrical state.

Eight fixed independent seed pairs: each uses **70 training** stimulus trajectories with 0–3 early events and a late probe around 19 ms; **24 test** trajectories with three ordered earlier events and a later probe around 21 ms. Total: **560 training, 192 held-out** sequences. Crucially, the learner is **told the externally applied conductance history** and trained against soma traces; it is **not** given internal site voltages or cable state. No unknown external stimulus is recovered here.

The learned maps are ridge readouts (`λ=3`, intercept unpenalized) of predeclared causal features:

| Model | Features | Interpretation |
| --- | ---: | --- |
| Current only | 6 | Instantaneous input + intercept |
| Linear history | 31 | Input plus fixed exponential traces (0.3, 1, 3, 8, 20 ms for each of 5 sites) |
| Quadratic history | 51 | Above plus 20 input-history products |

Means and standard deviations used for feature normalization are computed **only on training responses' associated input features**, not on held-out episodes. No response is fed in as an autoregressive predictor input at test time. This is a fixed temporal feature basis with trained readout, not a fully trained recurrent neural model.

## Frozen exploratory results

Whole soma-trace RMSE (mV), **mean across 8 independent runs**:

| Model | RMSE |
| --- | ---: |
| Current only | 9.725039 |
| Linear history | 5.227602 |
| Quadratic history | **3.289170** |

Both ordered comparisons win **8/8** splits. Note the quadratic model also has more features; this protocol does not match parameter counts or optimize the baselines. The split design and feature choice were exploratory, *not prospectively preregistered*, and the test is only on simulated dynamics from this one graph and a narrow family of pulse histories. No confidence interval or real-neural generalization claim is warranted.

Three full held-out example traces and all eight seed scores are persisted. Their labels and truth remain separate from the training path in `experiments/temporal_gate.py`.

## Single-sample soma aliasing (deliberately constructed)

Two early stimulus histories target different dendritic sites. A bisection adjusts the second amplitude to match the first history's **one soma voltage** at the onset of the common probe, 19 ms.

- Conditioner A: site index 0 in the five-input array; AMPA peak 0.08000 µS.
- Conditioner B: site index 3; AMPA peak ~0.428360755 µS.
- Single matching soma voltage: ~9.453975311 mV depolarization.
- Difference between the *single* soma readings: ~1.7053×10⁻¹³ mV.
- Probe: site index 2, AMPA peak 0.16 µS, identical in both histories.
- Each incremental response is calculated by subtracting the no-probe baseline from the corresponding probed trace.
- Maximum difference between the two probe increments: **4.281340387 mV**.

This establishes a counterexample to a **memoryless scalar soma readout**, not a theorem about full output histories. A stronger observer using an entire preceding somatic trace could potentially distinguish the histories before probing. Conditioner strengths were *chosen using the simulator*; the equality was constructed, not discovered in a naturally occurring sample.

## Reproduction and limitations

```sh
python -m pip install numpy==2.1.3
python experiments/temporal_gate.py --operaattori /path/to/Operaattori --check data/temporal_reference.json
node --test tests/*.test.mjs
```

The CI workflow pins the real Operaattori source SHA and checks the complete saved arrays to 1e-7 mV, then runs Node tests. The web site does not run the cable solver or retrain these estimators.

**Not established:** real EEG inverse recovery, inferential identification of dendritic anatomy, biological QKV, unknown-stimulus reconstruction, a general best model class, or an advantage for active probes. The next sharper gate would deliberately conceal some injected stimulus events and attempt to infer them from noisy soma measurements plus chosen future probes, using untouched held-out stimulus families.
