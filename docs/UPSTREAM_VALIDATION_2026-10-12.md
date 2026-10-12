# Original Operaattori runtime verification — 12 October 2026

**Result: PASS — all 72 output values match exactly in the pinned validation environment.**

- [GitHub Actions execution and full logs](https://github.com/anttiluode/MondayRepo/actions/runs/38208577227) — successful, job `original-runtime`
- Original upstream repository: [anttiluode/Operaattori](https://github.com/anttiluode/Operaattori)
- Exact upstream commit: `f2398939dc2ef2ec829093953c20637c74e82cd0`
- Source file: [`reduced/causal_graph_circuit.py`](https://github.com/anttiluode/Operaattori/blob/f2398939dc2ef2ec829093953c20637c74e82cd0/reduced/causal_graph_circuit.py)
- Actual upstream Git blob SHA-1: `433621a1a5a39a6ea3c2dcf47082c8e2a9085a47` (checked using `git hash-object`)
- Python 3.11; NumPy 2.1.3 on GitHub-hosted Ubuntu runner
- MondayRepo source branch verification commit: `e90ec6d0010084ed818214a3aac7c91495e5306f`

## Method

The CI workflow [`verify-operaattori.yml`](../.github/workflows/verify-operaattori.yml) checks out **both repositories independently**. The generator [`scripts/generate_cable_fixture.py`](../scripts/generate_cable_fixture.py) imports `CausalGraphCircuit` from the upstream checkout, builds the seven-compartment illustrative passive graph and runs 6 conditioning contexts × 4 dendritic port programs. For each program it computes three scalar features of **the same soma voltage trace**, after subtracting the conditioner-only baseline.

The CI run first generated a new upstream result file (`/tmp/upstream_cable.json`) from the original module. It compared the full `responses` array against the existing checked-in [`data/cable_reference.json`](../data/cable_reference.json), then executed the generator's `--check` option. The original reference was **not supplied to the upstream generator** before response computation.

## Observed values (from GitHub Actions logs)

| Measurement | Result |
| --- | ---: |
| Response tensor dimensions | `(6, 4, 3)` |
| Compared scalar features | `72` |
| Maximum absolute difference | `0.0 mV` |
| RMSE | `0.0 mV` |
| Largest number of Newton iterations | `3` (upstream and stored fixture) |
| Generator `--check` | `CHECK PASSED` |
| Upstream source identity | Git object `433621a1a5a39a6ea3c2dcf47082c8e2a9085a47` |

The equality is equality of the **reported floating-point results in that environment**, not a universal bitwise determinism claim for other BLAS or NumPy implementations. `--check` has tolerance `1e-8 mV`, while this particular run happened to produce exactly `0.0 mV` difference at displayed Python precision.

## What this establishes — and what it does not

**Established:** MondayRepo's pre-existing synthetic response grid matches the output of Operaattori's actual, pinned `CausalGraphCircuit` implementation on the declared illustrative probe protocol. Thus the prior independent reproduction did not introduce a numerical discrepancy detectable by this test. The verification is continuous via GitHub Actions on fixture, generator or workflow changes.

**Not established:** a measurement of an actual neuron; biological correctness of the cable parameters; anatomical QKV; unique recovery of hidden synaptic operators; recovery of a full time-dependent neural state; independent validation of Operaattori's comparison against external NEURON experiments. This check validates the synthetic **forward response fixture**, not the blind inverse estimator's wider generalization claims.

## Reproduction

```sh
python -m pip install numpy==2.1.3
python scripts/generate_cable_fixture.py --operaattori /path/to/Operaattori --out /tmp/upstream_cable.json
python scripts/generate_cable_fixture.py --operaattori /path/to/Operaattori --check data/cable_reference.json
```

Use the pinned upstream commit above; the CI workflow performs this checkout automatically. The pre-existing `--check` mode exits with failure if the maximum absolute discrepancy exceeds `1e-8 mV`.
