# MondayRepo — The Observer's Matrix

**Two linked inverse-modeling labs: known-candidate identification and blind response recovery.** [Open the GitHub Pages demo](https://anttiluode.github.io/MondayRepo/).

What can a measurement tell us about the computation that produced it? In this small synthetic system you can select the context and input port, collect noisy responses from three output channels, and try to identify which of two hidden circuits is generating them. A second pair has *different internal coordinates* but **identical observable responses for every allowed intervention**.

The experiment is deliberately smaller and more falsifiable than claiming that neurons literally implement transformer Q, K, and V matrices. The core lesson: **recovering a response operator is not the same thing as uniquely identifying its implementation.**

## Experiment 2: blind inverse-modeling (added 12 October)

The **Blind inverse** section learns a context-conditioned response operator from observations with no catalog of candidate circuits. It initially knows only that the response should be approximately affine in context: `H(c) = H0 + c H1`, and that individual input ports are controllable. The learner has 24 unknown numerical coefficients (3 outputs × 4 ports × 2 context coefficients), a zero-mean isotropic Gaussian prior (precision 0.25) and a declared independent Gaussian observation-noise level. This is Bayesian linear regression, not identification of individual synapses.

The active selector chooses a context/port by maximizing expected reduction in predictive variance over `c = −1, 0, +1`. It uses **only the collected measurement coordinates and the assumed noise law**; no candidate answer matrix or sealed truth is provided to the estimator. Therefore it is an uncertainty-sampling *design*, not an information-theoretic solution to arbitrary nonlinear inversion. Read `blind.mjs`.

An optional rank-two reconstruction projects both learned coefficient matrices into the dominant shared two-dimensional output subspace, fitted from the **estimated matrices** by small-matrix eigendecomposition. It is not fed a privileged hidden basis. The sealed audit measures the response predictions at contexts `−0.5`, `+0.5`, `+1.5` and compares constant, affine, rank-2 and rank-1 estimators. Unlike a neural decoder, this toy has only four known input locations and fully observed three-dimensional outputs. The audit explicitly uses sealed ground truth for **scoring only**, not choosing queries, and is separate from the training path. All samples are generated in-browser using seeded pseudorandom draws. The source is open, so “sealed” is a software-interface distinction, not a cryptographic guarantee.

### Cable-model fixture and actual Operaattori adapter

`data/cable_reference.json` supplies six conditioning-context grids of one seven-compartment **illustrative** passive cable with local AMPA/NMDA conductances. Each of four dendritic ports receives a known conductance pulse. Three reported scalar outputs are the soma voltage difference (probe minus conditioner alone) at its peak, at 12.5 ms, and averaged over 13–23 ms. These are **three features of the same soma trace**, not recordings from three neurons. Browser interpolation over this fixed grid is a lookup; it does not re-run the differential equation on every click.

The fixture was generated with a **standalone numerical reproduction of Operaattori's documented passive backward-Euler plus implicit NMDA equations**. The supplied independent reference is not included as a distributed copy of Operaattori source. The verification adapter `scripts/generate_cable_fixture.py` is designed to import **the real** `reduced.causal_graph_circuit.CausalGraphCircuit` from an explicitly provided local checkout, regenerate all values and compare against the checked-in fixture. This cross-repository check has **not yet been executed here**; the adapter therefore must not be presented as a verified match to the original until a user runs it. The synthetic fixture is not a real neuron morphology, an electrophysiological recording or biological QKV.

```bash
# With a separate local checkout of anttiluode/Operaattori:
python -m pip install numpy
python scripts/generate_cable_fixture.py --operaattori ../Operaattori --check data/cable_reference.json
# To deliberately regenerate the fixture:
python scripts/generate_cable_fixture.py --operaattori ../Operaattori --out data/cable_reference.json
```

### Reproducible blind-gate benchmark

```bash
node --test tests/*.test.mjs
node experiments/benchmark.mjs 12
```

Twenty fixed seeds per condition, four context-zero observations, 12 additional measurements per policy, identical training and scoring budgets; values below are **mean held-out RMSE** (not training error or statistical significance):

| Substrate | Unknown-covariance selector | Balanced fixed schedule | Random schedule |
| --- | ---: | ---: | ---: |
| Factorized, affine prediction | 0.103856 | **0.096612** | 0.157593 |
| Cable fixture, affine prediction (mV) | 0.012889 | **0.012233** | 0.023327 |

The selector uses a known noise variance and unknown response coefficients; it cannot outperform an optimal balanced fixed design in this stationary two-feature experiment. The fixed schedule did modestly better in this run. This is a limitation of adaptive design, not a broken implementation. The exact complete table and seed protocol are saved in `data/benchmark.json`. Rank-two projection slightly improves the mean in both families, but not universally. Neither method recovers a unique latent QKV realization, a neural network or a real EEG signal.

## What to try

1. Open the demo and send a pulse with context `0`. Both explanations produce the same response operator, so your Bayesian belief should remain close to 50/50 (finite measurement noise cannot systematically favor either).
2. Click **Ask best discriminating question**. It searches the *known* candidate pairs over contexts `−1, 0, +1` and the four ports. The chosen pulse should strongly separate the response-different pair.
3. Switch to **Gauge twins**. The two candidates are different in their internal `K`, `V`, and `G` coordinates but exactly equivalent under all allowed probes. The posterior remains 50/50 even after many measurements. Here *not identifiable* is the correct answer.
4. Under **The heatmap can lie**, change the common audio-like nuisance level. Inspect how the passive sensor correlation matrix differs from an intervention-defined response operator. The right heatmap subtracts a *perfectly known* simulated contaminant—an **oracle control**, not an actual audio-recovery algorithm.

No login, external models, data uploads, analytics, or third-party runtime libraries. Everything executes in your browser. Controls are deterministic for a chosen seed, apart from deliberate measurement noise generated by the seeded PRNG. The app exposes the hidden generator only when you click **Reveal hidden circuit**.

## Mathematical contract

Three output channels, four independently controllable input ports, two latent dimensions. An experiment at context `c` produces the signed response operator

\[
H(c) = V G(c) K^T, \quad G(c)=G_0+cG_1,
\]

with \(V\in\mathbb R^{3\times2}\), \(G\in\mathbb R^{2\times2}\), and \(K\in\mathbb R^{4\times2}\). A unit pulse at port \(j\) measures column \(H(c)_{:,j}\) with independent \(\mathcal{N}(0,\sigma^2)\) noise per output. **This is not transformer softmax attention or a simulation of a spiking neuron.** It is an elementary factorized, context-conditioned operator, serving as an analogy for some Q/K/V *functions*.

The first alternative `B` has the same \(G_0\) as `A`, but different \(G_1\). At context zero, \(H_A(0)=H_B(0)\); other contexts can distinguish them. The active question maximizes the squared Euclidean difference of the two response predictions over the allowed contexts and ports. Because output noise is isotropic Gaussian with equal variance, this also maximizes pairwise KL divergence for one measurement. The selector is **privileged**: it knows both candidate models, their noise law, and the allowed probe family. It is not a blind biological controller.

The second alternative uses the invertible transform \(R\):

\[
V'=VR,\quad K'=KR^{-T},\quad G'(c)=R^{-1}G(c)R.
\]

Then for **all** \(c\), \(V'G'(c)K'^T=VG(c)K^T\). The matrices differ, but every possible response in this instrument is unchanged. Repeated measurements or lower Gaussian noise cannot resolve this exact gauge.

Beliefs start at \(P(A)=P(B)=1/2\) and use a known-variance two-hypothesis Gaussian likelihood:

\[
\log\frac{P(A|\mathcal D)}{P(B|\mathcal D)}
=\sum_{(c,j,y)\in\mathcal D}\frac{\|y-H_B(c)_{:,j}\|^2-\|y-H_A(c)_{:,j}\|^2}{2\sigma^2}.
\]

To prevent numerical overflows in the interface we clip displayed accumulated log-odds to ±24; this has no bearing on the exact gauge conclusion.

## Passive heatmap: reference confound

The synthetic recording contains a common known signal mixed into the three recorded outputs, independent synthetic drivers and sensor noise. The heatmap is a **Pearson correlation matrix** over time. It is symmetric, has unity diagonal (for nonconstant signals) and can be dominated by the known common source. Neither its strong off-diagonal entries nor its self-correlation diagonal reveals a wiring graph or attention weights. The oracle subtraction control deletes only the exact simulated common component; the two maps should not be interpreted as evidence about real EEG.

## Reproduce and verify

Requirements: a current browser; Node.js ≥20 for the tests.

```bash
npm test
python -m http.server 8000
# open http://localhost:8000
```

`node --test tests/*.test.mjs` covers dimensions, identical rest responses, informative interventions, exact gauge equivalence across multiple contexts, Bayesian nonidentifiability after 80 pulses, correct controlled identification under noise and the passive correlation confound.

GitHub Pages uses the existing `.github/workflows/static.yml` on pushes to `main`. No bundler is needed.

## Evidence ladder & limitations

| Claim | Status | Scope |
| --- | --- | --- |
| Different context responses with same reference response | **PASS, exact algebra** | Synthetic operator family |
| Appropriate pulses distinguish these two known hypotheses | **PASS, deterministic test** | Prespecified Gaussian likelihood, explicit candidate catalog |
| Gauge-related internals are observationally equivalent | **PASS, exact algebra** | This entire family of unit-port/context response experiments |
| Known common source changes passive correlation | **PASS, synthetic test** | Not source separation in real recordings |
| Infer real neuronal Q/K/V | **NOT TESTED** | No real neuron, EEG, connectome or biological circuit data |
| Learn a general inverse model from recordings | **NOT TESTED** | Observer receives the true two-model candidate set |
| Distinguish other classes of hidden circuits | **NOT TESTED** | Linear context-conditioned operator only |

## Why this repository exists

The project follows recurring questions in [BrainLoops](https://github.com/anttiluode/BrainLoops) (temporal recurrence versus analysis artifacts), [HeadAsResonator](https://github.com/anttiluode/HeadAsResonator) (known audio leaking into a measured signal), [LensLuotainTarget](https://github.com/anttiluode/LensLuotainTarget) (selecting a discriminating measurement), [Operaattori](https://github.com/anttiluode/Operaattori) (known physical forward models and nonlinear neuronal responses), [TransientWaveCompiler](https://github.com/anttiluode/TransientWaveCompiler) (response-equivalent realizations) and [NeuralAlgorithmDecoding](https://github.com/anttiluode/NeuralAlgorithmDecoding) (recover a compact executable description that survives intervention).

**Next scientific gate:** run the included Operaattori adapter against the exact upstream runtime, then move beyond offline cable-response grids to continuous traces and multiple stimulus histories. Withhold joint temporal programs and compare the affine estimator with a state-dependent recurrent alternative. No unique QKV or anatomical inference follows merely from low held-out error. The current blind gate tests new contexts at previously stimulated ports, not new ports. Retain negative outcomes and report observational equivalence when identification is impossible.

MIT. Browser: `index.html`, `style.css`, `app.js`, `blind-app.js`, `model.mjs`, `blind.mjs`. Tests: `tests/*.test.mjs`. Benchmarks and cable responses: `experiments/`, `scripts/`, `data/`.
