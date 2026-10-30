# Action Required: Send to MerLin Team

## Summary

Six items came out of a MerLin adaptation of Abbas et al., "The power of quantum neural
networks" (arXiv:2011.00027):
- one documentation error that silently produces the wrong input state (`EmitterState`
  GHZ recipe);
- one packaging gap: `EmitterState` is absent from the PyPI release;
- two API-clarity items: `ComputationSpace.UNBUNCHED` is not threshold detection, and the
  parameter-to-column order of `input_parameters` is only reachable through internals;
- one `FeedForwardBlock` usability item;
- one hardware-modelling observation about emitter-generated cluster inputs, which raise
  Fisher capacity but limit output confidence on larger chips.

## Context

- **Paper:** it compares Fisher-information capacity (the "effective dimension") and
  trainability of qubit QNNs and classical networks at d = 40 parameters, 4 inputs and 2
  outputs.
- **Adaptation:** the reproduction built a native linear-optical MerLin version with no
  dual-rail gates:
  - m modes, m/2 photons;
  - a Fock input or a linear-cluster input from one spin emitter;
  - the paper's data map as phases, re-uploaded between trainable MZI meshes;
  - parity of threshold clicks as readout.
- **Usage:** these models went through per-sample Fisher matrices, design searches and
  learning-rate-tuned training on Iris, breast cancer and MNIST 3 vs 5.
- **Environments:**
  - local: MerLin 0.5.0.dev0, the photonic_engines fork (MerLin 0.4.1 API), with
    perceval 1.2.4, torch float64, CPU;
  - remote: PyPI `merlinquantum` 0.4.2 with perceval-quandela 1.3.1, torch 2.12.1+cu130,
    NVIDIA L4.
- **Why it matters:** the items affect any MerLin user who builds entangled emitter inputs,
  models detectors, or needs per-sample gradients (Fisher, influence, Jacobian studies).

## Items to Send

### High — `EmitterState.caterpillar`: the docstring's GHZ recipe produces a product state

- **Type:** bug (documentation).
- **What happened:** the docstring of `EmitterState.caterpillar`
  (`merlin/core/emitter_state.py`, line 69 in 0.5.0.dev0) gives two recipes:
  - "`spin_gate(pi/2, 0)` everywhere: the linear cluster state" (correct);
  - "`phi = pi`: GHZ" (wrong). `spin_gate(pi/2, pi)` before every emission gives the
    **product** Fock state |1010…⟩ exactly.

  A GHZ state needs `spin_gate(pi/2, 0)` for the first photon and identity gates
  (`spin_gate(0, 0)`) after it, with the emitter projected on |+⟩ (`final = (1/√2, 1/√2)`).
  Also, with `final=None` the emitter is traced out. For a GHZ that leaves a 50/50 classical
  mixture of |101010⟩ and |010101⟩ (difference 7e-17), not an entangled photon state. The
  docstring mentions tracing out but not this consequence.
- **Evidence:**
  - the product-state result and the traced mixture were checked numerically
    (`papers/power_of_QNN/LOG.md`, "Amendment 2a", 2026-10-24: product state to 1e-16);
  - the corrected recipe is verified against an explicit GHZ `StateVector` to 8e-17
    (`papers/power_of_QNN/tests/test_photonic_merlin.py::test_ghz_input_equals_explicit_ghz_state_vector`,
    passing);
  - the input builder is `papers/power_of_QNN/lib/photonic_merlin.py` (`_input_state`).
- **Why it matters:** following the docstring silently feeds a classical (product or mixed)
  state into an experiment meant to test entangled inputs. Nothing fails, so the error only
  surfaces as a wrong scientific result. Our first design screen was about to run with it.
- **Suggested action:**
  - correct the GHZ recipe;
  - state that `final=None` yields a classical mixture for GHZ-like states, and that a pure
    entangled photonic state needs a projection via `final`;
  - add a docstring example that builds both states and checks the GHZ against an explicit
    `StateVector`.

### Medium — Packaging: `EmitterState` is not in the PyPI release

- **Type:** integration blocker.
- **What happened:** `ML.EmitterState` exists in 0.5.0.dev0 (photonic_engines fork) but not in
  PyPI `merlinquantum` 0.4.2. The 0.4.x line also requires torch < 2.13.
- **Evidence:**
  - every emitter-input run had to stay on a local CPU, although the comparison runs were
    planned for remote GPUs (`LOG.md`, Phase 04 and Phase 05 plans);
  - the reproduction's emitter tests skip when the attribute is missing
    (`tests/test_photonic_merlin.py`);
  - on the same smoke config, remote GPU (PyPI) and local CPU (fork) agree to 3e-8 for
    non-emitter models.
- **Why it matters:** emitter-generated cluster/GHZ inputs were the most effective capacity
  lever in this study. Users on the public release cannot run them, so results cannot be
  reproduced from `pip install merlinquantum` alone.
- **Suggested action:** ship `EmitterState` in the next PyPI release, or document which release
  will include it and how to install a build that has it.

### Medium — `ComputationSpace.UNBUNCHED` is postselection, not threshold detection

- **Type:** API confusion, with a requested improvement.
- **What happened:** `UNBUNCHED` keeps the collision-free outcomes and **renormalizes**. Physical
  threshold detectors instead register bunched events as clicks. In the reproduction:
  - a 6-mode / 3-photon probe kept 34.9% of the probability mass;
  - the design runs kept 37.4% (hard encoding) and 43.3% (easy encoding).

  Real threshold detection had to be built by hand: simulate `FOCK`, then map each Fock
  outcome in `layer.output_keys` to its click pattern.
- **Evidence:**
  - `tests/test_photonic_merlin.py::test_unbunched_is_renormalized_collision_free_fock`
    (passing): UNBUNCHED equals FOCK restricted to collision-free states and renormalized;
  - `LOG.md`, "MerLin API Notes", item 2, and the Phase 04 hardware table (postselection
    success rates);
  - the effective dimension changes when UNBUNCHED replaces threshold detection
    (`results/phase04_tables.md`, UNBUNCHED ablation).

  The workflow cookbook used by this team calls UNBUNCHED "threshold detectors,
  hardware-aligned". Whether MerLin's own docs say the same was **not checked**.
- **Why it matters:** treating UNBUNCHED as threshold detection hides a 57-65% postselection
  loss and changes the statistical model. Fisher, capacity and training results then differ
  from what threshold hardware would give.
- **Suggested action:**
  - state explicitly in the `ComputationSpace` docs that UNBUNCHED postselects and
    renormalizes;
  - consider a built-in threshold-detector measurement (a click-pattern grouping of FOCK
    outputs, optionally with PNR or threshold per mode), so users do not hand-roll the
    mapping.

### Medium — Per-sample trainable parameters: no public way to map columns to circuit parameters

- **Type:** improvement (API).
- **What happened:** Fisher-information studies need one parameter row per sample. The
  high-level `CircuitBuilder` / `add_entangling_layer` path registers trainable phases as
  shared layer parameters, so it cannot express that. The working pattern passes trainable
  phases as an extra `input_parameters` group. Assembling the input tensors in the right column
  order, however, required the internal attribute
  `layer.computation_process.converter.spec_mappings`. Columns follow (zero-padded) name order,
  not circuit order.
- **Evidence:**
  - `lib/photonic_merlin.py` and `lib/qloq_merlin.py` (`_inputs`: columns assembled by
    name via `spec_mappings`);
  - `LOG.md`, "MerLin API Notes" (pattern borrowed from `papers/BP_QNN`);
  - outputs match Perceval SLOS to ~1e-16
    (`tests/test_photonic_merlin.py::test_matches_perceval_slos_with_threshold_click_parity`).
- **Why it matters:** per-sample Jacobians (Fisher, effective dimension, influence functions,
  natural gradient) are common in QML research. The workaround depends on an internal attribute
  and on a naming convention that is easy to break: a wrong order silently permutes parameters.
- **Suggested action:**
  - expose a public accessor for the parameter order of each `input_parameters` group;
  - or support per-sample (batched) trainable parameters directly, e.g. a `torch.func`-friendly
    functional call;
  - document the pattern in the user guide.

### Low — `FeedForwardBlock`: per-sample gradients and provider offsets are hard to use

- **Type:** pain point and missing documentation.
- **What happened** (a 6-mode feed-forward model with mid-circuit PNR detection on mode 0 and
  two conditional meshes):
  - classical inputs reach only the first stage, and later-stage phases are shared layer
    parameters, so per-sample Jacobians need a manual per-row loop;
  - `FFCircuitProvider` offsets are relative to the measured mode (negative offsets count from
    the far end);
  - Perceval refuses to reuse one parameterized circuit in two configurations (a duplicate-name
    error), so each outcome needs its own circuit copy.
- **Evidence:**
  - `lib/feedforward_merlin.py` (`summed_fisher_shared`);
  - `tests/test_photonic_merlin.py`: the feed-forward Fisher is checked against finite
    differences;
  - `LOG.md`, Amendment 3 (the feed-forward design, about 98 min, dominated by the per-sample loop).
- **Why it matters:** feed-forward is a distinctive photonic capability. Its API makes the
  standard per-sample analyses slow and error-prone.
- **Suggested action:**
  - document the offset convention and the one-circuit-per-configuration rule with an example;
  - consider letting classical inputs reach later stages and supporting per-sample parameters
    there.

### Low — Observation: emitter cluster inputs raise Fisher capacity but cap output confidence on larger chips

- **Type:** other (hardware-modelling observation).
- **What happened:** with a linear-cluster input from one spin emitter and 3 data re-uploads,
  a 10-mode / 5-photon classifier (d = 40, click-parity readout) reached a higher normalized
  effective dimension than the qubit QNN:
  - 0.9096 ± 0.0018 vs 0.8926 ± 0.0011 at n = 1e5 (5 fresh draws).

  It could not output confident class probabilities, though:
  - for a single input, optimizing all 40 phases reaches p(correct) ≈ 0.78-0.81;
  - the same chip with a Fock input reaches 0.999, and a 6-mode chip with either input
    reaches 1.0.

  As a result, it had the worst held-out loss of all models. A 6-mode / 3-photon cluster
  design kept qubit-level capacity (0.8970) and was the best quantum learner on Iris 1/2,
  breast cancer and MNIST 3 vs 5. It still trailed simple classical baselines.
- **Evidence:**
  - `README.md`, "Photonic Assessment and MerLin Extension";
  - `results/phase04_design_screens.md` and `results/phase04_amendment5_tables.md`;
  - `results/phase05_tables.md`;
  - `LOG.md`, Amendment 5, which includes the single-input margin diagnostic. That diagnostic
    was an exploratory script, not a stored artifact: 200 Adam steps, 8 inits.
- **Uncertainty:**
  - only click-parity readout was studied, so the cause (cluster input × chip size × readout)
    is not isolated;
  - the result comes from ideal, lossless simulation.
- **Why it matters:** emitter-based inputs are a natural hardware selling point. This suggests
  capacity metrics alone can favour designs that learn poorly. Examples or benchmarks that
  showcase emitter inputs should report an output-margin or training metric next to capacity.
- **Suggested action:** none required. Possibly add a note to emitter-state examples.
