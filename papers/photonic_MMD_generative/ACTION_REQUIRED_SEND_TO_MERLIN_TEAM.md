# Action Required: Send to MerLin Team

## Summary

Reproducing a photonic generative-model paper at m = 100–256 modes, n = 10–16 photons, needed
MerLin as a *unitary compiler and sampler* rather than as a SLOS simulator. That worked well
through internal entry points of the `photonic_engines` fork, but those entry points are
undocumented. Six items:
- Medium: expose a public way to get a differentiable unitary at large m.
- Medium: document the identity-at-π convention of `CircuitBuilder` meshes.
- Medium: the fork cannot be installed from PyPI, so remote GPU venvs need a hand-shipped wheel.
- Medium: the Perceval dependency's `Clifford2017Backend` ignores `pcvl.random_seed`.
- Low: document the seeded GPU Clifford sampler as a direct entry point.
- Low: note in the docs that the native mesh trains much worse than a QR/Haar parametrization
  for this kind of model.

## Context

Paper: Gottlieb et al., *Efficient training of photonic quantum generative models*
(arXiv:2603.08793). A photonic QCBM (n single photons into an m-mode interferometer, PNR
detection) is trained classically: the MMD loss is estimated from permanents of n×n submatrices
of U†W_kU (Gurvits estimator), so training only needs the m×m unitary and its gradient, and
deployment is boson sampling. Phase 04 of the reproduction (`README.md`, "Photonic Assessment
and MerLin Extension") ported training to PyTorch on MerLin-compiled circuits, sampled trained
models with the fork's GPU Clifford sampler, and ran a small exact noise/detector study with
SLOS. Environment: MerLin benchmarking fork `merlinquantum 0.5.0.dev0` (commit `0b5ec1c`,
`merlin.photonic_engines`), Perceval 1.2.4, torch 2.12, NVIDIA L4 and Apple M5.

## Items to Send

### Medium — Circuit compilation: no public API for a differentiable unitary at large m

- Type: API confusion / missing documentation
- What happened: building a `QuantumLayer` at m = 100, n = 10 means a SLOS output space of about
  1.7e13 Fock states, which is infeasible. What the application needed was only the trainable
  m×m unitary. The working route was the internal
  `merlin.backend.converter.CircuitConverter(circuit, input_specs=["el"], dtype=torch.complex128).to_tensor(params)`,
  found by reading `backend/converter.py`.
- Evidence: `papers/photonic_MMD_generative/lib/qcbm_merlin.py` (`MerlinInterferometer`);
  `LOG.md`, "Phase 04 Results" (source files inspected and why). With this route, training is
  identical to the JAX reference to about 1e-16: same losses, stopping step and trained unitary
  on Fig. 3a and Fig. 3b, 5 seeds each.
- Why it matters: permanent-, Gaussian- or estimator-based methods (generative models, boson
  sampling, classical-training schemes) need the unitary, not the output distribution. Without a
  public entry point, users either hit the SLOS cost or depend on internals.
- Suggested action: expose and document a public "compile to unitary" call, for example
  `CircuitBuilder(...).to_unitary(params)` or a documented `CircuitConverter`, with gradient
  support and dtype control.

### Medium — CircuitBuilder meshes: identity point is at internal phase π

- Type: missing documentation
- What happened: the `add_entangling_layer()` MZI mesh is the identity when the internal phases
  `el_*_li*` equal π, not 0. This was found by probing the compiled unitary (2026-10-14).
  Near-identity initialization, which the paper uses, had to be written as li = π + U[0, ε),
  lo = U[0, ε).
- Evidence: `lib/qcbm_merlin.py` (module docstring and `initial_parameters`);
  `tests/test_merlin_adaptation.py::test_merlin_mesh_is_unitary_has_identity_point_and_gradients`.
- Why it matters: near-identity initialization is common for trainable interferometers. A user
  who sets all phases to 0 expecting the identity silently starts from a different, mixing
  unitary (not characterized further here).
- Suggested action: state the MZI convention and the identity point in the `CircuitBuilder`
  docs, or offer an `init="identity"` helper.

### Medium — Packaging: photonic_engines fork not installable on remote machines

- Type: integration blocker
- What happened: the fork exists only as a local editable checkout. Remote GPU venvs are built
  from `requirements.txt`, so the fork had to be shipped as a hand-built pure-Python wheel. The
  steps were `pip wheel <benchmarking>/merlin --no-deps`, copying the wheel into a git-ignored
  data folder, and adding `--find-links` with a Linux-only marker to `requirements.txt`.
- Evidence: `requirements.txt` (comments and markers); `LOG.md`, "Phase 4 Adaptation Plan",
  Amendment (wheel from commit `0b5ec1cd`).
- Why it matters: every reproduction that needs `photonic_engines` (GPU Clifford sampler) on a
  remote machine repeats this. It is also easy to overwrite the fork with PyPI
  `merlinquantum`.
- Suggested action: publish the fork, or the `photonic_engines` extra, as a versioned package
  or pre-release, or document a supported install-from-git line.

### Medium — Perceval dependency: Clifford2017Backend ignores `pcvl.random_seed`

- Type: bug (in Perceval, a MerLin dependency; not a MerLin defect)
- What happened: two draws with `Clifford2017Backend.samples(N)` after the same
  `pcvl.random_seed(s)` differ. This held in one local process and across machines, with
  Perceval 1.2.4.
- Evidence: `LOG.md`, "Sweep Records", Determinism / cross-machine check (2026-10-14). Training
  and unitaries were bitwise identical across machines, but QCBM sample-based MMD² differed by
  0.3–0.7%, with everything else equal.
- Why it matters: sample-based evaluation of photonic models is not reproducible via the
  documented seeding call. Users of MerLin's Perceval path inherit this.
- Suggested action: forward to the Perceval team; meanwhile document the MerLin GPU sampler
  (next item) as the seeded alternative.

### Low — photonic_engines Clifford sampler: seeded, exact, but only reachable internally

- Type: improvement / missing documentation
- What happened: `merlin.photonic_engines.clifford.fock_sample` with a `torch.Generator` gives
  bitwise-reproducible boson samples on CUDA. Two draws with the same seed were equal and a
  different seed differed. Photon number was conserved. Per-mode occupations of 20,000 shots
  matched exact values (max |z| = 1.72 over 16 modes, m = 16, n = 4). Sample-based MMD agreed
  with Perceval within 2 SE at m = 256, n = 16. It was reached by reading
  `photonic_engines/clifford/{__init__,sampler}.py` and `core/merlin_processor.py`
  (`local("clifford")`).
- Evidence: `utils/check_merlin_sampler.py` → `outdir/merlin_sampler_check.json`;
  `results/phase4_fig3a_backend_comparison.json`.
- Why it matters: it is the only seeded large-m boson sampler found here, and it is fast on GPU.
- Suggested action: document `fock_sample` (or `local("clifford")`) as a public sampling API
  with its seeding contract, CUDA requirement and input format (unitary + input modes).

### Low — Native mesh vs QR/Haar parametrization for generative training

- Type: other (usage observation)
- What happened: on the paper's sushi task (m = 100, n = 10), the `CircuitBuilder` rectangular
  mesh started near identity reached a test MMD² of 5.9e-2 ± 9.7e-3. A QR/Haar-parametrized
  unitary reached 7.0e-4 under the same estimator, optimizer, steps and seeds. This matches the
  paper's own finding that local meshes near identity propagate photons poorly. Per step,
  MerLin is about 1.3–1.5× slower than JAX for the Haar parametrization and faster for meshes
  (L4).
- Evidence: `LOG.md`, "Phase 04 Results", M1/M2 table; sweep
  `outdir/sweep_fig3b_sushi_merlin_20261014-105935/`.
- Why it matters: users building generative models with the default mesh may get poor training
  without knowing that the initialization or parametrization is the cause. This is not a defect.
- Suggested action: a short docs note or example on initialization and parametrization choices
  for interferometer-only models.
