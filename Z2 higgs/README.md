# Z2 higgs: resonant sideband trains for the Z2 lattice hop

Which train of resonant sideband pulses should make the conditional hop `exp(-i theta Z (a^dag b + b^dag a))` of the
4-cavity, 3-transmon Z2 lattice? Ideal, lossless simulation (two-level transmons, pulse-area gates, no chi, no loss).

**Start with `resonant_train_comparison.ipynb`.** Its first cell is the answer; the rest builds up to it.

## Files

| File | What it is |
| --- | --- |
| `resonant_train_comparison.ipynb` | The study: the six trains, how the optimised ones were found, one link (one hop, CSWAP), the full lattice, calibration errors, plain vs symmetric Trotter step, summary, recommended step with flat times. |
| `resonant_trains.py` | The trains (textbook, paired, half-ends, symmetric 8/12/16), pulse timing, the one-link simulation (one hop, CSWAP with echo), calibration-error helper. Only numpy/scipy. |
| `lattice_gate_fidelity.py` | Lattice gate fidelity on every input with exactly N photons (1320 for N = 8), with a photon cutoff raised until converged, plus Gauss and photon-number loss. |
| `optimise_trains.py` | How the optimised trains were found: `free`, `bounded`, `symmetric` searches, written to `trains/`. |
| `trains/symmetric_trains.json` | The symmetric trains used by the notebook (every angle, 8/12/16 pulses). |
| `trains/free_search.json`, `trains/bounded_search.json` | The two earlier searches, shown in the notebook's Section 2. |
| `checks/check_link.py` | Independent check of the one-link numbers (full-space matrix exponentials, 16 cavity levels). |
| `checks/check_lattice.py` | Independent check of the lattice gate fidelity (sparse Hamiltonians, `expm_multiply`): all metrics and the 8 Gauss-sector fidelities, different trains on each link (the calibration-error path), the cutoff loop and the cache. |
| `results/` | Every table of the notebook as csv; `lattice_gate_cache/` holds finished lattice runs. |

## How to run (from this folder, `$ACADIA_ENV` = `boson/acadia_env`)

    $ACADIA_ENV/bin/jupyter nbconvert --to notebook --execute --inplace resonant_train_comparison.ipynb   # ~1 min with the cache
    $ACADIA_ENV/bin/python checks/check_link.py
    $ACADIA_ENV/bin/python checks/check_lattice.py
    $ACADIA_ENV/bin/python optimise_trains.py symmetric        # rewrites trains/symmetric_trains.json (~20 s on 32 cores)

The lattice cache is keyed by the pulse list, the physical parameters and the SHA-256 of `lattice_gate_fidelity.py`.
Changing any pulse or that file recomputes the affected lattice runs (all 255 take a few hours on 24 cores). The folder
also holds 112 entries from one run of the notebook in another Python environment, whose calibrated scales differ in the
last bits; they are harmless (a cache hit requires the exact pulse list).
The notebook's text quotes numbers from the run; if parameters change, update the text from `results/`.

## Conventions

* `H = J sum Z (a^dag b + b^dag a) + h sum X`, Z = +1 for g; `J T = pi/2`; h/J = 0.5, 1.5, 3.0.
* Sideband swap 250 ns: g = (pi/2) / 250 ns. Fixed sideband scale: area = g (flat + 5 ns), so every pulse has
  area >= g x 5 ns = 0.031 rad. Pulse length = area / g + 5 ns. R_x / echo slot 30 ns.
* Symmetric step `R_x(h dt) hop_Q1Q3(theta/2) hop_Q2(theta) hop_Q1Q3(theta/2) R_x(h dt)`; s steps use s + 1 R_x slots.
* Echo: from step 2 a pi on every transmon in the R_x slot; leg A at +pi after an odd number of pi's; net pi removed.
* One link: gate fidelity `|Tr(U_target^dag U)|^2 / 90^2` on 90 inputs (g/e, n_a + n_b <= 8).
* Lattice: gate fidelity `|Tr(V^dag P U P)|^2 / d^2` on all d = 8 C(N+3, 3) inputs with exactly N photons; states are
  evolved with up to (cutoff) photons, so photon leakage lowers F. Gauss loss = P(any G_i changed), number loss =
  P(photon number changed), both averaged over all inputs.

## History

* 2026-11-13: beam-splitter + dynamical-chi comparison (`z2_trotter_comparison.ipynb`, `z2_trotter.py`) and the
  `faster_cswap/` studies removed: resonant trains are the way forward. Backup of the whole old folder:
  `~/backups/z2_higgs_before_cleanup_2026-11-13.tar.gz` (includes Codex's audit of 2026-11-13).
* Codex audit, 2026-11-13: lattice scores changed from a 3-state average to the full 1320-input gate fidelity with
  converged photon cutoff; CSWAP cavity cutoff converged; finite symmetric runs need s + 1 R_x slots; the Magnus
  explanation corrected (textbook flips first at 5th order, paired at 7th, half-ends at 3rd; confirmed independently).
* Cleanup, 2026-11-13: code consolidated into `resonant_trains.py`, `lattice_gate_fidelity.py` (unchanged),
  `optimise_trains.py`; trains bit-identical to before (all 63), so all 255 cached lattice runs were reused.
  Checks: `checks/check_link.py` 19 gates x 90 inputs, largest difference 4.4e-16; `checks/check_lattice.py` 320
  sparse-reference + 320 exact-hop input columns (largest difference 3.1e-15), 342720 amplitudes at the N = 8 cutoff
  (3.8e-17). The symmetric search reran to identical trains (30/30). The free and bounded searches were rerun with
  `optimise_trains.py` (local searches: slightly different trains from the old script, same conclusion); the files in
  `trains/` are those reruns.

* Independent verification, 2026-11-14 (two separate reviewers, no project simulation code used for the physics):
  - Re-implementation from the model description: one hop 36 trains (gate error to 5e-10 relative), CSWAP 8 cases
    (9e-13), exact-hop Trotter reference 6 cases (7e-10), and 15 full 1320-input lattice runs / 58 quantities at
    photon cutoffs 11-17: symmetric 12 and 16 at 12 steps, symmetric 12 at 4 steps, order 1, h/J 0.5, N = 2, and the
    calibration errors areas +1 %, r_eff +0.25 ns, leg B +2 deg (gate error, Gauss and number loss, time to <= 4e-10).
    Paired and half-ends (calibrated scale) agree to 1-3e-6 relative only, because `calibrate` stops at an absolute
    scale tolerance of 1e-8; results built from calibrated trains are reproducible to ~1e-5 relative, not better.
  - Code and claims review: no bug changing a headline number. Fixed after it: free-search trains now merge neighbouring
    same-leg pulses (the "14-pulse" free train is 12 pulses; bounded search rerun from it: 12 pulses = symmetric 12 to
    2e-5 rad); notebook claims corrected (99.9 % CSWAP point, "symmetry found by the optimiser" -> "best trains found
    are symmetric, from the symmetric start", Omega_1 residual of the symmetric trains, Gauss/number ratio only at
    <= 12 steps, calibration ranges, cross-references); Magnus orders and Omega_1 now computed in the notebook;
    `checks/check_lattice.py` extended. The net echo pi at the end is not played (readout), so it costs no time.
