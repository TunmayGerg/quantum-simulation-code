# Fast CD and 8 dB conditional squeeze with the 2026-11-20 sideband freedoms

Simulation only. No board, runtime, config, NCO or SDK change. Code is
[`code/new_freedoms_cd.py`](../../code/new_freedoms_cd.py). Every number below
comes from the JSON files in this folder (`summary.json` lists them all).

## Rules used

- The sideband phase may turn along a 10 ns constant-radius quintic arc at any
  time. In the recommended gates every arc turns by at most pi.
- Alternatively, the sideband may dip down and up over 30 ns under one qubit
  pulse. Its phase may jump at the zero.
- Otherwise the sideband amplitude is constant.
- Peak |g| ≤ π/(2·200 ns) = 7.854 rad/µs after the RFDC filter, including ramp
  overshoot.
- Every qubit pulse is the same 30 ns Hann π pulse. Only its phase changes.
- Identical π pulses force an even pulse count: an odd number of equatorial π
  rotations is a net π rotation, so the qubit cannot close. Between 310 ns
  (10 pulses) and 370 ns (12 pulses) only "10 pulses plus waits" exists.
- Figure of merit: coherent process fidelity |Tr(T†V)|²/18² on g/e × Fock 0..8.
  Relative phases and leakage count. Cutoffs converge for target and output.
- One final virtual qubit Rz is allowed, declared and reported (as in the
  squeeze reference).

## Result table

"Certified" means three things:

- SDK int16 words and the full RFDC 8x filter were used.
- Target and output columns converged to below 1e-9.
- An independent order-16 propagator agrees, and the words compile to two
  waveforms with one trigger (`*_certificate.json`).

"Design" means SDK/RFDC at a fixed cutoff (32 levels for β=1, 64–72 for β=2,
80 for squeeze), not cutoff-converged.

| Construction | Duration | 18-input process F | Vacuum F (g0 / e0) | Peak equiv. swap | Verified |
|---|---:|---:|---:|---:|---|
| **CD β=1** | | | | | |
| Reference (X X −X −X)×3, fixed sideband | 370 ns | 99.2278 % | 99.920 / 99.902 % | 281.3 ns | yes (folder verifier, replayed exactly) |
| **NEW: 10 π + 29 contiguous 10 ns arcs, each ≤ π** (`n10_maxfree_grid8_s6_bounded`) | **310 ns** | **99.3064 %** | 99.914 / 99.932 % | 200.6 ns | **yes** (96 levels) |
| 10 π + arcs, unbounded (some arcs turn 5.5 rad) (`n10_maxfree_grid8_s3`) | 310 ns | 99.3945 % | 99.915 / 99.923 % | 200.6 ns | yes, but filtered ripple ±0.5 %; not recommended |
| 10 π, qubit phases only (baseline, same optimizer) | 310 ns | 97.924 % | 99.84 / 99.83 % | 234.3 ns | design |
| 10 π + arcs, block of 5 pulses repeated (22 params) | 310 ns | 98.715 % | 99.82 / 99.79 % | 200.6 ns | design |
| 10 π + one symmetric mid-pulse phase bump per pulse (7 / 12 params) | 310 ns | 97.03 / 97.38 % | 99.39 % | 200.6 ns | design |
| 10 π + two 30 ns dips with phase jumps | 310 ns | 97.17 % (dips at 4,5), 95.27 % (0,9) | 99.8 / 99.3 % | 211 ns | design |
| 8 π + 30 ns of waits + arcs | 280 ns | 95.25 % | 98.8 / 99.1 % | 200.6 ns | design |
| **CD β=2** | | | | | |
| Folder rule (X X −X −X)×9, one g | 1090 ns | 99.179 % | 99.848 / 99.848 % | 420.9 ns | design (72 levels) |
| Folder rule (X X −X −X)×5 | 610 ns | 93.22 % | 98.997 / 98.993 % | 231.9 ns | design |
| **NEW: 28 π + 83 arcs ≤ π** (`b2_n28_maxfree_s1`) | **850 ns** | **99.1017 %** | 99.765 / 99.807 % | 289.4 ns | **yes** (96 levels) |
| 24 π + arcs ≤ π | 730 ns | 96.67 % | 99.24 / 99.22 % | 200.6 ns | design |
| Two copies of the 310 ns gate, no refit | 620 ns | 95.899 % | **99.672 / 99.722 %** | 200.6 ns | SDK/RFDC + converged cutoff (96), no order-16 check |
| Three copies (β=3), no refit | 930 ns | 84.13 % | 98.62 / 98.71 % | 200.6 ns | SDK/RFDC + converged cutoff (128) |
| **Squeeze 8 dB** | | | | | |
| Reference 160 π table, fixed sideband | 4810 ns | 99.0183 % | — | 351.6 ns | yes (earlier certificate) |
| 144 π, qubit phases only | 4330 ns | 98.065 % | (g0 squeeze 99.66 %) | 330.2 ns | design |
| 144 π + contiguous arcs, 300 evaluations | 4330 ns | 98.334 % | (99.72 %) | 330.6 ns | design |
| 144 π + arcs, continued (+500 evaluations) | 4330 ns | 98.725 % (still rising slowly) | (99.81 %) | 330.5 ns | design |
| 128 π, qubit phases only | 3850 ns | 97.07 % | (99.54 %) | 307.7 ns | design |
| 128 π + arcs from a random-arc start | 3850 ns | 89.3 % (did not recover) | | 294.1 ns | design |

The reference replay gives F = 0.9922779449668537 exactly. That is 18 columns
at 96 levels, 3,456 amplitudes, independent difference 1e-25
(`reference_replay.json`).

## The recommended 310 ns CD (β=1)

These files give the literal controls:

- `n10_maxfree_grid8_s6_bounded_recipe.json` (every pulse phase, arc centre and
  step);
- `_controls.npz` (800 MS/s samples);
- `_words.npz` (SDK int16);
- `_pulses.csv`.

The gate parameters:

- **Qubit:** ten back-to-back 30 ns Hann π pulses, one phase each.
- **Sideband:** constant scale at g = 7.779 rad/µs (nominal swap 201.9 ns; the
  filtered peak, including ramp overshoot, corresponds to 200.6 ns). It has the
  usual 5 ns outer ramps.
- **Phase moves:** 29 quintic constant-radius arcs, each 10 ns long, centred
  every 10 ns (at 5, 15 and 25 ns into every pulse). Each arc turns by at most π.
- **Final frame:** declared final qubit Rz = 4.74095 rad.
- **Filtered sideband magnitude** on the plateau: 0.99959–1.00036 of nominal,
  i.e. the same ±0.04 % as an isolated arc. It is propagated in the simulation.
- **Fidelity:** weakest basis input 97.79 %.

On the device the per-pulse freedom is only the qubit phase (10 numbers), plus
a programmed sideband phase path (one step angle per 10 ns arc).

This is a compiled table with 41 parameters:

- 10 qubit phases;
- ψ0 plus 29 arc steps;
- g.

It is not a few-parameter Magnus block (see "What failed").

Robustness of the search: 4 of 12 independent 310 ns arc fits exceeded
98.9 %. Two certified above 99 %, and a third (grid12_s3, design) reached
99.01 %. Fits that start at zero arcs stay at the qubit-only optimum, because
zero is a stationary point; random arc starts are needed.

## Physics

**Speed bound.** Heisenberg's equation gives d⟨a⟩/dt = −i s*(t)⟨σ−⟩, and
|⟨σ−⟩| ≤ 1/2. Therefore

|⟨a⟩(T) − ⟨a⟩(0)| ≤ ½ ∫|s(t)| dt

for any qubit control and any sideband phase. The CD must take |+,0> to
|+,β>, so β ≤ ½∫|g|dt ≤ g_max T/2.

- At the cap, β=1 needs 255 ns of full-strength sideband.
- With the 5 ns ramps and the filter overshoot that is about 263 ns.
- Phase arcs keep |s|, so they cannot beat the bound. Dips lower ∫|s|.
- Only an additional coupling such as a blue sideband (σ+a†, which adds a σ+
  term to da/dt) could exceed it.

The numerical check (`speed_bound_check.json`) covers 63 waveforms: 60 random
controls with 582 arcs and dips, plus 3 rotary trains at the cap. The maximum
of |⟨a⟩| / (½∫|g|) is 0.9918.

- The X-locked rotary train reaches 98.7–99.2 % of the bound.
- 8 pulses at the cap (250 ns) give |⟨a⟩| = 0.944 < 1, so they are impossible.
- With identical π pulses the next even count is 10 (310 ns).
- 8 π plus waits (280 ns) fails at 95.2 %, because the qubit idles unlocked
  during the waits.

**Why the 310 ns gate needs the arcs.**

- With back-to-back X/−X pulses the σx branches stay locked: CD efficiency is
  about 1.
- The error is the transverse term (g/√2) x σ⊥(θ), which rotates at the π-pulse
  Rabi rate.
- The rotary word cancels its second order.
- What remains is a photon-number-dependent displacement shortfall. The qubit
  tilts by about g·x/Ω away from the locking axis, and the force is cos(tilt).
  This shortfall is monotone in n in every gate computed here.

Re⟨a⟩ for |+,n>, n = 0 → 8:

| Gate | Re⟨a⟩ at n = 0 → 8 | Spread |
|---|---|---:|
| Qubit-only, 310 ns | 1.009 → 0.936 | 0.073 |
| Reference, 370 ns | 1.008 → 0.965 | 0.042 |
| Certified arc gates, 310 ns | 1.012 → 0.951 or 0.958 | 0.055–0.061 |

Because ⟨σx⟩ ≤ 1, the leading shortfall cannot change sign. This is the
reviewer's sign-definite c3 for X-only paths, and here it holds in all
solutions found.

How the arcs help:

- **What they are physically.** In the frame that follows the sideband phase,
  an arc inside a pulse is a 10 ns change of the qubit drive phase, plus a short
  detuning. Each 30 ns pulse becomes a shaped composite rotation that is no
  longer a fixed-axis π.
- **What they cost.** The CD is then not perfectly locked. The gate spends 16 %
  of the bound's displacement on cancellation: β / (½∫|g|) = 0.84, with g at
  the cap.
- **What they remove.** The displacement-direction error (Im⟨a⟩ = −0.011 in the
  qubit-only gate) and part of the shortfall.

Arcs centred exactly at qubit zeros are equivalent to qubit phase shifts plus a
final frame. That is why the earlier zeros-only study gained only 0.1 % at
370 ns. The new gain comes from mid-pulse arcs.

**Dips** remove CD area: about 15 ns of force per 30 ns dip. Both dip
placements were worse than no dips.

**Generalization.** CD(β)^k = CD(kβ) exactly. Playing the β=1 gate k times,
with copy j phase-offset by −j·Rz (both qubit and sideband samples), is a
zero-refit rule for any integer β. It is excellent from vacuum (β=2: 99.67 % in
620 ns) but compounds the process error (95.9 % at β=2, 84.1 % at β=3).

For the 18-input metric at larger β, refit the same freedom class: 28 π plus
arcs gives β=2 at 850 ns, certified 99.10 %. The folder rule needs 1090 ns.

The shortfall scales like β³ per gate (process infidelity ~ β⁶/T⁴), which is
why β=2 needs about 2.7× the β=1 time on this metric.

## What failed

- **No ≥99 % CD below 310 ns.**
  - 8 pulses cannot reach β=1 at the cap.
  - 8 pulses plus waits (280 ns) gives 95.2 %.
- **Few-parameter structured families stayed below 99 %.**
  - Mirror-symmetric mid-pulse bumps (7 or 12 parameters): 97.0–97.4 %.
  - A repeated 2-pulse motif (10 parameters): ≤ 91.9 %.
  - A repeated 5-pulse motif (22 parameters): 98.7 %.
  - Low-order Magnus cancellation explains the failure modes but did not by
    itself produce the 310 ns gate. The accepted gate is a compiled
    41-parameter table.
- **Dips** cost displacement area and lowered fidelity.
- **Squeezing.**
  - Arcs gave a real but small gain at 4.33 µs: 98.065 → 98.334 % after 300
    evaluations and 98.725 % after 800, still rising slowly.
  - A cold start with random arcs on 128 pulses lost the basin (89.3 %).
  - No squeeze shorter than the 4.81 µs reference reached 99 % in this budget.
  - The photon-number-dependent squeezing saturation is sign-definite in the
    earlier word catalogue. It is the analogue of the CD shortfall, and the arcs
    did not remove it.
- **Wrapped arcs.** Wrapping the unbounded 99.39 % gate's 5.5 rad arcs into ±π
  changes the transients and lost the basin (98.5 %). The bounded gate was
  found from a fresh start.

## Reproduce

From the project root, with OPENBLAS_NUM_THREADS=1:

```bash
../../acadia_env/bin/python code/new_freedoms_cd.py gradient          # 4 directions, max error 1.8e-7
../../acadia_env/bin/python code/new_freedoms_cd.py reference         # 370 ns replay
../../acadia_env/bin/python code/new_freedoms_cd.py bound             # 63 waveforms
../../acadia_env/bin/python code/new_freedoms_cd.py cd_search --set n10      # also n10more, n10b, b2, sq1, sq2
../../acadia_env/bin/python code/new_freedoms_cd.py certify runs/new_freedoms_2026_11_20/n10_maxfree_grid8_s6_bounded_recipe.json
../../acadia_env/bin/python code/new_freedoms_cd.py repeat n10_maxfree_grid8_s6_bounded 2
../../acadia_env/bin/python code/new_freedoms_cd.py summary
```

Design fits optimize the unquantized samples, because the int16 staircase
stalls L-BFGS line searches. Every reported number uses the SDK words. Squeeze
designs use the 1.25 ns DAC-rate model. It reproduces the folder's cell model
to 5e-14, and each squeeze result is re-scored once on the full RFDC model.

Compute used: about 1.1 CPU-hours. Logs are in `logs/`.
