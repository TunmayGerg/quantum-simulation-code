# Strategic squeezing: current findings

All gates here target **8 dB**, $r=8\ln(10)/20$, on all 18 qubit g/e × Fock n=0…8 inputs. Qubit envelopes are 30 ns. Coupling never exceeds the 200 ns vacuum-swap bound. Scores below are native sampled-model search scores unless a certificate is explicitly named.

## Debate and falsification

1. **Fixed-pump physical-Z bridge.** Hypothesis: a qubit-Z conjugation preserves the desired $\sigma_z(a^2-a^{\dagger2})$ generator and reverses qubit-off-diagonal cubic errors. Every bridge pulse has JC evolution, so the full 24-pulse native word must be checked. The parent's 1024-word Magnus catalogue was screened here for six words × four repeat counts: **24 gates, 4482 propagated input columns**. Best fidelity was only **39.064%**. Rejected as a fast gate; bridge costs/errors matter.

2. **True JC sign cycling.** Hypothesis: $g\to-g$ preserves second-order squeezing and reverses every odd-in-g term, including the dominant cubic. Implemented as constant-radius sideband-IQ phase arcs, never a Cartesian chord. Native 10 ns arcs add a conditional number rotation. Reviewer independently predicted that term; the explicit polynomial calculation confirmed it. Sixteen fixed-target gates, 2592 propagated columns, reached at best **92.819% at 4.915 μs**. Rejected.

3. **Opposite arc chirality and balanced compensation.** Two sign cycles with opposite arc winding cancel the arc-induced $i\sigma_z n$ term. Eight midpoint phase samples occupy each 10 ns qubit-off interval; equal negative-JC dwells in opposite-chirality cycles cancel the total 5 ns outer-ramp area. On n=0…3 columns, the full finite-pulse series has first-order norm **6.56e−17**, second-order unwanted norm **2.10e−15**, and $i\sigma_z n$ coefficient **−3.51e−18 μs²**. The cubic norm remains **3.28e−4 μs³**, and quartic terms remain. Eight fixed-target gates/1296 columns still reach only **86.101% at 3.935 μs**. This is a useful cancellation test, not an accepted short gate.

4. **Why more phase cycles fail.** A complete 36-operator fit identifies the dominant quartic correction as $\sigma_z(a^{\dagger3}a-a^\dagger a^3)$, photon-dependent squeezing. Fit residual **5.68e−19**, quartic norm **4.32e−5**. Across **all 16** proper positive-squeezing eight-pulse words, its coefficient has the same sign, **1.0416e−6…1.45645e−6 μs⁴**. Positive mixtures and target-preserving Pauli/parity cycles cannot cancel this operator. A four-block mixture reduced some other quartic terms but reached only **86.346% at 3.935 μs** (2 gates/324 columns). To cancel the remaining curvature requires a genuinely different block family or a signed higher-order composition with its physical inverse and timing costs.

## Correct fixed-target convergence

The balanced sign-cycle family converges when coupling is reduced and duration increased while retaining the same 8 dB target:

| Vacuum swap | Total time | Native coherent gate fidelity |
|---|---:|---:|
| 529.10 ns | 7.855 μs | 97.1195% |
| 655.87 ns | 11.775 μs | 98.6666% |
| 761.64 ns | 15.695 μs | 99.1890% |

These 3 gates used 486 propagated input columns. This family is slower than the already verified fixed-pump **4.81 μs / 99.0183%** reference. No improved fast squeezing winner has been found.

## Verification and deliverables

- Full native waveform series independently compared with a separate 16th-order propagator: **384 complex amplitudes** across g=0.1, 0.2, 0.4; degree-four remainder grows at least as g⁵. Errors **3.98e−10, 1.27e−8, 5.68e−7**. File `series_validation.json`.
- Exported rejected 3.935 μs waveform: `rejected_balanced_raw.npz`, metadata `rejected_balanced.json`: **128 explicit 30 ns π envelopes**, **8 explicit 10 ns qubit-off phase arcs**, **3148 DAC samples**. Modelled peak RFDC coupling **4.37556 rad/μs**, below the cap. This demonstrates why a symmetry argument alone is insufficient.
- Currently replaying the slow 15.695 μs / 99.189% family through actual SDK int16 words and RFDC filters, automatically increasing both output and target cutoffs, with an independent final propagator comparison. Log `converged_reference_certificate.log`. This gate will remain a convergence reference, not the recommended fastest gate.

No board jobs, runtime calibration, SDK implementation, or shared model files were changed. Source: `code/strategic_squeezing.py`.
