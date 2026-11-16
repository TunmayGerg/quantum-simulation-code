# Conditional Gaussian gates and reusable fast CD

Open the executed [coupler notebook](coupler_phase_transients.ipynb) for the
driven-coupler calculation, actual programmed I/Q, pulse tables, convergence
checks and saved figures. The executed [Gaussian notebook](conditional_gaussian_gates.ipynb)
and [CD notebook](fast_conditional_displacement.ipynb) contain the complete
effective-model searches, individual sequences, per-input errors and Wigner plots.

The audit found and fixed a **qubit DAC phase-sign error** in the replacement
exporter. For example, the old physical words for CD library entry 1 gave
0.4041% fidelity, despite the Hamiltonian-axis calculation giving 99.6562%.
The corrected words reproduce 99.6562%. All exported sequences were regenerated.
The Hamiltonian parameters themselves did not need refitting.

## Targets, fidelity and scope

We assume a perfect two-level qubit, 30 ns Hann pi envelopes, and no dissipation.
The pi duration is never shortened. With sigma_z=+1 on g, the targets are

$$
S_z(r)=\exp\!\left[\frac r2\sigma_z(a^2-a^{\dagger2})\right],\qquad
r=\frac{8\ln10}{20}=0.921034,
$$

$$
U_\chi=\exp[-i(\pi/2)\sigma_z a^\dagger a],\qquad
\mathrm{CD}(\beta)=\exp[\beta\sigma_x(a^\dagger-a)].
$$

Thus U_chi gives a **pi relative rotation** between qubit branches. The coherent
process fidelity uses **all** g/e times n=0..8 inputs (d=18); the effective CD
study uses the stricter n=0..10 range (d=22) and additionally reports d=18.
For target columns T and propagated columns V, including any declared final
frame updates,

$$F_{\rm proc}=\frac{|\operatorname{Tr}(T^\dagger V)|^2}{d^2}.$$

Relative input phases count. Leakage remains in $V$ and is never renormalized
away. This metric is not the mean of basis-state fidelities or a worst-input
guarantee. The notebooks show basis-input minima separately. If a Haar-average
state fidelity is wanted, with $M=T^\dagger V$ it is

$$\overline F_{\rm state}=\frac{\operatorname{Tr}(M^\dagger M)+|\operatorname{Tr}M|^2}{d(d+1)}.$$

The effective model is resonant JC plus the finite qubit drive, without static
chi. Its gate scores do not certify the actual coupler during a rapid phase change.

## What is optimized and what was verified

The original library searches 545 Gaussian and 696 CD candidates with L-BFGS-B
and exact gradients of a cell-matrix model. The parameters are qubit/pump axes
per cell, with chosen pulse counts, gaps and sideband strengths. Each accepted
Gaussian/library CD includes the explicit common final qubit Rz printed in its
notebook. Its target is fixed; the Rz is a physical frame correction, not an
independent phase fitted to each input.

Every one of the 1,241 candidates was also replayed through SDK packing and RFDC
interpolation: 25,122 input columns and 3,378,624 complex amplitudes. Exploratory
scores retain their recorded cutoffs; accepted gates have a separate certificate.
For the 16 accepted library gates, 320 input columns and 90,240 amplitudes were
compared against an independent propagator (largest difference 1.17e-14).
Output **and target** column changes must both fall below 1e-9 as cutoffs grow;
some squeeze/rotation cases need 256 cavity levels even though inputs stop at 8.
The largest accepted surrogate-versus-playback fidelity difference is below
3.2e-5. These are coherent model checks, not board measurements.

The new shared CD fit optimizes only nine pattern variables: four first-half
qubit axes, four mirrored pump offsets and one common gain. A simpler five-variable
fit keeps the pump phase constant. Its objective is mean infidelity plus maximum
infidelity at beta=0.2, 0.4, 1 and 2, evaluated through actual SDK+RFDC replay.
The seeds are the standard echo and reproducible small phase perturbations.
The initial 0.5-per-motif designs failed the 99% family goal; 0.4-per-motif
designs succeeded. The original shared-motif search made 10,936 waveform solves
(240,592 columns), rather than optimizing arbitrary samples.

Phases are rounded to 0.1 degree and gain to six decimals before certification.
The constant-pump and 10 ns Cartesian families each pass 20 signed-strength
tests: 40 gates, 880 columns and 174,592 independently compared amplitudes.
The all-column 1e-9 cutoff condition is checked for each. This is a tested
strength grid, not a uniform proof between points.

## Literal reusable pulse sequences

Each row below is **one full pi rotation with a 30 ns envelope**. Phase sets its
axis, not its angle. All eight pulses touch, with no added idle gaps. Play one
5 ns pump turn-on before the whole train and one 5 ns turn-off after it. Repeat
the motif without inserting extra ramps. The qubit DAC peak scale remains
0.4377916667, with the original supplied channel/pulse convention.

The first-half Hamiltonian axes are [0,0,pi,pi] plus four small offsets; the
second half reverses them and adds pi. The pump phases are mirrored. This
rotary echo cancels much of the qubit drive while retaining the accumulated
conditional force. Finite pulse evolution and RFDC filtering are still simulated;
a phase-label palindrome is not an exact time-symmetry or a proof of Magnus order.

**Constant pump**, rounded motif. Peak pump DAC scale is 0.05754339 times the
signed multiplier s below. The phase stays at 270 degrees throughout the motif.

| Pulse | Qubit DAC phase (degrees) | Pump DAC phase (degrees) |
|---|---:|---:|
| 1 | 0.3 | 270.0 |
| 2 | 2.4 | 270.0 |
| 3 | 181.6 | 270.0 |
| 4 | 175.6 | 270.0 |
| 5 | 355.6 | 270.0 |
| 6 | 1.6 | 270.0 |
| 7 | 182.4 | 270.0 |
| 8 | 180.3 | 270.0 |

**Cartesian 10 ns pump transitions**, rounded motif. Peak plateau DAC scale is
0.05957622 times s. At each unequal neighboring phase, interpolate the complex
I/Q with $h(u)=10u^3-15u^4+6u^5$ over the last 5 ns/first 5 ns of those cells.
This uses eight 1.25 ns interface samples inside the existing pulse windows.
The RFDC then filters that programmed transition; its analog duration is not
assumed to remain exactly 10 ns.

| Pulse | Qubit DAC phase (degrees) | Pump DAC phase (degrees) |
|---|---:|---:|
| 1 | 357.6 | 289.5 |
| 2 | 357.6 | 285.0 |
| 3 | 180.1 | 253.2 |
| 4 | 178.7 | 258.0 |
| 5 | 358.7 | 258.0 |
| 6 | 0.1 | 253.2 |
| 7 | 177.6 | 285.0 |
| 8 | 177.6 | 289.5 |

For $b=|\beta|$, choose

$$
M=\begin{cases}\lceil b/0.4\rceil,&0<b\le2,\\
\max\{\lceil b/0.4\rceil,\lceil5(b/2)^2\rceil\},&b>2,
\end{cases}\qquad s=\frac{\beta}{0.4M}.
$$

Repeat the eight rows M times and multiply only the pump by s; keep the qubit
envelopes unchanged. A negative s adds 180 degrees to the pump. Skip beta=0.
The quadratic branch above 2 reduces local force as intermediate photon number
grows; a less conservative scaling was tested and failed at beta=4. The final
qubit Rz for these two rounded effective-model families is fixed at zero.

$$T_{\rm command}=0.240M+0.010\;\mu\mathrm{s}.$$

Add 0.04421875 us to obtain the full modeled RFDC tail window. These times
exclude preparation/readout and unspecified channel synchronization overhead.

| Family | beta | Motifs | Command (us) | Fidelity, n<=10 | Fidelity, n<=8 |
|---|---:|---:|---:|---:|---:|
| constant | 0.4 | 1 | 0.25 | 99.8219% | 99.8577% |
| constant | 1 | 3 | 0.73 | 99.8642% | 99.8890% |
| constant | 2 | 5 | 1.21 | 99.2715% | 99.4846% |
| constant | 4 | 20 | 4.81 | 99.5264% | 99.6467% |
| cartesian | 0.4 | 1 | 0.25 | 99.8146% | 99.8599% |
| cartesian | 1 | 3 | 0.73 | 99.8633% | 99.8836% |
| cartesian | 2 | 5 | 1.21 | 99.4061% | 99.5874% |
| cartesian | 4 | 20 | 4.81 | 99.5942% | 99.6692% |

The amplitude-matched standard 600 ns-swap echo gives beta=1 in 0.81 us at
99.8027%, and beta=2 in 1.61 us at 99.2190%. Both shared families are faster
and have higher process fidelity at those targets in the effective model.
The fixed-strength optimized library also offers beta=1 in 0.49 us at 99.6562%
and beta=2 in 0.97 us at 99.3255%; those separate sequences are not claimed to
follow the shared scaling rule. The old qubit-only sweep fixed the pump at zero,
which was an unfair test of the natural 270-degree constant-pump CD reference.

Use [smooth_motif](coupler_phase_model.py) with parameters saved in
[the family manifest](runs/transition_families/manifest.json). Exact int16 words
and output columns for every tested strength are in `runs/transition_families/`.

## What the played waveform does to a dynamical coupler

Following the ideal LINC circuit in the
[primary paper](https://arxiv.org/html/2501.18025v3), the microscopic model keeps
the coupler ladder and the full driven cosine in an assumed normal-mode basis:

$$
H(t)=H_0+2M_L^2E_J\sin\Phi(t)\,\mathcal P_2
\cos(\Theta/M_L)\mathcal P_2+H_{q,\mathrm{RF}}(t),\quad
\Theta=\lambda_c(c+c^\dagger)+\lambda_a(a+a^\dagger)+\lambda_q(b+b^\dagger).
$$

Here M_L counts arrayed loops and P_2 restricts the qubit to g/e. It is not an
instantaneous JC replacement: coupler excitation, micromotion, counter-rotating
terms and driven frequency shifts are generated by propagation. The infinite
oscillator cosine is analytically projected using Laguerre matrix elements.
Only an irrelevant common scalar is removed.

Assuming 6.8 GHz is the linear-bias frequency and 4 GHz the full tuning span,
these measurements imply 2E_J/E_L=0.56221719 and (E_C/h)(E_L/h)=5.78 GHz^2.
They do not fix E_C or the mode participations. We assume E_C/h=100 MHz, one
loop and illustrative 120/50 MHz bare couplings. Gain/resonances are calibrated
using one-period maps to the 600 ns swap and carrier references. These are
model assumptions/calibrations, not measurements of the unknown circuit details.
The participations are first-order weak-mixing estimates,
$\lambda_{q,a}\simeq\lambda_c g_{qc,ac}/(f_c-f_{q,a})$; an exact static circuit
fit has not been inferred from the two supplied frequency numbers.

Four 30-degree boundary tests converge below 1e-6 in output-column norm as
cutoffs increase to 20 cavity/10 coupler levels. They compare g/e times n=0,4,8
(96 propagated columns across the cutoff sequence). Maximum residual coupler
excitation is 0.3503% for an abrupt switch, 0.3430% for a 10 ns constant-magnitude
phase ramp, 0.2763% for a Cartesian ramp, and 0.00184% for 5 ns down/5 ns up.
Those are coherent residual populations, not heating rates. The down/up shape
helps this boundary test, but its tested CD motif failed 99% and is not selected.
Ten nanoseconds is not a universal guarantee of adiabatic Floquet following.

For the full beta=0.4 motif, the weak-swap calibration alone gives 96.30%
(constant pump) and 94.56% (Cartesian). We then calibrate stronger-drive
resonances/gain from constant-drive maps, displacement strength/direction from
the two vacuum inputs, and two common final frames from known observables on
the fixed coherent references |+x> times |+/-0.5>. The observable calibration
uses cached columns by linearity; no target matrix or process fidelity enters
that frame calculation. Its truncated coherent-input tail norm is below 2.9e-6.
No separate phase is fitted to each Fock input. The fixed CD target is retained.

| Programmed family, beta=0.4 | Before observable frames | With declared frames | Worst basis input, corrected | Max residual coupler excitation |
|---|---:|---:|---:|---:|
| constant | 97.7180% | 99.3644% | 98.9308% | 0.000160686 |
| cartesian | 97.3966% | 99.2851% | 98.8052% | 0.000284632 |

The above-99% scores **include** the following final rotations,
$R_z(\zeta)=e^{-i\zeta\sigma_z/2}$ and $R_c(\varphi)=e^{-i\varphi n}$. Apply or track them
explicitly; omitting them gives the lower scores in the second column. The
corrected process scores do not imply every individual input exceeds 99%.

| Family | Total final qubit Rz (rad) | Total final cavity R (rad) | Peak pump DAC scale |
|---|---:|---:|---:|
| constant | -0.062625901 | 0.011521202 | 0.052816973 |
| cartesian | -0.092119557 | 0.023204723 | 0.054369526 |

These microscopic programs include small explicit I/Q chirps and an independently
calibrated pump scale/phase. They are different from the rounded effective-model
words above. Exact words, detunings, eight-row base axes and frame updates are in
`runs/coupler/programmed_<family>_*` and `vacuum_calibrated_<family>.json`;
`programmed_words` in [calibrate_programmed_family.py](calibrate_programmed_family.py)
reconstructs them. Base axes plus the recorded chirp define the entire pulse,
including phase evolution within each cell, rather than a hidden NCO change.
The explicit eight-row CSVs are
[constant pump](runs/coupler/programmed_constant_sequence.csv) and
[Cartesian ramps](runs/coupler/programmed_cartesian_sequence.csv).

All 18 inputs are propagated at 20 cavity/8 coupler levels and 0.005 ns steps:
two gates, 36 columns, 11,520 amplitudes. The highest-photon g8/e8 inputs are
additionally checked at 24/10 levels and half-sized steps; see
[the numerical convergence record](runs/coupler/programmed_convergence.json).
The largest of those four output-column changes is $2.25\times10^{-7}$.
This representative microscopic check is not the effective model's all-column
1e-9 certificate. No full-coupler gate certification is claimed for every beta
in the scaling grid or for the long squeezing/rotation sequences.

Sparse C++ commutator-free Magnus propagation with bounded Chebyshev exponential
actions makes these checks practical. Fixed circuit operators are cached, only
needed input columns are evolved, and one-period maps handle calibration.
An independent adaptive DOP853 integration compares 96 amplitudes per step size:
maximum errors are 3.90e-6, 2.45e-7 and 5.52e-8 at 0.01/0.005/0.0025 ns.
Norm preservation is checked in addition to these accuracy comparisons.

The input is real SDK-packed int16 I/Q, the documented RFDC FIR output, then an
ideal linear analog reconstruction multiplied by continuous carriers. A measured
DAC-to-flux transfer function, asymmetry, parasitics and losses have not been
supplied. Therefore the calculated full-coupler scores are conditional predictions
for the stated circuit, not verified device fidelities. A closed system also
has no dissipative Floquet steady state or physical heating-rate prediction.

## FPGA playback and phase audit

The supplied programmed model has qubit I/Q multiplying the **lowering**
operator. The Hamiltonian-axis model instead multiplies raising, so export
qubit DAC phase as minus its model phase. Export pump phase with the same sign.
The physical replay applies the inverse conversion when entering the solver.

Each channel is one merged complex array loaded with `use_stretch=False`, scale
1 and no extra sequencer detuning/phase. The offline test uses the unchanged real
SDK `compute_pulse`/`load_pulse`, substituting memory/clock access only, and checks
every signed word. The real SDK also compiles the two-channel sequence: two
arbitrary DMA commands, one synchronizer trigger. Interior phase changes are
successive I/Q samples, not a request to update the sequencer phase every cell.
The original DAC7/DAC10 and supplied pulse names are preserved.

The configured interface is 800 MS/s into 8x interpolation at a 6.4 GS/s DAC.
The modeled three-halfband filters and 22.65625 ns group delay follow
[AMD PG269](https://docs.amd.com/r/en-US/pg269-rf-data-converter/Interpolation-Filter-Details-Gen-3/DFE).
Converter internal rounding and the unmeasured analog line are not claimed
bit-exact. No board job was launched, and SDK/runtime/configuration files were
not modified. Digital compilation is not a substitute for an analog loopback.

## Gaussian library and reproduction

These are corrected effective-model SDK+RFDC results, with the listed final
qubit frame update in each notebook. The static chi term is absent.

| Gate / index | Swap reference (ns) | Command (us) | SDK + RFDC fidelity |
|---|---:|---:|---:|
| 8 dB squeeze [0] | 600 | 7.61 | 99.9304% |
| 8 dB squeeze [1] | 600 | 6.73 | 99.6880% |
| Squeeze, constant pump [2] | 600 | 7.61 | 99.4438% |
| 8 dB squeeze [3] | 300 | 4.81 | 99.9549% |
| Relative-pi rotation [4] | 600 | 8.81 | 99.9682% |
| Relative-pi rotation [5] | 600 | 7.61 | 99.5973% |
| Rotation, constant pump [6] | 600 | 12.81 | 99.9286% |
| Relative-pi rotation [7] | 300 | 4.81 | 99.9824% |

Use `runs/sequence_<index>.csv` / `runs/words_<index>.npz` for Gaussian gates,
and `runs/cd_sequence_<index>.csv` / `runs/cd_words_<index>.npz` for CD library
gates. CSVs distinguish physical DAC phases from Hamiltonian phases. Each
notebook explains its optimizer variables, seed families, objective and selected
result; numerical optimization alone is not a proof of a Trotter order.

From this directory, using the existing environment:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
../../acadia_env/bin/python audit_cd.py
../../acadia_env/bin/python verify_transition_families.py
../../acadia_env/bin/python verify_coupler.py independent
../../acadia_env/bin/python verify_coupler.py boundaries
../../acadia_env/bin/python verify_programmed_family.py
../../acadia_env/bin/python check_programmed_convergence.py
PATH="../../acadia_env/bin:$PATH" ../../acadia_env/bin/python execute_notebooks.py
PATH="../../acadia_env/bin:$PATH" ../../acadia_env/bin/python execute_notebooks.py coupler_phase_transients.ipynb
../../acadia_env/bin/python check_markdown_math.py
../../acadia_env/bin/python finalize_audit.py
```

Saved manifests in `runs/audit/`, `runs/transition_families/` and `runs/coupler/`
record comparisons, cutoffs, words and source hashes. Full microscopic checks
take several minutes; the notebook reuses those records and also executes a
fresh short full-Hamiltonian replay and a one-period calibration check.
Math uses Markdown/LaTeX; no HTML report is required.
The [final coupler manifest](runs/coupler/final_coupler_manifest.json) and
[notebook execution record](runs/audit/notebook_execution.json) contain the
final counts: three executed notebooks, 40 code cells and 100 embedded figures.

The pre-audit replacement is preserved in `runs/audit/before_playback_fix.tar.gz`.
The earlier whole study remains in the user's `~/backups/` archive.
