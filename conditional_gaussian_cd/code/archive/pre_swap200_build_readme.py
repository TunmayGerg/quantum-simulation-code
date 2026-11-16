"""Write the concise result/recipe README from the current certificates."""
from pathlib import Path
import json
import math

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'runs/constant_pump_gaussian';V=DATA/'verified'


def certificate(name):
    path=DATA.parent/'higher_order_cd/verified'/(name+'.json')
    if not path.exists():path=V/(name+'.json')
    return json.loads(path.read_text())


def main():
    squeeze_name='strength_squeeze_swap300'
    for name in ['compressed_squeeze_n120','compressed_squeeze_n124']:
        path=DATA.parent/'higher_order_cd/verified'/(name+'.json')
        if path.exists():
            candidate=certificate(name)
            if candidate['accepted'] and candidate['command_us']<certificate(squeeze_name)['command_us']:
                squeeze_name=name
    names=[('CD beta=1','cd_binary_scaled_b1'),('8 dB conditional squeeze',squeeze_name)]
    result_table='| Gate | Vacuum swap time | Command duration | Full RFDC window | Gate fidelity | Sideband phase |\n'
    result_table+='|---|---:|---:|---:|---:|---|\n'
    for label,name in names:
        r=certificate(name)
        source=DATA/r['source_record']
        if not source.exists():source=DATA.parent/'higher_order_cd'/r['source_record']
        a=json.loads(source.read_text())
        swap=a.get('equivalent_swap_ns',a['job'].get('swap',0)*1000)
        if not swap:swap=1000*math.pi/(2*max(a['parameters'][8:]))
        constant=r.get('pump_phase_constant',not a['job'].get('pump_control',False))
        result_table+=f"| {label} | {swap:.2f} ns | {r['command_us']:.3f} us | {r['RFDC_window_us']:.6f} us | {100*r['fidelity']:.4f}% | "+('Constant' if constant else 'Smooth 10 ns I/Q')+' |\n'
    rule=json.loads((DATA/'binary_rule_summary.json').read_text())
    scale_table='| Target beta | Four-pulse cycles | Command time | Gate fidelity |\n|---:|---:|---:|---:|\n'
    for beta in [.2,.5,.75,1.,1.25,1.5,2.,3.,4.]:
        a=next(a for a in rule['results'] if a['job']['strength']==beta)
        r=certificate(a['job']['name'])
        scale_table+=f"| {beta:g} | {a['job']['cycles']} | {1000*r['command_us']:.0f} ns | {100*r['fidelity']:.4f}% |\n"
    text=r'''# Fast conditional gates: results and pulse recipes

The hardware limit is **200 ns minimum vacuum swap time** throughout the
waveform: $|g(t)|\leq\pi/(2\times0.200\,\mu\mathrm{s})$.
The archived 260 ns CD needs about 35 ns swap coupling and is **outside this
limit**. It is not a device-feasible recommendation.

The executed [higher-order comparison](higher_order_cd_and_squeezing.ipynb)
retains those stronger-coupling results with their feasibility limitation.
Read the [baseline CD notebook](fast_conditional_displacement.ipynb) for the
four-pulse construction and strength rule. Read the executed
[Gaussian notebook](conditional_gaussian_gates.ipynb) for the Trotter seeds,
finite-pulse corrections, repeated blocks and shorter smooth-phase gates.
The follow-up has seven executed code cells and six figures; each earlier
notebook has six code cells and five figures. The larger old
notebooks are preserved under `runs/audit/pre_story_notebooks/`.

## Short gates above the 99% goal

__RESULT_TABLE__

These are ideal, coherent **played-waveform** simulations: a perfect two-level
qubit, resonant JC interaction, no static chi or dissipation. Every qubit pi
envelope is 30 ns. The input space is all 18 g/e times Fock n=0..8 states.
Both target and output cutoffs grow automatically; accepted cases need up to
320 oscillator levels. No coupler calculation is used for the new results.

For exact target columns T and propagated columns V, including the declared
common final qubit frame,

$$F_{\rm proc}=|\operatorname{Tr}(T^\dagger V)|^2/18^2.$$

Relative phases count and leaked population is retained. This process metric
is not a guarantee of 99% fidelity for each individual input; the notebooks
also show those input errors. Command time and the full causal RFDC response
window are listed separately: the allowed baseline beta=1 gate has 370 ns of programmed
envelopes. A common DAC delay does not change fidelity and is not a gate cost.
Its nominal vacuum swap time is 283 ns, safely weaker than the 200 ns limit.

For the older 370 ns baseline, at 400 ns from buffer start the evolution has 99.2278% gate
fidelity. The remaining filter tail changes the gate on the prepared input
subspace by less than $8\times10^{-7}$ in operator norm, independently checked.
No played samples are deleted or reprogrammed.

## Original CD protocol and its tested scaling rule

Play **X, X, Xbar, Xbar**, each a 30 ns pi pulse, and repeat it K times.
The physical qubit DAC phases are **0, 0, 180, 180 degrees**. There are no gaps.
The sideband phase is fixed at **270 degrees** throughout, including its
single 5 ns turn-on and 5 ns turn-off. For negative beta use 90 degrees instead.
There are no internal pump ramps, phase changes, chirps or NCO updates.

With $b=|\beta|>0$, use the tested rule

$$K=\left\lceil\max(2.5b,\;2.25b^2)\right\rceil,\qquad
\mathcal A_K=(0.120K+0.005)\,\mu\mathrm{s},$$

$$g_0=2b/\mathcal A_K,\qquad g=g_0(1+c g_0^2),\qquad
c=0.0004071933463\;\mu\mathrm{s}^2.$$

Here g is in rad/us. The coefficient c is calibrated once from the beta=1
strength fit and held fixed at every other beta. The targets remain fixed;
no achieved output is used to redefine beta. Skip the gate for beta=0.

$$t_{\rm swap}=\pi/(2g),\qquad T_{\rm command}=0.120K+0.010\;\mu\mathrm{s}.$$

In the present calibrated linear-gain model, the pump plateau DAC scale is
$0.045(0.600\,\mu\mathrm{s})/t_{\rm swap}$. Keep the supplied qubit pi scale
0.4377916667 and its 30 ns Hann envelope unchanged.

__SCALE_TABLE__

The rule passes all 20 tested signed strengths from |beta|=0.05 to 4, with
minimum process fidelity 99.1038%. This is a verified grid, not a proof for
every real beta. Exact words, closing frames and complete pulse tables are
saved in `runs/constant_pump_gaussian/verified/`, including the
[beta=1 CSV](runs/constant_pump_gaussian/verified/cd_binary_scaled_b1_sequence.csv).

The mechanism includes evolution during the qubit pulses. Alternating X/Xbar
leaves a large transverse-force moment. All-X cancels that first moment but
leaves a quadratic error. XX Xbar Xbar suppresses both in the measured
finite-drive toggling-frame diagnostic. The complete beta=1 gate fidelities
at the same strength/time are 59.68%, 88.88% and 99.23%, respectively.
Higher-order terms still grow with sideband strength and photon population;
extra cycles at larger beta keep them below the error budget.

## Archived stronger-coupling comparison: outside the hardware limit

The new fixed-phase CD uses four symmetric JC/anti-JC steps, with eight 30 ns
pi pulses. Its chronological interaction coefficients start from

$$A/8,\ B/4,\ A/4,\ B/4,\ A/4,\ B/4,\ A/4,\ B/4,\ A/8,$$

where A is dimensionless JC at the chosen quadrature and B is its qubit-X
conjugate. Nine interaction amplitudes and eight qubit axes are corrected
against the full target; no per-sample waveform optimization is used. The
5 ns amplitude ramps overlap the low-amplitude edges of the pi envelopes.
The numerical result takes **260 ns at 99.3713%**, with a fixed pump phase and
peak vacuum swap reference about **35 ns**. The stronger ideal coupling is an
assumed linear extension of the calibrated model. It violates the user's
200 ns minimum swap time and is excluded from current recommendations.

A controlled phase-only fit keeps all amplitudes and elapsed intervals fixed;
it changes fidelity to **99.3775%**. Also allowing transverse interaction weights
gives **99.4539% at 260 ns**. Explicit fourth-order compositions were checked;
their extra sign reversals and switches did not beat this shorter construction.

A direct dynamically-corrected-gate objective penalizes the leading qubit-gain
error generator. It lowers its measured norm from 1.0251 to 0.8977 at the same
260 ns, retaining 99.3141% fidelity. It does not make that derivative zero, so
no exact DCG order is claimed. Full pulse tables, I/Q and magnitudes, fixed
strength curves, failed shorter gates and independent replay are in the
[focused notebook](higher_order_cd_and_squeezing.ipynb).

## Conditional squeezing and number rotation

The targets are

$$S_z(r)=e^{(r/2)\sigma_z(a^2-a^{\dagger2})},\quad r=8\ln(10)/20,
\qquad U_\chi=e^{-i(\pi/2)\sigma_z n}.$$

The number gate gives a pi relative rotation between qubit branches. Constant
sideband phase is sufficient: qubit-only corrections give an 8 dB gate in
4.81 us at 99.0183%, and the number gate in 7.69 us at 99.3033%.
There are also repeated constructions: a 96-pulse block played twice gives
8 dB in 5.77 us at 99.0333%; a 128-pulse block played twice gives the number
gate in 7.69 us at 99.0514%. Their complete words are the
[squeeze CSV](runs/constant_pump_gaussian/verified/squeeze_lifted_n192_b96_sequence.csv)
and [rotation CSV](runs/constant_pump_gaussian/verified/rotation_lifted_n256_b128_sequence.csv).

Allowing smooth pump-phase changes gave the shorter gates in the first table.
For neighboring complex I/Q plateaus, interpolate for 10 ns using

$$z(u)=[1-h(u)]z_0+h(u)z_1,\qquad h(u)=10u^3-15u^4+6u^5.$$

This transition occupies the last 5 ns / first 5 ns of adjacent cells. The
RFDC filters those actual samples; no instantaneous sideband jump is assumed.
The shortest complete prescriptions are the
[squeeze CSV](runs/higher_order_cd/verified/compressed_squeeze_n120_sequence.csv)
and [number-rotation CSV](runs/constant_pump_gaussian/verified/strength_rotation_swap150_sequence.csv).
Apply the one common final qubit Rz printed in each certificate/notebook after all repetitions.

The construction uses commutators of qubit-conjugated JC Hamiltonians.
Pairing opposite commutator loops cancels the cubic ideal error, which
motivates the seeds. Higher formal order can have a large coefficient and
does not automatically give a faster finite-pulse gate. Repeated-block fits
follow that construction; the fastest smooth-phase fits continue the best
previous Claude/Codex solutions. The fit propagates the finite pi pulses and
corrects the remaining terms. No exact Trotter order is claimed for a fitted
phase list without a separate scaling demonstration.

## Strength plots and search coverage

Shorter swap time means stronger sideband coupling. The notebooks distinguish
two questions: retune phases at each fixed strength, or hold a selected
sequence fixed and measure its strength sensitivity. These must not be read
as the same curve.

![CD: fixed-strength fits and calibration sensitivity](runs/constant_pump_gaussian/figures/cd_strength_comparison.png)

The search includes:

| Family | Coverage |
|---|---|
| Previous simulations | All 1,241 stored Gaussian/CD fits audited through SDK+RFDC replay; used as starting points |
| Binary CD words | All 2,728 words with 4/6/8/10/12 pi pulses, modulo exact global-sign symmetry; 40 scalar-strength refinements |
| Native commutator constructions | 20 coupling/strength continuation paths, plus repeated-block and gap-compression studies |
| Fixed-strength CD | 44 actual-waveform fits: 310/370 ns, constant or smooth pump, 150–600 ns swap grid |
| Fractional qubit angles | 20 additional pi/2, pi/3 and pi/4 variants; each envelope still 30 ns |
| Gaussian strength scan | Six fixed strengths for each short Gaussian gate, with cell continuation followed by actual 10 ns playback corrections |
| Strength sensitivity | 176 points across eight selected sequences, all 18 inputs, with automatic cutoff checks |
| Reusable CD rule | 20 signed strengths, 360 input columns, independently checked |
| Higher-order and corrected products | 64 new final candidates checked; first/second/fourth-order products, overlap, phase-only controls, direct DCG sensitivity and squeezing pair removal |

The earlier continuous-pump 8/10-pulse beta=1 alternatives did not reach 99%.
The new segmented eight-pulse gate does. Continuous phase
and angle searches are bounded local searches; a finite study cannot establish
that every continuous pulse sequence has been exhausted or prove a global
minimum time. Unsuccessful cases remain available in the saved data, rather
than cluttering the main notebooks.

## Playback and verification

Every forward optimization score packs actual SDK int16 I/Q and propagates
the full documented RFDC response. No individual DAC sample is an optimization
variable. A C++ adjoint makes phase fitting practical; its continuous-control
gradient approximation is separately checked, and final scores use independent
forward propagation. Accepting a gate requires output **and target** column
changes below $10^{-9}$, followed by an order-16 independent amplitude comparison.

Physical qubit DAC phase is minus its Hamiltonian-axis phase; pump sign is
unchanged. The prior export sign error was fixed, and exports use that convention.
The unchanged real SDK compiles two merged `use_stretch=False` waveforms with
one common trigger. Interior changes are I/Q samples, not sequencer phase
instructions. Runtime/configuration/NCO files and `qmsmt.py` were not changed.
No board experiment was run; analog-line errors and dissipation are outside
the stated ideal model.

The [short-gate manifest](runs/constant_pump_gaussian/verified/manifest.json)
contains words, frames, cutoffs and independent comparison counts. From this
directory, use the existing environment to regenerate the concise notebooks:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
../../acadia_env/bin/python binary_cd_scaling.py rule
../../acadia_env/bin/python verify_short_gates.py --family binary --workers 4
../../acadia_env/bin/python binary_cd_scaling.py completion
../../acadia_env/bin/python make_story_data.py
../../acadia_env/bin/python build_clear_notebooks.py
PATH="../../acadia_env/bin:$PATH" ../../acadia_env/bin/python execute_notebooks.py
../../acadia_env/bin/python build_readme.py
../../acadia_env/bin/python check_markdown_math.py
../../acadia_env/bin/python finalize_audit.py
```

The follow-up is reproduced by `higher_order_cd.py` (product, overlap,
phasecycle and dcg stages), `dcg_derivative_cd.py`, `phase_only_product_cd.py`,
and `compress_squeeze_blocks.py` (then `--nearby`). Its final checks are
`verify_higher_order_cd.py`, `higher_order_story.py`,
`build_higher_order_notebook.py`, execution of `higher_order_cd_and_squeezing.ipynb`,
and `audit_higher_order_cd.py`. The 64 records compare 1,152 input columns and
269,568 independent amplitudes. Accepted and rejected cases are both retained.

The earlier microscopic coupler notebook remains an appendix. It is not
needed to reproduce or interpret the ideal gate results above.
'''
    text=text.replace('__RESULT_TABLE__',result_table.rstrip()).replace('__SCALE_TABLE__',scale_table.rstrip())
    (ROOT/'README.md').write_text(text)
    print('Wrote concise README from current gate certificates.')


if __name__=='__main__':main()
