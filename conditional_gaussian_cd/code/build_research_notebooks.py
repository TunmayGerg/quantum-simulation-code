"""Two current fixed-pump notebooks; historical modulation stays in archive/."""
import shutil
import re
import nbformat as nb
from project_paths import ROOT,NOTEBOOKS,prepare_notebook

M=nb.v4.new_markdown_cell;C=nb.v4.new_code_cell
SETUP=r'''
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from project_paths import ROOT
import cd_trains as cd
import hardware_check as hw
from notebook_figures import read,certificate,drives,finish,waveform_figure,individual_overlaps,DATA
'''


def build(name,cells):
    path=NOTEBOOKS/name
    old=NOTEBOOKS/'archive'/('before_fixed_'+name)
    if path.exists() and not old.exists():shutil.copy2(path,old)
    for cell in cells:
        if cell.cell_type=='markdown':
            # Keep prose readable without changing code, filenames or equations.
            for old,new in [('all18','all 18'),('All20','All 20'),('all20','all 20'),
                            ('tested16','tested 16'),('tested20','tested 20'),
                            ('enumerated65,536','enumerated 65,536'),('Only16','Only 16'),
                            ('the160','the 160'),('This160','This 160'),('its160','its 160'),
                            ('to8.5','to 8.5'),('referenceis','reference is')]:
                cell.source=cell.source.replace(old,new)
            cell.source=re.sub(r'(?<![\w/])(\d+(?:\.\d+)?)(ns|us|dB|degrees)(?!\w)',r'\1 \2',cell.source)
            cell.source=re.sub(r'(?<=[a-z])(?=99%)',' ',cell.source)
    book=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    book=prepare_notebook(book,path);nb.validate(book);nb.write(book,path)
    print('BUILT',name,len(cells),'cells;',sum(c.cell_type=='code' for c in cells),'code cells')


cd_cells=[M(r'''# Conditional displacement with a fixed sideband

The spacing comparison targets **the same CD beta=1 gate** at a fixed total
duration per curve, on all qubit g/e × Fock n=0..8 inputs. A separate
strength comparison allows weaker coupling a longer gate. The signed-beta
scaling section explicitly varies the target. No curve intentionally changes
the operation by under-displacing when coupling becomes weaker.

The verified reference gives **370 ns and 99.2278% coherent gate fidelity**.
Its sideband phase and plateau amplitude remain fixed. Every qubit pi pulse
is 30 ns, and only the outer sideband ramps change amplitude, for 5 ns each.
Peak coupling respects the **200 ns minimum vacuum-swap time**.

New mixed-axis, fractional-angle and timing designs were tested. None has yet
passed99% in less time. The reference is the fastest verified result in this
finite study, not a global optimum. The calculations below explain what
different cancellation strategies remove and why their native fidelity matters.

The model is a perfect two-level qubit with resonant JC, no loss or static chi.
The qubit drive must be calibrated at the chosen **steady pump-on frequency**.
The ideal model does not predict an unknown device-dependent frequency shift
during turn-on/off.'''),C(SETUP+r'''
ref=certificate('cd_constant_reference')
display(pd.DataFrame([{k:ref[k] for k in ['command_us','fidelity','worst_basis_fidelity','peak_swap_ns','levels']}]))
print('Independent replay:',ref['columns'],'input columns;',ref['complex_amplitudes'],
      'complex amplitudes; max difference',ref['independent_max_difference'])
'''),M(r'''## 1. Fixed beta, weaker sideband, longer gate

The word **X, X, −X, −X** repeats K times. These are qubit pi pulses with
physical DAC axes 0,0,180,180 degrees. The pump has one fixed quadrature.

For each K, one **constant coupling** is calibrated against the same beta=1
target. Quibit phases are unchanged. A rough area rule is

$$\beta\simeq\frac g2(0.120K+0.005\,\mu\mathrm{s}),\qquad
t_{\rm swap}=\frac{\pi}{2g},\qquad T=0.120K+0.010\,\mu\mathrm{s}.$$

The small strength correction is determined by complete quantum propagation;
the target is never redefined from the achieved output. The left plot shows
the coherent gate fidelity and the weakest basis-input fidelity. The right
plot shows the duration used at the same strength. Thus these are comparisons
of the same intended operation, not weaker displacements.'''),C(r'''
strengths=read('runs/reference_strengths/summary.json')
rows=pd.DataFrame(strengths['results'])
assert len(rows)==5 and all(rows.strength==1) and rows.columns.nunique()>0
fig,axes=plt.subplots(1,2,figsize=(10.5,3.5),constrained_layout=True)
axes[0].plot(rows.peak_swap_ns,100*rows.fidelity,'o-',label='Coherent gate fidelity')
axes[0].plot(rows.peak_swap_ns,100*rows.worst_basis_fidelity,'s--',label='Weakest basis input')
axes[0].axhline(99,c='k',ls=':');axes[0].set(xlabel='Peak equivalent swap time (ns)',ylabel='Fidelity (%)')
axes[0].legend(fontsize=7)
axes[1].plot(rows.peak_swap_ns,1000*rows.command_time_us,'o-')
for _,r in rows.iterrows():axes[1].annotate(f"{int(r.repeats)} cycles",(r.peak_swap_ns,1000*r.command_time_us),xytext=(4,4),textcoords='offset points',fontsize=8)
axes[1].set(xlabel='Peak equivalent swap time (ns)',ylabel='Gate duration for beta=1 (ns)')
for ax in axes:ax.grid(alpha=.3)
finish('cd_fixed_beta_strength_and_time')
display(rows[['repeats','nominal_swap_ns','command_time_us','fidelity','worst_basis_fidelity']])
print('Strength comparison:',strengths['recipes_compared'],'gates;',strengths['target_input_columns_compared'],
      'input columns;',strengths['independent_amplitudes_compared'],'independent amplitudes')
'''),M(r'''## 2. Which terms need cancellation?

At the CD pump quadrature,

$$H_{\rm JC}=\tfrac g2(\sigma_xP-\sigma_yX),\quad
X=a+a^\dagger,\quad P=i(a^\dagger-a).$$

Following the finite qubit drive defines a rotation angle theta(t). Its
toggling-frame Hamiltonian contains the wanted force plus transverse terms:

$$H_I=\tfrac g2\sigma_xP-\tfrac g2
[\sigma_y\cos\theta(t)-\sigma_z\sin\theta(t)]X.$$

First-order transverse cancellation requires the angular-path centroid
$\int e^{i\theta(t)}dt$ to vanish. A phase label alone does not establish this.
The plot shows the centroid path and the positive functional that enters a
surviving longitudinal cubic error. For controls about X only, that principal
cubic coefficient is proportional to

$$-\tfrac14\int_0^T\left|\int_0^t e^{i\theta(s)}ds\right|^2dt.$$

Its sign cannot be reversed by another X twirl. Timing can reduce it, but
mixed-axis trajectories or a genuinely different signed composition are needed
to cancel it. This is a statement about that perturbative coefficient, not a
global fidelity bound. Native coefficients independently match the functional
to8.5e−21. Instantaneous X and −X toggle JC identically; their benefit here is
**finite-pulse error cancellation**, not a claimed higher Trotter order.'''),C(r'''
# One finite four-pulse block; an explanatory qubit trajectory, not a fidelity score.
substeps=32;dt=cd.DT/substeps
q=np.repeat(np.r_[cd.pi_samples(),cd.pi_samples(),-cd.pi_samples(),-cd.pi_samples()],substeps)
theta=2*(np.cumsum(q)*dt-q*dt/2)
centroid=np.cumsum(np.exp(1j*theta))*dt
fig,axes=plt.subplots(1,2,figsize=(10.4,3.5),constrained_layout=True)
axes[0].plot(centroid.real*1000,centroid.imag*1000);axes[0].scatter([0],[0],c='k')
axes[0].set(xlabel='Re angular centroid (ns)',ylabel='Im angular centroid (ns)',title='First-order cancellation path')
axes[0].axis('equal')
axes[1].plot(np.arange(len(q))*dt*1000,abs(centroid)**2*1e6)
axes[1].set(xlabel='Time within four-pulse block (ns)',ylabel='|centroid(t)|² (ns²)',title='Positive cubic-error functional')
for ax in axes:ax.grid(alpha=.3)
finish('cd_magnus_geometry')
display(pd.DataFrame([read('runs/independent_review/same_axis_cubic.json')]))
'''),M(r'''## 3. Strategic alternatives were judged by full gate fidelity

The ten-pulse X-twirl construction has the correct first average, but leaves
second-order conditional number and squeezing terms. A separate strategy
uses a path followed by its inverse to make the actual toggling Hamiltonian
time symmetric and remove its second Magnus term. It then adjusts pi/N
rotations, qubit axes and physical dwells to cancel the first-force centroid.

Both exported examples below have the same beta=1 target and a fixed pump.
They were independently replayed through SDK words, RFDC and converged
oscillator cutoffs. They fail99% and are not recommended. A canonical
fourth-order signed-JC composition also has a967.6ns time floor in this hardware
model, so formal order alone does not make it competitive.

Time symmetry must be checked on the complete waveform, including its ramps,
padding and sample origin. An isolated flat-window Hann word can leave a
second-order term. A new direct quantum check of the complete370ns reference,
including its ramps, instead gives $\|\Omega_2/g^2\|=5.44\times10^{-16}$.
Thus this complete gate already cancels second order. Its unwanted first-order
ramp contribution and cubic/higher errors remain. Details and counted searches are in the
[CD research report](runs/strategic_cd/README.md).'''),C(r'''
comparison=[certificate(n) for n in ['best_proper10','best_mixed_geometric','cd_constant_reference']]
labels=['Proper first-average word','Mixed-axis geometric path','Verified reference']
fig,ax=plt.subplots(figsize=(7.5,3.5),constrained_layout=True)
for label,r in zip(labels,comparison):ax.scatter(r['command_us']*1000,100*r['fidelity'],s=80,label=label)
ax.axhline(99,c='k',ls='--');ax.set(xlabel='Gate duration at fixed beta=1 (ns)',ylabel='Coherent gate fidelity (%)',ylim=(96.8,99.6))
ax.grid(alpha=.3);ax.legend(fontsize=8);finish('cd_strategies_native_decision')
display(pd.DataFrame([{k:r[k] for k in ['name','command_us','fidelity','worst_basis_fidelity','accepted']} for r in comparison]))
'''),M(r'''## 4. Literal reference sequence and constant pump

Play twelve30ns pi pulses with axes **0,0,180,180 degrees repeated three
times**. Use a constant sideband plateau with nominal swap283.21ns and fixed
phase270degrees, with one5ns turn-on and one5ns turn-off. The qubit train begins
after the initial ramp. No internal pump modulation or final qubit Rz is used.

The [complete pulse table](runs/research_verified/cd_constant_reference_pulses.csv)
and [exact SDK words](runs/research_verified/cd_constant_reference_words.npz)
give the complete recipe. The plot aligns the common digital-filter delay for
display; full filtered shapes are propagated. Magnitude is shown separately
for both channels.'''),C(r'''
display(pd.read_csv(DATA/'cd_constant_reference_pulses.csv'))
waveform_figure('cd_constant_reference');finish('cd_constant_pump_IQ')
'''),M(r'''## 5. Fresh independent replay on all18 inputs

The acceptance score is $|\operatorname{Tr}(T^\dagger V)|^2/18^2$, where T and V
are exact and simulated output columns. Relative phases and leakage remain.
The individual-state plot is diagnostic; it does not replace this coherent
metric. In particular,99% process fidelity does not imply99% for every basis
input. This cell independently evolves all18 columns at a larger cutoff.'''),C(r'''
q,s,qf,sf,words=drives('cd_constant_reference')
space=cd.Space(128,9);initial=np.eye(256,dtype=complex)[:,space.inputs]
V=hw.propagate(qf,sf,initial,cd.DT/8,order=16);T=cd.target_columns(space,'cd',1.)
saved=np.load(DATA/'cd_constant_reference_columns.npz')['output'];pad=np.zeros_like(V);pad[:len(saved)]=saved
error=float(np.max(abs(V-pad)));F=float(abs(np.vdot(T,V)/18)**2)
assert V.shape==(256,18) and error<1e-10 and F>.99
print('FRESH CD CHECK:18columns;',V.size,'amplitudes;maxdifference',error,'F',F)
overlap=np.sum(T.conj()*V,axis=0)
fig,ax=plt.subplots(figsize=(7.5,3.4),constrained_layout=True)
for offset,label in [(0,'Input g'),(9,'Input e')]:ax.plot(np.arange(9),100*abs(overlap[offset:offset+9])**2,'o-',label=label)
ax.set(xlabel='Input photon number',ylabel='Basis-state fidelity (%)',title=f'CD beta=1; coherent gate F={100*F:.4f}%')
ax.grid(alpha=.3);ax.legend();finish('cd_all_input_diagnostics')
''')]

cd_cells[4:4]=[M(r'''## 1b. A reusable recipe for smaller and larger displacements

Let $b=|\beta|$. Repeat the same four-pulse word K times; change only one
constant sideband strength for the whole gate. The checked empirical rule is

$$K=\left\lceil\max(2.5b,2.25b^2)\right\rceil,\qquad
L=0.120K+0.005\;\mu\mathrm{s},$$

$$g_0=2b/L,\qquad g=g_0(1+c g_0^2),\qquad
c=0.0004071933463\;\mu\mathrm{s}^2.$$

For positive beta use pump phase270 degrees; negative beta adds180 degrees.
The qubit axes remain0,0,180,180 degrees and each physical envelope stays30ns.
This is a calibrated scaling law, not a theorem or an optimizer rerun at every
beta. The single coefficient c comes from beta=1. All20 listed signed targets
were independently checked on all18 inputs with automatic cutoff growth.
The lowest checked process fidelity is99.1038%. Arbitrary interpolated beta
still needs the full simulation; do not extrapolate a guaranteed99% bound.

Here different beta values deliberately correspond to different target gates.
This scaling plot is separate from the fixed-beta strength comparison above.'''),C(r'''
rule=read('runs/constant_pump_gaussian/binary_rule_summary.json')
manifest=read('runs/constant_pump_gaussian/verified/manifest.json')
names={r['job']['name'] for r in rule['results']}
checked=[r for r in manifest['results'] if r['name'] in names]
assert len(checked)==20 and sum(r['columns'] for r in checked)==360
assert min(r['fidelity'] for r in checked)>.99
assert all(r['pump_phase_constant'] and r['equivalent_swap_ns']>=200 for r in checked)
assert max(r['independent_max_difference'] for r in checked)<1e-10
scaling=pd.DataFrame([dict(beta=r['job']['strength'],cycles=r['job']['cycles'],
    duration_ns=r['command_us']*1000,nominal_swap_ns=r['job']['swap']*1000,
    fidelity=r['fidelity']) for r in rule['results']]).sort_values('beta')
positive=scaling[scaling.beta>0]
fig,axes=plt.subplots(1,2,figsize=(10.5,3.5),constrained_layout=True)
axes[0].plot(positive.beta,positive.duration_ns,'o-')
axes[0].set(xlabel='Target displacement beta',ylabel='Gate duration (ns)',title='Same word; integer repeat scaling')
axes[1].plot(positive.beta,100*positive.fidelity,'o-');axes[1].axhline(99,c='k',ls='--')
axes[1].set(xlabel='Target displacement beta',ylabel='Coherent gate fidelity (%)')
for ax in axes:ax.grid(alpha=.3)
finish('cd_signed_beta_scaling');display(scaling)
print('SCALING CHECK:20 signed targets;360 input columns;',
      sum(r['complex_amplitudes'] for r in checked),'independently compared amplitudes')
''')]

sq_cells=[M(r'''# Conditional squeezing: fixed sideband, strategic cancellation tests

The target is the complete **8dB conditional squeezing unitary** on qubit g/e
× Fock n=0..8, not vacuum squeezing. The fastest currently verified fixed-pump
reference takes **4.81us at99.0183% coherent process fidelity**. The pump phase
and plateau amplitude stay constant; only its outer5ns ramps change amplitude.
All qubit pi envelopes remain30ns and the peak coupling respects the200ns
minimum swap time.

This160-pulse reference is a compiled qubit-phase solution. It is not explained
by assigning a low Magnus order to the whole long train. We use short-block
Magnus calculations to propose simpler cancellation mechanisms, then let the
full unitary simulation decide. None of the new tested strategic constructions
has yet beaten the reference at99%.

The reference optimization varied the160 physical qubit pi-pulse axes at fixed
sideband strength, amplitude, phase and pulse duration. Its objective was the
same coherent18-column target score, allowing one declared final qubit Z
frame. It used continuation from longer fitted trains and motivated
commutator/phase-cycle starts. The saved solution is a local numerical result;
its literal axes are exported below. SDK packing, filtering and independent
cutoff checks are subsequent acceptance steps, rather than fitted targets.

The ideal qubit drive is assumed calibrated at the chosen steady pump-on
frequency. Unknown power-dependent detuning during turn-on/off is not modeled.
Historical amplitude-modulated and phase-modulated faster gates are archived;
they are not fixed-pump recommendations.'''),C(SETUP+r'''
ref=certificate('squeeze_constant_reference')
display(pd.DataFrame([{k:ref[k] for k in ['command_us','fidelity','worst_basis_fidelity','peak_swap_ns','qubit_frame_Rz','levels']}]))
print('Independent full-gate check:',ref['columns'],'input columns;',ref['complex_amplitudes'],
      'amplitudes;max difference',ref['independent_max_difference'])
'''),M(r'''## 1. Why the full input subspace matters

The exact target is

$$S_z(r)=\exp[\tfrac r2\sigma_z(a^2-a^{\dagger2})],\qquad r=8\ln(10)/20.$$

The coherent process fidelity includes all18 columns and their relative
phases. The state fidelities below show where error grows with photon number.
Output cutoff convergence is required separately from preparing inputs only
up to8 photons; squeezing can populate much higher Fock states.'''),C(r'''
overlaps=individual_overlaps('squeeze_constant_reference')
assert abs(abs(overlaps.mean())**2-ref['fidelity'])<1e-13
fig,ax=plt.subplots(figsize=(7.5,3.4),constrained_layout=True)
for offset,label in [(0,'Input g'),(9,'Input e')]:ax.plot(np.arange(9),100*abs(overlaps[offset:offset+9])**2,'o-',label=label)
ax.set(xlabel='Input photon number',ylabel='Basis-state fidelity (%)',title=f"8dB; coherent gate F={100*ref['fidelity']:.4f}%")
ax.grid(alpha=.3);ax.legend();finish('squeeze_all_input_diagnostics')
'''),M(r'''## 2. Work out what a symmetry removes before simulating the gate

The short-block search enumerated65,536 eight-pulse X/Y/−X/−Y words.
680 close the qubit operation and cancel the first-order JC average. Only16
produce the desired pure second-order squeezing term. Simply cycling qubit
phases removes only part of their cubic error; the rest survives.

An optional sign-cycle branch uses constant-radius sideband phase arcs:
$I=A\cos\psi(t)$ and $Q=A\sin\psi(t)$, changing phase over10ns while the
qubit envelope is zero. This preserves the programmed magnitude. Straight
Cartesian I/Q interpolation would change it and is not used.

Those phase arcs add a conditional-number term at second order. Pairing
opposite chiralities cancels that term, as the calculation below shows. Yet
the full fast gates still fail99%; cancellation of one coefficient does not
establish gate fidelity. This optional branch is therefore not recommended
over the simpler fixed-pump reference. It also has small filtered magnitude
ripple, which cannot be called perfectly constant at the device.'''),C(r'''
before=read('runs/strategic_squeezing/chirality_magnus.json')
after=read('runs/strategic_squeezing/balanced_magnus.json')
print('Arc/chirality terms (complete saved calculations):');display(before);display(after)
# Compare the same two-repeat supercycle before/after chirality balancing.
chi_before=abs(next(r['second_chi'] for r in after if r['block_index']==0 and not r['chirality_cycle']))
chi_after=abs(next(r['second_chi'] for r in after if r['block_index']==0 and r['chirality_cycle']))
fig,ax=plt.subplots(figsize=(7.5,3.4),constrained_layout=True)
ax.bar(['Before balancing','Opposite-chirality balance'],[chi_before,chi_after])
ax.set(yscale='log',ylabel='Unwanted second-order chi coefficient (us²)',title='A real cancellation; higher-order errors remain')
ax.grid(axis='y',alpha=.3);finish('squeeze_arc_chi_cancellation')
'''),M(r'''## 3. The surviving error changes with photon number

In the tested16 proper positive eight-pulse blocks, a quartic term contains

$$E_4\propto\sigma_z(a^{\dagger3}a-a^\dagger a^3).$$

Its upward two-photon matrix element is n times that of the intended
$\sigma_z(a^{\dagger2}-a^2)$ generator. A single coupling calibration can
adjust the mean squeeze strength but cannot remove that n dependence.
All16 blocks have the same sign for this coefficient, so their positive
mixtures and tested target-preserving sign/Pauli cycles cannot cancel it.
This obstruction is limited to that family, not a proof against arbitrary
fractional-angle or timing designs.

The plot compares generator matrix-element shapes only. It is not a gate
infidelity curve. At working coupling, all higher orders must be tested by
full quantum propagation.'''),C(r'''
n=np.arange(9);desired=np.sqrt((n+1)*(n+2));unwanted=n*desired
fig,axes=plt.subplots(1,2,figsize=(10.5,3.4),constrained_layout=True)
axes[0].plot(n,desired,'o-',label='Target two-photon generator')
axes[0].plot(n,unwanted,'s-',label='Photon-dependent quartic generator')
axes[0].set(xlabel='Input photon number',ylabel='Upward matrix element (coefficient=1)');axes[0].legend(fontsize=7)
axes[1].plot(n,n,'o-');axes[1].set(xlabel='Input photon number',ylabel='Unwanted / target matrix element',title='Different n cannot share one correction')
for ax in axes:ax.grid(alpha=.3)
finish('squeeze_photon_dependent_error')
print('Operator-fit residual:',read('runs/strategic_squeezing/quartic_operators.json')['residual'])
'''),M(r'''## 4. Literal fixed-pump reference

The sideband has a constant plateau, nominal swap353.95ns and phase0degrees,
with outer5ns amplitude ramps. The [full160-pulse table](runs/research_verified/squeeze_constant_reference_pulses.csv)
lists every physical qubit axis. Each pulse is30ns and a full pi rotation.
Apply the single final qubit Rz printed below after the complete train; this
tracked frame is included in the coherent gate fidelity.

The qubit plot zooms into the first300ns for legibility. The sideband plot
shows the entire4.81us gate and its constant magnitude. All pulses are present
in the [exact SDK words](runs/research_verified/squeeze_constant_reference_words.npz).
This is the previously verified numerical reference; its160 phase parameters
are not presented as a simple higher-order product formula.'''),C(r'''
pulses=pd.read_csv(DATA/'squeeze_constant_reference_pulses.csv')
assert len(pulses)==160 and all(pulses.duration_ns==30)
display(pulses.head(12))
print('All160phases are in the linked CSV. Final qubit Rz(rad):',ref['qubit_frame_Rz'])
waveform_figure('squeeze_constant_reference',qubit_zoom_ns=300);finish('squeeze_constant_pump_IQ')
'''),M(r'''## 5. Fresh boundary-input replay and the decision from the experiments

The full reference certificate independently compares all18 columns at a
converged output cutoff. This cell additionally replays g8/e8 at384 levels
and compares complex amplitudes against that certificate. No leakage is
discarded and no target is fitted to the achieved state.

The strategic tests produced useful falsifications: correct first/second
Magnus terms do not guarantee accuracy at finite coupling; bridge pulses and
phase arcs contribute their own terms; and a phase symmetry cannot cancel an
error it preserves. A slower, fully checked phase-cycle construction reaches
99.19% only at15.695us, so it loses the speed comparison. Composite pi/N
qubit sandwiches cancel their added second-order chi when paired, but the
completed faster tests remain below99%.

The final proposed alternative was a primitive ending in qubit Z, so that
repeating it could cancel qubit-X/Y cubic errors without extra bridge pulses.
A search over all eight- and ten-pulse words made of ±X/±Y pi and pi/2
envelopes found no Z-ending primitive with zero first JC average. Exact
nearest-neighbor matching tests the 855,296 and 50,039,808 relevant endpoint
pairs without simulating a billion full gates. The first-moment calculation
was separately checked against 2,048 oscillator/qubit Cauchy coefficients at
each length. Thus this discrete family fails before any fidelity optimization;
other rotation fractions and timings remain outside that result. Full counts
are saved in [the open-Z search](runs/open_z_squeezing/length10.json).

Research decisions and comparisons are recorded in [WORK_STATUS](WORK_STATUS.md)
and the [independent review](runs/independent_review/REVIEW.md). The current
recommendation stays4.81us until a shorter gate passes the same full metric.'''),C(r'''
q,s,qf,sf,words=drives('squeeze_constant_reference')
space=cd.Space(384,9);initial=np.eye(768,dtype=complex)[:,[16,17]]
V=hw.propagate(qf,sf,initial,cd.DT/8,order=16)
V*=np.exp(-1j*ref['qubit_frame_Rz']*space.z/2)[:,None]
T=cd.target_columns(space,'squeeze',cd.R_8DB)[:,[8,17]]
saved=np.load(DATA/'squeeze_constant_reference_columns.npz')['output'][:,[8,17]]
pad=np.zeros_like(V);pad[:len(saved)]=saved
error=float(np.max(abs(V-pad)));assert V.shape==(768,2) and error<1e-10
print('FRESH SQUEEZE CHECK:2boundaryinputs;',V.size,'amplitudes;maxdifference',error)
fig,axes=plt.subplots(1,2,figsize=(10,3.3),constrained_layout=True)
for j,ax in enumerate(axes):
    exact=(abs(T[:,j])**2).reshape(-1,2).sum(axis=1)
    actual=(abs(V[:,j])**2).reshape(-1,2).sum(axis=1)
    ax.semilogy(exact,'--',label='Exact8dB target');ax.semilogy(actual,label='4.81us reference')
    ax.set(title=['Input g8','Input e8'][j],xlabel='Output photon number',ylabel='Probability',xlim=(0,200),ylim=(1e-9,1))
    ax.grid(alpha=.3);ax.legend(fontsize=8)
finish('squeeze_boundary_input_replay')
''')]

cd_cells[2:2]=[M(r'''## Pulse spacing at fixed beta and fixed total duration

Here the center-to-center spacing is $\Delta=30\,\mathrm{ns}+t_{\rm idle}$.
The physical pi envelope stays30ns. More repetitions of the same X,X,−X,−X
cycle fill a fixed-duration window as spacing decreases. Each curve has
exactly the same total duration, including the two5ns sideband ramps.

The intended gate remains beta=1 on all18 inputs. One constant coupling is
normalized from the wanted **first Magnus coefficient of the actual packed
and RFDC-filtered waveform**. Pulse phases are not refitted and fidelity is
not optimized. DAC resolution leaves less than0.3% coefficient mismatch.
The earlier reference table uses its separately calibrated coupling; this
spacing sweep deliberately uses the same leading-generator normalization
throughout each fixed-duration curve.

The middle panel separates the wanted coefficient from the achieved vacuum
mean. Errors in the achieved operation remain in the fidelity; they do not
redefine the target. The right panel confirms that changing pulse density
does not covertly change the CD sideband strength within a duration group.
The direction to the right is decreasing spacing;30ns is the hardware floor.

At370ns the spacings90,45,30ns give77.4136%,97.4220%,99.1035%. At730ns,
the30ns point gives99.9149%. These compare fixed targets and durations,
unlike a sweep that simply shortens the gate as more pulses are removed.
The [complete data](runs/pulse_spacing/summary.json) and
[example timing table](runs/pulse_spacing/cd_T370_N12_pulses.csv) are saved.'''),C(r'''
from pulse_spacing_figures import cd_figure,load as spacing_data
spacing_report,spacing_rows=spacing_data()
display(cd_figure().assign(process_fidelity_percent=lambda r:100*r.fidelity).drop(columns='fidelity'))
cd_spacing=spacing_rows[spacing_rows.gate=='cd']
assert len(cd_spacing)==8 and sum(cd_spacing.columns=='fidelity')==1
print('CD SPACING CHECK:8 points;144 input columns;',int(cd_spacing.independent_amplitudes.sum()),
      'independent complex amplitudes;fixed duration per curve;no phase/fidelity fit')
''')]

sq_cells[2:2]=[M(r'''## Pulse spacing at fixed8dB target and fixed4.81us duration

This is a controlled Trotter comparison: repeat the same proper eight-pulse
cycle, with operator axes **Y,X,Y,X,−Y,−X,−Y,−X**, while changing its idle time
and repeat count. Physical DAC angles are270,0,270,0,90,180,90,180degrees.
Every pi envelope is30ns, and every point lasts exactly4.810us including
outer ramps. The spacing is center-to-center, not an adjustable pi width.

For each spacing, normalize one constant JC strength so the wanted second
Magnus coefficient of the **packed/RFDC waveform** is $r/2$, with
$r=8\ln(10)/20$. Pulse axes and coupling are not optimized for fidelity.
As in the reference, one declared final qubit Z frame is calibrated and
included, with the same angle applied to all18 inputs. The wanted quadratic
generator therefore stays8dB to the DAC resolution; every fidelity still compares with
the same exact8dB target on all18 inputs.

The dashed8dB line is the wanted generator strength. The solid vacuum curve
shows implementation error in an observable; it is not used to choose or
rename the target. The black star is the separate optimized160-axis reference,
at the same duration and30ns spacing. Its phases differ from the repeated
eight-pulse Trotter word; do not join it into that word's convergence curve.

Reducing spacing from300ns to37.5ns improves this repeated-cycle fidelity
from30.64% to87.74%; the30ns point is86.29%. This last turnover is possible
because the pi duration is fixed: keeping8dB in the same time requires
stronger JC as spacing decreases, increasing JC evolution during each pulse.
Thus this is not a limit in which every control timescale shrinks together.
It does not meet99% at any spacing tested; the optimized reference does.

In the instantaneous-pulse commutator limit,

$$r\propto g^2 T\Delta,\qquad g\propto\Delta^{-1/2}
\quad\text{at fixed }r,T.$$

Here $\tau_\pi=30\,\mathrm{ns}$ is fixed, so $g\tau_\pi$ increases rather than
vanishing. The exact normalization uses the played-waveform coefficient,
not this approximate rate formula.

Both target and output cutoffs are independently converged. The
[full data](runs/pulse_spacing/summary.json) and
[dense-cycle timing table](runs/pulse_spacing/squeeze_T4810_N160_pulses.csv)
give every physical wait and axis.'''),C(r'''
from pulse_spacing_figures import squeeze_figure,load as spacing_data
spacing_report,spacing_rows=spacing_data()
table=squeeze_figure()
display(table.assign(process_fidelity_percent=lambda r:100*r.fidelity,
    nominal_gate_dB=lambda r:40*r.wanted_coefficient_actual/np.log(10)).drop(columns=['fidelity','wanted_coefficient_actual']))
sq_spacing=spacing_rows[spacing_rows.gate=='squeeze']
assert len(sq_spacing)==10 and max(abs(sq_spacing.command_us-4.810))<1e-12
print('SQUEEZING SPACING CHECK:10 points;180 input columns;',int(sq_spacing.independent_amplitudes.sum()),
      'independent complex amplitudes;all targets8dB;all durations4.810us')
''')]

build('fast_conditional_displacement.ipynb',cd_cells)
build('conditional_gaussian_gates.ipynb',sq_cells)
