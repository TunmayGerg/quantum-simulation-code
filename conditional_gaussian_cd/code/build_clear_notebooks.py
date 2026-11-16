"""Build two concise, executed-result stories; keep earlier notebooks archived."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path, prepare_notebook
from pathlib import Path
import shutil
import nbformat as nb


ARCHIVE=NOTEBOOKS/'archive';ARCHIVE.mkdir(exist_ok=True)
for name in ['conditional_gaussian_gates.ipynb','fast_conditional_displacement.ipynb','README.md']:
    if document_path(name).exists() and not (ARCHIVE/name).exists():shutil.copy2(document_path(name),ARCHIVE/name)

SETUP=r'''
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
import cd_trains as cd
import hardware_check as hw
from acadia.sample_arithmetic import sample_to_complex
ROOT=Path.cwd();DATA=ROOT/'runs/constant_pump_gaussian';V=DATA/'verified'
FIG=DATA/'figures';FIG.mkdir(exist_ok=True)
def read(name):return json.loads((DATA/name).read_text())
def certificate(name):return json.loads((V/(name+'.json')).read_text())
def source(name):return read(name+'.json')
def finish(name):
    plt.gcf().savefig(FIG/(name+'.png'),dpi=160,bbox_inches='tight');plt.show()
def replay(name):
    r=certificate(name);w=np.load(V/(name+'_words.npz'))
    digest=hashlib.sha256(w['qubit'].tobytes()+w['sideband'].tobytes()).hexdigest()
    assert digest==r['word_sha256']
    qs,ss=hw.dac_scales(.6)
    q=hw.rfdc_interpolate(sample_to_complex(w['qubit'])).conj()/qs*(np.pi/cd.PI_TIME)
    s=hw.rfdc_interpolate(sample_to_complex(w['sideband']))/ss*(np.pi/(2*.6))
    sp=cd.Space(r['levels'],9);initial=np.eye(2*r['levels'],dtype=complex)[:,sp.inputs]
    output=hw.propagate(q,s,initial,cd.DT/8,order=16)
    output*=np.exp(-1j*r['qubit_frame_Rz']*sp.z/2)[:,None]
    saved=np.load(V/(name+'_columns.npz'));target=saved['target']
    difference=float(np.max(abs(output-saved['output'])))
    fidelity=float(abs(np.vdot(target,output)/18)**2)
    assert difference<1e-10 and abs(fidelity-r['fidelity'])<1e-12
    print('FRESH INDEPENDENT REPLAY',name,'18inputcolumns;',output.size,
          'amplitudes;maxdifference',difference,'F',fidelity)
    return r,output,target
'''


def build(name,cells):
    notebook=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python (acadia_env)')))
    nb.validate(notebook);nb.write(prepare_notebook(notebook,NOTEBOOKS/'archive'/('previous_'+name)),NOTEBOOKS/'archive'/('previous_'+name))
    print('BUILT',name,len(cells),'cells;',sum(c.cell_type=='code' for c in cells),'code cells')


M=nb.v4.new_markdown_cell;C=nb.v4.new_code_cell
gaussian=[M(r'''# Short conditional Gaussian gates: construction, corrections, results

For the latest higher-order, phase-cycle and shorter squeezing comparisons,
read [the focused follow-up](higher_order_cd_and_squeezing.ipynb).

The current minimum swap time is **200 ns**. This historical sweep includes
stronger couplings; use the focused notebook for current feasible CD and
squeezing recommendations.

Goal: 8 dB conditional squeezing and a pi-relative conditional number rotation,
with at least **99% coherent gate fidelity over the complete g/e × n=0..8 input
subspace**. Relative phases count. Every qubit pi envelope lasts 30 ns.

Constant pump phase allows simpler repeated qubit blocks. Smooth **10 ns pump
I/Q transitions** gave these earlier short gates. The table below contains
independently verified played-waveform results in the ideal, lossless JC model
with a perfect two-level qubit and zero static chi.'''),C(SETUP+r'''
names=['squeeze_compression_07','squeeze_lifted_n192_b96','strength_squeeze_swap300',
       'rotation_lifted_n256_b128','strength_rotation_swap150']
records=[certificate(n) for n in names]
rows=[]
for name,r in zip(names,records):
    a=source(name);job=a['job']
    rows.append(dict(gate='8dB squeeze' if r['gate']=='squeeze' else 'pi-relative rotation',
                     pattern=(str(job['block'])+' pi pulses, repeated twice' if 'lifted' in name else
                              'individually corrected pi axes'),
                     sideband='constant phase' if r['pump_phase_constant'] else '10ns smooth IQ',
                     swap_ns=(a.get('equivalent_swap_ns') or job['swap']*1000),
                     command_us=r['command_us'],full_RFDC_window_us=r['RFDC_window_us'],
                     coherent_gate_F=r['fidelity'],worst_basis_F=r['worst_basis_fidelity']))
display(pd.DataFrame(rows))
'''),M(r'''## 1. Targets and the Trotter seed

With $\sigma_z=+1$ on g, the fixed targets are

$$S_z(r)=e^{(r/2)\sigma_z(a^2-a^{\dagger2})},\qquad
r=\frac{8\ln10}{20}=0.921034,\qquad U_\chi=e^{-i(\pi/2)\sigma_z n}.$$

For the 18 target columns $T$ and propagated columns $V$,

$$F_{\mathrm{proc}}=\frac{|\operatorname{Tr}(T^\dagger V)|^2}{18^2}.$$

Leaked amplitude is retained. This process metric is not a worst-input
guarantee. Each sequence includes one declared final qubit $R_z$.

Write $H_{\mathrm{JC}}/g=(\sigma_xX+\sigma_yP)/2$, with
$X=a+a^\dagger$ and $P=i(a^\dagger-a)$. Qubit conjugations change the effective
JC Hamiltonian while the pump phase stays fixed:

| Target | Hermitian generators | Commutator |
|---|---|---|
| Squeezing | $A=H_{\mathrm{JC}}/g$, $B=-\sigma_xA\sigma_x$ | $\sigma_z(a^2-a^{\dagger2})$ |
| Number rotation | $A=H_{\mathrm{JC}}/g$, $B=R_z(-\pi/2)AR_z^\dagger(-\pi/2)$ | $-i[\sigma_z(2n+1)-I]$ |

The number-rotation seed has a known scalar and qubit-Z correction.
Two pi pulses about axes separated by $\varphi/2$ synthesize a qubit Z rotation
of $\varphi$ in this instantaneous-control construction.

A chronological $A,B,-A,-B$ loop gives

$$C_\epsilon=e^{i\epsilon B}e^{i\epsilon A}e^{-i\epsilon B}e^{-i\epsilon A},
\qquad \log C_\epsilon=\epsilon^2[A,B]+O(\epsilon^3).$$

Pairing $C_\epsilon$ with $C_{-\epsilon}$ cancels the cubic term. The reference
below uses instantaneous controls: higher order improves asymptotic convergence
but can have a large coefficient, so it is not automatically faster at a finite
number of loops.'''),C(r'''
reference=pd.DataFrame(read('instantaneous_commutator_reference.json'))
fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
for ax,gate in zip(axes,['squeeze','rotation']):
    for method,group in reference[reference.gate==gate].groupby('method'):
        ax.loglog(group.loops,1-group.fidelity,'o-',label=method.replace('_',' '))
    ax.axhline(.01,color='k',ls='--',label='99% gate fidelity')
    ax.set(title=gate,xlabel='Ideal commutator loops',ylabel='Process infidelity')
    ax.grid(alpha=.3);ax.legend(fontsize=7)
finish('gaussian_trotter_reference')
'''),M(r'''## 2. Correct the errors while the qubit pulse is on

JC evolution continues during every 30 ns pi pulse. The fit evolves both
controls simultaneously. A loop-area argument or a palindrome of phase labels
does not replace the full gate calculation.

The constant-phase repeated designs begin with the native commutator seed,
continue coupling and target strength together, and correct the axes in two
**identical superblocks**. They have 96 or 128 independent qubit axes.

The shortest smooth-phase designs continue the best previous Gaussian fits.
Their variables are the qubit axes, pump plateau phases and one common pump
strength. These fastest rows are continuations of earlier solutions; they are
not proofs of a particular Trotter order.

Each smooth-phase forward objective packs SDK int16 words, applies the RFDC
filters and propagates all 18 inputs. No individual waveform sample is a fit
variable. The unsuccessful shorter motifs remain in the saved search data.'''),C(r'''
fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
for ax,gate in zip(axes,['squeeze','rotation']):
    selected=[r for r in records if r['gate']==gate]
    ax.scatter([r['command_us'] for r in selected],[100*r['fidelity'] for r in selected],s=70)
    for r in selected:
        label='constant' if r['pump_phase_constant'] else '10ns IQ'
        if 'lifted' in r['name']:label+='; repeated block'
        ax.annotate(label,(r['command_us'],100*r['fidelity']),xytext=(4,6),textcoords='offset points',fontsize=8)
    ax.axhline(99,color='k',ls='--');ax.set(title=gate,xlabel='Command duration (us)',ylabel='Coherent gate fidelity (%)',ylim=(98.97,99.65))
    ax.grid(alpha=.3)
finish('gaussian_useful_results')
print('Search-cutoff starting score and converged final certificate:')
for name in ['strength_squeeze_swap300','strength_rotation_swap150']:
    r=source(name)
    print(name,'initial F',r['initial_fidelity'],'final F',certificate(name)['fidelity'],
          'actual-playback evaluations',r['evaluations'])
'''),M(r'''## 3. Literal pulse prescription

Every qubit envelope is a 30 ns pi pulse; its phase sets the axis. A repeated
sequence plays its listed half twice. The pump has one 5 ns ramp at each outer
end, with no added ramp between repeats.

For neighboring pump plateaus $z_0,z_1$, the smooth designs program

$$z(u)=[1-h(u)]z_0+h(u)z_1,\qquad
h(u)=10u^3-15u^4+6u^5,\qquad 0\le u\le1,$$

over the last 5 ns / first 5 ns of adjacent cells. The RFDC filters those samples.
The physical qubit DAC phase is minus its model axis; the pump sign is unchanged.
NCOs stay fixed and no extra phase chirp is inserted.'''),C(r'''
fig,axes=plt.subplots(2,2,figsize=(12,6),constrained_layout=True)
for row,name in zip(axes,['strength_squeeze_swap300','strength_rotation_swap150']):
    r=certificate(name);table=pd.read_csv(V/(name+'_sequence.csv'))
    row[0].plot(table.pulse,table.qubit_DAC_deg,'.',label='Qubit DAC phase')
    row[0].plot(table.pulse,table.pump_plateau_DAC_deg,'.',label='Pump plateau phase')
    row[0].set(title=r['gate'],xlabel='30ns pi pulse number',ylabel='Phase (degrees)');row[0].legend(fontsize=7)
    words=np.load(V/(name+'_words.npz'))
    for key,color in [('qubit','C0'),('sideband','C3')]:
        programmed=sample_to_complex(words[key]);played=hw.rfdc_interpolate(programmed)
        t=np.arange(len(played))*.15625-22.65625
        row[1].plot(t,played.real,color=color,label=key+' I')
        row[1].plot(t,played.imag,color=color,ls='--',label=key+' Q')
    row[1].set(xlim=(55,175),xlabel='Aligned time (ns)',ylabel='Played DAC scale',title='Four-pulse detail after RFDC')
    row[1].legend(fontsize=7,ncol=2)
    for ax in row:ax.grid(alpha=.25)
    print(name,'apply final qubit Rz(rad)=',r['qubit_frame_Rz'])
    display(Markdown(f"Complete pulse table: [CSV](runs/constant_pump_gaussian/verified/{name}_sequence.csv)."))
finish('gaussian_played_sequences')
for name in ['squeeze_lifted_n192_b96','rotation_lifted_n256_b128']:
    print(name,': repeat the identical half twice; final Rz(rad)=',certificate(name)['qubit_frame_Rz'])
    display(Markdown(f"[Repeated-block CSV](runs/constant_pump_gaussian/verified/{name}_sequence.csv)"))
'''),M(r'''## 4. Strength sweep: retuned gates versus calibration sensitivity

Shorter swap time means stronger coupling: $g=\pi/(2t_{\mathrm{swap}})$.

The upper row retunes phases at each fixed strength, including the 10 ns pump
transitions. Grey curves show the final shorter-duration screens. These are
achieved local-fit scores, not global maxima or impossibility bounds.
Every plotted map point has separately converged output and target cutoffs.

The lower row holds each selected sequence, duration and final frame fixed.
It measures calibration tolerance, rather than a reoptimized envelope.'''),C(r'''
curves=read('strength_sensitivity.json')['results']
maps=read('gaussian_strength_played_summary.json')
shorter=read('shorter_played_summary.json')
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
chosen=['squeeze_compression_07','strength_squeeze_swap300',
        'rotation_compression_10','strength_rotation_swap150']
for col,gate in enumerate(['squeeze','rotation']):
    group=sorted([r for r in maps if r['job']['gate']==gate],key=lambda r:r['equivalent_swap_ns'])
    duration=group[0]['command_us']
    axes[0,col].plot([r['equivalent_swap_ns'] for r in group],
                     [100*r['converged_fidelity'] for r in group],'o-',label=f'Retuned {duration:.2f}us')
    for n in sorted(set(r['job']['cells'] for r in shorter if r['job']['gate']==gate)):
        group=sorted([r for r in shorter if r['job']['gate']==gate and r['job']['cells']==n],
                     key=lambda r:r['equivalent_swap_ns'])
        axes[0,col].plot([r['equivalent_swap_ns'] for r in group],
                        [100*r['converged_fidelity'] for r in group],'x--',label=f'Screened {group[0]["command_us"]:.2f}us')
    for item in [c for c in curves if c['gate']==gate and c['name'] in chosen]:
        frame=pd.DataFrame(item['results']).sort_values('swap_ns')
        r=certificate(item['name'])
        label=('constant phase' if r['pump_phase_constant'] else '10ns IQ')+f", {r['command_us']:.2f}us"
        nominal=item['nominal_swap_ns'];near=frame[abs(frame.swap_ns/nominal-1)<.05]
        axes[1,col].plot(100*(near.swap_ns/nominal-1),100*near.fidelity,'o-',label=label)
    axes[0,col].set(title=gate+': retune at fixed strength',
                    xlabel='Sideband vacuum swap time (ns)',ylabel='Gate fidelity (%)',ylim=(0,100))
    axes[1,col].set(title='Hold selected sequence fixed',xlabel='Swap-time error from nominal (%)',
                    ylabel='Gate fidelity (%)',ylim=(95,100))
    for ax in axes[:,col]:ax.axhline(99,color='k',ls='--');ax.grid(alpha=.25);ax.legend(fontsize=7)
finish('gaussian_strength_dependence')
print('Retuned Gaussian map:',len(maps),'strength points. Additional shorter screens:',len(shorter))
'''),M(r'''## 5. Fresh independent simulation

The following cell reloads the integer words and independently evolves all 18
inputs again. Accepted gates increase the output cutoff until both output and
target columns change by less than $10^{-9}$ in norm. Squeezing requires 256
levels here; one repeated rotation requires 320. The tested input range remains
n=0..8.'''),C(r'''
fig,axes=plt.subplots(1,2,figsize=(11,3.5),constrained_layout=True)
amplitudes=0
for ax,name in zip(axes,['strength_squeeze_swap300','strength_rotation_swap150']):
    r,out,T=replay(name);amplitudes+=out.size
    errors=1-abs(np.sum(T.conj()*out,axis=0))**2
    ax.bar(np.arange(18),errors)
    ax.set(title=r['gate'],ylabel='Individual basis-input infidelity',xlabel='Input')
    ax.set_xticks(range(18),[f'g{n}' for n in range(9)]+[f'e{n}' for n in range(9)],rotation=90);ax.grid(alpha=.25)
finish('gaussian_input_errors')
print('Fresh Gaussian totals:2gates;36inputcolumns;',amplitudes,'complex amplitudes compared.')
'''),M(r'''The 99% process goal is reached in **3.85 us for 8 dB squeezing** and
**2.89 us for the pi-relative number rotation**, with smooth 10 ns pump I/Q.
Constant-phase alternatives and identical-superblock constructions are shown
above. These are achieved results within the stated search, not global minimum
times. The [README](README.md) links complete words, CSVs and search coverage.''')]
build('conditional_gaussian_gates.ipynb',gaussian)

displacement=[M(r'''# Baseline CD: a reusable four-pulse word

The minimum allowed vacuum swap time is **200 ns**. The current
[fast-gate notebook](higher_order_cd_and_squeezing.ipynb) gives a verified
300 ns CD within this limit. This notebook preserves the 370 ns baseline
and its tested signed-beta scaling rule.

Play **X, X, Xbar, Xbar**, each a 30 ns pi pulse, and repeat. Keep the sideband
phase fixed. Three cycles give **CD(beta=1) in 370 ns command time at 99.23%
coherent gate fidelity**. The entire modeled RFDC response is 414.22 ns.

The input space is all g/e × Fock n=0..8 states, including relative phases.
The model has a perfect two-level qubit, no loss and no static chi.
This notebook explains the cancellation during finite pulses, gives the
repeat/strength rule, and compares stronger-sideband alternatives.'''),C(SETUP+r'''
simple=certificate('cd_binary_scaled_b1')
refined=certificate('played_cd_n12_pump1_seed0')
display(pd.DataFrame([dict(scheme=label,command_ns=r['command_us']*1000,
                         full_RFDC_window_ns=r['RFDC_window_us']*1000,
                         gate_fidelity=r['fidelity'],worst_basis_fidelity=r['worst_basis_fidelity'])
                      for label,r in [('Fixed pump, four-pulse word',simple),('10ns IQ, corrected phases',refined)]]))
'''),M(r'''## 1. Cancel errors while the qubit pulse is on

The target is $\mathrm{CD}(\beta)=e^{\beta\sigma_x(a^\dagger-a)}$.
A constant pump phase gives

$$H_{\mathrm{JC}}=\frac g2(\sigma_xP-\sigma_yX),\qquad
X=a+a^\dagger,\quad P=i(a^\dagger-a).$$

For the real qubit drive $\Omega(t)\sigma_x$, define
$\theta(t)=2\int_0^t\Omega(u)\,du$. Following this finite drive gives

$$H_I=\frac g2\sigma_xP
-\frac g2[\sigma_y\cos\theta(t)-\sigma_z\sin\theta(t)]X.$$

The first term produces CD. The transverse terms must cancel, including during
the pi pulses. For equal envelopes, a leading moment is proportional to
$\sum_j(-1)^j s_j$, with $s_j=+1$ for X and $-1$ for Xbar. **XX Xbar Xbar**
balances that moment and reverses the qubit drive. Its actual played-waveform
second-order coefficient norm is also strongly suppressed.

Xbar is a pi pulse with the opposite drive phase. An isolated X and Xbar
have the same endpoint up to global phase, but their trajectories while JC is
on differ. That difference supplies the useful error cancellation.

The comparison below holds target beta, pump strength and duration fixed.
Moments explain the construction; complete unitary propagation certifies it.
The metric is $F_{\mathrm{proc}}=|\operatorname{Tr}(T^\dagger V)|^2/18^2$.
No leakage is removed, and this is not a worst-input guarantee.'''),C(r'''
diagnostics=pd.DataFrame(read('cd_finite_pulse_diagnostics.json'));display(diagnostics)
fig,axes=plt.subplots(1,2,figsize=(11,3.5),constrained_layout=True)
x=np.arange(3)
axes[0].bar(x-.18,np.maximum(diagnostics.unwanted_first_moment,1e-16),.36,label='Unwanted first moment')
axes[0].bar(x+.18,np.maximum(diagnostics.quadratic_diagnostic,1e-16),.36,label='Second-order coefficient norm')
axes[0].set(yscale='log',xticks=x,xticklabels=diagnostics.pattern,ylabel='Dimensionless coefficient norm')
axes[0].legend(fontsize=7)
axes[1].bar(x,100*diagnostics.fidelity);axes[1].axhline(99,color='k',ls='--')
axes[1].set(xticks=x,xticklabels=diagnostics.pattern,ylabel='Full gate fidelity (%)',ylim=(0,100))
for ax in axes:ax.grid(alpha=.25)
finish('cd_why_four_pulses')
'''),M(r'''## 2. The complete reusable sequence

| Pulse within a cycle | Physical qubit DAC phase | Angle | Duration |
|---|---:|---:|---:|
| 1 | 0 degrees | pi | 30 ns |
| 2 | 0 degrees | pi | 30 ns |
| 3 | 180 degrees | pi | 30 ns |
| 4 | 180 degrees | pi | 30 ns |

There are no gaps. Add one 5 ns pump ramp at either end of the entire train.
The pump phase is fixed at **270 degrees**, or 90 degrees for negative beta.
The supplied qubit pi amplitude and envelope stay unchanged.

For $b=|\beta|>0$, the tested rule is

$$K=\left\lceil\max(2.5b,\;2.25b^2)\right\rceil,\qquad
\mathcal A_K=(0.120K+0.005)\,\mu\mathrm{s},$$

$$g_0=\frac{2b}{\mathcal A_K},\qquad
g=g_0(1+c g_0^2),\qquad c=0.0004071933463\,\mu\mathrm{s}^2.$$

Here K is the number of four-pulse cycles and g is in rad/us. The single
correction coefficient c is calibrated at beta=1 and then held fixed.
The target is never redefined using an achieved output.

$$t_{\mathrm{swap}}=\frac{\pi}{2g},\qquad
T_{\mathrm{command}}=0.120K+0.010\;\mu\mathrm{s}.$$

Extra cycles reduce the finite-pulse error as beta and intermediate photon
population grow. The quadratic repeat rule is a tested safeguard, not a
universal error bound. Skip beta=0.'''),C(r'''
rule=read('binary_rule_summary.json');rows=[]
for a in rule['results']:
    r=certificate(a['job']['name'])
    rows.append(dict(beta=r['strength'],cycles=a['job']['cycles'],pi_pulses=a['job']['cells'],
                     swap_ns=1000*a['job']['swap'],command_ns=r['command_us']*1000,
                     gate_fidelity=r['fidelity']))
table=pd.DataFrame(rows).sort_values('beta');display(table)
assert len(table)==20 and min(table.gate_fidelity)>.99
print('One calibrated coefficient;20signed strengths;360input columns independently checked.')
fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
positive=table[table.beta>0]
axes[0].plot(positive.beta,100*positive.gate_fidelity,'o-');axes[0].axhline(99,color='k',ls='--')
axes[0].set(xlabel='Target beta',ylabel='Coherent gate fidelity (%)',ylim=(99,100))
axes[1].plot(positive.beta,positive.command_ns,'o-')
axes[1].set(xlabel='Target beta',ylabel='Command duration (ns)')
for ax in axes:ax.grid(alpha=.3)
finish('cd_reusable_strength_rule')
display(Markdown('[beta=1 full pulse CSV](runs/constant_pump_gaussian/verified/cd_binary_scaled_b1_sequence.csv)'))
'''),M(r'''## 3. Programmed waveform, played waveform and the 400 ns gate

For beta=1, three cycles contain 12 pi pulses and take 370 ns to program.
The constant pump needs no internal phase ramp or chirp. The plots retain
the real RFDC delay, with time measured from the beginning of the buffer.

The entire causal filter response lasts 414.22 ns. The gate is already at
**99.2278% fidelity at 400 ns**. Its remaining tail changes the input-subspace
gate operator by less than $8\times10^{-7}$ in spectral norm. We evaluate the
actual evolution at that time; no waveform samples are deleted or reprogrammed.'''),C(r'''
w=np.load(V/'cd_binary_scaled_b1_words.npz')
fig,axes=plt.subplots(3,1,figsize=(11,7),constrained_layout=True)
for ax,key in zip(axes[:2],['qubit','sideband']):
    programmed=sample_to_complex(w[key]);played=hw.rfdc_interpolate(programmed)
    ts=np.arange(len(programmed))*1.25;tf=np.arange(len(played))*.15625
    ax.step(ts,programmed.real,where='post',alpha=.4,label='Programmed I')
    ax.step(ts,programmed.imag,where='post',alpha=.4,label='Programmed Q')
    ax.plot(tf,played.real,label='RFDC I');ax.plot(tf,played.imag,label='RFDC Q')
    ax.set(title=key,xlabel='Time since buffer start (ns)',ylabel='DAC scale');ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=4)

completion=read('cd_completion_curve.json');points=pd.DataFrame(completion['points'])
axes[2].plot(points.time_ns,100*points.fidelity,'o-')
axes[2].axhline(99,color='k',ls='--');axes[2].axvline(400,color='C3',ls=':')
axes[2].set(xlim=(370,415),ylim=(90,100),xlabel='Time since buffer start (ns)',
            ylabel='Coherent gate fidelity (%)',title='Actual evolution through the end of the gate')
axes[2].grid(alpha=.25)
axes[2].annotate('400ns:99.2278%; remaining tail norm <8e-7',(400,99.2278),
                 xytext=(382,93),arrowprops=dict(arrowstyle='->'),fontsize=8)

finish('cd_simple_played_waveform')
'''),M(r'''## 4. Stronger sidebands: retune and compare

Shorter swap time means stronger coupling. The left plot retunes phases at
each fixed strength, for 310 ns or 370 ns command duration. It compares a
constant pump with smooth 10 ns I/Q transitions. These are achieved local-fit
scores within the stated seed and evaluation budget, not global maxima.

The right plot holds a selected sequence and final frame fixed while varying
strength; it measures calibration sensitivity. The exact-target bound
$|d\langle a\rangle/dt|\le |g|/2$ explains why a weak sideband cannot create
a large CD arbitrarily quickly. It is not itself a 99% process-fidelity bound.'''),C(r'''
scan=pd.DataFrame(read('strength_played_summary.json'))
fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
for cells in [10,12]:
    for pump in [False,True]:
        group=[r for r in read('strength_played_summary.json') if r['job']['cells']==cells and r['job']['pump_control']==pump]
        group=sorted(group,key=lambda r:r['equivalent_swap_ns'])
        for ax in axes[:,0]:
            ax.plot([r['equivalent_swap_ns'] for r in group],[100*r.get('converged_fidelity',r['fidelity']) for r in group],
                    'o-',label=f'{cells*30+10}ns; '+('10ns IQ' if pump else 'fixed pump'))
for item in read('strength_sensitivity.json')['results']:
    if item['gate']!='cd':continue
    data=pd.DataFrame(item['results']).sort_values('swap_ns')
    label='simple constant pump' if 'binary' in item['name'] else 'corrected 10ns IQ'
    axes[0,1].plot(data.swap_ns,100*data.fidelity,'o-',ms=3,label=label)
    nominal=item['nominal_swap_ns'];near=data[abs(data.swap_ns/nominal-1)<.05]
    axes[1,1].plot(100*(near.swap_ns/nominal-1),100*near.fidelity,'o-',ms=3,label=label)
for ax in axes.ravel():
    ax.axhline(99,color='k',ls='--');ax.grid(alpha=.25);ax.legend(fontsize=7)
    ax.set(xlabel='Sideband vacuum swap time (ns)',ylabel='Coherent gate fidelity (%)',ylim=(0,100))
axes[0,0].set_title('Retune phases at each fixed strength');axes[0,1].set_title('Hold selected sequence fixed')
axes[1,0].set(ylim=(95,100),xlim=(150,320),title='99% threshold in detail')
axes[1,1].set(ylim=(95,100),xlabel='Swap-time error from nominal (%)',title='Calibration tolerance')
finish('cd_strength_comparison')
print('Actual-playback fixed-strength fits:',len(scan),';each includes all18inputs.')
binary=read('binary_cd_summary.json');angles=read('angles_played_summary.json')
print('Enumerated',binary['unique_patterns'],'binary X/Xbar words, modulo exact global-sign symmetry.')
print('Also tested',len(angles),'fractional-angle cases;best F=',max(r['fidelity'] for r in angles))
print('The tested 8/10-pulse schemes did not reach99%;this does not prove impossibility.')
'''),M(r'''## 5. Fresh independent simulation

The next cell reloads the beta=1 words and propagates all 18 columns with the
independent order-16 solver. The family also passes automatic output/target
cutoff convergence. Individual basis-input errors remain visible: the 99%
goal is coherent process fidelity, rather than a worst-input guarantee.'''),C(r'''
r,out,T=replay('cd_binary_scaled_b1')
fig,ax=plt.subplots(figsize=(9,3))
ax.bar(np.arange(18),1-abs(np.sum(T.conj()*out,axis=0))**2)
ax.set_xticks(range(18),[f'g{n}' for n in range(9)]+[f'e{n}' for n in range(9)],rotation=90)
ax.set(ylabel='Individual input infidelity',title='beta=1, constant sideband phase');ax.grid(alpha=.25)
finish('cd_individual_inputs')

qs,ss=hw.dac_scales(.6);words=np.load(V/'cd_binary_scaled_b1_words.npz')
q=hw.rfdc_interpolate(sample_to_complex(words['qubit'])).conj()/qs*(np.pi/cd.PI_TIME)
s=hw.rfdc_interpolate(sample_to_complex(words['sideband']))/ss*(np.pi/(2*.6))
space=cd.Space(r['levels'],9);initial=np.eye(2*r['levels'],dtype=complex)[:,space.inputs]
count=round(.400/(cd.DT/8))
at400=hw.propagate(q[:count],s[:count],initial,cd.DT/8,order=16)
F400=float(abs(np.vdot(T,at400)/18)**2)
tail=float(np.linalg.norm(out-at400,2))
completion=read('cd_completion_curve.json')
assert abs(F400-completion['independent_fidelity'])<1e-12 and tail<1e-6
print('FRESH ACTUAL400ns CHECK:18inputs;',at400.size,'amplitudes;F',F400,
      'remainingtail operatornorm',tail)
'''),M(r'''The simple four-pulse word and one calibrated strength rule give a compact
family. At beta=1, pump-phase corrections improve 99.23% to 99.33% at the same
duration, so the fixed-phase word is the simpler choice for a 99% goal.

The tested faster 8/10-pulse alternatives remain below 99%; this does not prove
impossibility. The [README](README.md) links exact words, the calibration
coefficient and counted search coverage.''')]
build('fast_conditional_displacement.ipynb',displacement)
