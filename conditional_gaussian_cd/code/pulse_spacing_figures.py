"""Plots of pulse spacing with the target and total duration held fixed."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from project_paths import ROOT

OUT=ROOT/'runs/pulse_spacing';FIGURES=OUT/'figures';FIGURES.mkdir(exist_ok=True)


def load():
    report=json.loads((OUT/'summary.json').read_text());assert report['points']==18
    rows=pd.DataFrame(report['results'])
    assert all(rows.relative_coefficient_error<.003)
    assert all(rows.qubit_envelope_ns==30) and all(rows.peak_swap_ns>=200)
    assert not any(rows.phases_reoptimized) and not any(rows.fidelity_optimized)
    return report,rows


def spacing_axis(ax,ticks):
    ax.set_xscale('log');ax.set_xticks(ticks);ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.invert_xaxis();ax.tick_params(axis='x',labelrotation=35)
    ax.set_xlabel('π center spacing Δ (ns; denser →)');ax.grid(alpha=.25)


def cd_figure():
    report,rows=load();data=rows[rows.gate=='cd'].copy();assert len(data)==8
    fig,axes=plt.subplots(1,3,figsize=(13.5,3.8),constrained_layout=True)
    for index,duration in enumerate(sorted(data.duration_us.unique())):
        group=data[data.duration_us==duration].sort_values('spacing_ns',ascending=False)
        assert max(abs(group.command_us-duration))<1e-12
        label=f'Total {duration*1000:g} ns'
        color=f'C{index}'
        axes[0].plot(group.spacing_ns,100*group.fidelity,'o-',label=label,color=color)
        axes[1].plot(group.spacing_ns,group.wanted_coefficient_actual,'--',alpha=.55,color=color)
        axes[1].plot(group.spacing_ns,group.vacuum_displacement_real,'o-',label=label,color=color)
        axes[2].plot(group.spacing_ns,group.nominal_swap_ns,'o-',label=label,color=color)
    axes[0].axhline(99,c='k',ls=':',label='99%');axes[0].set(ylabel='Coherent process fidelity (%)',ylim=(70,100.2))
    axes[0].legend(fontsize=8)
    axes[1].axhline(1,c='k',ls=':');axes[1].set(ylabel='Displacement beta',title='Dashed: wanted coefficient; solid: vacuum mean')
    axes[2].axhline(200,c='k',ls=':');axes[2].set(ylabel='Nominal vacuum-swap time (ns)',title='Constant strength within each duration')
    for ax in axes:spacing_axis(ax,[30,45,60,90,180])
    fig.suptitle('CD target beta=1; fixed pulse cycle; 30 ns π envelopes; total duration fixed per curve')
    fig.savefig(FIGURES/'cd_fixed_duration_spacing.png',dpi=170,bbox_inches='tight')
    plt.show()
    return data[['duration_us','cells','spacing_ns','gap_ns','nominal_swap_ns','fidelity',
        'wanted_coefficient_actual','vacuum_displacement_real','levels']].sort_values(['duration_us','spacing_ns'],ascending=[True,False])


def squeeze_figure():
    report,rows=load();data=rows[rows.gate=='squeeze'].sort_values('spacing_ns',ascending=False).copy();assert len(data)==10
    assert max(abs(data.command_us-4.810))<1e-12
    ref=json.loads((ROOT/'runs/research_verified/squeeze_constant_reference.json').read_text())
    saved=np.load(ROOT/'runs/research_verified/squeeze_constant_reference_columns.npz')['output'][:,0].reshape(-1,2)
    n=np.arange(len(saved));mean_n=np.sum(n[:,None]*abs(saved)**2)
    mean_a=np.sum(np.sqrt(n[1:])[:,None]*saved[:-1].conj()*saved[1:])
    a2=np.sum(np.sqrt(n[2:]*(n[2:]-1))[:,None]*saved[:-2].conj()*saved[2:])
    reference_dB=-10*np.log10(float(2*mean_n+1+2*a2.real-4*mean_a.real**2))
    nominal_dB=40*data.wanted_coefficient_actual/np.log(10)
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.2),constrained_layout=True)
    axes[0,0].plot(data.spacing_ns,100*data.fidelity,'o-',label='Repeated proper 8-pulse cycle')
    axes[0,0].scatter([30],[100*ref['fidelity']],s=130,marker='*',c='k',label='Separate optimized 160-axis reference')
    axes[0,0].axhline(99,c='k',ls=':');axes[0,0].set(ylabel='Coherent process fidelity (%)',ylim=(25,101))
    axes[0,0].legend(fontsize=7)
    axes[0,1].plot(data.spacing_ns,data.nominal_swap_ns,'o-');axes[0,1].axhline(200,c='k',ls=':')
    axes[0,1].set(ylabel='Nominal vacuum-swap time (ns)',title='Stronger JC needed at smaller spacing')
    axes[1,0].plot(data.spacing_ns,nominal_dB,'--',label='Wanted Gaussian coefficient')
    axes[1,0].plot(data.spacing_ns,data.vacuum_target_axis_squeezing_dB,'o-',label='Actual vacuum X variance')
    axes[1,0].scatter([30],[reference_dB],s=100,marker='*',c='k',label='Optimized reference: actual variance')
    axes[1,0].axhline(8,c='k',ls=':');axes[1,0].set(ylabel='Squeezing (dB)',ylim=(0,9))
    axes[1,0].legend(fontsize=7)
    kappa=data.g*.030*3 # sqrt(n+1)=3 for the highest e8 JC input.
    axes[1,1].plot(data.spacing_ns,kappa,'o-')
    axes[1,1].set(ylabel='g × 30 ns × sqrt(9)',title='JC evolution during each finite π pulse grows')
    for ax in axes.ravel():spacing_axis(ax,[30,37.5,50,75,100,150,200,300])
    fig.suptitle('Conditional squeezing target: 8 dB; total duration 4.810 μs at EVERY point')
    fig.savefig(FIGURES/'squeeze_fixed_duration_spacing.png',dpi=170,bbox_inches='tight')
    plt.show()
    return data[['cells','spacing_ns','gap_ns','nominal_swap_ns','fidelity',
        'wanted_coefficient_actual','vacuum_target_axis_squeezing_dB','qubit_frame_Rz','levels']]
