"""Small plotting helpers for the two current research notebooks."""
import json
import numpy as np
import matplotlib.pyplot as plt
from project_paths import ROOT
import cd_trains as cd
import hardware_check as hw

DATA=ROOT/'runs/research_verified'
FIGURES=ROOT/'runs/research_figures';FIGURES.mkdir(exist_ok=True)


def read(relative):return json.loads((ROOT/relative).read_text())
def certificate(name):return read('runs/research_verified/'+name+'.json')


def drives(name):
    words=np.load(DATA/(name+'_words.npz'))
    qs,ss=hw.dac_scales(.6)
    q=hw.sample_to_complex(words['qubit']).conj()/qs*(np.pi/cd.PI_TIME)
    s=hw.sample_to_complex(words['sideband'])/ss*(np.pi/(2*.6))
    return q,s,hw.rfdc_interpolate(q),hw.rfdc_interpolate(s),words


def finish(name):
    plt.savefig(FIGURES/(name+'.png'),dpi=160,bbox_inches='tight');plt.show()


def waveform_figure(name,qubit_zoom_ns=None):
    r=certificate(name);q,s,qf,sf,words=drives(name)
    delay=np.argmax(abs(hw.rfdc_interpolate(np.array([1.]))))*cd.DT/8
    qs,ss=hw.dac_scales(.6)
    played=[qf.conj()/(np.pi/cd.PI_TIME)*qs,sf/(np.pi/(2*.6))*ss]
    fig,axes=plt.subplots(2,2,figsize=(11,5.2),constrained_layout=True)
    for row,(key,z,label) in enumerate(zip(['qubit','sideband'],played,['Qubit','Sideband'])):
        raw=hw.sample_to_complex(words[key]);t=np.arange(len(raw))*cd.DT*1000
        tf=(np.arange(len(z))*cd.DT/8-delay)*1000
        axes[row,0].step(t,raw.real,where='post',alpha=.3,c='C0',label='Programmed I')
        axes[row,0].step(t,raw.imag,where='post',alpha=.3,c='C1',label='Programmed Q')
        axes[row,0].plot(tf,z.real,c='C0',label='Played I');axes[row,0].plot(tf,z.imag,c='C1',label='Played Q')
        axes[row,1].step(t,abs(raw),where='post',alpha=.4,label='Programmed magnitude')
        axes[row,1].plot(tf,abs(z),c='k',label='Played magnitude')
        end=qubit_zoom_ns if row==0 and qubit_zoom_ns else r['command_us']*1000
        for ax in axes[row]:
            ax.set(title=label+(' — first pulses' if row==0 and qubit_zoom_ns else ''),
                   xlabel='Device-envelope time (ns)',ylabel='DAC scale',xlim=(-5,end+5))
            ax.grid(alpha=.3);ax.legend(fontsize=6)
    return fig


def individual_overlaps(name):
    data=np.load(DATA/(name+'_columns.npz'))
    return np.sum(data['target'].conj()*data['output'],axis=0)
