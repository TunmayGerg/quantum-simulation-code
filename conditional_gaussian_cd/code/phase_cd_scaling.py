"""Check whether the phase-refined CD has a useful low-parameter scaling law.

Keep every internal relative phase and pulse width fixed. At each independently
specified beta, fit only one constant coupling and two global carrier-axis
shifts. The target never follows the achieved displacement. All scores decode
SDK words and propagate the modeled RFDC waveform on all18 inputs.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time
import numpy as np
from scipy.optimize import minimize
from project_paths import ROOT
import cd_trains as cd
import hardware_check as hw

OUT=ROOT/'runs/phase_cd_scaling';OUT.mkdir(exist_ok=True)


def main():
    source='arc_sym_n12_magnus4';meta=json.load(open(ROOT/'runs/research_verified'/f'{source}.json'))
    data=np.load(ROOT/meta['controls_file']);q0,s0=data['q'],data['s'];qs,ss=hw.dac_scales(.6)
    cap=np.pi/(2*.200)/max(abs(hw.rfdc_interpolate(s0)))*.998
    space=cd.Space(64,9);P=np.eye(128,dtype=complex)[:,space.inputs];results=[]
    for beta in [.25,.5,.75,1.,1.25]:
        T=cd.target_columns(space,'cd',beta);history=[]
        def cost(p):
            q=q0*np.exp(1j*p[0]);s=s0*p[2]*np.exp(1j*(p[0]+p[1]))
            _,qd=hw.to_words_and_back(q,np.pi/cd.PI_TIME,qs,qubit=True)
            _,sd=hw.to_words_and_back(s,np.pi/(2*.6),ss)
            qf,sf=hw.rfdc_interpolate(qd),hw.rfdc_interpolate(sd)
            assert max(abs(sf))<=np.pi/(2*.200)
            V=hw.propagate(qf,sf,P,cd.DT/8,order=8,fast=True)
            F=float(abs(np.vdot(T,V)/18)**2)
            history.append(dict(parameters=list(p),fidelity=F));return 1-F
        initial=np.array([0.,0.,min(beta,cap)]);initialF=1-cost(initial)
        fit=minimize(cost,initial,method='Powell',bounds=[(-.3,.3),(-.3,.3),(.75*beta,min(1.25*beta,cap))],
            options=dict(maxfev=150,xtol=1e-3,ftol=1e-9))
        best=max(history,key=lambda r:r['fidelity'])
        results.append(dict(beta=beta,initial_fidelity=initialF,fidelity=best['fidelity'],parameters=best['parameters'],
            evaluations=len(history),command_us=len(q0)*cd.DT,columns=18))
        print('PHASE CD SCALING',beta,'Fbefore',initialF,'Fafter',best['fidelity'],'parameters',best['parameters'],
            'evaluations',len(history),flush=True)
        (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    assert len(results)==5
    print('PHASE SCALING:5 fixed targets;',18*sum(r['evaluations'] for r in results),'input-column evaluations',flush=True)


if __name__=='__main__':main()
