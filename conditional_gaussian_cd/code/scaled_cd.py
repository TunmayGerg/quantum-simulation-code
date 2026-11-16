"""One phase block for a family of CD strengths; all scores use actual SDK I/Q.

For beta >= 0, play M=ceil(beta/beta_block) copies, with pump gain
s=beta/(M*beta_block). Qubit pi envelopes stay 30 ns at their calibrated scale.
The block's closing Z is carried forward into BOTH drive phases, rather than
silently inserting zero-time physical rotations between blocks.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,copy
from pathlib import Path
import numpy as np
import cd_trains as cd
import hardware_check as hw
from functools import lru_cache


@lru_cache(maxsize=64)
def target(beta,levels,n_input=11):
    space=cd.Space(levels,n_input)
    return space,np.eye(2*levels,dtype=complex)[:,space.inputs],cd.target_columns(space,'cd',beta)


def align_z(output,target):
    zg=np.vdot(target[0::2],output[0::2]);ze=np.vdot(target[1::2],output[1::2])
    zeta=float(np.angle(zg)-np.angle(ze))
    return zeta


def arrays(result,gain=1.,repeats=1,zeta=None):
    j=result['job'];tr=cd.Train('cd',j['strength'],j['swap'],j['cells'],j['gap'],16)
    _,q,s=cd.waveform(tr,np.r_[result['qubit_phases'],result['pump_phases']])
    if zeta is None:zeta=result['qubit_frame_Rz']
    # Rz(zeta) V repeated M times equals Rz(M*zeta) times a train
    # whose m-th block is Rz(-m*zeta) V Rz(m*zeta).
    q=np.concatenate([q*np.exp(-1j*m*zeta) for m in range(repeats)])
    s=np.concatenate([s*gain*np.exp(-1j*m*zeta) for m in range(repeats)])
    qs,ss=hw.dac_scales(tr.swap)
    qw,q=hw.to_words_and_back(q,np.pi/cd.PI_TIME,qs,qubit=True)
    sw,s=hw.to_words_and_back(s,tr.g,ss)
    return hw.rfdc_interpolate(q),hw.rfdc_interpolate(s),dict(qubit=qw,sideband=sw)


def evaluate(result,gain=1.,repeats=1,zeta=None,levels=96,fit_z=False,outputs=False):
    q,s,words=arrays(result,gain,repeats,zeta)
    beta=result['job']['strength']*gain*repeats
    space,initial,desired=target(beta,levels)
    out=hw.propagate(q,s,initial,cd.DT/8,order=8,fast=True)
    if fit_z:
        assert repeats==1
        correction=align_z(out,desired)
    else:correction=repeats*(result['qubit_frame_Rz'] if zeta is None else zeta)
    out*=np.exp(-1j*correction*space.z/2)[:,None]
    overlaps=np.sum(desired.conj()*out,axis=0)
    d=len(overlaps);M=desired.conj().T@out
    report=dict(beta=beta,pump_gain=gain,repeats=repeats,block_Rz_rad=correction/repeats,
                final_Rz_rad=correction,fidelity=float(abs(overlaps.mean())**2),
                worst_basis_fidelity=float(np.min(abs(overlaps)**2)),
                Haar_average_state_fidelity=float((np.linalg.norm(M)**2+abs(np.trace(M))**2)/(d*(d+1))),
                command_us=len(words['qubit'])*cd.DT,RFDC_window_us=len(q)*cd.DT/8,
                columns=22,complex_amplitudes=out.size,levels=levels,
                norm_error=float(np.max(abs(np.sum(abs(out)**2,axis=0)-1))))
    return (report,out,desired,words) if outputs else report


def motif_waveform(parameters,repeats,gain=1.,swap=.3):
    """Eight-pulse rotary echo: opposed reversed qubit axes, mirrored pump.

    Four qubit offsets, four pump offsets and a common pump scale are the only
    nine design variables. The waveform is merged BEFORE RFDC interpolation.
    One outer 5 ns ramp at either end; no ramps inserted between motif copies.
    """
    p=np.asarray(parameters,float)
    qhalf=np.array([0.,0.,np.pi,np.pi])+p[:4]
    phalf=-np.pi/2+p[4:8]
    qphase=np.tile(np.r_[qhalf,qhalf[::-1]+np.pi],repeats)
    pphase=np.tile(np.r_[phalf,phalf[::-1]],repeats)
    cells=8*repeats;g=np.pi/(2*swap)
    # Explicit outer ramps take the endpoint phase; they are never phase 0
    # unless requested by the motif. All per-cell q envelopes remain 30 ns.
    q=np.r_[np.zeros(4),np.concatenate([cd.pi_samples()*np.exp(1j*v) for v in qphase]),np.zeros(4)]
    rampup=np.sin(np.pi*np.arange(4)/8)**2;rampdown=np.cos(np.pi*np.arange(4)/8)**2
    s=np.r_[rampup*np.exp(1j*pphase[0]),np.repeat(np.exp(1j*pphase),24),
            rampdown*np.exp(1j*pphase[-1])]*g*p[8]*gain
    qs,ss=hw.dac_scales(swap)
    qw,q=hw.to_words_and_back(q,np.pi/cd.PI_TIME,qs,qubit=True)
    sw,s=hw.to_words_and_back(s,g,ss)
    return hw.rfdc_interpolate(q),hw.rfdc_interpolate(s),dict(qubit=qw,sideband=sw)


def motif_score(parameters,repeats,gain=1.,levels=64,outputs=False,frame=None,beta_unit=.5):
    beta=beta_unit*repeats*gain
    q,s,words=motif_waveform(parameters,repeats,gain)
    space,initial,desired=target(beta,levels)
    out=hw.propagate(q,s,initial,cd.DT/8,order=8,fast=True)
    correction=align_z(out,desired) if frame is None else frame
    out*=np.exp(-1j*correction*space.z/2)[:,None]
    overlaps=np.sum(desired.conj()*out,axis=0)
    subset=np.r_[np.arange(9),11+np.arange(9)]
    result=dict(beta=beta,beta_unit=beta_unit,repeats=repeats,pump_gain=gain,final_Rz_rad=correction,
                fidelity=float(abs(overlaps.mean())**2),worst_basis_fidelity=float(min(abs(overlaps)**2)),
                fidelity_nmax8=float(abs(overlaps[subset].mean())**2),
                command_us=len(words['qubit'])*cd.DT,RFDC_window_us=len(q)*cd.DT/8,
                levels=levels,columns=22,complex_amplitudes=out.size)
    return (result,out,desired,words) if outputs else result


def optimise_motif(job):
    """Multi-strength local fit of nine interpretable echo-pattern variables."""
    from scipy.optimize import minimize
    seed,maxeval=job[:2];beta_unit=job[2] if len(job)>2 else .5
    rng=np.random.default_rng(seed)
    p0=np.r_[np.zeros(8),.85*beta_unit/.5]
    if seed:p0[:8]=rng.normal(0,.15,8)
    training=([(1,.5),(1,1.),(2,1.),(4,1.)] if beta_unit==.5 else
              [(1,.5),(1,1.),(3,1/(3*beta_unit)),(5,1.)])
    best=[np.inf,None,None];history=[]
    def objective(p):
        if not .2<p[8]<1.01:return 1+abs(p[8]-.85)
        scores=[motif_score(p,m,s,beta_unit=beta_unit) for m,s in training]
        errors=np.array([1-r['fidelity'] for r in scores])
        cost=float(errors.mean()+errors.max())
        history.append(dict(parameters=p.tolist(),cost=cost,scores=scores))
        if cost<best[0]:best[:]=[cost,p.copy(),scores]
        if len(history)%50==0:
            print('MOTIF',beta_unit,seed,'calls',len(history),'best',best[0],
                  'F',[round(r['fidelity'],8) for r in best[2]],flush=True)
        return cost
    simplex=np.vstack([p0,p0+np.diag(np.r_[np.full(8,.08),.03])])
    fit=minimize(objective,p0,method='Nelder-Mead',options=dict(maxfev=maxeval,
                 initial_simplex=simplex,xatol=1e-5,fatol=1e-10))
    record=dict(seed=seed,beta_unit=beta_unit,parameters=best[1].tolist(),training_scores=best[2],objective=best[0],
                waveform_evaluations=4*len(history),input_column_comparisons=88*len(history),
                optimizer_message=str(fit.message),history=history)
    tag='' if beta_unit==.5 else f'_unit{beta_unit:g}'
    (ROOT/'runs/audit'/f'scalable_motif{tag}_seed{seed}.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def probe():
    final=json.loads((ROOT/'runs/cd_final.json').read_text());rows=[]
    for index in [0,2,4,6]:
        base=final[index]
        for gain in [.25,.5,.75,1.]:
            report=evaluate(base,gain,fit_z=True)
            report['base_index']=index;rows.append(report)
            print('SCALE PROBE',index,gain,'F',report['fidelity'],'Rz',report['block_Rz_rad'],flush=True)
        for repeats in [2,3,4]:
            report=evaluate(base,repeats=repeats,levels=192)
            report['base_index']=index;rows.append(report)
            print('REPEAT PROBE',index,repeats,'beta',report['beta'],'F',report['fidelity'],flush=True)
    (ROOT/'runs/audit/scaling_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert len(rows)==28
    print('SCALING PROBE COUNTS',len(rows),'waveforms;',22*len(rows),'input columns',flush=True)


if __name__=='__main__':probe()
