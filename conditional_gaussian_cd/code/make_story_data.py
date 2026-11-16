"""Small reference calculations explaining the useful pulse constructions."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json
import numpy as np
import cd_trains as cd
import hardware_check as hw
from verify_short_gates import arrays,OUT


def commutator_reference():
    # In this reference the qubit flips are instantaneous and each JC stretch
    # consists of independent 2x2 blocks. Use their exact analytic actions,
    # rather than constructing and exponentiating large dense propagators.
    def jc(state,angle):
        levels=len(state)//2;out=state.copy()
        c=np.cos(angle*np.sqrt(np.arange(1,levels)))[:,None]
        s=-1j*np.sin(angle*np.sqrt(np.arange(1,levels)))[:,None]
        out[1:-1:2]=c*state[1:-1:2]+s*state[2::2]
        out[2::2]=c*state[2::2]+s*state[1:-1:2]
        return out
    rng=np.random.default_rng(103);probe=rng.normal(size=(32,18))+1j*rng.normal(size=(32,18))
    from scipy.linalg import expm
    direct=expm(-.13j*cd.Space(16,9).hamiltonian(1.,0.))@probe
    assert probe.size==576 and np.max(abs(jc(probe,.13)-direct))<1e-12
    rows=[]
    for gate,strength in [('squeeze',cd.R_8DB),('rotation',np.pi/2)]:
        for loops in [8,16,32,64,128,256,512,1024]:
            eps=np.sqrt(strength/(2*loops))
            for method in ['commutator','cubic_cancelled_pair']:
                previous=None
                for levels in ([192,256,320,384] if gate=='squeeze' else [32,64,96]):
                    space=cd.Space(levels,9)
                    state=np.eye(2*levels,dtype=complex)[:,space.inputs]
                    def B(state,angle):
                        if gate=='squeeze':
                            swap=np.arange(2*levels)^1
                            return jc(state[swap],-angle)[swap]
                        R=np.exp(1j*np.pi/4*space.z)[:,None]
                        return R*jc(R.conj()*state,angle)
                    for loop in range(loops):
                        angle=-eps if method=='cubic_cancelled_pair' and loop%2 else eps
                        state=jc(state,angle);state=B(state,angle)
                        state=jc(state,-angle);state=B(state,-angle)
                    state*=np.exp(1j*(strength if gate=='rotation' else 0)*space.z/2)[:,None]
                    T=cd.target_columns(space,gate,strength)
                    if previous is not None:
                        old,oldT=previous
                        embedded=np.zeros_like(state);embedded[:len(old)]=old
                        embeddedT=np.zeros_like(T);embeddedT[:len(oldT)]=oldT
                        change=float(max(np.linalg.norm(state-embedded,axis=0)))
                        target_change=float(max(np.linalg.norm(T-embeddedT,axis=0)))
                        if max(change,target_change)<1e-9:break
                    previous=state,T
                else:raise AssertionError((gate,loops,'ideal reference cutoff failed'))
                fidelity=float(abs(np.vdot(T,state)/18)**2)
                rows.append(dict(gate=gate,loops=loops,method=method,fidelity=fidelity,
                                 levels=levels,output_change=change,target_change=target_change))
        assert rows[-1]['fidelity']>.99,(gate,rows[-1])
    assert len(rows)==32 and max(max(r['output_change'],r['target_change']) for r in rows)<1e-9
    (OUT/'instantaneous_commutator_reference.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('IDEAL COMMUTATOR REFERENCE:32comparisons,18inputs each;instantaneous controls only',flush=True)


def cd_diagnostics():
    original=json.loads((OUT/'cd_binary_n12_0.json').read_text())
    rows=[];patterns={'alternating X/Xbar':np.resize([0,np.pi],12),
                     'all X':np.zeros(12),'XX Xbar Xbar':np.resize([0,0,np.pi,np.pi],12)}
    for name,phases in patterns.items():
        record=dict(original,qubit_phases=phases.tolist())
        q,s,words,_,_,_=arrays(record);dt=cd.DT/8
        theta=2*np.cumsum(q.real)*dt-q.real*dt
        coefficients=np.c_[s/2,-1j*s*np.cos(theta)/2,1j*s*np.sin(theta)/2]
        increments=coefficients*dt;previous=np.cumsum(increments,axis=0)-increments
        K=np.cross(increments,previous).sum(axis=0)
        R=np.cross(increments,previous.conj()).real.sum(axis=0)
        first=increments.sum(axis=0)
        space=cd.Space(96,9);T=cd.target_columns(space,'cd',1.)
        initial=np.eye(192,dtype=complex)[:,space.inputs]
        output=hw.propagate(q,s,initial,dt,order=8,fast=True)
        row=dict(pattern=name,fidelity=float(abs(np.vdot(T,output)/18)**2),
                 unwanted_first_moment=float(np.linalg.norm(first[1:])),
                 quadratic_diagnostic=float(np.sqrt(sum(abs(K)**2)+4*sum(R**2))))
        rows.append(row);print('FINITE PULSE CD DIAGNOSTIC',row,flush=True)
    assert len(rows)==3
    (OUT/'cd_finite_pulse_diagnostics.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    commutator_reference();cd_diagnostics()
