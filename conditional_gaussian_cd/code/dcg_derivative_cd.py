"""Direct projective error-generator penalty for the 260 ns CD candidate.

Unlike averaging nominal gate errors, this explicitly minimizes the distance
between gates with opposite small qubit-gain errors. In the small-error limit
it is the squared leading error-generator norm (modulo global phase).
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import minimize
import cd_trains as cd
import hardware_check as hw
from higher_order_cd import OUT,ProductGate


def pair_cost(model,p,gradient=True,quantized=True):
    eps=.01
    data=[]
    for gain in [1-eps,1+eps]:
        model.qubit_gain=gain
        _,_,q,s,_=model.arrays(p,quantized)
        V=hw.propagate(q,s,model.initial,cd.DT/8,order=8,fast=True)
        data.append((q,s,V,gain))
    model.qubit_gain=1.
    trace=np.vdot(data[0][2],data[1][2])
    value=float((1-abs(trace/18)**2)/(4*eps**2))
    if not gradient:return value
    result=np.zeros_like(p)
    for j in [0,1]:
        q,s,V,gain=data[j];adjoint=np.ascontiguousarray(data[1-j][2])
        tr=trace if j==1 else trace.conjugate()
        G=np.zeros((len(q),4),float)
        model.backward(model.levels,18,len(q),8,cd.DT/8,float(tr.real),float(tr.imag),
                       q.ctypes.data,s.ctypes.data,V.ctypes.data,adjoint.ctypes.data,G.ctypes.data)
        Gq=fftconvolve(G[:,0]+1j*G[:,1],model.filter[::-1],mode='valid')[::8]
        Gs=fftconvolve(G[:,2]+1j*G[:,3],model.filter[::-1],mode='valid')[::8]
        keep=model.qcell>=0
        qb=model.qshape[keep]*np.exp(1j*p[:model.cells][model.qcell[keep]])
        dphi=gain*np.bincount(model.qcell[keep],weights=np.real(Gq[keep].conj()*1j*qb),minlength=model.cells)
        da=np.real((model.W.T@Gs).conj()*(-1j))
        result+=np.r_[dphi,da]/(4*eps**2)
    return value,result


def check():
    model=ProductGate('strang',2,50,24,4,1)
    p=np.r_[model.seed_phases+.1,model.seed_amplitudes]
    _,g=pair_cost(model,p,quantized=False)
    rng=np.random.default_rng(130);errors=[]
    for _ in range(3):
        v=rng.normal(size=len(p));v/=np.linalg.norm(v);e=1e-4
        fd=(pair_cost(model,p+e*v,False,False)-pair_cost(model,p-e*v,False,False))/(2*e)
        errors.append(float(abs(fd-g@v)))
    assert len(errors)==3 and max(errors)<5e-4,errors
    print('DCG GENERATOR GRADIENT:3directions;maxerror',max(errors),flush=True)
    (OUT/'dcg_generator_gradient.json').write_text(json.dumps(dict(directions=3,errors=errors),indent=2)+'\n')


def fit(weight):
    source=json.loads((OUT/'overlap_strang_r4_s35.json').read_text())
    model=ProductGate('strang',4,35,48,4,1)
    p=np.array(source['parameters']);p[:8]+=.05*np.sin(np.linspace(-np.pi,np.pi,8))
    history=[];best=[np.inf,p.copy()];started=time.monotonic()
    def objective(p):
        nominal,grad=model.evaluate(p)
        sensitivity,dg=pair_cost(model,p)
        cost=nominal+weight*sensitivity
        history.append(dict(nominal_fidelity=1-nominal,sensitivity=sensitivity,cost=cost))
        if cost<best[0]:best[:]=[cost,p.copy()]
        return cost,grad+weight*dg
    def callback(p):
        if len(history)>=100 or time.monotonic()-started>140:raise StopIteration
    result=minimize(objective,p,jac=True,method='L-BFGS-B',bounds=[(None,None)]*8+[(-45.,45.)]*9,
                    callback=callback,options=dict(maxiter=150,maxcor=20,maxls=25,ftol=1e-12,gtol=1e-8))
    report,_,_,_=model.evaluate(best[1],False,outputs=True)
    name=f'dcg_generator_weight{weight:g}'
    report.update(job=dict(name=name,method='strang',repeats=4,swap_ns=35,overlap_samples=4,sample_grain=1,
                          generator_weight=weight),parameters=best[1].tolist(),
                  initial_fidelity=history[0]['nominal_fidelity'],history=history,
                  evaluations=len(history),generator_penalty=pair_cost(model,best[1],False),
                  command_us=.260,seconds=time.monotonic()-started,optimizer_message=str(result.message))
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('DCG GENERATOR RESULT',name,'F',report['fidelity'],'penalty',report['generator_penalty'],
          'evaluations',len(history),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:check()
    else:
        print('START DIRECT DCG GENERATOR FITS:2weights;all18inputs;3forwards/objective',flush=True)
        with ProcessPoolExecutor(2) as pool:rows=list(pool.map(fit,[.005,.02]))
        assert len(rows)==2
        (OUT/'dcg_generator_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('COMPLETED DIRECT DCG GENERATOR FITS',len(rows),'fits;',54*sum(r['evaluations'] for r in rows),'inputcolumns',flush=True)
