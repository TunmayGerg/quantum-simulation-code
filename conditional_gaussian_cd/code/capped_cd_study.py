"""CD search with the user's hard 200 ns minimum vacuum-swap time.

All qubit pi pulses keep their calibrated 30 ns Hann envelope. A small
number of pump coefficients (two per cell) controls evolution during and
between those pulses. Pump I/Q changes use 10 ns quintic transitions.
Every objective uses the SDK words and the full RFDC response. A linear-map
norm bounds the whole response, rather than bounding I and Q separately.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
import cd_trains as cd
import hardware_check as hw
from higher_order_cd import CorrectedTrain,ROOT

OUT=ROOT/'runs/swap200';OUT.mkdir(exist_ok=True)
MIN_SWAP_NS=200.
MAX_G=np.pi/(2*MIN_SWAP_NS*.001)


class CappedCD(CorrectedTrain):
    def __init__(self,cells,gap_ns,phase_control=False,levels=48,edge_overlap=False):
        super().__init__(cells,levels,True,False,'cd')
        self.phase_control=phase_control
        half_gap=int(round(gap_ns*.001/(2*cd.DT)))
        assert abs(2*half_gap*cd.DT*1000-gap_ns)<1e-10
        cell_samples=24+2*half_gap
        self.half_gap=half_gap;self.cell_samples=cell_samples
        outer=0 if edge_overlap else 4
        count=2*outer+cells*cell_samples
        self.qshape=np.zeros(count,complex);self.qcell=np.full(count,-1,int)
        for j in range(cells):
            start=outer+j*cell_samples+half_gap
            self.qshape[start:start+24]=cd.pi_samples();self.qcell[start:start+24]=j
        self.knots=2*cells
        W=np.zeros((count,self.knots))
        half_cell=cell_samples//2
        assert 2*half_cell==cell_samples
        for j in range(self.knots):W[outer+j*half_cell:outer+(j+1)*half_cell,j]=1
        W[:4,0]=np.sin(np.pi*np.arange(4)/8)**2
        W[-4:,-1]=np.cos(np.pi*np.arange(4)/8)**2
        u=np.arange(8)/8;h=10*u**3-15*u**4+6*u**5
        for j in range(1,self.knots):
            start=outer+j*half_cell-4
            W[start:start+8]=0
            W[start:start+8,j-1]=1-h;W[start:start+8,j]=h
        self.W=csr_matrix(W)
        # For all complex knot coefficients z, |K z| <= ||z||_infinity
        # times the row sum of |K|. This enforces a radial, physical bound
        # even while the phase changes and after the interpolation filters.
        K=np.column_stack([hw.rfdc_interpolate(W[:,j]) for j in range(self.knots)])
        self.filter_gain_bound=float(np.max(np.sum(abs(K),axis=1)))
        self.radius_bound=MAX_G*.998/self.filter_gain_bound  # includes int16 rounding margin

    def expand(self,p):
        n=self.cells;m=self.knots
        radius=p[n:n+m]
        angle=p[n+m:] if self.phase_control else np.zeros(m)
        return p[:n],-1j*radius*np.exp(1j*angle)

    def arrays(self,p,quantized=True):
        result=super().arrays(p,quantized)
        assert max(np.max(abs(result[1])),np.max(abs(result[3])))<=MAX_G*(1+1e-12)
        return result

    def evaluate(self,p,gradient=True,quantized=True,outputs=False):
        result=super().evaluate(p,gradient,quantized,outputs)
        if not gradient or outputs:return result
        value,g=result;n=self.cells;m=self.knots
        radius=p[n:n+m];angle=p[n+m:] if self.phase_control else np.zeros(m)
        da=g[n:n+m];db=g[n+m:]
        dr=da*np.cos(angle)+db*np.sin(angle)
        dangle=radius*(-da*np.sin(angle)+db*np.cos(angle))
        return value,np.r_[g[:n],dr,dangle if self.phase_control else []]


def check_gradient():
    errors=[];rng=np.random.default_rng(200)
    for phases in [False,True]:
        model=CappedCD(4,5,phases,24)
        p=np.r_[rng.normal(0,.1,4),np.full(8,5.),rng.normal(0,.1,8) if phases else []]
        _,g=model.evaluate(p,quantized=False)
        for _ in range(3):
            v=rng.normal(size=len(p));v/=np.linalg.norm(v);e=1e-5
            fd=(model.evaluate(p+e*v,False,False)-model.evaluate(p-e*v,False,False))/(2*e)
            errors.append(float(abs(fd-g@v)))
    assert len(errors)==6 and max(errors)<2e-5,errors
    (OUT/'gradient_check.json').write_text(json.dumps(dict(directions=6,errors=errors),indent=2)+'\n')
    print('CAPPED GRADIENT:6directions;maxerror',max(errors),flush=True)


def fit(job):
    start=time.monotonic();n=job['cells'];phases=job['phase_control']
    model=CappedCD(n,job['gap_ns'],phases,edge_overlap=job.get('edge_overlap',False))
    if n in [8,10]:
        rows=json.loads((ROOT/'runs/constant_pump_gaussian/binary_cd_summary.json').read_text())['results']
        source=max((r for r in rows if r['job']['cells']==n),key=lambda r:r['fidelity'])
        phi=np.asarray(source['qubit_phases'])
    else:phi=np.resize([0.,0.,np.pi,np.pi],n)
    if phases:phi=phi+.05*np.sin(np.linspace(-np.pi,np.pi,n))
    duration=(8+n*model.cell_samples)*cd.DT
    initial_g=min(2/(duration-.005),model.radius_bound*.98)
    initial=np.r_[phi,np.full(model.knots,initial_g),
                  .08*np.sin(np.linspace(-np.pi,np.pi,model.knots)) if phases else []]
    if 'initial' in job:
        initial=np.asarray(job['initial'])
        initial[n:n+model.knots]=np.minimum(initial[n:n+model.knots],model.radius_bound)
    history=[];best=[np.inf,initial.copy()]
    def objective(p):
        value,g=model.evaluate(p);history.append(value)
        if value<best[0]:best[:]=[value,p.copy()]
        if len(history)%40==0:
            print('CAPPED FIT',job['name'],'eval',len(history),'bestF',1-best[0],flush=True)
        return value,g
    def callback(p):
        if best[0]<.005 or len(history)>=180 or time.monotonic()-start>100:raise StopIteration
    bounds=[(None,None)]*n+[(0.,model.radius_bound)]*model.knots
    if phases:bounds += [(None,None)]*model.knots
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,
                    callback=callback,options=dict(maxiter=220,maxcor=25,maxls=25,ftol=1e-13,gtol=1e-8))
    report,_,_,words=model.evaluate(best[1],False,outputs=True)
    qraw,sraw,q,s,_=model.arrays(best[1])
    peak=float(max(abs(s)))
    report.update(job=job,parameters=best[1].tolist(),history=history,evaluations=len(history),
                  initial_fidelity=1-history[0],command_us=len(qraw)*cd.DT,
                  programmed_peak_g=float(max(abs(sraw))),played_peak_g=peak,
                  played_min_swap_ns=1000*np.pi/(2*peak),
                  hard_min_swap_ns=MIN_SWAP_NS,filter_gain_bound=model.filter_gain_bound,
                  coefficient_radius_bound=model.radius_bound,seconds=time.monotonic()-start,
                  optimizer_message=str(result.message))
    (OUT/(job['name']+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('CAPPED RESULT',job['name'],'F',report['fidelity'],'Tns',report['command_us']*1000,
          'minswapns',report['played_min_swap_ns'],'eval',len(history),flush=True)
    return report


def study(workers):
    jobs=[]
    for n,gap,phases in [(8,5,False),(8,10,False),(10,0,False),(10,2.5,False),(10,5,False),
                         (10,0,True),(10,2.5,True),(10,5,True),(12,0,False)]:
        jobs.append(dict(name=f'cd_n{n}_gap{gap:g}_phase{int(phases)}',cells=n,gap_ns=gap,
                         phase_control=phases,min_swap_ns=MIN_SWAP_NS))
    print('START CAPPED STUDY:',len(jobs),'fits;all18inputs;minswap200ns',flush=True)
    with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(fit,jobs))
    assert len(rows)==len(jobs)==9
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('COMPLETED CAPPED STUDY:',len(rows),'fits;',18*sum(r['evaluations'] for r in rows),'inputcolumns',flush=True)


def edge_study(workers):
    jobs=[]
    for n,gap,phases in [(10,0,False),(10,0,True),(8,5,True)]:
        job=dict(name=f'edge_n{n}_gap{gap:g}_phase{int(phases)}',cells=n,gap_ns=gap,
                 phase_control=phases,min_swap_ns=MIN_SWAP_NS,edge_overlap=True)
        if n==10:
            source=json.loads((OUT/f'cd_n10_gap0_phase{int(phases)}.json').read_text())
            job['initial']=source['parameters']
        jobs.append(job)
    print('START EDGE-RAMP STUDY:',len(jobs),'fits;all18inputs;minswap200ns',flush=True)
    with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(fit,jobs))
    assert len(rows)==len(jobs)==3
    (OUT/'edge_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('COMPLETED EDGE-RAMP STUDY:',len(rows),'fits;',18*sum(r['evaluations'] for r in rows),'inputcolumns',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['gradient','fit','edge']);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.stage=='gradient':check_gradient()
    elif args.stage=='fit':study(args.workers)
    else:edge_study(args.workers)
