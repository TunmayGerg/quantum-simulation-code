"""Constant sideband phase and amplitude; optimize qubit controls only.

Each gate chooses one coupling and keeps it constant throughout the plateau.
The only pump shaping is a 5 ns ramp at each outer end. The qubit is assumed
calibrated at the steady pump-on frequency. Unknown power-dependent detuning
during turn-on/off is not a simulated calibration.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from project_paths import ROOT,CODE,NOTEBOOKS
import cd_trains as cd
import hardware_check as hw
from shaped_pulse_optimisation import PlayedGate
from constant_pump_gaussian import FixedPumpTrain
from compress_squeeze_blocks import delete_pair

OUT=ROOT/'runs/fixed_sideband';OUT.mkdir(exist_ok=True)
MAX_G=np.pi/(2*.200)


class ConstantGate(PlayedGate):
    def __init__(self,gate,cells,levels=64,gap_ns=0.):
        super().__init__(gate,cells,levels,False,gate!='cd')
        half_gap=round(gap_ns*.001/(2*cd.DT))
        assert abs(half_gap*2*cd.DT*1000-gap_ns)<1e-10
        cell=24+2*half_gap;count=8+cells*cell
        self.qshape=np.zeros(count,complex);self.qcell=np.full(count,-1,int)
        W=np.zeros((count,cells))
        for j in range(cells):
            start=4+j*cell+half_gap
            self.qshape[start:start+24]=cd.pi_samples();self.qcell[start:start+24]=j
            W[4+j*cell:4+(j+1)*cell,j]=1.
        W[:4,0]=np.sin(np.pi*np.arange(4)/8)**2
        W[-4:,-1]=np.cos(np.pi*np.arange(4)/8)**2
        self.W=csr_matrix(W)
        self.pump_shape=np.asarray(self.W.sum(axis=1)).ravel()
        self.maximum_g=MAX_G*.998/max(abs(hw.rfdc_interpolate(self.pump_shape)))
        self.gap_ns=gap_ns

    def arrays(self,p,quantized=True):
        result=super().arrays(p,quantized)
        raw=result[1]
        assert np.all(raw[4:-4]==raw[4]),'Pump plateau must be exactly constant.'
        assert max(max(abs(raw)),max(abs(result[3])))<=MAX_G
        return result


def fit(job):
    start=time.monotonic();n=job['cells']
    model=ConstantGate(job.get('gate','squeeze'),n,job.get('levels',64),job.get('gap_ns',0.))
    initial=np.r_[job['qubit_phases'],np.log(min(job['g'],model.maximum_g))]
    history=[];best=[np.inf,initial.copy()]
    def objective(p):
        cost,g=model.evaluate(p);history.append(cost)
        if cost<best[0]:best[:]=[cost,p.copy()]
        if len(history)%40==0:
            print('FLAT PUMP FIT',job['name'],'eval',len(history),'bestF',1-best[0],flush=True)
        return cost,g
    def callback(p):
        if (best[0]<job.get('stop_infidelity',.004) or len(history)>=job.get('budget',220)
                or time.monotonic()-start>job.get('seconds',300)):
            raise StopIteration
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',
                    bounds=[(None,None)]*n+[(np.log(1.),np.log(model.maximum_g))],
                    callback=callback,options=dict(maxiter=300,maxcor=30,maxls=25,ftol=1e-13,gtol=1e-8))
    p=best[1];report,_,_,words=model.evaluate(p,False,outputs=True)
    report.update(job=job,parameters=p.tolist(),initial_fidelity=1-history[0],history=history,
                  evaluations=len(history),g=float(np.exp(p[-1])),
                  nominal_swap_ns=float(1000*np.pi/(2*np.exp(p[-1]))),
                  command_us=len(words['qubit'])*cd.DT,seconds=time.monotonic()-start,
                  optimizer_message=str(result.message),pump_phase_constant=True,pump_amplitude_constant=True)
    (OUT/(job['name']+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('FLAT PUMP RESULT',job['name'],'F',report['fidelity'],'Tus',report['command_us'],
          'swapns',report['nominal_swap_ns'],'eval',len(history),flush=True)
    return report


def warm_study(workers):
    data=ROOT/'runs/constant_pump_gaussian'
    jobs=[]
    for source,multiplier in [('squeeze_compression_07',1.),('squeeze_lifted_n192_b96_half',np.sqrt(2))]:
        r=json.loads((data/(source+'.json')).read_text());n=r['job']['cells']
        jobs.append(dict(name=f'squeeze_flat_n{n}_warm',gate='squeeze',cells=n,
                         qubit_phases=r['qubit_phases'],g=float(np.pi/(2*r['job']['swap'])*multiplier),
                         seed=('verified8dB' if multiplier==1 else 'double4dBcommutatorarea'),
                         source=source,budget=220,seconds=300))
    print('START FLAT PUMP WARM STUDY:2fits;8dBtarget;all18inputs',flush=True)
    with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(fit,jobs))
    assert len(rows)==2
    (OUT/'warm_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('COMPLETED FLAT WARM:',len(rows),'fits;',18*sum(r['evaluations'] for r in rows),'inputcolumns',flush=True)


def pair_seeds():
    """Delete the least damaging pair, preserve leading N*g^2, correct axes."""
    r=json.loads((ROOT/'runs/constant_pump_gaussian/squeeze_compression_07.json').read_text())
    phi=np.asarray(r['qubit_phases']);n0=len(phi);g0=np.pi/(2*r['job']['swap'])
    train=FixedPumpTrain('squeeze',cd.R_8DB,r['job']['swap'],n0,0.,64)
    train.pump_phase=0.;history=[]
    for n in range(n0-2,127,-2):
        x,deleted,score=delete_pair(train,np.r_[phi,np.zeros(len(phi))])
        phi=x[:n];g=g0*np.sqrt(n0/n)
        train=FixedPumpTrain('squeeze',cd.R_8DB,np.pi/(2*g),n,0.,64);train.pump_phase=0.
        best=[np.inf,phi.copy()];evaluations=[0]
        def objective(p):
            value,gradient=train.objective(p);evaluations[0]+=1
            if value<best[0]:best[:]=[value,p.copy()]
            return value,gradient
        minimize(objective,phi,jac=True,method='L-BFGS-B',options=dict(maxiter=180,maxcor=25,maxls=25,ftol=1e-12,gtol=1e-8))
        phi=best[1]
        row=dict(cells=n,g=float(g),qubit_phases=phi.tolist(),deletion_start=deleted,
                 deletion_fidelity=score,corrected_cell_fidelity=1-best[0],evaluations=evaluations[0])
        history.append(row)
        (OUT/'pair_seeds.json').write_text(json.dumps(history,indent=2)+'\n')
        print('FLAT PAIR REMOVAL',n,'before',score,'corrected',1-best[0],flush=True)
    assert len(history)==16
    print('COMPLETED FLAT PAIR REMOVAL:16pairs;',18*sum(r['evaluations'] for r in history),'inputcolumns',flush=True)


def shortened_study(workers):
    seeds=json.loads((OUT/'pair_seeds.json').read_text());jobs=[]
    for n in [128,144,152]:
        row=next(r for r in seeds if r['cells']==n)
        jobs.append(dict(name=f'squeeze_flat_n{n}_pairs',gate='squeeze',cells=n,g=row['g'],
                         qubit_phases=row['qubit_phases'],seed='qubitonlypairremoval',budget=220,seconds=300))
    print('START SHORTENED FLAT STUDY:',len(jobs),'fits;all18inputs',flush=True)
    with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(fit,jobs))
    assert len(rows)==3
    (OUT/'shortened_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('COMPLETED SHORTENED FLAT STUDY:',len(rows),'fits;',18*sum(r['evaluations'] for r in rows),'inputcolumns',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['warm','seeds','shorten'])
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    if args.stage=='warm':warm_study(args.workers)
    elif args.stage=='seeds':pair_seeds()
    else:shortened_study(args.workers)
