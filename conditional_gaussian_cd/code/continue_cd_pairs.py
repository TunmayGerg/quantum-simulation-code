"""Remove one inverse qubit-pulse pair by discrete-angle continuation.

Each nonzero physical pulse remains30ns and rotates pi/N. A zero-angle cell
is an explicit qubit-off interval, not a shortened pi pulse. After reaching
zero angle its dwell is shortened in DAC-sized increments. Sideband phase
changes retain10ns constant-radius arcs at qubit zeros. The same beta1 target
and all18 columns are used at every step.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize
from project_paths import ROOT
from phase_arc_cd import ArcCD,fit,export,MAX_G
import cd_trains as cd

OUT=ROOT/'runs/cd_pair_continuation';OUT.mkdir(exist_ok=True)


class CompressedIdleCD(ArcCD):
    def __init__(self,pair,idle_ticks=24,levels=32):
        super().__init__(12,levels,True)
        assert 8<=idle_ticks<=24 and idle_ticks%2==0
        self.rotation_fractions[[pair,11-pair]]=0
        lengths=np.full(12,24,int);lengths[[pair,11-pair]]=idle_ticks
        self.starts=4+np.r_[0,np.cumsum(lengths[:-1])]
        self.count=int(8+sum(lengths));self.gaps=np.zeros(12,int)
        self.qshape=np.zeros(self.count,complex);self.qcell=np.full(self.count,-1,int)
        W=np.zeros((self.count,12))
        for j in range(12):
            start=self.starts[j];length=lengths[j];W[start:start+length,j]=1
            if self.rotation_fractions[j]>0:
                shape=cd.pi_samples() if j<6 else cd.pi_samples()[::-1]
                self.qshape[start:start+24]=shape;self.qcell[start:start+24]=j
        W[:4,0]=1;W[-4:,-1]=1
        self.arc_centers=[]
        for j in range(1,12):
            if j==6:continue
            center=self.starts[j]-(1 if j>6 else 0)
            assert self.qshape[center]==0
            u=np.arange(9)/8;h=10*u**3-15*u**4+6*u**5
            W[center-4:center+5]=0;W[center-4:center+5,j-1]=1-h;W[center-4:center+5,j]=h
            self.arc_centers.append(center)
        self.phase_W=W
        self.envelope=np.ones(self.count);ramp=np.sin(np.pi*(np.arange(4)+.5)/8)**2
        self.envelope[:4]=ramp;self.envelope[-4:]=ramp[::-1]


def half_pairs():
    r=json.loads((ROOT/'runs/phase_arc_cd/arc_sym_n12_magnus4.json').read_text());jobs=[]
    for pair in range(6):
        fractions=np.ones(12);fractions[[pair,11-pair]]=.5
        jobs.append(dict(name=f'arc_pairhalf_{pair}',cells=12,symmetric=True,fractions=fractions.tolist(),
            parameters=r['parameters'],budget=260,stop=.003,seed_reason='Reduce one inverse pulse pair to pi/2; continue the verified-neighbor phase geometry.'))
    with ProcessPoolExecutor(3) as pool:results=list(pool.map(fit,jobs))
    assert len(results)==6
    (OUT/'half_pairs.json').write_text(json.dumps(results,indent=2)+'\n')
    print('HALF PAIR COMPARISONS:6fits;',18*sum(r['evaluations'] for r in results),'input-column evaluations',flush=True)


def shrink_pair():
    rows=json.loads((OUT/'half_pairs.json').read_text());best=max(rows,key=lambda r:r['fidelity'])
    pair=int(best['name'].rsplit('_',1)[1]);p=best['parameters'];results=[]
    for N in [3,4,8,16,64,0]:
        fractions=np.ones(12);fractions[[pair,11-pair]]=0. if N==0 else 1/N
        r=fit(dict(name=f'arc_pair{pair}_N{N}',cells=12,symmetric=True,fractions=fractions.tolist(),
            parameters=p,budget=320,stop=.003,seed_reason='Discrete pi/N continuation of one inverse pair; N=0 denotes a physical idle.'))
        results.append(r);p=r['parameters']
        (OUT/'angle_continuation.json').write_text(json.dumps(results,indent=2)+'\n')
    print('ANGLE CONTINUATION:6fits;',18*sum(r['evaluations'] for r in results),'input-column evaluations',flush=True)


def compress():
    rows=json.loads((OUT/'angle_continuation.json').read_text());source=rows[-1]
    pair=int(source['name'].split('_')[1].replace('pair',''));p=np.asarray(source['parameters']);results=[]
    for ticks in [24,22,20,18,16,14,12,10,8]:
        model=CompressedIdleCD(pair,ticks);previous=source['command_us'] if not results else results[-1]['command_us']
        initial=p.copy();initial[-1]+=np.log(previous/(model.count*cd.DT))
        initial[-1]=min(initial[-1],np.log(model.maximum_g))
        history=[];best=[float('inf'),initial.copy()]
        def objective(x):
            cost,grad=model.evaluate(x);history.append(cost)
            if cost<best[0]:best[:]=[cost,x.copy()]
            return cost,grad
        def callback(x):
            if best[0]<.003 or len(history)>=300:raise StopIteration
        m=6
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',
            bounds=[(None,None)]*m+[(-np.pi/2-2.5,-np.pi/2+2.5)]*m+[(np.log(2),np.log(model.maximum_g))],
            callback=callback,options=dict(maxiter=400,maxcor=30,maxls=25,ftol=1e-13,gtol=1e-8))
        p=best[1];report,V,T,words=model.evaluate(p,False,outputs=True)
        name=f'arc_compressed_pair{pair}_ticks{ticks}'
        recipe=export(model,p,name,dict(seed_reason='Shorten a qubit-off interval after discrete-angle pair removal.',
            qubit_frame_Rz=report['qubit_frame_Rz']))
        report.update(name=name,pair=pair,idle_ticks=ticks,parameters=p.tolist(),command_us=model.count*cd.DT,
            evaluations=len(history),recipe_file=str(recipe.relative_to(ROOT)),
            worst_basis_fidelity=float(min(abs(np.sum(T.conj()*V,axis=0))**2)))
        results.append(report)
        (OUT/'compressed.json').write_text(json.dumps(results,indent=2)+'\n')
        print('PAIR COMPRESSION',pair,'idle_ticks',ticks,'Tns',model.count*cd.DT*1000,'F',report['fidelity'],flush=True)
    assert len(results)==9
    print('PAIR COMPRESSION:9 durations;',18*sum(r['evaluations'] for r in results),'input-column evaluations',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['half','shrink','compress']);args=parser.parse_args()
    if args.stage=='half':half_pairs()
    elif args.stage=='shrink':shrink_pair()
    else:compress()
