"""A signed-JC dwell inside a mirrored fractional rotary sequence.

Start with the 12-pulse rotary CD. Replace its central inverse pulse pair by
an adjustable qubit-off interval. A pi/2 envelope near that interval rotates
the unwanted first force into the direction corrected by a reversed-JC dwell.
The pump turns out and back through10ns constant-radius arcs, wholly centered
on qubit zeros. Mirror symmetry cancels Omega2; Omega1 and cubic errors are
explicitly fitted as low-dimensional seeds before the full quantum objective.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import least_squares,minimize
from project_paths import ROOT
from phase_arc_cd import ArcCD,symbol,MAX_G
import cd_trains as cd

OUT=ROOT/'runs/central_arc_cd';OUT.mkdir(exist_ok=True)


class CentralArcCD(ArcCD):
    def __init__(self,gap_ns=40.,half_fractions=None,levels=32):
        f=np.ones(5) if half_fractions is None else np.asarray(half_fractions)
        ticks=round(gap_ns/1.25);assert ticks>=16 and ticks%4==0
        gaps=[0]*5+[ticks]+[0]*4
        super().__init__(10,levels,True,np.r_[f,f[::-1]],gaps)
        first=4+24*5+3;second=self.count-1-first
        self.centers=[first,second]
        profile=np.zeros(self.count)
        u=np.arange(9)/8;h=10*u**3-15*u**4+6*u**5
        profile[first-4:first+5]=h;profile[first+5:second-4]=1
        profile[second-4:second+5]=1-h
        assert max(abs(profile-profile[::-1]))<1e-14
        assert all(self.qshape[c]==0 for c in self.centers)
        self.phase_W=np.stack([np.ones(self.count),profile],axis=1)

    def expand(self,p):
        return np.r_[p[:5],p[:5][::-1]+np.pi],p[5:7],float(np.exp(p[-1]))

    def reduce_gradient(self,dq,ds):return dq[:5]+dq[5:][::-1],ds


def seeds():
    rows=[];started=time.time()
    for gap in [30.,40.,50.]:
        for index in range(5):
            f=np.ones(5);f[index]=.5
            model=CentralArcCD(gap,f,24)
            for word in [0,1]:
                phi=np.resize([0.,0.,np.pi,np.pi],5) if word==0 else np.zeros(5)
                p=np.r_[phi,-np.pi/2,np.pi,np.log(min(2/(.305-gap*.001+0.020),model.maximum_g))]
                def residual(x):
                    q,s=model.raw(x);g=np.exp(x[-1]);A,B,C=symbol(q,s/g)
                    desired=np.zeros((2,3));desired[1,0]=1
                    return np.r_[8*(g*A-desired).ravel(),20*g**3*C.ravel()]
                result=least_squares(residual,p,bounds=(np.r_[np.full(5,-3*np.pi),-np.pi,0.,np.log(2.)],
                    np.r_[np.full(5,3*np.pi),0.,2*np.pi,np.log(model.maximum_g)]),max_nfev=220,
                    ftol=1e-10,xtol=1e-10,gtol=1e-10)
                p=result.x;q,s=model.raw(p);g=np.exp(p[-1]);A,B,C=symbol(q,s/g,16)
                r=dict(gap_ns=gap,half_fractions=f.tolist(),word=word,parameters=p.tolist(),
                    fidelity=1-model.evaluate(p,False),first_error=float(np.linalg.norm(g*A-np.array([[0,0,0],[1,0,0]]))),
                    second_norm=float(np.linalg.norm(B)),scaled_cubic=float(g**3*np.linalg.norm(C)),
                    evaluations=result.nfev,command_us=model.count*cd.DT)
                rows.append(r);print('CENTRAL ARC SEED',gap,index,word,'F',r['fidelity'],'first',r['first_error'],
                    'cubic',r['scaled_cubic'],flush=True)
    assert len(rows)==30
    (OUT/'seeds.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('CENTRAL ARC SEEDS:30 moment fits;540actual-filter input columns;',time.time()-started,'seconds',flush=True)


def fit(seed):
    started=time.time();model=CentralArcCD(seed['gap_ns'],seed['half_fractions'],32)
    initial=np.asarray(seed['parameters']);history=[];best=[float('inf'),initial.copy()]
    def objective(p):
        value,gradient=model.evaluate(p);history.append(value)
        if value<best[0]:best[:]=[value,p.copy()]
        return value,gradient
    def callback(p):
        if best[0]<.006 or len(history)>=320:raise StopIteration
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',
        bounds=[(None,None)]*5+[(-np.pi,0),(0,2*np.pi),(np.log(2.),np.log(model.maximum_g))],
        callback=callback,options=dict(maxiter=400,maxcor=30,maxls=25,ftol=1e-13,gtol=1e-8))
    p=best[1];report,V,T,words=model.evaluate(p,False,outputs=True);q,s=model.raw(p)
    name=f"central_arc_gap{seed['gap_ns']:g}_half{np.argmin(seed['half_fractions'])}_word{seed['word']}"
    controls=OUT/(name+'_controls.npz');np.savez_compressed(controls,q=q,s=s)
    meta=dict(name=name,gate='cd',strength=1.,sideband_mode='phase_arc',qubit_frame_Rz=0.,
        controls_file=str(controls.relative_to(ROOT)),
        magnus_justification='Mirrored finite qubit path cancels Omega2; a signed central JC dwell targets weighted cubic error.',
        phase_arcs=[dict(center_sample=int(c),duration_ns=10.,shape='quintic constant-radius',
            start_phase_rad=float(p[5]+(p[6] if j else 0)),end_phase_rad=float(p[5]+(0 if j else p[6])))
            for j,c in enumerate(model.centers)],
        pulses=[dict(start_sample=int(model.starts[j]),samples=24,angle_fraction=float(model.rotation_fractions[j]),
            phase_rad=float(p[j] if j<5 else p[9-j]+np.pi),reversed_envelope=bool(j>=5)) for j in range(10)])
    recipe=OUT/(name+'_recipe.json');recipe.write_text(json.dumps(meta,indent=2)+'\n')
    report.update(name=name,seed=seed,parameters=p.tolist(),command_us=model.count*cd.DT,
        evaluations=len(history),history=history,worst_basis_fidelity=float(min(abs(np.sum(T.conj()*V,axis=0))**2)),
        recipe_file=str(recipe.relative_to(ROOT)),seconds=time.time()-started)
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('CENTRAL ARC FULL',name,'F',report['fidelity'],'Tns',1000*report['command_us'],'eval',len(history),flush=True)
    return report


def refine():
    rows=json.loads((OUT/'seeds.json').read_text());selected=[]
    for gap in [30.,40.,50.]:selected.extend(sorted([r for r in rows if r['gap_ns']==gap],key=lambda r:-r['fidelity'])[:2])
    with ProcessPoolExecutor(3) as pool:results=list(pool.map(fit,selected))
    assert len(results)==6
    (OUT/'refinement.json').write_text(json.dumps(results,indent=2)+'\n')
    print('CENTRAL ARC REFINEMENT:6 fits;',18*sum(r['evaluations'] for r in results),'input-column evaluations',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['seeds','refine']);args=parser.parse_args()
    seeds() if args.stage=='seeds' else refine()
