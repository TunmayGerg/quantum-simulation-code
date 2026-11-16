"""Structured local JC phase windings at qubit-drive zeros.

A full2pi phase traversal occupies10ns and returns to the same pump direction.
The mirrored traversal has opposite chirality, so the whole pump is even and
the qubit inverse trajectory is odd. This retains exact second-Magnus
cancellation. Finite pi/2 envelopes provide extra qubit zeros inside a logical
pi rotation, allowing a winding at a different accumulated-error location.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,itertools,json,time
import numpy as np
from scipy.optimize import minimize
from scipy.special import betainc
from project_paths import ROOT
from phase_arc_cd import ArcCD,symbol,MAX_G
import cd_trains as cd
import hardware_check as hw

OUT=ROOT/'runs/winding_cd';OUT.mkdir(exist_ok=True)


class WindingCD(ArcCD):
    def __init__(self,cells=10,levels=32,fractions=None,gaps=None,eta=3):
        super().__init__(cells,levels,True,fractions,gaps)
        self.eta=eta;self.phase_W=np.ones((self.count,1));self.set_windings(np.zeros(self.nfree-1,int))

    def set_windings(self,windings):
        self.windings=np.asarray(windings,int);assert len(windings)==self.nfree-1
        phase=np.zeros(self.count)
        for j,k in enumerate(windings,1):
            center=self.starts[j]
            u=np.arange(9)/8;h=betainc(self.eta,self.eta,u)
            phase[center-4:center+5]+=2*np.pi*k*h
            phase[center+5:]+=2*np.pi*k
        phase[self.count//2:]=phase[:self.count//2][::-1]
        self.phase_fixed=phase

    def raw(self,p):
        m=self.nfree;phi=np.r_[p[:m],p[:m][::-1]+np.pi]
        q=self.qshape.copy();keep=self.qcell>=0;q[keep]*=np.exp(1j*phi[self.qcell[keep]])
        s=np.exp(p[-1])*self.envelope*np.exp(1j*(p[m]+self.phase_fixed))
        assert max(abs(q+q[::-1]))<1e-12 and max(abs(s-s[::-1]))<1e-12
        return q,s

    def reduce_gradient(self,dq,ds):
        m=self.nfree;return dq[:m]+dq[m:][::-1],ds


def catalogue():
    started=time.time();m=5;records=[]
    # Two pi/2 pulses replace one full angular traversal while leaving a
    # drive-zero at its midpoint. Enumerate their placement, not random OCT.
    fraction_sets=[np.ones(m)]
    for i,j in itertools.combinations(range(m),2):
        f=np.ones(m);f[[i,j]]=.5;fraction_sets.append(f)
    patterns=list(itertools.product([0,1,-1],repeat=m-1))
    signwords=[np.r_[0,np.pi*np.array(bits)] for bits in itertools.product([0,1],repeat=m-1)]
    for fi,half_fraction in enumerate(fraction_sets):
        fractions=np.r_[half_fraction,half_fraction[::-1]]
        model=WindingCD(10,24,fractions)
        for windings in patterns:
            model.set_windings(windings)
            for phi in signwords:
                p=np.r_[phi,0.,0.];q,s=model.raw(p)
                A,_,_=symbol(q,s,4)
                phase=np.arctan2(-A[0,0],A[1,0]);p[m]=phase
                q,s=model.raw(p);A,B,C=symbol(q,s,4)
                force=A[1,0]
                if force<=0:continue
                g=min(1/force,model.maximum_g)
                target=np.zeros((2,3));target[1,0]=1
                first_error=float(np.linalg.norm(g*A-target));cubic=float(g**3*np.linalg.norm(C))
                records.append(dict(fraction_index=fi,half_fractions=half_fraction.tolist(),windings=list(windings),
                    half_phi=phi.tolist(),phase=float(phase),g=float(g),first_error=first_error,
                    second_norm=float(np.linalg.norm(B)),scaled_cubic_norm=cubic,
                    target_curvature=float(g**3*C[1,0]),score=4*first_error+20*cubic,
                    parameters=np.r_[phi,phase,np.log(g)].tolist()))
        print('WINDING CATALOGUE fraction',fi,'/',len(fraction_sets),'count',len(records),flush=True)
    assert len(records)==11*81*16
    eligible=[r for r in records if r['first_error']<.08]
    chosen=[]
    # Keep diverse geometric recipes; a +/- chirality duplicate is not a new idea.
    for row in sorted(eligible,key=lambda r:r['score']):
        if any(row['fraction_index']==r['fraction_index'] and
               np.all(np.array(row['windings'])==-np.array(r['windings'])) and
               row['half_phi']==r['half_phi'] for r in chosen):continue
        chosen.append(row)
        if len(chosen)==48:break
    print('WINDING SYMBOL FILTER:',len(records),'paths;',len(eligible),'near-correct first averages;',len(chosen),'full waveforms',flush=True)
    for index,row in enumerate(chosen):
        fractions=np.r_[row['half_fractions'],row['half_fractions'][::-1]]
        model=WindingCD(10,32,fractions);model.set_windings(row['windings'])
        p=np.asarray(row['parameters']);row['fidelity']=1-model.evaluate(p,False)
        print('WINDING FULL',index,'F',row['fidelity'],'first',row['first_error'],'cubic',row['scaled_cubic_norm'],
              'fractions',row['half_fractions'],'wind',row['windings'],flush=True)
    payload=dict(enumerated_paths=len(records),eligible_first_moments=len(eligible),full_waveforms=len(chosen),
        full_input_columns=18*len(chosen),seconds=time.time()-started,results=chosen,
        minimum_symbol_score=min(r['score'] for r in records),
        target_curvature_range=[min(r['target_curvature'] for r in eligible),max(r['target_curvature'] for r in eligible)])
    (OUT/'catalogue.json').write_text(json.dumps(payload,indent=2)+'\n')
    # Preserve the finite scan compactly, without thousands of narrative tables.
    np.savez_compressed(OUT/'symbol_catalogue.npz',fraction_index=[r['fraction_index'] for r in records],
        half_phi=[r['half_phi'] for r in records],windings=[r['windings'] for r in records],
        first_error=[r['first_error'] for r in records],cubic=[r['scaled_cubic_norm'] for r in records],
        curvature=[r['target_curvature'] for r in records],score=[r['score'] for r in records])
    print('WINDING COMPLETE:',len(records),'symbol paths;',18*len(chosen),'actual-filter input columns;',payload['seconds'],'seconds',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['catalogue']);args=parser.parse_args();catalogue()
