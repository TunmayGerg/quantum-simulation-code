"""Enumerate simple X/Xbar CD words before fitting arbitrary phase corrections.

For an even number of pi pulses the qubit returns to itself up to global phase.
The toggling-frame transverse force has a leading finite-pulse moment
sum_j (-1)^j sign_j. Balanced signs cancel it. The exact finite-pulse unitary
decides which balanced word also suppresses higher terms.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import cd_trains as cd
from constant_pump_gaussian import FixedPumpTrain, OUT


def train_for(cells,swap,levels=48):
    tr=FixedPumpTrain('cd',1.,swap,cells,0.,levels)
    tr.pump_phase=-np.pi/2;tr.fixed_frame=True
    phase=np.exp(-1j*tr.pump_phase*tr.space.n)
    tr.up=phase[:,None]*tr.up*phase.conj()[None,:]
    tr.down=phase[:,None]*tr.down*phase.conj()[None,:]
    matrices=[]
    for phi in [0.,np.pi]:
        D=np.exp(1j*(phi*tr.space.k-tr.pump_phase*tr.space.n))
        matrices.append(D[:,None]*tr.U0*D.conj()[None,:])
    return tr,matrices


def score(tr,matrices,bits):
    state=tr.up@tr.inputs
    for bit in bits:state=matrices[bit]@state
    state=tr.down@state
    return float(abs(np.vdot(tr.target,state)/18)**2)


def main():
    started=time.monotonic();grid=[];refined=[];columns=0
    for cells in [4,6,8,10,12]:
        # Global sign inversion is an exact conjugation by cavity parity times
        # qubit Z, which leaves target and input subspace invariant. Fix first X.
        patterns=[np.r_[0,[(i>>j)&1 for j in range(cells-1)]] for i in range(2**(cells-1))]
        reference=np.pi*cells*.03/4
        best={}
        for factor in [.96,1.,1.04]:
            swap=reference*factor;tr,matrices=train_for(cells,swap)
            values=[score(tr,matrices,bits) for bits in patterns];columns+=18*len(values)
            for i,F in enumerate(values):
                if i not in best or F>best[i]['fidelity']:
                    signs=1-2*patterns[i]
                    best[i]=dict(cells=cells,swap=swap,fidelity=F,bits=patterns[i].tolist(),
                                 alternating_moment=int(signs@((-1.)**np.arange(cells))),
                                 palindrome=bool(np.array_equal(signs,signs[::-1])),
                                 opposed_reverse=bool(np.array_equal(signs,-signs[::-1])))
        top=sorted(best.values(),key=lambda r:-r['fidelity'])[:8]
        grid.extend(best.values())
        for index,row in enumerate(top):
            evaluations=[]
            def cost(swap):
                tr,matrices=train_for(cells,swap)
                F=score(tr,matrices,np.asarray(row['bits']));evaluations.append(F)
                return 1-F
            result=minimize_scalar(cost,bounds=(reference*.90,reference*1.10),method='bounded',
                                   options=dict(xatol=1e-9,maxiter=35))
            swap=float(result.x);tr,_=train_for(cells,swap)
            phases=np.pi*np.asarray(row['bits'])
            record=dict(job=dict(name=f'cd_binary_n{cells}_{index}',gate='cd',strength=1.,swap=swap,
                                 cells=cells,gap=0.,pump_phase=-np.pi/2,fixed_frame=True,
                                 family='binary',seed='balanced X/Xbar enumeration'),
                        **tr.report(np.r_[phases,np.full(cells,-np.pi/2)]),
                        qubit_phases=phases.tolist(),pump_phases=np.full(cells,-np.pi/2).tolist(),
                        binary_pattern=row['bits'],alternating_moment=row['alternating_moment'],
                        palindrome=row['palindrome'],opposed_reverse=row['opposed_reverse'],
                        amplitude_evaluations=len(evaluations),initial_fidelity=row['fidelity'])
            (OUT/(record['job']['name']+'.json')).write_text(json.dumps(record,indent=2)+'\n')
            refined.append(record);columns+=18*len(evaluations)
        winner=max(refined[-8:],key=lambda r:r['fidelity'])
        print('BINARY CD',cells,'patterns',len(patterns),'bestF',winner['fidelity'],
              'swapns',winner['swap_ns'],'word',winner['binary_pattern'],
              'alternatingmoment',winner['alternating_moment'],'T',winner['time_us'],flush=True)
    assert len(grid)==sum(2**(n-1) for n in [4,6,8,10,12]) and len(refined)==40
    summary=dict(unique_patterns=len(grid),refined_patterns=len(refined),
                 input_columns_evaluated=columns,results=refined,grid=grid,
                 seconds=time.monotonic()-started)
    (OUT/'binary_cd_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('FINAL BINARY CD COUNTS',len(grid),'unique words;',40,'amplitude refinements;',columns,'input columns;',summary['seconds'],'seconds',flush=True)


if __name__=='__main__':main()
