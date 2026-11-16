"""Test explicitly signed JC plateaus in closed fractional-angle CD words.

Eleven30ns envelopes use two pi/2 rotations so the X-only starting word
closes. A30ns negative-JC cell is reached by10ns constant-radius phase arcs;
same-chirality0->pi->2pi arcs cancel their leading orthogonal force. This is
a signed-coupling alternative to positive-g cubic cancellation, not a raw
sample-based control search. Full-word refinement uses only pulse axes and
circular-phase plateau values.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from project_paths import ROOT
from phase_arc_cd import fit,MAX_G

OUT=ROOT/'runs/phase_arc_cd'


def main():
    jobs=[];n=11
    for label,indices in [('ends',[0,10]),('center',[4,5]),('quarter',[1,9])]:
        fractions=np.ones(n);fractions[indices]=.5
        for negative in [None,4,5,6]:
            phi=np.resize([0.,0.,np.pi,np.pi],n)
            psi=np.full(n,-np.pi/2)
            if negative is not None:
                psi[negative]=np.pi/2;psi[negative+1:]=3*np.pi/2
            initial_g=min(2/(n*.03+.005-(.060 if negative is not None else 0)),MAX_G*.974)
            jobs.append(dict(name=f'arc_n11_{label}_neg{negative}',cells=n,symmetric=False,
                fractions=fractions.tolist(),parameters=np.r_[phi,psi,np.log(initial_g)].tolist(),budget=320,
                phase_lower=(psi-1.25).tolist(),phase_upper=(psi+1.25).tolist(),
                seed_reason='Closed pi/2 rotary word; one signed JC plateau with same-chirality phase arcs to alter the cubic error.'))
    print('START REVERSED JC:12 structured fits;340ns;all18inputs',flush=True)
    with ProcessPoolExecutor(3) as pool:results=list(pool.map(fit,jobs))
    assert len(results)==12
    (OUT/'reversed_jc_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('REVERSED JC COMPARISONS:12 fits;',18*sum(r['evaluations'] for r in results),'input-column evaluations;',
          sum(r['fidelity']>=.99 for r in results),'above99% design scores',flush=True)


if __name__=='__main__':main()
