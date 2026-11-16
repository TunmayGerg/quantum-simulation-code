"""A qubit-closed eleven-pulse seed from the improved twelve-pulse gate.

Delete the exactly inverse central pair, then split one remaining pi pulse
into two30ns pi/2 pulses about the same axis. The qubit-only endpoint is
unchanged, whereas the finite JC trajectory is changed. This is a structured
pair-removal continuation, not a random-start search or a shortened pi pulse.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize
from phase_arc_cd import ArcCD,fit,OUT,MAX_G


def main():
    r=json.loads((OUT/'arc_sym_n12_magnus4.json').read_text());p=np.asarray(r['parameters'])
    phi=np.r_[p[:6],p[:6][::-1]+np.pi];psi=np.r_[p[6:12],p[6:12][::-1]]
    phi=np.delete(phi,[5,6]);psi=np.delete(psi,[5,6]);seeds=[]
    for split in range(10):
        q=np.insert(phi,split,phi[split]);s=np.insert(psi,split,psi[split]);fractions=np.ones(11);fractions[split:split+2]=.5
        model=ArcCD(11,32,False,fractions);model.frame=True
        initial=np.array([0.,0.,p[-1]+np.log(.370/.340)])
        def expand(x):return np.r_[q+x[0],s+x[1],x[2]]
        # Only two carrier-axis offsets and one constant g in this seed repair.
        def cost(x):return model.evaluate(expand(x),False)
        result=minimize(cost,initial,method='Powell',bounds=[(-.6,.6),(-.6,.6),(np.log(2.),np.log(model.maximum_g))],
            options=dict(maxfev=160,xtol=1e-3,ftol=1e-8))
        seed=dict(split=split,parameters=expand(result.x).tolist(),fidelity=1-result.fun,
            fractions=fractions.tolist(),evaluations=result.nfev)
        seeds.append(seed);print('PRUNE/SPLIT SEED',split,'F',seed['fidelity'],'eval',result.nfev,flush=True)
    (OUT/'prune_split_seeds.json').write_text(json.dumps(seeds,indent=2)+'\n')
    jobs=[]
    for row in sorted(seeds,key=lambda r:-r['fidelity'])[:4]:
        psi=np.asarray(row['parameters'][11:22])
        jobs.append(dict(name=f"arc_prune_split11_{row['split']}",cells=11,symmetric=False,
            fractions=row['fractions'],parameters=row['parameters'],frame=True,budget=400,
            phase_lower=(psi-1.5).tolist(),phase_upper=(psi+1.5).tolist(),
            seed_reason='Delete an inverse pair from the high-fidelity12-pulse gate, split one remaining pi into two pi/2 envelopes; qubit-only closure is retained.'))
    with ProcessPoolExecutor(3) as pool:results=list(pool.map(fit,jobs))
    assert len(seeds)==10 and len(results)==4
    (OUT/'prune_split_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('PRUNE/SPLIT COMPARISONS:10 three-parameter seed repairs;',18*sum(r['evaluations'] for r in seeds),
        'seed-column evaluations;4full fits;',18*sum(r['evaluations'] for r in results),'fit-column evaluations',flush=True)


if __name__=='__main__':main()
