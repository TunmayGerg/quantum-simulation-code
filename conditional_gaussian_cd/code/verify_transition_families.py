"""Converged effective-model certificates for the simpler/smoothed CD families."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,hashlib
import numpy as np
from coupler_phase_model import *
from verify_scaled_cd import rule

OUT=ROOT/'runs/transition_families';OUT.mkdir(exist_ok=True,parents=True)


def verify(job):
    kind,beta=job
    name='transition_fit_constant.json' if kind=='constant' else 'smooth_cartesian.json'
    p=np.array(json.loads((ROOT/'runs/audit'/name).read_text())['parameters'])
    p[:8]=np.radians(np.round(np.degrees(p[:8]),1));p[8]=round(p[8],6)
    M,gain=rule(beta);shape='phase' if kind=='constant' else 'cartesian'
    history=[];previous=None
    for levels in [64,96,128,192,256]:
        r,out,T,words=effective_score(p,M,gain,shape,levels,True)
        item=dict(levels=levels,fidelity=r['fidelity'],columns=22)
        if previous is not None:
            old,oldT=previous;embedded=np.zeros_like(out);embedded[:len(old)]=old
            embeddedT=np.zeros_like(T);embeddedT[:len(oldT)]=oldT
            item['output_column_change']=float(max(np.linalg.norm(out-embedded,axis=0)))
            item['target_column_change']=float(max(np.linalg.norm(T-embeddedT,axis=0)))
        history.append(item)
        if previous is not None and max(item['output_column_change'],item['target_column_change'])<1e-9:break
        previous=out,T
    else:raise AssertionError('cutoff did not converge')
    qs,ss=hw.dac_scales(.3)
    q=hw.rfdc_interpolate(sample_to_complex(words['qubit'])).conj()/qs*(np.pi/cd.PI_TIME)
    s=hw.rfdc_interpolate(sample_to_complex(words['sideband']))/ss*np.pi/(2*.3)
    space,initial,_=target(beta,levels)
    independent=hw.propagate(q,s,initial,cd.DT/8,order=16)
    error=float(max(abs(independent-out).ravel()));assert error<1e-10
    assert r['fidelity']>.99,(kind,beta,r)
    tag=f'{kind}_beta{beta:g}'
    np.savez_compressed(OUT/(tag+'_words.npz'),**words)
    np.savez_compressed(OUT/(tag+'_columns.npz'),output=out,target=T)
    r.update(kind=kind,beta=beta,parameters=p.tolist(),cutoff_history=history,
             complex_amplitudes=out.size,independent_max_difference=error,
             word_sha256=hashlib.sha256(words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest())
    (OUT/(tag+'.json')).write_text(json.dumps(r,indent=2)+'\n')
    return r


if __name__=='__main__':
    betas=json.loads((ROOT/'runs/scalable_cd/recipe.json').read_text())['tested_strengths']
    jobs=[(kind,beta) for kind in ['constant','cartesian'] for beta in betas]
    print('START',len(jobs),'rounded transition-family gates, all22inputs, automaticcutoffs, independentorder16',flush=True)
    rows=[]
    with ProcessPoolExecutor(4) as pool:
        pending=[pool.submit(verify,job) for job in jobs]
        for f in as_completed(pending):
            r=f.result();rows.append(r)
            print('VERIFIED FAMILY',len(rows),'/',len(jobs),r['kind'],r['beta'],'F',r['fidelity'],
                  'amplitudes',r['complex_amplitudes'],'maxdiff',r['independent_max_difference'],flush=True)
    assert len(rows)==40
    manifest=dict(gates=40,columns=22*len(rows),independent_amplitudes=sum(r['complex_amplitudes'] for r in rows),results=rows)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FINAL TRANSITION FAMILY COUNTS',40,'gates;',manifest['columns'],'columns;',manifest['independent_amplitudes'],'amplitudes',flush=True)
