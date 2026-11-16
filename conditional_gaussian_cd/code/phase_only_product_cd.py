"""Controlled phase-only comparison: all durations and JC magnitudes fixed."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize
from higher_order_cd import OUT,ProductGate


def fit(repeats):
    source=json.loads((OUT/f'overlap_strang_r{repeats}_s35.json').read_text())
    model=ProductGate('strang',repeats,35,48,4,1,True)
    n=model.cells;m=model.section_count
    magnitude=np.asarray(source['parameters'][n:])
    initial=np.r_[source['parameters'][:n],.2*np.sin(np.linspace(-np.pi/2,np.pi/2,m))]
    def expand(p):return np.r_[p[:n],magnitude*np.cos(p[n:]),magnitude*np.sin(p[n:])]
    history=[];best=[np.inf,initial.copy()];started=time.monotonic()
    def cost(p):
        value,g=model.evaluate(expand(p));history.append(value)
        gradient=np.r_[g[:n],-g[n:n+m]*magnitude*np.sin(p[n:])+g[n+m:]*magnitude*np.cos(p[n:])]
        if value<best[0]:best[:]=[value,p.copy()]
        return value,gradient
    def callback(p):
        if len(history)>=220 or time.monotonic()-started>110:raise StopIteration
    result=minimize(cost,initial,jac=True,method='L-BFGS-B',callback=callback,
                    options=dict(maxiter=250,maxcor=20,maxls=25,ftol=1e-13,gtol=1e-8))
    p=expand(best[1]);report,_,_,words=model.evaluate(p,False,outputs=True)
    name=f'phase_only_r{repeats}'
    report.update(job=dict(name=name,method='strang',repeats=repeats,swap_ns=35,
                          overlap_samples=4,sample_grain=1,complex_pump=True,
                          magnitudes_fixed=True),parameters=p.tolist(),
                  optimizer_parameters=best[1].tolist(),fixed_section_magnitudes=magnitude.tolist(),
                  source_record=source['job']['name']+'.json',evaluations=len(history),
                  history=history,command_us=len(words['qubit'])*.00125,
                  seconds=time.monotonic()-started,optimizer_message=str(result.message))
    assert np.max(abs(np.hypot(p[n:n+m],p[n+m:])-magnitude))<1e-12
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('PHASE ONLY RESULT',name,'baselineF',source['fidelity'],'F',report['fidelity'],
          'durationus',report['command_us'],'fixedmagnitudes',m,flush=True)
    return report


if __name__=='__main__':
    print('START PHASE-ONLY CONTROLLED COMPARISON:2fits;durationsandmagnitudesfixed',flush=True)
    with ProcessPoolExecutor(2) as pool:rows=list(pool.map(fit,[3,4]))
    assert len(rows)==2
    (OUT/'phase_only_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('COMPLETED PHASE-ONLY COMPARISON',len(rows),'fits;',18*sum(r['evaluations'] for r in rows),'inputcolumns',flush=True)
