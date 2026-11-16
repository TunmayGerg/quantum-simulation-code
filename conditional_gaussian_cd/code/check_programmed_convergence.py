"""Larger-cutoff and half-step checks on the most strongly occupied inputs."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from coupler_phase_model import *
from calibrate_programmed_family import programmed_words,in_calibrated_frame

DATA=ROOT/'runs/coupler'


def check(job):
    kind,mode=job
    record=json.loads((DATA/f'vacuum_calibrated_{kind}.json').read_text())
    na,nc,dt=(24,10,.005) if mode=='cutoff' else (20,8,.0025)
    words=programmed_words(record)
    result,out=microscopic(words,Circuit(**record['reference_circuit']),na,nc,dt,inputs=[(0,8),(1,8)])
    out=in_calibrated_frame(out,record,na,nc,result['duration_ns'])
    prior=np.load(DATA/f'programmed_{kind}_a20_c8_dt0.005_columns.npz')['output'][:,[8,17]]
    embedded=np.zeros((na,nc,2,2),complex)
    embedded[:20,:8]=prior.reshape(20,8,2,2)
    embedded=embedded.reshape(out.shape)
    global_phase=np.angle(np.vdot(embedded,out))
    difference=float(max(np.linalg.norm(out*np.exp(-1j*global_phase)-embedded,axis=0)))
    assert difference<1e-5,(kind,mode,difference)
    result.update(kind=kind,check=mode,compared_input_columns=2,
                  compared_amplitudes=out.size,maximum_output_column_change=difference)
    print('MICROSCOPIC CONVERGENCE',kind,mode,'columns2;amplitudes',out.size,'change',difference,
          'seconds',result['elapsed_seconds'],flush=True)
    return result


if __name__=='__main__':
    jobs=[(kind,mode) for kind in ['constant','cartesian'] for mode in ['cutoff','timestep']]
    print('START4controlledconvergencecomparisons, g8/e8, larger24x10andhalfdt',flush=True)
    rows=[]
    with ProcessPoolExecutor(4) as pool:
        for result in pool.map(check,jobs):rows.append(result)
    assert len(rows)==4 and sum(r['compared_input_columns'] for r in rows)==8
    (DATA/'programmed_convergence.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('FINAL MICROSCOPIC CONVERGENCE COUNTS',4,'comparisons;',8,'inputcolumns;',
          sum(r['compared_amplitudes'] for r in rows),'amplitudes',flush=True)
