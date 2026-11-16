"""Replay every saved search result through corrected SDK words and RFDC FIRs.

Exploratory scores use their recorded search cutoff. Only accepted gates get
the separate automatic column-cutoff and independent-solver certificate.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import csv,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import cd_trains as cd
import hardware_check as hw


def replay(job):
    group,index,result=job;j=result['job'];gate=j['gate'];n=j.get('n_input',9)
    strength=j.get('strength',cd.R_8DB if gate=='squeeze' else np.pi/2)
    levels=j.get('levels',128);space=cd.Space(levels,n)
    train=cd.Train(gate,strength,j['swap'],j['cells'],j['gap'],16)
    q,s,words=hw.played_arrays(train,np.r_[result['qubit_phases'],result['pump_phases']])
    initial=np.eye(2*levels,dtype=complex)[:,space.inputs]
    out=hw.propagate(q,s,initial,cd.DT/8,order=8,fast=True)
    out*=np.exp(-1j*result['qubit_frame_Rz']*space.z/2)[:,None]
    target=cd.target_columns(space,gate,strength)
    overlaps=np.sum(target.conj()*out,axis=0)
    return dict(group=group,index=index,gate=gate,levels=levels,columns=2*n,
                amplitudes=out.size,RFDC_samples=len(q),fidelity=float(abs(overlaps.mean())**2),
                worst_basis_fidelity=float(np.min(abs(overlaps)**2)),
                model_fidelity=result['fidelity'],word_sha256=hashlib.sha256(
                    words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest())


if __name__=='__main__':
    expected={'sweep':237,'round2':120,'round3':188,'cd':696};jobs=[]
    for group,count in expected.items():
        records=json.loads((ROOT/'runs'/f'{group}.json').read_text());assert len(records)==count
        jobs.extend((group,i,r) for i,r in enumerate(records))
    print('START ALL SEARCH PLAYBACKS',len(jobs),'records',flush=True)
    rows=[]
    with ProcessPoolExecutor(6) as pool:
        pending={pool.submit(replay,j):j[:2] for j in jobs}
        for future in as_completed(pending):
            row=future.result();rows.append(row)
            if len(rows)%50==0:
                print('SEARCH PLAYBACKS',len(rows),'/',len(jobs),'columns',sum(r['columns'] for r in rows),flush=True)
    assert len(rows)==sum(expected.values())==1241
    rows.sort(key=lambda r:(r['group'],r['index']))
    path=ROOT/'runs/audit/search_playback.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    counts=dict(records=len(rows),columns=sum(r['columns'] for r in rows),
                complex_amplitudes=sum(r['amplitudes'] for r in rows),
                RFDC_samples=sum(r['RFDC_samples'] for r in rows),
                max_model_playback_fidelity_change=max(abs(r['fidelity']-r['model_fidelity']) for r in rows),
                expected_group_counts=expected)
    assert counts['columns']==25122
    (ROOT/'runs/audit/search_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    print('FINAL SEARCH PLAYBACK COUNTS',counts,flush=True)
