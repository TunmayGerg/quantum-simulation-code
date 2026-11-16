"""All-input check after primitive, vacuum and observable-frame calibrations."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from coupler_phase_model import *
from calibrate_programmed_family import programmed_words,in_calibrated_frame,observer_frames
from audit_cd import compile_words

OUT=ROOT/'runs/coupler'


def finalize_frames(kind,na=20,nc=8,dt=.005):
    tag=f'programmed_{kind}_a{na}_c{nc}_dt{dt:g}'
    data=np.load(OUT/(tag+'_columns.npz'));out=data['output'];T=data['target']
    frames=observer_frames(out,na,nc)
    n=np.repeat(np.arange(na),2*nc);z=np.tile([1.,-1.],na*nc)
    corrected=out*np.exp(-1j*frames['cavity_R_rad']*n-1j*frames['qubit_Rz_rad']*z/2)[:,None]
    overlaps=np.sum(T.conj()*corrected,axis=0)
    frames.update(coherent_gate_fidelity=float(abs(overlaps.mean())**2),
                  worst_basis_fidelity=float(min(abs(overlaps)**2)))
    record=json.loads((OUT/(tag+'.json')).read_text())
    # The propagations arrive in six-column chunks. Recompute aggregate fields
    # from ALL cached columns rather than retaining the first chunk's list.
    population=np.sum(abs(out.reshape(na,nc,2,out.shape[1])[:,1:])**2,axis=(0,1,2))
    record['final_coupler_excitation']=population.tolist()
    record['max_final_coupler_excitation']=float(max(population))
    record['norm_error']=float(max(abs(np.sum(abs(out)**2,axis=0)-1)))
    if 'elapsed_seconds' in record:
        record['first_chunk_elapsed_seconds']=record.pop('elapsed_seconds')
    record.pop('amplitude_and_phase_calibrated_only_on_vacuum',None)
    record['strength_and_direction_calibrated_only_on_vacuum']=True
    record['observer_frame_calibration']=frames
    record['total_final_qubit_Rz_rad']=record['final_qubit_Rz_rad']+frames['qubit_Rz_rad']
    record['total_final_cavity_R_rad']=record['final_cavity_R_rad']+frames['cavity_R_rad']
    (OUT/(tag+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    np.savez_compressed(OUT/(tag+'_frame_corrected_columns.npz'),output=corrected,target=T)
    print('OBSERVABLE FRAME CALIBRATION',kind,frames,flush=True)
    return record


def chunk(job):
    kind,start,na,nc,dt=job
    record=json.loads((OUT/f'vacuum_calibrated_{kind}.json').read_text())
    last=record['vacuum_calibration_history'][-1]
    assert abs(complex(last['conditional_beta_real'],last['conditional_beta_imag'])-.4)<.002
    words=programmed_words(record)
    allinputs=[(q,n) for q in range(2) for n in range(9)]
    result,out=microscopic(words,Circuit(**record['reference_circuit']),na,nc,dt,
                           inputs=allinputs[start:start+6])
    out=in_calibrated_frame(out,record,na,nc,result['duration_ns'])
    print('MICROSCOPIC COLUMN CHUNK',kind,'columns',start,start+6,'basis',na,nc,'dt',dt,
          'Pcoupler',result['max_final_coupler_excitation'],'seconds',result['elapsed_seconds'],flush=True)
    return kind,start,result,out


def main(na=20,nc=8,dt=.005):
    jobs=[(kind,start,na,nc,dt) for kind in ['constant','cartesian'] for start in [0,6,12]]
    print('START microscopic all18input verification AFTER vacuum-only calibration',len(jobs),'chunks',flush=True)
    rows=[]
    with ProcessPoolExecutor(4) as pool:
        for item in pool.map(chunk,jobs):rows.append(item)
    reports=[]
    for kind in ['constant','cartesian']:
        pieces=sorted([r for r in rows if r[0]==kind],key=lambda r:r[1]);assert len(pieces)==3
        out=np.concatenate([p[3] for p in pieces],axis=1)
        T=cd.target_columns(cd.Space(na,9),'cd',.4)
        desired=np.zeros_like(out);desired.reshape(na,nc,2,18)[:,0]=T.reshape(na,2,18)
        overlaps=np.sum(desired.conj()*out,axis=0)
        r=pieces[0][2].copy()
        r.update(kind=kind,beta=.4,columns=18,complex_amplitudes=out.size,
                 coherent_gate_fidelity=float(abs(overlaps.mean())**2),
                 worst_basis_fidelity=float(min(abs(overlaps)**2)),
                 max_final_coupler_excitation=max(p[2]['max_final_coupler_excitation'] for p in pieces),
                 elapsed_worker_seconds=sum(p[2]['elapsed_seconds'] for p in pieces),
                 amplitude_and_phase_calibrated_only_on_vacuum=True)
        calibration=json.loads((OUT/f'vacuum_calibrated_{kind}.json').read_text())
        duration=r['duration_ns'];dq=calibration['qubit_IQ_detune_GHz']
        da=dq+calibration['pump_IQ_detune_GHz']
        r.update(final_qubit_Rz_rad=2*np.pi*dq*duration,final_cavity_R_rad=-2*np.pi*da*duration)
        words=programmed_words(calibration)
        r['word_sha256']=hashlib.sha256(words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest()
        r['compiled']=compile_words(words,OUT/f'programmed_{kind}_compiled')
        tag=f'programmed_{kind}_a{na}_c{nc}_dt{dt:g}'
        np.savez_compressed(OUT/(tag+'_columns.npz'),output=out,target=desired)
        np.savez_compressed(OUT/(tag+'_words.npz'),**words)
        (OUT/(tag+'.json')).write_text(json.dumps(r,indent=2)+'\n');reports.append(r)
        print('ALL-INPUT MICROSCOPIC RESULT',kind,'F',r['coherent_gate_fidelity'],'worstbasis',r['worst_basis_fidelity'],
              'Pcoupler',r['max_final_coupler_excitation'],'amplitudes',out.size,flush=True)
    reports=[finalize_frames(kind,na,nc,dt) for kind in ['constant','cartesian']]
    manifest=dict(gates=2,columns=36,amplitudes=sum(r['complex_amplitudes'] for r in reports),
                  results=reports,source_sha256={name:hashlib.sha256((CODE / name).read_bytes()).hexdigest() for name in
                    ['coupler_phase_model.py','coupler_replay.cpp','calibrate_coupler.py','calibrate_programmed_family.py','verify_programmed_family.py']})
    (OUT/f'programmed_manifest_a{na}_c{nc}_dt{dt:g}.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FINAL PROGRAMMED MICROSCOPIC COUNTS',2,'gates;',36,'inputcolumns;',manifest['amplitudes'],'amplitudes',flush=True)


if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='frames':
        for kind in ['constant','cartesian']:finalize_frames(kind)
        raise SystemExit(0)
    main(int(sys.argv[1]) if len(sys.argv)>1 else 20,
         int(sys.argv[2]) if len(sys.argv)>2 else 8,
         float(sys.argv[3]) if len(sys.argv)>3 else .005)
