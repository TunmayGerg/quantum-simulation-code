"""Certify short gates on all18inputs, automatic cutoffs and independent replay."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,csv,hashlib,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import cd_trains as cd
import hardware_check as hw
from audit_cd import compile_words
from shaped_pulse_optimisation import PlayedGate,ROOT,OUT

VERIFIED=OUT/'verified';VERIFIED.mkdir(exist_ok=True)


def arrays(record):
    job=record['job'];n=job['cells']
    if 'RFDC_window_us' in record and 'parameters' in record and 'g' in record:
        model=PlayedGate(job['gate'],n,16,job.get('pump_control',True),job.get('frame',True),job.get('angle_fractions'))
        _,_,q,s,words=model.arrays(np.asarray(record['parameters']))
        phases=np.asarray(record['parameters'])[:n]
        pump=(np.asarray(record['parameters'])[n:2*n] if job.get('pump_control',True)
              else np.full(n,-np.pi/2 if job['gate']=='cd' else 0.))
        gap=0.
    else:
        train=cd.Train(job['gate'],job.get('strength',dict(cd=1.,squeeze=cd.R_8DB,rotation=np.pi/2)[job['gate']]),
                       job['swap'],n,job.get('gap',0.),16)
        phases=np.asarray(record['qubit_phases']);pump=np.asarray(record['pump_phases'])
        _,q,s=cd.waveform(train,np.r_[phases,pump])
        assert np.max(abs(np.exp(1j*pump)-np.exp(1j*pump[0])))<1e-12
        s=abs(s)*np.exp(1j*pump[0]) # same phase at turn-on, plateau and turn-off
        qs,ss=hw.dac_scales(.6)
        qw,q=hw.to_words_and_back(q,np.pi/cd.PI_TIME,qs,qubit=True)
        sw,s=hw.to_words_and_back(s,np.pi/(2*.6),ss)
        words=dict(qubit=qw,sideband=sw);q=hw.rfdc_interpolate(q);s=hw.rfdc_interpolate(s)
        gap=job.get('gap',0.)
    return q,s,words,phases,pump,gap


def verify(name):
    record=json.loads((OUT/(name+'.json')).read_text());job=record['job']
    q,s,words,phases,pump,gap=arrays(record);previous=None;history=[]
    strength=job.get('strength',dict(cd=1.,squeeze=cd.R_8DB,rotation=np.pi/2)[job['gate']])
    for levels in [64,96,128,160,192,224,256,320,384,512]:
        space=cd.Space(levels,9);initial=np.eye(2*levels,dtype=complex)[:,space.inputs]
        target=cd.target_columns(space,job['gate'],strength)
        raw=hw.propagate(q,s,initial,cd.DT/8,order=8,fast=True)
        frame=np.exp(-1j*record['qubit_frame_Rz']*space.z/2)[:,None]
        output=frame*raw;overlaps=np.sum(target.conj()*output,axis=0)
        F=float(abs(overlaps.mean())**2);entry=dict(levels=levels,fidelity=F)
        if previous is not None:
            old,T=previous;embedded=np.zeros_like(output);embedded[:len(old)]=old
            embeddedT=np.zeros_like(target);embeddedT[:len(T)]=T
            entry.update(output_change=float(max(np.linalg.norm(output-embedded,axis=0))),
                         target_change=float(max(np.linalg.norm(target-embeddedT,axis=0))))
        history.append(entry)
        if previous is not None and max(entry['output_change'],entry['target_change'])<1e-9:break
        previous=output,target
    else:raise AssertionError((name,'cutoff failed'))
    independent=frame*hw.propagate(q,s,initial,cd.DT/8,order=16)
    error=float(np.max(abs(independent-output)))
    assert error<1e-10 and F>.99,(name,F,error)
    if not job.get('pump_control',False):
        assert np.all(words['sideband'][:,0 if job['gate']=='cd' else 1]==0)
    compiled=compile_words(words,VERIFIED/(name+'_compiled'))
    np.savez_compressed(VERIFIED/(name+'_words.npz'),**words)
    np.savez_compressed(VERIFIED/(name+'_columns.npz'),output=output,target=target)
    with (VERIFIED/(name+'_sequence.csv')).open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['pulse','pi_duration_ns','half_gap_ns','qubit_DAC_deg','pump_plateau_DAC_deg'])
        for j,(phi,psi) in enumerate(zip(phases,pump),1):writer.writerow([j,30,gap*500,np.degrees(-phi)%360,np.degrees(psi)%360])
    result=dict(name=name,gate=job['gate'],strength=strength,fidelity=F,
                worst_basis_fidelity=float(min(abs(overlaps)**2)),columns=18,levels=levels,
                complex_amplitudes=output.size,independent_max_difference=error,
                command_us=len(words['qubit'])*cd.DT,RFDC_window_us=len(q)*cd.DT/8,
                qubit_frame_Rz=record['qubit_frame_Rz'],cutoff_history=history,compiled=compiled,
                pump_phase_constant=not job.get('pump_control',False),
                pump_transition_ns=10 if job.get('pump_control',False) else 0,
                word_sha256=hashlib.sha256(words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest(),
                source_record=name+'.json')
    g=float(np.exp(record['parameters'][-1])) if 'g' in record and 'RFDC_window_us' in record else float(np.pi/(2*job['swap']))
    qs,ss=hw.dac_scales(.6)
    result.update(g_rad_per_us=g,equivalent_swap_ns=1000*np.pi/(2*g),
                  qubit_pi_DAC_scale=qs,pump_plateau_DAC_scale=ss*g/(np.pi/(2*.6)),
                  programmed_NCO_changes=False)
    (VERIFIED/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('VERIFIED SHORT GATE',name,'F',F,'columns18;amplitudes',output.size,'cutoff',levels,
          'independenterror',error,'commandus',result['command_us'],'RFDCwindowus',result['RFDC_window_us'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('names',nargs='*');parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--family',choices=['binary'])
    args=parser.parse_args()
    if args.family:
        assert not args.names,'Choose explicit records or one complete family.'
        args.names=[r['job']['name'] for r in json.loads((OUT/'binary_rule_summary.json').read_text())['results']]
        assert len(args.names)==20
    assert args.names,'Supply record names or --family binary.'
    print('START SHORT GATE CERTIFICATION',len(args.names),'gates;all18inputs',flush=True)
    with ProcessPoolExecutor(args.workers) as pool:rows=list(pool.map(verify,args.names))
    assert len(rows)==len(args.names)
    allrows=[json.loads(p.read_text()) for p in VERIFIED.glob('*.json') if p.name!='manifest.json']
    manifest=dict(gates=len(allrows),columns=18*len(allrows),independent_amplitudes=sum(r['complex_amplitudes'] for r in allrows),
                  results=allrows,source_sha256={name:hashlib.sha256((CODE / name).read_bytes()).hexdigest() for name in
                  ['verify_short_gates.py','shaped_pulse_optimisation.py','waveform_adjoint.cpp','constant_pump_gaussian.py']})
    (VERIFIED/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FINAL SHORT CERTIFICATION COUNTS',len(rows),'newgates;',18*len(rows),'columns;',sum(r['complex_amplitudes'] for r in rows),'independentamplitudes',flush=True)
