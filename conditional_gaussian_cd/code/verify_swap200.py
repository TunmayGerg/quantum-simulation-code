"""Independent gate and peak-coupling checks under the 200 ns swap limit."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,csv,hashlib,json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cd_trains as cd
import hardware_check as hw
from audit_cd import compile_words
from capped_cd_study import OUT,ROOT,MAX_G,MIN_SWAP_NS,CappedCD
from verify_short_gates import arrays as original_arrays
from verify_higher_order_cd import make_model as old_model

V=OUT/'verified';V.mkdir(exist_ok=True)


def source_and_arrays(name):
    if name=='baseline_cd':
        path=ROOT/'runs/constant_pump_gaussian/cd_binary_scaled_b1.json'
        record=json.loads(path.read_text());q,s,words,_,_,_=original_arrays(record)
        qs,ss=hw.dac_scales(.6)
        qraw=hw.sample_to_complex(words['qubit']).conj()/qs*(np.pi/cd.PI_TIME)
        sraw=hw.sample_to_complex(words['sideband'])/ss*(np.pi/(2*.6))
        return record,(qraw,sraw,q,s,words),None
    if name=='squeeze_8db':
        path=ROOT/'runs/higher_order_cd/compressed_squeeze_n120.json'
        record=json.loads(path.read_text());model=old_model(record)
        return record,model.arrays(np.asarray(record['parameters'])),model
    record=json.loads((OUT/(name+'.json')).read_text());j=record['job']
    model=CappedCD(j['cells'],j['gap_ns'],j['phase_control'],16,j.get('edge_overlap',False))
    return record,model.arrays(np.asarray(record['parameters'])),model


def verify(name):
    record,(qraw,sraw,q,s,words),model=source_and_arrays(name)
    gate='squeeze' if name=='squeeze_8db' else 'cd'
    strength=cd.R_8DB if gate=='squeeze' else 1.
    angle=record.get('qubit_frame_Rz',0.)
    peak=float(max(max(abs(sraw)),max(abs(s))))
    assert peak<=MAX_G,(name,peak,MAX_G)
    previous=None;history=[]
    for levels in [64,96,128,192,256,320,384,512]:
        space=cd.Space(levels,9);initial=np.eye(2*levels,dtype=complex)[:,space.inputs]
        T=cd.target_columns(space,gate,strength)
        frame=np.exp(-1j*angle*space.z/2)[:,None]
        output=frame*hw.propagate(q,s,initial,cd.DT/8,order=8,fast=True)
        row=dict(levels=levels)
        if previous is not None:
            old,oldT=previous;pad=np.zeros_like(output);pad[:len(old)]=old
            padT=np.zeros_like(T);padT[:len(oldT)]=oldT
            row.update(output_change=float(max(np.linalg.norm(output-pad,axis=0))),
                       target_change=float(max(np.linalg.norm(T-padT,axis=0))))
        history.append(row)
        if previous is not None and max(row['output_change'],row['target_change'])<1e-9:break
        previous=output,T
    else:raise AssertionError((name,'cutoff failed'))
    independent=frame*hw.propagate(q,s,initial,cd.DT/8,order=16)
    error=float(np.max(abs(independent-output)));assert error<1e-10
    overlaps=np.sum(T.conj()*output,axis=0);F=float(abs(overlaps.mean())**2)
    ideal=frame*hw.propagate(np.repeat(qraw,8),np.repeat(sraw,8),initial,cd.DT/8,order=8,fast=True)
    result=dict(name=name,gate=gate,fidelity=F,accepted=bool(F>.99),hardware_eligible=True,
                ideal_device_fidelity=float(abs(np.vdot(T,ideal)/18)**2),
                command_us=len(qraw)*cd.DT,qubit_frame_Rz=angle,columns=18,levels=levels,
                independent_amplitudes=output.size,independent_max_difference=error,
                cutoff_history=history,hard_min_swap_ns=MIN_SWAP_NS,peak_g_rad_per_us=peak,
                actual_min_swap_ns=1000*np.pi/(2*peak),
                programmed_peak_g_rad_per_us=float(max(abs(sraw))),played_peak_g_rad_per_us=float(max(abs(s))),
                worst_basis_fidelity=float(min(abs(overlaps)**2)),
                pump_phase_constant=not record['job'].get('phase_control',name=='squeeze_8db'),
                word_sha256=hashlib.sha256(words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest())
    if result['accepted']:
        np.savez_compressed(V/(name+'_words.npz'),**words)
        np.savez_compressed(V/(name+'_columns.npz'),output=output,target=T)
        result['compiled']=compile_words(words,V/(name+'_compiled'))
        if name not in ['baseline_cd','squeeze_8db']:
            p=np.asarray(record['parameters']);n=model.cells;m=model.knots
            with (V/(name+'_sequence.csv')).open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['pulse','start_ns','duration_ns','qubit_DAC_deg'])
                for j in range(n):
                    start=np.flatnonzero(model.qcell==j)[0]
                    writer.writerow([j+1,start*1.25,30.,-np.degrees(p[j])%360])
            with (V/(name+'_pump_knots.csv')).open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['knot','g_rad_per_us','pump_DAC_deg','transition_ns'])
                psi=p[n+m:] if record['job']['phase_control'] else np.zeros(m)
                for j in range(m):writer.writerow([j+1,p[n+j],(-90+np.degrees(psi[j]))%360,10])
    (V/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('VERIFIED200',name,'F',F,'Tns',result['command_us']*1000,
          'minswapns',result['actual_min_swap_ns'],'18columns;',output.size,'amplitudes;',
          'independenterror',error,'accepted',result['accepted'],flush=True)
    return result


def aligned_end_checks():
    """Evaluate at the actual envelope end after removing common group delay."""
    rows=[];delay=np.argmax(abs(hw.rfdc_interpolate(np.array([1.]))))*cd.DT/8
    names=['edge_n10_gap0_phase1','cd_n10_gap0_phase0','cd_n12_gap0_phase0','squeeze_8db']
    for name in names:
        record,(qr,sr,q,s,w),model=source_and_arrays(name)
        cert=json.loads((V/(name+'.json')).read_text());space=cd.Space(cert['levels'],9)
        initial=np.eye(2*space.levels,dtype=complex)[:,space.inputs]
        stop=round((len(qr)*cd.DT+delay)/(cd.DT/8))
        output=hw.propagate(q[:stop],s[:stop],initial,cd.DT/8,order=16)
        output*=np.exp(-1j*cert['qubit_frame_Rz']*space.z/2)[:,None]
        T=cd.target_columns(space,cert['gate'],cd.R_8DB if cert['gate']=='squeeze' else 1.)
        fidelity=float(abs(np.vdot(T,output)/18)**2)
        full=np.load(V/(name+'_columns.npz'))['output']
        assert output.shape[1]==18 and fidelity>.99
        row=dict(name=name,aligned_device_end_ns=len(qr)*cd.DT*1000,
                 common_delay_removed_ns=delay*1000,fidelity_at_aligned_end=fidelity,
                 columns=18,amplitudes=output.size,
                 remaining_tail_input_operator_norm=float(np.linalg.norm(full-output,2)))
        rows.append(row);print('ALIGNED DEVICE END',row,flush=True)
    assert len(rows)==4
    (OUT/'aligned_end_checks.json').write_text(json.dumps(rows,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('names',nargs='*');parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--aligned-end',action='store_true')
    args=parser.parse_args();names=args.names
    if args.aligned_end:
        aligned_end_checks();return
    if not names:
        for filename in ['summary.json','edge_summary.json']:
            if (OUT/filename).exists():names.extend(r['job']['name'] for r in json.loads((OUT/filename).read_text()))
        names+=['baseline_cd','squeeze_8db']
    assert names and len(names)==len(set(names))
    print('START200CERTIFICATION',len(names),'gates;all18inputs;peaknormbound',MAX_G,flush=True)
    with ProcessPoolExecutor(args.workers) as pool:rows=list(pool.map(verify,names))
    assert len(rows)==len(names)
    allrows=[json.loads(p.read_text()) for p in V.glob('*.json') if p.name!='manifest.json']
    feasible=[r for r in allrows if r['accepted'] and r['hardware_eligible']]
    selected={g:min((r for r in feasible if r['gate']==g),key=lambda r:(r['command_us'],-r['fidelity']))['name']
              for g in ['cd','squeeze'] if any(r['gate']==g for r in feasible)}
    manifest=dict(records=len(allrows),columns=18*len(allrows),
                  independent_amplitudes=sum(r['independent_amplitudes'] for r in allrows),
                  minimum_swap_ns=MIN_SWAP_NS,maximum_g_rad_per_us=MAX_G,selected=selected,results=allrows)
    (V/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FINAL200COMPARISONS',len(rows),'records;',18*len(rows),'columns;',
          sum(r['independent_amplitudes'] for r in rows),'amplitudes',flush=True)
    if all((V/(name+'.json')).exists() for name in ['edge_n10_gap0_phase1','cd_n10_gap0_phase0','cd_n12_gap0_phase0','squeeze_8db']):
        aligned_end_checks()


if __name__=='__main__':main()
