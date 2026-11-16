"""Validate the rounded reusable eight-pulse motif at held-out CD strengths."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,csv,hashlib
from pathlib import Path
import numpy as np
import cd_trains as cd
import hardware_check as hw
from scaled_cd import motif_score,motif_waveform,target
from audit_cd import compile_words


FOLDER=ROOT/'runs/scalable_cd'


def rule(beta):
    """Round up the motif count, then scale ONLY the pump amplitude.

    At beta>2, use quadratically more motifs to lower the local sideband
    strength as the intermediate photon population grows. This is a tested
    conservative heuristic, not a universal order or error bound.
    beta=0 means skip the gate entirely.
    """
    if beta==0:return 0,0.
    size=abs(beta)
    repeats=int(np.ceil(size/.4-1e-12))
    if size>2:repeats=max(repeats,int(np.ceil(5*(size/2)**2-1e-12)))
    return repeats,beta/(.4*repeats)


def verify(beta,parameters):
    repeats,gain=rule(beta);assert repeats>0 and abs(gain)<=1+1e-12
    history=[];previous=None
    for levels in [64,96,128,192,256,384]:
        score,out,T,words=motif_score(parameters,repeats,gain,levels,True,frame=0.,beta_unit=.4)
        entry=dict(levels=levels,fidelity=score['fidelity'],columns=22)
        if previous is not None:
            old,oldT=previous;embed=np.zeros_like(out);embed[:len(old)]=old
            embedT=np.zeros_like(T);embedT[:len(oldT)]=oldT
            entry['output_column_change']=float(np.max(np.linalg.norm(out-embed,axis=0)))
            entry['target_column_change']=float(np.max(np.linalg.norm(T-embedT,axis=0)))
        history.append(entry)
        if previous is not None and max(entry['output_column_change'],entry['target_column_change'])<1e-9:break
        previous=out,T
    else:raise AssertionError('cutoff did not converge')
    q,s,_=motif_waveform(parameters,repeats,gain)
    space,initial,_=target(beta,levels)
    independent=hw.propagate(q,s,initial,cd.DT/8,order=16)
    error=float(np.max(abs(independent-out)));assert error<1e-10
    assert score['fidelity']>.99,(beta,score)
    tag=f'beta_{beta:g}'
    np.savez_compressed(FOLDER/(tag+'_words.npz'),**words)
    np.savez_compressed(FOLDER/(tag+'_columns.npz'),output=out,target=T)
    score.update(cutoff_history=history,independent_max_difference=error,
                 word_sha256=hashlib.sha256(words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest(),
                 training_point=bool(beta in [.2,.4,1.,2.]),words_file=tag+'_words.npz')
    if beta in [.125,1.,2.,4.]:score['compiled']=compile_words(words,FOLDER/(tag+'_compiled'))
    (FOLDER/(tag+'.json')).write_text(json.dumps(score,indent=2)+'\n')
    print('SCALED VERIFIED beta',beta,'M',repeats,'s',gain,'F',score['fidelity'],
          'nmax8',score['fidelity_nmax8'],'levels',levels,'amplitudes',out.size,'maxdiff',error,flush=True)
    return score


def main():
    FOLDER.mkdir(exist_ok=True,parents=True)
    # Select the best actual-playback multi-strength fit, then round ALL phase
    # offsets to 0.1 degree and the common gain to six decimals. These rounded
    # values themselves, not the unrounded optimiser score, are certified.
    fits=[json.loads(p.read_text()) for p in (ROOT/'runs/audit').glob('scalable_motif_unit0.4_seed*.json')]
    assert len(fits)==2
    best=min(fits,key=lambda r:r['objective']);p=np.asarray(best['parameters'])
    p[:8]=np.radians(np.round(np.degrees(p[:8]),1));p[8]=round(p[8],6)
    phases=np.r_[np.array([0.,0.,np.pi,np.pi])+p[:4]]
    qubit=np.r_[phases,phases[::-1]+np.pi]
    phalf=-np.pi/2+p[4:8];pump=np.r_[phalf,phalf[::-1]]
    with (FOLDER/'motif.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['pulse','qubit_DAC_phase_deg','pump_DAC_phase_deg','duration_ns'])
        writer.writeheader()
        for j,(q,s) in enumerate(zip(qubit,pump)):
            writer.writerow(dict(pulse=j+1,qubit_DAC_phase_deg=float(np.degrees(-q)%360),
                                 pump_DAC_phase_deg=float(np.degrees(s)%360),duration_ns=30))
    betas=[-.5,-1.,-2.,.05,.1,.125,.2,.3,.4,.5,.75,1.,1.25,1.5,1.75,2.,2.25,2.5,3.,4.]
    print('START REUSABLE MOTIF VERIFICATION',len(betas),'strengths, all22inputs; no fitted final frame',flush=True)
    results=[verify(beta,p) for beta in betas]
    assert len(results)==20
    baseline=json.loads((ROOT/'runs/audit/matched_baseline.json').read_text())
    for beta in [1.,2.]:
        old=next(r for r in baseline if r['swap_ns']==600 and r['beta']==beta)
        new=next(r for r in results if r['beta']==beta)
        assert new['command_us']<old['command_us'] and new['fidelity']>=old['fidelity']
    sources=['scaled_cd.py','verify_scaled_cd.py','hardware_check.py','cd_trains.py','fast_waveform_replay.cpp','native_waveform_check.cpp']
    report=dict(parameters=p.tolist(),beta_unit=.4,peak_pump_DAC_scale=.09*p[8],
                equivalent_swap_ns=300/p[8],final_Rz_rad=0.,motif_pulses=8,
                fit_seed=best['seed'],fit_training_betas=[.2,.4,1.,2.],
                tested_strengths=betas,gates=len(results),columns=22*len(results),
                independent_amplitudes=sum(r['complex_amplitudes'] for r in results),
                results=results,source_sha256={name:hashlib.sha256((CODE / name).read_bytes()).hexdigest() for name in sources})
    (FOLDER/'recipe.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL REUSABLE MOTIF COUNTS',len(results),'gates;',report['columns'],'columns;',
          report['independent_amplitudes'],'independent amplitudes; minimum F',min(r['fidelity'] for r in results),flush=True)


if __name__=='__main__':main()
