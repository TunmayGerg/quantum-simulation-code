"""Actual-playback strength sensitivity of the selected, fixed pulse sequences.

These curves hold phases, durations and the declared final qubit frame fixed.
They are calibration/sensitivity curves, not globally reoptimized envelopes.
The separately saved fixed-strength CD optimization grid answers the latter
question within its explicitly bounded local search.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import copy,json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cd_trains as cd
import hardware_check as hw
from verify_short_gates import arrays,OUT,VERIFIED


def curve(name):
    record=json.loads((OUT/(name+'.json')).read_text())
    certificate=json.loads((VERIFIED/(name+'.json')).read_text())
    shaped='g' in record and 'RFDC_window_us' in record
    nominal=(record['equivalent_swap_ns'] if shaped else 1000*record['job']['swap'])
    grid=np.unique(np.r_[[100,125,150,175,200,225,250,275,300,350,400,500,600],
                         nominal*(1+np.linspace(-.04,.04,9))])
    rows=[]
    for swap_ns in grid:
        candidate=copy.deepcopy(record)
        if shaped:candidate['parameters'][-1]=float(np.log(np.pi/(2*swap_ns*.001)))
        else:candidate['job']['swap']=float(swap_ns*.001)
        q,s,_,_,_,_=arrays(candidate)
        prior=None;levels=certificate['levels']
        for attempt in range(5):
            space=cd.Space(levels,9)
            initial=np.eye(2*levels,dtype=complex)[:,space.inputs]
            T=cd.target_columns(space,certificate['gate'],certificate['strength'])
            output=hw.propagate(q,s,initial,cd.DT/8,order=8,fast=True)
            output*=np.exp(-1j*certificate['qubit_frame_Rz']*space.z/2)[:,None]
            if prior is not None:
                old,oldT=prior;embedded=np.zeros_like(output);embedded[:len(old)]=old
                embeddedT=np.zeros_like(T);embeddedT[:len(oldT)]=oldT
                change=float(max(np.linalg.norm(output-embedded,axis=0)))
                target_change=float(max(np.linalg.norm(T-embeddedT,axis=0)))
                if max(change,target_change)<1e-8:break
            prior=output,T;levels+=32 if levels<192 else 64
        else:raise AssertionError((name,swap_ns,'curve cutoff did not converge'))
        F=float(abs(np.vdot(T,output)/18)**2)
        rows.append(dict(swap_ns=float(swap_ns),fidelity=F,levels=levels,
                         output_change=change,target_change=target_change,columns=18,
                         amplitudes=output.size,command_us=certificate['command_us']))
    result=dict(name=name,gate=certificate['gate'],nominal_swap_ns=nominal,
                fixed_sequence=True,fixed_final_frame=True,results=rows)
    (OUT/(name+'_strength_curve.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('STRENGTH SENSITIVITY',name,len(rows),'strengths;',18*len(rows),'columns;',
          sum(r['amplitudes'] for r in rows),'amplitudes;maxcutoffchange',max(r['output_change'] for r in rows),flush=True)
    return result


def validate_map(filename):
    """Converge every plotted local-fit point, including the unsuccessful ones."""
    rows=json.loads((OUT/filename).read_text());amplitudes=0
    for record in rows:
        q,s,_,_,_,_=arrays(record);gate=record['job']['gate']
        strength=dict(cd=1.,squeeze=cd.R_8DB,rotation=np.pi/2)[gate]
        previous=None;history=[]
        levels=256 if gate=='squeeze' else 96
        for attempt in range(5):
            sp=cd.Space(levels,9);initial=np.eye(2*levels,dtype=complex)[:,sp.inputs]
            target=cd.target_columns(sp,gate,strength)
            output=hw.propagate(q,s,initial,cd.DT/8,order=8,fast=True)
            output*=np.exp(-1j*record['qubit_frame_Rz']*sp.z/2)[:,None]
            entry=dict(levels=levels)
            if previous is not None:
                old,T=previous;embedded=np.zeros_like(output);embedded[:len(old)]=old
                embeddedT=np.zeros_like(target);embeddedT[:len(T)]=T
                entry.update(output_change=float(max(np.linalg.norm(output-embedded,axis=0))),
                             target_change=float(max(np.linalg.norm(target-embeddedT,axis=0))))
            history.append(entry)
            if previous is not None and max(entry['output_change'],entry['target_change'])<1e-8:break
            previous=output,target;levels+=64 if levels>=192 else 32
        else:raise AssertionError((record['job']['name'],'map cutoff not converged'))
        record['converged_fidelity']=float(abs(np.vdot(target,output)/18)**2)
        record['plot_cutoff_history']=history
        amplitudes+=output.size
    (OUT/filename).write_text(json.dumps(rows,indent=2)+'\n')
    print('VALIDATED STRENGTH MAP',filename,len(rows),'points;',18*len(rows),
          'columns;',amplitudes,'comparison amplitudes',flush=True)
    return dict(file=filename,points=len(rows),columns=18*len(rows),amplitudes=amplitudes)


if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='maps':
        names=['strength_played_summary.json','gaussian_strength_played_summary.json','shorter_played_summary.json']
        print('START cutoff validation of all local-fit map points,successfulANDfailed',flush=True)
        with ProcessPoolExecutor(3) as pool:result=list(pool.map(validate_map,names))
        assert len(result)==3
        (OUT/'strength_map_validation.json').write_text(json.dumps(result,indent=2)+'\n')
        print('FINAL MAP VALIDATION COUNTS',sum(r['points'] for r in result),'points;',
              sum(r['columns'] for r in result),'columns;',sum(r['amplitudes'] for r in result),'amplitudes',flush=True)
        raise SystemExit(0)
    names=['cd_binary_n12_0','played_cd_n12_pump1_seed0',
           'squeeze_compression_07','played_squeeze_smooth',
           'rotation_compression_10','played_rotation_smooth',
           'strength_squeeze_swap300','strength_rotation_swap150']
    print('START actualSDK/RFDC strength sensitivity;',len(names),'fixedschemes;automaticcutoffs;all18inputs',flush=True)
    with ProcessPoolExecutor(4) as pool:rows=list(pool.map(curve,names))
    assert len(rows)==len(names)
    manifest=dict(schemes=len(rows),strength_points=sum(len(r['results']) for r in rows),
                  columns=sum(18*len(r['results']) for r in rows),results=rows)
    (OUT/'strength_sensitivity.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FINAL SENSITIVITY COUNTS',len(rows),'schemes;',manifest['strength_points'],'points;',manifest['columns'],'columns',flush=True)
