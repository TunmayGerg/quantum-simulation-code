"""Independent full-column certification of explicit research waveforms.

Input q,s are Hamiltonian coefficients at800MS/s. Pulse metadata must account
for every qubit sample. The target is fixed independently of the waveform.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from project_paths import ROOT,CODE
import cd_trains as cd
import hardware_check as hw
from audit_cd import compile_words
from verify_short_gates import arrays as legacy_arrays

OUT=ROOT/'runs/research_verified';OUT.mkdir(exist_ok=True)
MAX_G=np.pi/(2*.200)


def check_phase_arcs(meta,q,s):
    """Assert radius,10ns transitions and their programmed qubit-zero centers."""
    arcs=meta.get('phase_arcs');assert isinstance(arcs,list) and arcs
    phase=np.full(len(s),np.angle(s[4]));previous=-1
    for arc in sorted(arcs,key=lambda a:a['center_sample']):
        center=arc['center_sample'];assert int(center)==center
        assert arc['duration_ns']==10. and abs(q[center])<1e-10
        left,right=center-4,center+4
        assert left>=previous and left>=4 and right<len(s)-4
        begin,end=arc['start_phase_rad'],arc['end_phase_rad']
        assert abs(np.exp(1j*phase[left])-np.exp(1j*begin))<1e-10
        u=np.arange(9)/8;h=10*u**3-15*u**4+6*u**5
        assert arc['shape']=='quintic constant-radius'
        phase[left:right+1]=begin+(end-begin)*h;phase[right+1:]=end
        previous=right
    expected=abs(s)*np.exp(1j*phase)
    assert max(abs(expected-s))<1e-9,'Phase path is not the declared smooth circular IQ trajectory.'
    return len(arcs)


def reference(name,source):
    r=json.loads((ROOT/'runs/constant_pump_gaussian'/(source+'.json')).read_text())
    q,s,w,phi,psi,gap=legacy_arrays(r);qs,ss=hw.dac_scales(.6)
    qr=hw.sample_to_complex(w['qubit']).conj()/qs*(np.pi/cd.PI_TIME)
    sr=hw.sample_to_complex(w['sideband'])/ss*(np.pi/(2*.6))
    np.savez_compressed(OUT/(name+'_controls.npz'),q=qr,s=sr,qubit_words=w['qubit'],sideband_words=w['sideband'])
    meta=dict(name=name,gate=r['job']['gate'],strength=1. if r['job']['gate']=='cd' else cd.R_8DB,
              controls_file=str((OUT/(name+'_controls.npz')).relative_to(ROOT)),
              sideband_mode='fixed',qubit_frame_Rz=r['qubit_frame_Rz'],source=source,
              pulses=[dict(start_sample=4+j*(24+round(gap/cd.DT))+round(gap/(2*cd.DT)),
                           samples=24,angle_fraction=1.,phase_rad=float(p)) for j,p in enumerate(phi)])
    path=OUT/(name+'_recipe.json');path.write_text(json.dumps(meta,indent=2)+'\n')
    return path


def certify(path):
    path=Path(path);path=path if path.is_absolute() else ROOT/path
    meta=json.loads(path.read_text());name=meta['name']
    controls=Path(meta['controls_file']);controls=controls if controls.is_absolute() else ROOT/controls
    data=np.load(controls)
    q=np.asarray(data['q'] if 'q' in data else data['qubit'],complex)
    s=np.asarray(data['s'] if 's' in data else data['sideband'],complex)
    assert q.shape==s.shape and q.ndim==1 and len(q)%4==0
    assert meta['gate'] in ['cd','squeeze']
    assert abs(meta['strength']-(1. if meta['gate']=='cd' else cd.R_8DB))<1e-12
    assert len(meta['pulses'])>0
    expected=np.zeros_like(q);covered=np.zeros(len(q),bool)
    for pulse in meta['pulses']:
        start=pulse['start_sample'];count=pulse['samples'];fraction=pulse['angle_fraction']
        assert count>=24 and int(count)==count and 0<fraction<=1
        assert abs(1/fraction-round(1/fraction))<1e-8
        assert not np.any(covered[start:start+count])
        offset=.5 if pulse.get('centered_envelope',meta.get('centered_envelopes',False)) else 0.
        shape=np.pi/(count*cd.DT)*np.sin(np.pi*(np.arange(count)+offset)/count)**2*fraction
        if pulse.get('reversed_envelope',False):shape=shape[::-1]
        expected[start:start+count]=shape*np.exp(1j*pulse['phase_rad'])
        covered[start:start+count]=True
    assert max(abs(q-expected))<.009,'Unaccounted qubit waveform or wrong envelope.'
    mode=meta.get('sideband_mode','fixed')
    if mode=='fixed':assert max(abs(s[4:-4]-s[4]))<1e-10
    else:
        assert mode=='phase_arc' and meta.get('magnus_justification')
        assert max(abs(abs(s[4:-4])-abs(s[4])))<1e-9
        check_phase_arcs(meta,q,s)
    qs,ss=hw.dac_scales(.6)
    if 'qubit_words' in data and 'sideband_words' in data:
        words=dict(qubit=data['qubit_words'],sideband=data['sideband_words'])
        assert max(abs(q-hw.sample_to_complex(words['qubit']).conj()/qs*(np.pi/cd.PI_TIME)))<1e-12
        assert max(abs(s-hw.sample_to_complex(words['sideband'])/ss*(np.pi/(2*.6))))<1e-12
    else:
        qw,q=hw.to_words_and_back(q,np.pi/cd.PI_TIME,qs,qubit=True)
        sw,s=hw.to_words_and_back(s,np.pi/(2*.6),ss);words=dict(qubit=qw,sideband=sw)
    qf,sf=hw.rfdc_interpolate(q),hw.rfdc_interpolate(s)
    assert max(max(abs(s)),max(abs(sf)))<=MAX_G
    angle=meta.get('qubit_frame_Rz',0.);previous=None;history=[]
    for levels in [64,96,128,192,256,320,384,512]:
        space=cd.Space(levels,9);initial=np.eye(2*levels,dtype=complex)[:,space.inputs]
        T=cd.target_columns(space,meta['gate'],meta['strength'])
        frame=np.exp(-1j*angle*space.z/2)[:,None]
        V=frame*hw.propagate(qf,sf,initial,cd.DT/8,order=8,fast=True)
        row=dict(levels=levels)
        if previous is not None:
            old,oldT=previous;pad=np.zeros_like(V);pad[:len(old)]=old
            padT=np.zeros_like(T);padT[:len(oldT)]=oldT
            row.update(output_change=float(max(np.linalg.norm(V-pad,axis=0))),
                       target_change=float(max(np.linalg.norm(T-padT,axis=0))))
        history.append(row)
        if previous is not None and max(row['output_change'],row['target_change'])<1e-9:break
        previous=V,T
    else:raise AssertionError((name,'cutoff not converged'))
    independent=frame*hw.propagate(qf,sf,initial,cd.DT/8,order=16)
    error=float(max(abs(independent-V).ravel()));assert error<1e-10
    overlaps=np.sum(T.conj()*V,axis=0);F=float(abs(overlaps.mean())**2)
    result=dict(name=name,gate=meta['gate'],strength=meta['strength'],fidelity=F,
                worst_basis_fidelity=float(min(abs(overlaps)**2)),accepted=bool(F>=.99),
                sideband_mode=mode,command_us=len(q)*cd.DT,columns=18,levels=levels,
                complex_amplitudes=V.size,independent_max_difference=error,
                peak_g=float(max(abs(sf))),peak_swap_ns=float(1000*np.pi/(2*max(abs(sf)))),
                qubit_frame_Rz=angle,cutoff_history=history,recipe_file=str(path.relative_to(ROOT)),
                controls_file=str(controls.relative_to(ROOT)),
                word_sha256=hashlib.sha256(words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest())
    result['compiled']=compile_words(words,OUT/(name+'_compiled'))
    np.savez_compressed(OUT/(name+'_words.npz'),**words)
    np.savez_compressed(OUT/(name+'_columns.npz'),output=V,target=T)
    with (OUT/(name+'_pulses.csv')).open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['pulse','start_ns','duration_ns','rotation_pi_units','qubit_DAC_deg','reversed_envelope'])
        for j,pulse in enumerate(meta['pulses'],1):
            writer.writerow([j,pulse['start_sample']*1.25,pulse['samples']*1.25,pulse['angle_fraction'],
                             -np.degrees(pulse['phase_rad'])%360,pulse.get('reversed_envelope',False)])
    (OUT/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('RESEARCH CERTIFIED',name,'F',F,'worstbasis',result['worst_basis_fidelity'],
          'timeus',result['command_us'],'18columns;',V.size,'amplitudes;error',error,
          'mode',mode,'accepted',result['accepted'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('recipes',nargs='*');parser.add_argument('--references',action='store_true')
    args=parser.parse_args()
    paths=args.recipes
    if args.references:
        paths=[reference('cd_constant_reference','cd_binary_scaled_b1'),
               reference('squeeze_constant_reference','squeeze_compression_07')]
    assert paths
    print('START RESEARCH CERTIFICATION',len(paths),'gates;all18inputs',flush=True)
    results=[certify(p) for p in paths]
    allresults=[json.loads(p.read_text()) for p in OUT.glob('*.json') if not p.name.endswith('_recipe.json') and p.name!='manifest.json']
    allresults=[r for r in allresults if 'columns' in r]
    manifest=dict(gates=len(allresults),columns=sum(r['columns'] for r in allresults),
                  independent_amplitudes=sum(r['complex_amplitudes'] for r in allresults),results=allresults)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FINAL RESEARCH COMPARISONS',len(results),'gates;',18*len(results),'columns;',
          sum(r['complex_amplitudes'] for r in results),'amplitudes',flush=True)
