"""Independent numerical and SDK checks for the current two CD notebooks."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import hashlib,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.linalg import expm
import cd_trains as cd
import hardware_check as hw


OUT=ROOT/'runs/audit'


def compile_words(words,directory):
    """Two ordinary SDK arbitrary waveforms, one trigger, no board attach/run."""
    from acadia import Acadia
    from acadia_qmsmt.qmsmt import InputOutput
    from acadia.sample_arithmetic import sample_to_complex
    program=json.loads(hw.PROGRAM.read_text());a=Acadia();channels=[]
    for channel,key,pulsekey in [('DAC7','qubit','idle_pi'),('DAC10','sideband','pump')]:
        name=program[pulsekey]['name'];data=sample_to_complex(words[key])
        config=dict(channel=channel,pulses={name:dict(data=data,scale=1.,phase=None,
                                                    detune=None,use_stretch=False)})
        io=InputOutput(channel,a,config);io.get_waveform_memory(name)
        channels.append((io,name,len(data)))
    def sequence(acadia):
        with acadia.channel_synchronizer():
            for io,name,count in channels:io.schedule_pulse(name)
    directory.mkdir(exist_ok=True,parents=True)
    a.compile(sequence,output_directory=str(directory))
    log=(directory/'compiled.log').read_text()
    dma=[line for line in log.splitlines() if 'Command DMA for DAC' in line]
    assert len(dma)==2 and all('type 1:' in line for line in dma)
    assert log.count('Trigger DMAs')==1
    for io,name,count in channels:
        mem=io.get_waveform_memory(name)
        assert mem.nbytes==count*4
        assert (a.waveform_dma_command(mem)&65535)+1==count//4
    return dict(arbitrary_commands=2,triggers=1,int16_values_compared=sum(w.size for w in words.values()))


def fundamentals():
    rng=np.random.default_rng(1729);levels=16;space=cd.Space(levels,9)
    # Construct the physical Hamiltonian independently, with tensor products.
    a=np.diag(np.sqrt(np.arange(1,levels)),1)
    sp=np.array([[0,0],[1,0]],complex);eye=np.eye(levels)
    target_checks=0
    for gate,strength,generator in [
        ('squeeze',cd.R_8DB,np.kron((a@a-a.T@a.T)/2,np.diag([1,-1]))),
        ('rotation',np.pi/2,-1j*np.kron(np.diag(np.arange(levels)),np.diag([1,-1]))),
        ('cd',1.,np.kron(a.T-a,np.array([[0,1],[1,0]])))]:
        expected=expm(strength*generator)[:,space.inputs]
        actual=cd.target_columns(space,gate,strength)
        assert np.max(abs(actual-expected))<1e-12
        target_checks+=actual.size
    tr=cd.Train('squeeze',.3,.3,8,.0175,levels)
    x=rng.uniform(-np.pi,np.pi,16)
    out,_=tr.run(x)
    _,q,s=cd.waveform(tr,x)
    independent=hw.propagate(q,s,tr.inputs,cd.DT,order=16)
    diff=float(np.max(abs(independent-out)));assert diff<1e-11
    gradient_checks=0;largest=0.
    for kind,motif in [('free',None),('palindrome',None),('motif',4),('qubit only',None)]:
        A=cd.structure(8,kind,motif);p=rng.normal(size=A.shape[1])
        _,g=tr.infidelity(A@p)
        for _ in range(3):
            v=rng.normal(size=len(p));v/=np.linalg.norm(v);epsilon=1e-5
            fd=(tr.infidelity(A@(p+epsilon*v),False)-tr.infidelity(A@(p-epsilon*v),False))/(2*epsilon)
            error=abs(fd-g@A@v);largest=max(largest,error)
            assert error<1e-8;gradient_checks+=1
    result=dict(independent_target_amplitudes=target_checks,
                cell_vs_direct_amplitudes=out.size,cell_vs_direct_max_difference=diff,
                gradient_directions=gradient_checks,gradient_max_difference=largest)
    print('FUNDAMENTALS',result,flush=True)
    return result


def verify_one(job):
    group,index,r=job;j=r['job'];gate=j['gate'];n=j.get('n_input',9)
    strength=j.get('strength',cd.R_8DB if gate=='squeeze' else np.pi/2)
    h,out,target=hw.check(r,start_levels=128 if gate=='squeeze' else 64,step=32,outputs=True)
    assert abs(h['fidelity']-r['hardware_check']['fidelity'])<1e-9
    tr=cd.Train(gate,strength,j['swap'],j['cells'],j['gap'],16)
    q,s,words=hw.played_arrays(tr,np.r_[r['qubit_phases'],r['pump_phases']])
    sp=cd.Space(h['levels'],n);initial=np.eye(2*h['levels'],dtype=complex)[:,sp.inputs]
    independent=hw.propagate(q,s,initial,cd.DT/8,order=16)
    independent*=np.exp(-1j*r['qubit_frame_Rz']*sp.z/2)[:,None]
    maxdiff=float(np.max(abs(independent-out)));assert maxdiff<1e-10
    assert h['fidelity']>.99
    compiled=compile_words(words,OUT/f'{group}_{index}_compiled')
    np.savez_compressed(ROOT/'runs'/r['words_file'],**words)
    np.savez_compressed(OUT/f'{group}_{index}_columns.npz',output=out,target=target)
    h.pop('words');h.update(group=group,index=index,independent_max_difference=maxdiff,
                           compiled=compiled,qubit_DAC_phase_sign=-1,
                           word_sha256=hashlib.sha256(words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest())
    return h


def main():
    OUT.mkdir(exist_ok=True,parents=True);basic=fundamentals();jobs=[]
    for group in ['final','cd_final']:
        jobs.extend((group,i,r) for i,r in enumerate(json.loads((ROOT/'runs'/f'{group}.json').read_text())))
    assert len(jobs)==16
    rows=[]
    print('START AUDIT',len(jobs),'complete gates, four CPU workers',flush=True)
    with ProcessPoolExecutor(4) as pool:
        tasks={pool.submit(verify_one,job):job for job in jobs}
        for future in as_completed(tasks):
            r=future.result();rows.append(r)
            (OUT/f"{r['group']}_{r['index']}.json").write_text(json.dumps(r,indent=2)+'\n')
            print('AUDITED',len(rows),'/',len(jobs),r['group'],r['index'],'F',r['fidelity'],
                  'levels',r['levels'],'columns',r['columns'],'amplitudes',r['complex_amplitudes'],
                  'maxdiff',r['independent_max_difference'],flush=True)
    assert len(rows)==16
    for group in ['final','cd_final']:
        path=ROOT/'runs'/f'{group}.json';saved=json.loads(path.read_text())
        for r in rows:
            if r['group']==group:saved[r['index']]['hardware_check']=r
        path.write_text(json.dumps(saved,indent=2)+'\n')
    manifest=dict(gates=len(rows),columns=sum(r['columns'] for r in rows),
                  independent_amplitudes=sum(r['complex_amplitudes'] for r in rows),
                  int16_values=sum(r['compiled']['int16_values_compared'] for r in rows),
                  arbitrary_commands=sum(r['compiled']['arbitrary_commands'] for r in rows),
                  triggers=sum(r['compiled']['triggers'] for r in rows),fundamentals=basic,results=rows,
                  source_sha256={name:hashlib.sha256((CODE / name).read_bytes()).hexdigest() for name in
                    ['cd_trains.py','hardware_check.py','native_waveform_check.cpp','fast_waveform_replay.cpp','audit_cd.py']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FINAL AUDIT COUNTS',manifest['gates'],'gates;',manifest['columns'],'columns;',
          manifest['independent_amplitudes'],'independent amplitudes;',manifest['int16_values'],'int16 entries',flush=True)


if __name__=='__main__':main()
