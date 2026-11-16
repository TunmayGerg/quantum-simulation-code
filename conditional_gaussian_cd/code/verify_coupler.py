"""Independent integrator, cutoff and timestep checks of the cosine coupler."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import cosm,sinm
from dataclasses import replace
from coupler_phase_model import *

OUT=ROOT/'runs/coupler';OUT.mkdir(exist_ok=True,parents=True)


def independent():
    circuit=Circuit();na=4;nc=3;inputs=[(0,0),(1,0),(0,1),(1,1)]
    words=boundary_words(delta=np.pi/3,shape='zero')
    # Add one calibrated 30ns pi pulse as an independent qubit-carrier check.
    q=np.r_[np.zeros(4),cd.pi_samples()*np.exp(.37j),np.zeros(20)]
    qs,_=hw.dac_scales(.3)
    words['qubit'],_=hw.to_words_and_back(q,np.pi/cd.PI_TIME,qs,qubit=True)
    p,C,E,discard=operators(circuit,na,nc)
    # Independent large-matrix cosine, projected AFTER evaluating the matrix
    # function; it must match the analytic infinite-oscillator Laguerre formula.
    biga,bigc=na+12,nc+12
    a=np.diag(np.sqrt(np.arange(1,biga)),1);c=np.diag(np.sqrt(np.arange(1,bigc)),1)
    phase=np.kron(p['lambda_a']*(a+a.T),np.eye(bigc))+np.kron(np.eye(biga),p['lambda_c']*(c+c.T))
    l=p['lambda_q'];cq=np.exp(-l*l/2)*np.diag([1,1-l*l]);sq=np.exp(-l*l/2)*l*np.array([[0,1],[1,0]])
    selection=np.array([i*bigc+j for i in range(na) for j in range(nc)])
    cos_full=cosm(phase)[np.ix_(selection,selection)];sin_full=sinm(phase)[np.ix_(selection,selection)]
    full=np.kron(cos_full,cq)-np.kron(sin_full,sq)
    ground=np.exp(-(p['lambda_c']**2+p['lambda_a']**2+p['lambda_q']**2)/2)
    full-=ground*np.eye(len(full))
    matrix_error=float(np.max(abs(full-C.toarray())));assert matrix_error<1e-12
    pump=hw.rfdc_interpolate(sample_to_complex(words['sideband']))
    qubit=hw.rfdc_interpolate(sample_to_complex(words['qubit']))
    sample_dt=cd.DT*1000/8;T=len(pump)*sample_dt;times=np.arange(len(pump))*sample_dt
    initial=np.zeros((len(E),len(inputs)),complex)
    for col,(q,n) in enumerate(inputs):initial[(n*nc)*2+q,col]=1
    factor=4*np.pi*p['EJ_GHz'];qgain=np.pi/(cd.PI_TIME*1000)/qs
    def env(values,t):
        return np.interp(t,times,values.real,right=0)+1j*np.interp(t,times,values.imag,right=0)
    def rhs(t,y):
        state=y.reshape(initial.shape)
        phi=-p['flux_per_DAC']*np.real(env(pump,t)*np.exp(2j*np.pi*(circuit.f_cavity_GHz-circuit.f_qubit_GHz)*t))
        drive=2*qgain*np.real(env(qubit,t).conjugate()*np.exp(-2j*np.pi*circuit.f_qubit_GHz*t))
        return (-1j*(E[:,None]*state+factor*np.sin(phi)*(full@state)+drive*state[np.arange(len(E))^1])).ravel()
    start=time.monotonic()
    reference=solve_ivp(rhs,(0,T),initial.ravel(),method='DOP853',rtol=2e-11,atol=2e-13)
    assert reference.success
    exact=reference.y[:,-1].reshape(initial.shape)*np.exp(1j*E*T)[:,None]
    checks=[]
    for dt in [.01,.005,.0025]:
        result,out=microscopic(words,circuit,na,nc,dt,inputs)
        difference=float(np.max(abs(out-exact)))
        checks.append(dict(timestep_ns=dt,amplitudes=out.size,max_difference=difference))
        print('INDEPENDENT COSINE',checks[-1],flush=True)
    assert checks[-1]['max_difference']<2e-7
    record=dict(independent_cosine_matrix_elements=full.size,cosine_max_difference=matrix_error,
                DOP853_derivative_evaluations=reference.nfev,checks=checks,
                elapsed_seconds=time.monotonic()-start)
    (OUT/'independent_integrator.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def boundary(job):
    shape,delta=job;words=boundary_words(delta,shape);previous=None;history=[]
    for na,nc,dt in [(12,4,.005),(16,6,.005),(20,8,.005),(20,10,.005),(20,12,.005)]:
        result,out=microscopic(words,na=na,nc=nc,dt_ns=dt)
        change=None
        if previous is not None:
            old,olda,oldc=previous
            embed=np.zeros((na,nc,2,out.shape[1]),complex)
            embed[:olda,:oldc]=old.reshape(olda,oldc,2,out.shape[1])
            # One global phase is irrelevant; no per-input or number phases.
            flat=embed.reshape(out.shape);phase=np.angle(np.vdot(flat,out))
            change=float(np.max(np.linalg.norm(out*np.exp(-1j*phase)-flat,axis=0)))
        history.append(dict(cavity_levels=na,coupler_levels=nc,timestep_ns=dt,
                            coupler_excitation=result['max_final_coupler_excitation'],column_change=change))
        print('BOUNDARY',shape,'delta',delta,'cutoff',na,nc,'dt',dt,'Pexc',result['max_final_coupler_excitation'],
              'column_change',change,'seconds',result['elapsed_seconds'],flush=True)
        previous=out,na,nc
        if change is not None and change<1e-6:break
    result.update(shape=shape,delta_rad=delta,convergence_history=history)
    # Record the actual convergence achieved, never silently call it exact.
    result['converged_1e6']=bool(history[-1]['column_change']<1e-6)
    tag=f'{shape}_delta{delta:.6f}'
    np.savez_compressed(OUT/(tag+'_columns.npz'),output=out)
    (OUT/(tag+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def full_motif(job):
    shape,dt=job
    path=ROOT/'runs/audit'/f'smooth_{shape}.json'
    p=np.array(json.loads(path.read_text())['parameters'])
    words=smooth_motif(p,1,1.,shape)
    inputs=[(q,n) for q in range(2) for n in range(9)]
    result,out=microscopic(words,na=24,nc=6,dt_ns=dt,inputs=inputs)
    space=cd.Space(24,9);T=cd.target_columns(space,'cd',.4)
    desired=np.zeros_like(out);desired.reshape(24,6,2,18)[:,0]=T.reshape(24,2,18)
    overlaps=np.sum(desired.conj()*out,axis=0)
    result.update(shape=shape,beta=.4,coherent_gate_fidelity=float(abs(overlaps.mean())**2),
                  worst_basis_fidelity=float(min(abs(overlaps)**2)))
    tag=f'full_motif_{shape}_dt{dt:g}'
    np.savez_compressed(OUT/(tag+'_columns.npz'),output=out,target=desired)
    (OUT/(tag+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('FULL MICROSCOPIC MOTIF',shape,'dt',dt,'F',result['coherent_gate_fidelity'],
          'Pcoupler',result['max_final_coupler_excitation'],'columns',18,'seconds',result['elapsed_seconds'],flush=True)
    return result


def motif_chunk(job):
    shape,na,nc,dt,start,stop,calibrated=job
    source=(ROOT/'runs/scalable_cd/recipe.json' if shape=='abrupt' else
            ROOT/'runs/audit'/(f'transition_fit_constant.json' if shape=='constant' else f'smooth_{shape}.json'))
    p=np.array(json.loads(source.read_text())['parameters'])
    words=smooth_motif(p,1,1.,'phase' if shape=='constant' else shape)
    all_inputs=[(q,n) for q in range(2) for n in range(9)]
    circuit=(Circuit(**json.loads((OUT/'floquet_calibration.json').read_text())['circuit']) if calibrated else Circuit())
    result,out=microscopic(words,circuit,na=na,nc=nc,dt_ns=dt,inputs=all_inputs[start:stop])
    if calibrated:
        # Report the gate in the fixed hardware-carrier frame, not the
        # circuit's slightly different bare-frequency frame.
        dq=circuit.qubit_drive_GHz-circuit.f_qubit_GHz
        da=circuit.qubit_drive_GHz+circuit.pump_drive_GHz-circuit.f_cavity_GHz
        phase=(np.arange(na)[:,None,None]*da+np.arange(2)[None,None,:]*dq)
        out*=np.exp(2j*np.pi*phase*np.ones((1,nc,1))*result['duration_ns']).reshape(-1,1)
    print('MICROSCOPIC CHUNK',shape,na,nc,dt,'columns',start,stop,
          'Pexc',result['max_final_coupler_excitation'],'seconds',result['elapsed_seconds'],flush=True)
    return shape,start,result,out


def split_motifs(na=20,nc=6,dt=.01,calibrated=False):
    shapes=['cartesian','constant'] if calibrated else ['abrupt','cartesian']
    jobs=[(shape,na,nc,dt,start,start+6,calibrated) for shape in shapes for start in [0,6,12]]
    rows=[]
    print('START SPLIT FULL MOTIFS',len(jobs),'independent6column chunks; basis',na,nc,'dt',dt,flush=True)
    with ProcessPoolExecutor(4) as pool:
        tasks=[pool.submit(motif_chunk,job) for job in jobs]
        for future in as_completed(tasks):rows.append(future.result())
    reports=[]
    for shape in shapes:
        selected=sorted([r for r in rows if r[0]==shape],key=lambda r:r[1])
        assert len(selected)==3 and [r[1] for r in selected]==[0,6,12]
        out=np.concatenate([r[3] for r in selected],axis=1)
        T=cd.target_columns(cd.Space(na,9),'cd',.4)
        desired=np.zeros_like(out);desired.reshape(na,nc,2,18)[:,0]=T.reshape(na,2,18)
        overlaps=np.sum(desired.conj()*out,axis=0)
        result=selected[0][2].copy()
        result.update(shape=shape,beta=.4,columns=18,complex_amplitudes=out.size,
                      coherent_gate_fidelity=float(abs(overlaps.mean())**2),
                      worst_basis_fidelity=float(min(abs(overlaps)**2)),
                      max_final_coupler_excitation=max(r[2]['max_final_coupler_excitation'] for r in selected),
                      elapsed_worker_seconds=sum(r[2]['elapsed_seconds'] for r in selected))
        tag=f'full_{shape}_a{na}_c{nc}_dt{dt:g}'+('_calibrated' if calibrated else '')
        np.savez_compressed(OUT/(tag+'_columns.npz'),output=out,target=desired)
        (OUT/(tag+'.json')).write_text(json.dumps(result,indent=2)+'\n');reports.append(result)
        print('FULL COUPLER GATE',shape,'F',result['coherent_gate_fidelity'],'columns18;amplitudes',out.size,
              'Pexc',result['max_final_coupler_excitation'],flush=True)
    assert len(reports)==2
    return reports


if __name__=='__main__':
    import sys
    mode=sys.argv[1] if len(sys.argv)>1 else 'independent'
    if mode=='independent':independent()
    elif mode=='boundaries':
        jobs=[(shape,np.pi/6) for shape in ['abrupt','phase','cartesian','zero']]
        print('START4FULLCOSINE BOUNDARY TESTS,3cutoff/timestep pairs each,6input columns',flush=True)
        with ProcessPoolExecutor(4) as pool:results=list(pool.map(boundary,jobs))
        (OUT/'boundary_summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print('FINAL BOUNDARY COUNTS',len(results),'shapes;',sum(len(r['convergence_history'])*r['columns'] for r in results),'propagated columns',flush=True)
    elif mode=='motifs':
        print('START full programmed CD motifs:fullcosine,18inputs',flush=True)
        with ProcessPoolExecutor(2) as pool:results=list(pool.map(full_motif,[(s,.01) for s in ['phase','cartesian']]))
        (OUT/'motif_summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print('FINAL MICROSCOPIC MOTIF COUNTS',len(results),'gates;',sum(r['columns'] for r in results),'input columns',flush=True)
    elif mode=='split_motifs':
        split_motifs(int(sys.argv[2]) if len(sys.argv)>2 else 20,
                     int(sys.argv[3]) if len(sys.argv)>3 else 6,
                     float(sys.argv[4]) if len(sys.argv)>4 else .01)
    elif mode=='calibrated_motifs':
        split_motifs(20,8,.005,True)
