"""Pulse-spacing comparisons at fixed target and fixed total gate duration.

CD repeats XX-X-X. Conditional squeezing repeats the previously verified
proper eight-pulse Magnus word. Pulse axes are NEVER reoptimized per point.
One constant JC coupling normalizes the wanted first/second Magnus coefficient
to the same exact18-column target. No fidelity optimization is performed.
Every30ns pi envelope, every physical wait, SDK rounding and the complete RFDC
response are propagated. Target/output cutoffs grow independently at the end.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,hashlib,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from project_paths import ROOT
import cd_trains as cd
import hardware_check as hw
from fixed_sideband import ConstantGate,MAX_G
from coupling_series import wanted

OUT=ROOT/'runs/pulse_spacing';OUT.mkdir(exist_ok=True)
CD_PHASES=np.array([0.,0.,np.pi,np.pi])
SQ_PHASES=np.array([1,0,1,0,3,2,3,2])*np.pi/2


def squeeze_coefficient(gap):
    """Exact finite-pulse quadratic coefficient of one eight-pulse cycle."""
    base=np.load(ROOT/'runs/strategic_squeezing/partial_pulse_series_N1.npz')['coefficients'][:3]
    space=cd.Space(8,4);d=16;J=space.hamiltonian(1.,0.).astype(complex)
    C=np.zeros((3,d,d),complex);C[0]=np.eye(d)
    def product(A,B):return np.array([sum(A[j]@B[k-j] for j in range(k+1)) for k in range(3)])
    A=-1j*(gap/2)*J;idle=np.array([np.eye(d),A,A@A/2])
    for phi in SQ_PHASES:
        D=np.exp(1j*phi*space.k);pulse=D[None,:,None]*base*D.conj()[None,None,:]
        C=product(idle,C);C=product(pulse,C);C=product(idle,C)
    C=np.array([C[0].conj().T@v for v in C]);L1=C[1];L2=C[2]-C[1]@C[1]/2
    a=np.diag(np.sqrt(np.arange(1,8)),1);G=np.kron(a@a-a.T@a.T,np.diag([1.,-1.]))
    low=np.arange(8);b=G[:,low].ravel();v=L2[:,low].ravel();c=float(np.vdot(b,v).real/np.vdot(b,b).real)
    assert c>0 and np.linalg.norm(L1[:,low])<1e-10
    return c


def moments(V,gate):
    psi=(V[:,0]+V[:,9])/np.sqrt(2) if gate=='cd' else V[:,0]
    psi=psi.reshape(-1,2);n=np.arange(len(psi))
    a=np.sum(np.sqrt(n[1:])[:,None]*psi[:-1].conj()*psi[1:])
    a2=np.sum(np.sqrt(n[2:]*(n[2:]-1))[:,None]*psi[:-2].conj()*psi[2:])
    number=float(np.sum(n[:,None]*abs(psi)**2))
    vx=float(2*number+1+2*a2.real-4*a.real*a.real)
    return dict(vacuum_displacement_real=float(a.real),vacuum_displacement_imag=float(a.imag),
        vacuum_target_axis_squeezing_dB=float(-10*np.log10(vx)),vacuum_photon_number=number,
        vacuum_ground_qubit_population=float(np.sum(abs(psi[:,0])**2)))


def jobs():
    result=[]
    for duration,counts in [(.370,[4,8,12]),(.730,[4,8,12,16,24])]:
        for n in counts:result.append(dict(gate='cd',duration_us=duration,cells=n))
    for n in [16,24,32,40,48,64,80,96,128,160]:
        result.append(dict(gate='squeeze',duration_us=4.810,cells=n))
    for j in result:
        block=4 if j['gate']=='cd' else 8
        assert j['cells']%block==0
        period=(j['duration_us']-.010)/j['cells'];gap=period-cd.PI_TIME
        assert gap>=-1e-12 and abs(gap/(2*cd.DT)-round(gap/(2*cd.DT)))<1e-8
        j['gap_ns']=max(0.,gap*1000);j['spacing_ns']=period*1000
        j['name']=f"{j['gate']}_T{j['duration_us']*1000:g}_N{j['cells']}"
    assert len(result)==18
    return result


def run(job,certify=True):
    started=time.time();gate=job['gate'];n=job['cells'];gap=job['gap_ns']*.001
    phases=np.tile(CD_PHASES if gate=='cd' else SQ_PHASES,n//(4 if gate=='cd' else 8))
    levels=64 if gate=='cd' else 192
    model=ConstantGate(gate,n,levels,job['gap_ns'])
    duration=len(model.qshape)*cd.DT;assert abs(duration-job['duration_us'])<1e-12
    c2=None
    estimate=2/(duration-.005)
    if gate=='squeeze':
        c2=squeeze_coefficient(gap);estimate=np.sqrt((cd.R_8DB/2)/(n/8*c2))
    assert estimate<model.maximum_g,(job,estimate,model.maximum_g)
    history=[];g=float(estimate);coefficient_target=1. if gate=='cd' else cd.R_8DB/2
    exponent=1 if gate=='cd' else 2
    print('START SPACING POINT',job['name'],'spacing',job['spacing_ns'],'Tns',duration*1000,'estimate_g',estimate,flush=True)
    # Calculate the coefficient from the actual packed/filter waveform, not
    # from its achieved output or a refitted target. A few global gain updates
    # correct the finite pulse/filter change. DAC quantization sets the floor.
    for iteration in range(4):
        parameters=np.r_[phases,np.log(g)]
        q,s,qf,sf,words=model.arrays(parameters)
        actual,terms=wanted(qf,sf,cd.DT/8,gate)
        assert actual>0
        relative=abs(actual/coefficient_target-1)
        history.append(dict(g=g,coefficient=actual,relative_error=relative,**terms))
        print('SPACING NORMALIZATION',job['name'],iteration,'coefficient',actual,'target',coefficient_target,
              'relative_error',relative,flush=True)
        if relative<.001:break
        if iteration<3:g*=float((coefficient_target/actual)**(1/exponent))
        assert g<=model.maximum_g
    assert relative<.003,'Wanted gate amount does not match the target within the DAC resolution.'
    fit,V,T,words=model.evaluate(parameters,False,outputs=True)
    angle=fit['qubit_frame_Rz'];q,s,qf,sf,_=model.arrays(parameters)
    result=dict(**job,target_strength=1. if gate=='cd' else cd.R_8DB,
        pulse_word_model_degrees=np.degrees(CD_PHASES if gate=='cd' else SQ_PHASES).tolist(),
        qubit_envelope_ns=30.,pump_phase_rad=-np.pi/2 if gate=='cd' else 0.,
        sideband_phase_constant=True,sideband_plateau_amplitude_constant=True,
        phases_reoptimized=False,fidelity_optimized=False,scalar_coupling_calibrated=True,g=g,
        nominal_swap_ns=float(1000*np.pi/(2*g)),peak_swap_ns=float(1000*np.pi/(2*max(abs(sf)))),
        command_us=duration,quadratic_cycle_coefficient_us2=c2,
        qubit_frame_Rz=angle,design_fidelity=fit['fidelity'],design_levels=levels,
        frame_calibration_scope='One declared final qubit Rz for squeezing, common to all18inputs; none for CD.',
        normalization_evaluations=len(history),normalization_history=history,
        wanted_coefficient_target=coefficient_target,wanted_coefficient_actual=actual,
        relative_coefficient_error=relative,
        calibration_scope='One constant g normalizes the wanted first/second Magnus coefficient of the actual SDK/RFDC waveform. No fidelity or phase fit.',
        **moments(V,gate))
    assert max(abs(sf))<=MAX_G
    np.savez_compressed(OUT/(job['name']+'_words.npz'),**words)
    np.savez_compressed(OUT/(job['name']+'_controls.npz'),q=q,s=s,parameters=parameters)
    if certify:
        previous=None;cutoffs=[]
        for L in ([48,64,96,128] if gate=='cd' else [128,192,256,320,384,512,640]):
            sp=cd.Space(L,9);initial=np.eye(2*L,dtype=complex)[:,sp.inputs];target=cd.target_columns(sp,gate,result['target_strength'])
            frame=np.exp(-1j*angle*sp.z/2)[:,None]
            output=frame*hw.propagate(qf,sf,initial,cd.DT/8,order=8,fast=True)
            row=dict(levels=L)
            if previous is not None:
                old,oldtarget=previous;pad=np.zeros_like(output);pad[:len(old)]=old
                padT=np.zeros_like(target);padT[:len(oldtarget)]=oldtarget
                row.update(output_change=float(max(np.linalg.norm(output-pad,axis=0))),
                    target_change=float(max(np.linalg.norm(target-padT,axis=0))))
            cutoffs.append(row)
            if previous is not None and max(row['output_change'],row['target_change'])<1e-9:break
            previous=output,target
        else:raise AssertionError((job['name'],'cutoff failed',cutoffs))
        independent=frame*hw.propagate(qf,sf,initial,cd.DT/8,order=16)
        error=float(np.max(abs(independent-output)));assert error<1e-10
        overlaps=np.sum(target.conj()*output,axis=0)
        result.update(fidelity=float(abs(overlaps.mean())**2),worst_basis_fidelity=float(min(abs(overlaps)**2)),
            cutoff_history=cutoffs,levels=L,columns=18,independent_amplitudes=output.size,independent_max_error=error,
            word_sha256=hashlib.sha256(words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest(),
            **moments(output,gate))
        np.savez_compressed(OUT/(job['name']+'_columns.npz'),output=output,target=target)
    result['seconds']=time.time()-started
    (OUT/(job['name']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('SPACING POINT',job['name'],'spacing',job['spacing_ns'],'F',result.get('fidelity',result['design_fidelity']),
        'swap',result['nominal_swap_ns'],'vacuum_beta',result['vacuum_displacement_real'],
        'vacuum_dB',result['vacuum_target_axis_squeezing_dB'],'seconds',result['seconds'],flush=True)
    return result


def main(pilot=False,workers=3):
    work=jobs()
    if pilot:work=[work[0],work[2],work[8],work[-1]]
    with ProcessPoolExecutor(workers) as pool:results=list(pool.map(run,work))
    assert len(results)==(4 if pilot else 18)
    report=dict(results=results,points=len(results),columns=18*len(results),
        independent_amplitudes=sum(r['independent_amplitudes'] for r in results),
        fixed_duration_groups=sorted({(r['gate'],r['duration_us']) for r in results}),
        scope='Pulse spacing at fixed duration and exact target; fixed pulse cycles; wanted Magnus coefficient normalized; no fidelity or phase optimization.')
    (OUT/('pilot.json' if pilot else 'summary.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('SPACING COMPLETE:',len(results),'points;',report['columns'],'input columns;',report['independent_amplitudes'],
        'independent complex amplitudes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();main(a.pilot,a.workers)
