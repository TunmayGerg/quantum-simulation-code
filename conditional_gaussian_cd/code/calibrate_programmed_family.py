"""Calibrate primitives and physical observables, then test a fixed CD target.

Constant-drive Floquet maps determine DAC amplitude and driven resonances
without using CD outputs. Vacuum first moments subsequently fix displacement
strength/direction, and fixed reference observables fix two common frames.
Those calibrations never optimize the gate trace or fit a new CD target.
Frequency corrections are explicit IQ chirps at the original NCOs.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import json
from dataclasses import replace
import numpy as np
from scipy.optimize import brentq
from coupler_phase_model import *
from calibrate_coupler import floquet_pair


def calibrate_family(kind):
    reference=json.loads((ROOT/'runs/coupler/floquet_calibration.json').read_text())
    base=Circuit(**reference['circuit'])
    source=ROOT/'runs/audit'/('transition_fit_constant.json' if kind=='constant' else 'smooth_cartesian.json')
    parameters=np.array(json.loads(source.read_text())['parameters'])
    desired=1/(4*300.)*parameters[8] # GHz, independently requested JC strength
    samples=[]
    def cost(scale):
        result=floquet_pair(base,scale=scale)
        samples.append(dict(DAC_scale=scale,**result))
        return result['coupling_GHz']-desired
    scale=brentq(cost,.035,.09,xtol=1e-8)
    drive=base
    for iteration in range(3):
        result=floquet_pair(drive,scale=scale)
        pump=result['cavity_on_GHz']-result['qubit_on_GHz']
        drive=replace(base,pump_drive_GHz=pump,qubit_drive_GHz=result['qubit_on_GHz'])
    checked=floquet_pair(drive,scale=scale,nc=10,dt=.0025)
    parameters[8]=scale/.09
    record=dict(kind=kind,parameters=parameters.tolist(),target_coupling_GHz=desired,
                reference_circuit=base.__dict__,driven_circuit=drive.__dict__,Floquet_check=checked,
                qubit_IQ_detune_GHz=drive.qubit_drive_GHz-base.qubit_drive_GHz,
                pump_IQ_detune_GHz=drive.pump_drive_GHz-base.pump_drive_GHz,
                pump_phase_correction_rad=-checked['coupling_phase_rad'],
                peak_DAC_scale=scale,samples=samples)
    path=ROOT/'runs/coupler'/f'family_calibration_{kind}.json';path.write_text(json.dumps(record,indent=2)+'\n')
    print('FAMILY PRIMITIVE CALIBRATION',kind,'scale',scale,'target_g_GHz',desired,'measured',checked,
          'IQdetunesGHz',record['qubit_IQ_detune_GHz'],record['pump_IQ_detune_GHz'],flush=True)
    return record


def programmed_words(record,repeats=1,gain=1.):
    """Exact SDK payload, with all detuning and phase corrections explicit."""
    shape='phase' if record['kind']=='constant' else 'cartesian'
    words=smooth_motif(record['parameters'],repeats,gain,shape)
    t=np.arange(len(words['qubit']))*cd.DT*1000
    delay=22.65625 # known baseband RFDC group delay, ns
    program=json.loads(hw.PROGRAM.read_text())
    result={}
    for key,channel,pulse,detune in [('qubit','DAC7','idle_pi',record['qubit_IQ_detune_GHz']),
                                   ('sideband','DAC10','pump',record['pump_IQ_detune_GHz'])]:
        values=sample_to_complex(words[key])*np.exp(2j*np.pi*detune*(t+delay))
        if key=='sideband':values*=np.exp(1j*record['pump_phase_correction_rad'])
        result[key],_=hw.sdk_load(values,channel,program[pulse]['name'])
    return result


def in_calibrated_frame(output,record,na,nc,duration_ns):
    base=Circuit(**record['reference_circuit']);driven=Circuit(**record['driven_circuit'])
    dq=driven.qubit_drive_GHz-base.f_qubit_GHz
    da=driven.qubit_drive_GHz+driven.pump_drive_GHz-base.f_cavity_GHz
    phase=(np.arange(na)[:,None,None]*da+np.arange(2)[None,None,:]*dq)*np.ones((1,nc,1))
    return output*np.exp(2j*np.pi*phase*duration_ns).reshape(-1,1)


def observer_frames(output,na,nc,alpha=.5):
    """Calibrate TWO common frames from fixed observable reference states.

    No target matrix or gate fidelity is an argument. By linearity the cached
    n<=8 columns exactly give the two normalized truncated coherent inputs
    |+x> tensor |+/-alpha>. This avoids repeating a costly full propagation.
    The omitted tail of a true coherent preparation is bounded explicitly.
    """
    from scipy.special import gammaln
    from scipy.stats import poisson
    means=[];coherences=[]
    for sign in [1,-1]:
        value=sign*alpha;n=np.arange(9)
        coefficients=np.exp(-abs(value)**2/2+n*np.log(abs(value))-.5*gammaln(n+1))*np.exp(1j*np.angle(value)*n)
        coefficients/=np.linalg.norm(coefficients)
        state=((output[:,:9]+output[:,9:])@coefficients/np.sqrt(2)).reshape(na,nc,2)
        means.append(np.sum(np.sqrt(np.arange(1,na))[:,None,None]*state[:-1].conj()*state[1:]))
        coherences.append(np.sum(state[:,:,0]*state[:,:,1].conj()))
    slope=(means[0]-means[1])/(2*alpha)
    return dict(reference_alpha=alpha,cavity_R_rad=float(np.angle(slope)),
                qubit_Rz_rad=float(np.angle(sum(coherences))),slope_magnitude=float(abs(slope)),
                input_coherent_tail_norm_bound=float(np.sqrt(poisson.sf(8,alpha**2))),
                calibration_observables='two cavity first moments and mean qubit coherence; no target or gate-trace fit')


def vacuum_strength_calibration(kind):
    """A physical first-moment calibration, separate from the all-input test.

    Only the two vacuum inputs |+x,0> and |-x,0> determine strength/direction.
    No full gate trace and no Fock n>=1 output enters this calibration.
    """
    record=json.loads((ROOT/'runs/coupler'/f'family_calibration_{kind}.json').read_text())
    base=Circuit(**record['reference_circuit']);history=[]
    for iteration in range(3):
        driven=Circuit(**record['driven_circuit']);scale=.09*record['parameters'][8]
        for _ in range(2):
            f=floquet_pair(driven,scale=scale)
            driven=replace(base,qubit_drive_GHz=f['qubit_on_GHz'],
                           pump_drive_GHz=f['cavity_on_GHz']-f['qubit_on_GHz'])
        record['driven_circuit']=driven.__dict__
        record['qubit_IQ_detune_GHz']=driven.qubit_drive_GHz-base.qubit_drive_GHz
        record['pump_IQ_detune_GHz']=driven.pump_drive_GHz-base.pump_drive_GHz
        words=programmed_words(record)
        na,nc=10,8;initial=np.zeros((2*na*nc,2),complex)
        initial[0]=1/np.sqrt(2);initial[1]=np.array([1,-1])/np.sqrt(2)
        result,out=microscopic(words,base,na,nc,.005,initial_state=initial)
        out=in_calibrated_frame(out,record,na,nc,result['duration_ns']).reshape(na,nc,2,2)
        means=np.sum(np.sqrt(np.arange(1,na))[:,None,None,None]*out[:-1].conj()*out[1:],axis=(0,1,2))
        beta=(means[0]-means[1])/2;unconditional=(means[0]+means[1])/2
        history.append(dict(iteration=iteration,peak_DAC_scale=scale,
                            conditional_beta_real=float(beta.real),conditional_beta_imag=float(beta.imag),
                            unconditional_alpha_real=float(unconditional.real),unconditional_alpha_imag=float(unconditional.imag),
                            columns=2,elapsed_seconds=result['elapsed_seconds']))
        print('VACUUM CALIBRATION',kind,history[-1],flush=True)
        if iteration<2:
            record['parameters'][8]*=.4/abs(beta)
            record['pump_phase_correction_rad']+=np.angle(beta)
    record['vacuum_calibration_history']=history
    record['calibration_mode']='two vacuum first moments; no n>=1 gate output fitted'
    (ROOT/'runs/coupler'/f'vacuum_calibrated_{kind}.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    import sys
    from concurrent.futures import ProcessPoolExecutor
    if len(sys.argv)>1 and sys.argv[1]=='vacuum':
        print('START vacuum first-moment strength/direction calibration; no all-input fidelity fitted',flush=True)
        with ProcessPoolExecutor(2) as pool:list(pool.map(vacuum_strength_calibration,['constant','cartesian']))
        raise SystemExit(0)
    print('START independent constant-drive calibrations, no CD target fitting',flush=True)
    with ProcessPoolExecutor(2) as pool:list(pool.map(calibrate_family,['constant','cartesian']))
