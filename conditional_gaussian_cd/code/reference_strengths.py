"""Fixed-beta CD strength/time reference, without phase optimization.

For each duration, only one constant sideband coupling is calibrated. Every
recipe targets the same beta=1 unitary on all 18 g/e x n=0..8 inputs. The pump
has one constant phase and amplitude on its plateau. Long weak gates are
therefore compared against short strong gates at the same target operation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import hashlib
import json
import numpy as np
from scipy.optimize import minimize_scalar
from project_paths import ROOT, CODE
import cd_trains as cd
import hardware_check as hw
from fixed_sideband import ConstantGate, MAX_G
from review_hypotheses import native_series

OUT=ROOT/'runs'/'reference_strengths'


def verify_four_pulse_average():
    space,Q,V,L=native_series([(0.,1.),(0.,1.),(np.pi,1.),(np.pi,1.)],ramps=False)
    a=np.diag(np.sqrt(np.arange(1,space.levels)),1)
    sx=np.array([[0,1],[1,0]])
    expected=-1j*.060*np.kron(a+a.T,sx)
    error=float(np.max(abs(L[0]-expected)))
    assert error<1e-12
    print('FOUR-PULSE AVERAGE:',expected.size,'complex amplitudes; maxerror',error,flush=True)
    return dict(complex_amplitudes_compared=expected.size,max_error=error,
                leading_logarithm='-i g (0.060 us) sigma_x X at pump phase 0',
                target_orientation='constant pump phase -pi/2 rotates X into P',
                formal_order_claim=False,
                note='Outer ramps add a small bare-JC first moment; all exact fidelity checks include it.')


def calibrate(k):
    n=4*k;model=ConstantGate('cd',n,64)
    phases=np.tile([0.,0.,np.pi,np.pi],k)
    duration=(24*n+8)*cd.DT
    estimate=2/(duration-.005)
    evaluations=[]
    def objective(g):
        p=np.r_[phases,np.log(g)]
        value=float(model.evaluate(p,False))
        evaluations.append(dict(g=float(g),fidelity=1-value))
        return value
    lo=.90*estimate;hi=min(1.12*estimate,model.maximum_g)
    result=minimize_scalar(objective,bounds=(lo,hi),method='bounded',
                           options=dict(xatol=1e-9,maxiter=60))
    # Quantization makes the forward objective a staircase. Keep the best
    # evaluated constant amplitude instead of assuming the final point is best.
    best=max(evaluations,key=lambda r:r['fidelity'])
    g=best['g'];parameters=np.r_[phases,np.log(g)]
    q,s,qf,sf,words=model.arrays(parameters)
    assert len(q)==24*n+8 and np.all(s[4:-4]==s[4])
    peak=float(np.max(abs(sf)))
    assert max(np.max(abs(s)),peak)<=MAX_G
    # Converge both target and propagated columns. The alternate propagator
    # is independently implemented in native_waveform_check.cpp.
    history=[];previous=None;final=None
    for levels in [48,64,80,96,128]:
        space=cd.Space(levels,9)
        target=cd.target_columns(space,'cd',1.)
        initial=np.eye(2*levels,dtype=complex)[:,space.inputs]
        out=hw.propagate(qf,sf,initial,cd.DT/8,order=8,fast=True)
        overlaps=np.sum(target.conj()*out,axis=0)
        item=dict(levels=levels,columns=18,complex_amplitudes=out.size,
                  fidelity=float(abs(np.sum(overlaps)/18)**2),
                  worst_basis_fidelity=float(np.min(abs(overlaps)**2)))
        if previous is not None:
            oldout,oldtarget=previous
            padded=np.zeros_like(out);padded[:len(oldout)]=oldout
            paddedtarget=np.zeros_like(target);paddedtarget[:len(oldtarget)]=oldtarget
            item['output_column_change']=float(np.max(np.linalg.norm(out-padded,axis=0)))
            item['target_column_change']=float(np.max(np.linalg.norm(target-paddedtarget,axis=0)))
        history.append(item)
        if previous is not None and max(item['output_column_change'],item['target_column_change'])<1e-10:
            independent=hw.propagate(qf,sf,initial,cd.DT/8,order=16,fast=False)
            backend_error=float(np.max(abs(independent-out)))
            assert backend_error<1e-11
            final=item;break
        previous=out,target
    assert final is not None,'cutoff did not converge'
    row=dict(name=f'cd_beta1_xxmm_k{k}',gate='cd',strength=1.,input_fock_max=8,
             qubit_levels=2,qubit_frame_Rz=0.,repeats=k,cells=n,qubit_phases=phases.tolist(),
             angle_fractions=np.ones(n).tolist(),qubit_envelope_ns=30.,
             pulse_word=['X','X','-X','-X'],pump_phase=-np.pi/2,
             pump_amplitude_constant=True,pump_phase_constant=True,
             outer_ramp_ns=5.,g_nominal=g,nominal_swap_ns=1000*np.pi/(2*g),
             actual_plateau_g=float(abs(s[4])),peak_played_g=peak,
             peak_swap_ns=1000*np.pi/(2*peak),command_time_us=duration,
             RFDC_window_us=len(qf)*cd.DT/8,
             speed_ratio_to_peak_force_bound=float(2/(duration*peak)),
             same_target_calibration='One constant g maximizes beta=1 coherent process fidelity; phases unchanged.',
             calibrated_beta_independently_fixed=True,
             search_bounds_g=[lo,hi],evaluations=len(evaluations),search_history=evaluations,
             fidelity=final['fidelity'],worst_basis_fidelity=final['worst_basis_fidelity'],
             cutoff_history=history,levels=final['levels'],columns=18,
             independent_backend_max_amplitude_error=backend_error,
             independent_amplitudes_compared=int(out.size),
             simulated_filter='SDK int16 decode and documented RFDC 8x interpolation, full tail',
             source_sha256=hashlib.sha256((CODE/'reference_strengths.py').read_bytes()).hexdigest())
    np.savez_compressed(OUT/(row['name']+'_controls.npz'),q=q,s=s,parameters=parameters,
                        qubit_words=words['qubit'],sideband_words=words['sideband'])
    (OUT/(row['name']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    print('REFERENCE',row['name'],'Tns',duration*1000,'swapns',row['nominal_swap_ns'],
          'peak_swapns',row['peak_swap_ns'],'F',row['fidelity'],'worst',row['worst_basis_fidelity'],
          'cutoffs',len(history),'backend_amplitudes',out.size,'maxerror',backend_error,flush=True)
    return row


if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True)
    leading=verify_four_pulse_average()
    rows=[calibrate(k) for k in [3,4,5,6,8]]
    assert len(rows)==5 and sum(r['columns'] for r in rows)==90
    payload=dict(target='CD beta=1 on qubit g/e x Fock n=0..8',leading_average=leading,
                 recipes_compared=5,target_input_columns_compared=90,
                 independent_amplitudes_compared=sum(r['independent_amplitudes_compared'] for r in rows),
                 total_design_input_column_evaluations=18*sum(r['evaluations'] for r in rows),results=rows)
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('REFERENCE COMPLETE',payload['recipes_compared'],'gates;',payload['target_input_columns_compared'],
          'targetcolumns;',payload['independent_amplitudes_compared'],'independentcomplexamplitudes;',
          payload['total_design_input_column_evaluations'],'designinputcolumnevaluations',flush=True)
