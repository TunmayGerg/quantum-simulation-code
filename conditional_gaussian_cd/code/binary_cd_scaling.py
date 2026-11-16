"""A reusable four-pulse CD word with one strength calibration and more repeats.

Every score uses actual signed SDK words and RFDC samples. Targets remain fixed;
only the integer number of XX Xbar Xbar cycles changes during the search.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json
import numpy as np
import cd_trains as cd
import hardware_check as hw
from verify_short_gates import arrays,OUT


def score(beta,cycles,coefficient):
    cells=4*cycles;area=cells*.03+.005
    linear=2*abs(beta)/area
    g=linear*(1+coefficient*linear**2)
    pump=-np.pi/2+(np.pi if beta<0 else 0)
    phi=np.resize([0.,0.,np.pi,np.pi],cells)
    record=dict(job=dict(name=f'cd_binary_scaled_b{beta:g}',gate='cd',strength=beta,cells=cells,
                         gap=0.,swap=float(np.pi/(2*g)),pump_phase=pump,pump_control=False,
                         cycles=cycles,calibration_coefficient=coefficient),
                qubit_phases=phi.tolist(),pump_phases=np.full(cells,pump).tolist(),qubit_frame_Rz=0.)
    q,s,words,_,_,_=arrays(record)
    levels=128 if abs(beta)>2 else 96
    space=cd.Space(levels,9);initial=np.eye(2*levels,dtype=complex)[:,space.inputs]
    T=cd.target_columns(space,'cd',beta)
    output=hw.propagate(q,s,initial,cd.DT/8,order=8,fast=True)
    overlaps=np.sum(T.conj()*output,axis=0)
    record.update(fidelity=float(abs(overlaps.mean())**2),worst_basis_fidelity=float(min(abs(overlaps)**2)),
                  command_us=len(words['qubit'])*cd.DT,played_window_us=len(q)*cd.DT/8,
                  levels=levels,columns=18,complex_amplitudes=output.size)
    return record


def completion_curve():
    """Evaluate the actual gate before the very last tiny FIR samples finish."""
    record=json.loads((OUT/'cd_binary_scaled_b1.json').read_text())
    q,s,words,_,_,_=arrays(record);dt=cd.DT/8
    levels=96;sp=cd.Space(levels,9)
    initial=np.eye(2*levels,dtype=complex)[:,sp.inputs]
    target=cd.target_columns(sp,'cd',1.)
    full=hw.propagate(q,s,initial,dt,order=8,fast=True)
    rows=[];at400=None
    for ns in [350,360,370,375,380,382,384,385,386,388,390,395,400,405,410,len(q)*dt*1000]:
        count=min(len(q),round(ns/(dt*1000)))
        output=hw.propagate(q[:count],s[:count],initial,dt,order=8,fast=True)
        rows.append(dict(time_ns=count*dt*1000,fidelity=float(abs(np.vdot(target,output)/18)**2),
                         remaining_tail_input_operator_norm=float(np.linalg.norm(full-output,2))))
        if ns==400:at400=output
    assert at400 is not None and len(rows)==16
    larger=cd.Space(128,9);P=np.eye(256,dtype=complex)[:,larger.inputs]
    count=round(.400/dt)
    independent=hw.propagate(q[:count],s[:count],P,dt,order=16)
    embedded=np.zeros_like(independent);embedded[:len(at400)]=at400
    difference=float(max(np.linalg.norm(independent-embedded,axis=0)))
    assert difference<1e-9
    T=cd.target_columns(larger,'cd',1.)
    F=float(abs(np.vdot(T,independent)/18)**2)
    selected=next(r for r in rows if r['time_ns']==400)
    assert F>.99 and selected['remaining_tail_input_operator_norm']<1e-6
    np.savez_compressed(OUT/'cd_400ns_columns.npz',output=independent,target=T)
    result=dict(points=rows,check_time_ns=400,independent_fidelity=F,
                independent_columns=18,independent_amplitudes=independent.size,
                independent_larger_cutoff_column_change=difference,
                remaining_tail_input_operator_norm=selected['remaining_tail_input_operator_norm'],
                note='Full waveform is still simulated; no FIR samples were deleted or reprogrammed.')
    (OUT/'cd_completion_curve.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ACTUAL CD AT400ns:',18,'inputcolumns;',independent.size,'independentamplitudes;F',F,
          'remainingtail operatornorm',selected['remaining_tail_input_operator_norm'],
          'largercutoffandindependentcolumnchange',difference,flush=True)


def reusable_rule():
    """Reproduce the documented family without changing its certified recipe."""
    base=json.loads((OUT/'cd_binary_n12_0.json').read_text())
    g=np.pi/(2*base['job']['swap']);linear=2/.365
    coefficient=float((g/linear-1)/linear**2)
    betas=[-.5,-1.,-2.,.05,.1,.125,.2,.3,.4,.5,.75,1.,1.25,1.5,1.75,2.,2.25,2.5,3.,4.]
    rows=[]
    for beta in betas:
        b=abs(beta);cycles=int(np.ceil(max(2.5*b,2.25*b*b)-1e-12))
        record=score(beta,cycles,coefficient);rows.append(record)
        assert record['fidelity']>.99
        (OUT/(record['job']['name']+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    assert len(rows)==20
    result=dict(rule='ceil(max(2.5*b,2.25*b*b))',calibration_coefficient=coefficient,
                results=rows,minimum_fidelity=min(r['fidelity'] for r in rows),
                waveform_evaluations=20,input_columns=360)
    (OUT/'binary_rule_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('REUSABLE RULE:20signed strengths;360inputcolumns;minimumF',result['minimum_fidelity'],flush=True)


if __name__=='__main__':
    import sys
    mode=sys.argv[1] if len(sys.argv)>1 else 'rule'
    if mode=='completion':
        completion_curve()
        raise SystemExit(0)
    if mode=='rule':
        reusable_rule()
        raise SystemExit(0)
    assert mode=='adaptive','Choose rule,completion,or the exploratory adaptive search.'
    base=json.loads((OUT/'cd_binary_n12_0.json').read_text())
    g=np.pi/(2*base['job']['swap']);linear=2/.365
    coefficient=float((g/linear-1)/linear**2)
    betas=[.1,.25,.5,.75,1.,1.25,1.5,2.,3.,4.]
    rows=[];history=[]
    print('START one four-pulse word;onebeta1strengthcalibration;all18inputs;no phase fitting',flush=True)
    for beta in betas:
        for cycles in range(max(1,int(np.ceil(beta))),int(np.ceil(3*max(beta,beta**2)))+3):
            record=score(beta,cycles,coefficient);history.append(dict(beta=beta,cycles=cycles,fidelity=record['fidelity']))
            if record['fidelity']>.991:break
        else:raise AssertionError((beta,'repeat search failed'))
        rows.append(record);name=record['job']['name']
        # Exploratory repeat minimization must not overwrite the documented
        # closed-rule source records that certificates refer to.
        name=name.replace('scaled','adaptive');record['job']['name']=name
        (OUT/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        print('BINARY SCALED',beta,'cycles',cycles,'T',record['command_us'],'F',record['fidelity'],flush=True)
    assert len(rows)==len(betas)
    result=dict(calibration_beta=1.,calibration_coefficient=coefficient,
                chosen_cycle_counts=[r['job']['cycles'] for r in rows],results=rows,
                search_history=history,waveform_evaluations=len(history),input_columns=18*len(history))
    (OUT/'binary_scaling_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FINAL BINARY SCALING COUNTS',len(rows),'strengths;',len(history),'waveforms;',18*len(history),'inputcolumns',flush=True)
