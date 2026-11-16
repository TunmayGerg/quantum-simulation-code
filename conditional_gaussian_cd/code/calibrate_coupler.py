"""Calibrate an assumed circuit to the measured 600ns swap with Floquet maps.

This avoids confusing a lowest-order g(phi) formula with a measured strong-drive
coupling. One-period propagation is far cheaper than fitting many 600ns traces.
Circuit energies/participations remain explicit assumptions.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
from dataclasses import replace
import json
import numpy as np
from scipy.linalg import eig,polar
from coupler_phase_model import *


def floquet_pair(circuit,scale=.045,na=3,nc=8,dt=.005):
    fp=(circuit.f_cavity_GHz-circuit.f_qubit_GHz if circuit.pump_drive_GHz is None else circuit.pump_drive_GHz)
    N=2*na*nc
    result,U=microscopic(None,circuit,na,nc,dt,initial_state=np.eye(N,dtype=complex),
                         rf_samples=(np.full(16,scale,complex),np.zeros(16,complex)),duration_ns=1/fp)
    _,_,E,_=operators(circuit,na,nc)
    U*=np.exp(-1j*E/fp)[:,None] # restore lab-frame one-period unitary
    values,vectors=eig(U)
    assert np.max(abs(abs(values)-1))<1e-9
    indices=[1,2*nc]
    weights=np.sum(abs(vectors[indices])**2,axis=0)
    pair=np.argsort(weights)[-2:]
    ground=int(np.argmax(abs(vectors[0])**2))
    energies=-np.angle(values)/(2*np.pi/fp)
    # H0 was globally shifted. Remove the ground quasienergy before lifting.
    relative=energies[pair]-energies[ground]
    relative+=np.round((circuit.f_qubit_GHz-relative)/fp)*fp
    rotation,_=polar(vectors[np.ix_(indices,pair)])
    effective=(rotation*relative)@rotation.conj().T
    return dict(qubit_on_GHz=float(effective[0,0].real),
                cavity_on_GHz=float(effective[1,1].real+fp),
                coupling_GHz=float(abs(effective[0,1])),
                coupling_phase_rad=float(np.angle(effective[0,1])),
                detuning_GHz=float((effective[1,1]-effective[0,0]).real),
                minimum_pair_weight=float(min(weights[pair])),
                columns=N,amplitudes=U.size,coupler_levels=nc,timestep_ns=dt)


def calibrate():
    circuit=Circuit();default=circuit.parameters();flux=default['flux_per_DAC']
    pump=circuit.f_cavity_GHz-circuit.f_qubit_GHz
    history=[]
    print('START PERIOD-MAP CALIBRATION: exact cosine, measured600ns reference',flush=True)
    # Fixed-point corrections are small. Each update enforces the measured
    # driven mode references and the measured on-resonance swap, not a target CD.
    for iteration in range(5):
        configured=replace(circuit,pump_drive_GHz=pump,qubit_drive_GHz=Circuit().f_qubit_GHz,
                           flux_per_DAC_override=flux)
        measured=floquet_pair(configured)
        desired=1/(4*600.)
        history.append(dict(iteration=iteration,circuit=configured.__dict__,floquet=measured))
        print('FLOQUET CALIBRATION',iteration,measured,'flux_per_DAC',flux,flush=True)
        flux*=desired/measured['coupling_GHz']
        circuit=replace(circuit,
            f_qubit_GHz=circuit.f_qubit_GHz+Circuit().f_qubit_GHz-measured['qubit_on_GHz'],
            f_cavity_GHz=circuit.f_cavity_GHz+Circuit().f_cavity_GHz-measured['cavity_on_GHz'])
    configured=replace(circuit,pump_drive_GHz=pump,qubit_drive_GHz=Circuit().f_qubit_GHz,
                       flux_per_DAC_override=flux)
    final=floquet_pair(configured,nc=10,dt=.0025)
    print('FINAL EXACT PERIOD CALIBRATION',final,flush=True)
    record=dict(circuit=configured.__dict__,final=final,history=history,
                assumption='measured6.8GHz and4GHz tuning; illustrativeEC and participations; calibrate600ns swap and driven carriers')
    path=ROOT/'runs/coupler/floquet_calibration.json';path.write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':calibrate()
