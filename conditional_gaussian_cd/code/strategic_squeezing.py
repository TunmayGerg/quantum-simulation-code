"""Falsify structured conditional-squeezing improvements before fitting phases.

The optional pump-phase cycle reverses the sign of the *entire* JC Hamiltonian.
For a proper block whose logarithm begins g**2 L2 + g**3 L3, alternating g
therefore preserves squeezing and reverses every cubic error operator. All
10 ns constant-radius phase arcs, finite 30 ns qubit pulses, outer ramps and
the small first-moment compensation dwell are included explicitly below.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

import cd_trains as cd
import hardware_check as hw
from project_paths import ROOT

OUT = ROOT / 'runs/strategic_squeezing'
OUT.mkdir(parents=True, exist_ok=True)
MAX_G = np.pi / (2 * .200)


def compensation_samples(repeat, repeats, balanced):
    if not balanced:
        return 4 if repeat==0 else 0
    # A chirality-conjugate pair gets equal negative-JC dwell, so its
    # conditional-number-rotation area cancels. Total dwell is always 5 ns.
    assert repeats % 2 == 0
    if repeats==4:
        return 1
    return 2 if repeat<2 else 0


def catalogue():
    return json.loads((ROOT / 'runs/fixed_sideband/magnus_catalogue.json').read_text())['results']


def waveform(word, repeats, g=1., compensation=True, arcs=True, chirality_cycle=False,
             balanced_arcs=False):
    """Return Hamiltonian IQ at 1.25 ns/sample; no Cartesian IQ chords.

    Every superblock is B(0), arc 0->pi, B(pi), B(pi), arc pi->2pi,
    B(2pi). The two opposite arcs have exactly opposite first moments.
    One 5 ns central negative-JC dwell cancels the total area of both outer
    sideband ramps. It is inserted only once, even when superblocks repeat.
    """
    qubit = []
    pump = []

    def append(q, s):
        qubit.extend(np.atleast_1d(q).tolist())
        pump.extend(np.atleast_1d(s).tolist())

    append(np.zeros(4), np.sin(np.pi * np.arange(4) / 8) ** 2)
    for repeat in range(repeats):
        chirality = -1 if chirality_cycle and repeat % 2 else 1
        for block, phase in enumerate([0., np.pi, np.pi, 2*np.pi]):
            active_word = word[(repeat*4+block) % len(word)] if np.ndim(word)==2 else word
            for axis in active_word:
                append(cd.pi_samples() * np.exp(1j * axis * np.pi / 2),
                       np.full(24, np.exp(1j*phase)))
            if block in (0, 2) and arcs:
                start = 0. if block == 0 else np.pi
                nodes = np.arange(8)+(.5 if balanced_arcs else 0.)
                angle = chirality*(start + np.pi*np.sin(np.pi*nodes/16)**2)
                append(np.zeros(8), np.exp(1j*angle))
            if block == 1 and compensation:
                count = compensation_samples(repeat,repeats,balanced_arcs)
                append(np.zeros(count), -np.ones(count))
    append(np.zeros(4), np.cos(np.pi * np.arange(4) / 8) ** 2)
    q, s = np.asarray(qubit, complex), np.asarray(pump, complex)
    assert q.shape == s.shape and len(q) % 4 == 0
    assert np.max(abs(abs(s[4:-4])-1)) < 3e-16
    return q, g*s


def pproduct(left, right):
    """Time-ordered truncated power-series multiplication."""
    return np.asarray([sum(left[j] @ right[k-j] for j in range(k+1))
                       for k in range(len(left))])


def series(word, repeats, compensation=True, arcs=True, chirality_cycle=False,
           balanced_arcs=False):
    """Exact native-pulse polynomial through degree four in constant g.

    Pulse coefficients come from the independently checked Cauchy derivatives
    of all 24 finite samples. Sideband-only ramps/arcs use their exact matrix
    exponential power series. Matrix dimensions are deliberately small here;
    only n<=3 columns are used to avoid a truncation-edge commutator artifact.
    """
    V = np.load(ROOT / 'runs/fixed_sideband/pulse_series.npz')['V'][0]
    d = V.shape[-1]
    levels = d//2
    space = cd.Space(levels, 3)
    J = space.hamiltonian(1., 0.).astype(complex)
    I = np.eye(d, dtype=complex)
    identity = np.zeros((5, d, d), complex)
    identity[0] = I
    C = identity.copy()

    def free(samples):
        nonlocal C
        for amp in samples:
            angle = np.angle(amp)
            D = np.exp(-1j*angle*space.n)
            H = D[:, None] * J * D.conj()[None, :]*abs(amp)
            A = -1j*cd.DT*H
            E = np.asarray([np.linalg.matrix_power(A,k)/math.factorial(k)
                            for k in range(5)])
            C = pproduct(E, C)

    free(np.sin(np.pi*np.arange(4)/8)**2)
    for repeat in range(repeats):
        chirality = -1 if chirality_cycle and repeat % 2 else 1
        for block, sign in enumerate([1, -1, -1, 1]):
            active_word = word[(repeat*4+block) % len(word)] if np.ndim(word)==2 else word
            for axis in active_word:
                phi = axis*np.pi/2
                Q2 = np.array([[0., -1j*np.exp(-1j*phi)],
                               [-1j*np.exp(1j*phi), 0.]])
                Q = np.kron(np.eye(levels), Q2)
                E = np.asarray([Q@V[axis,k]*sign**k for k in range(5)])
                C = pproduct(E, C)
            if block in (0, 2) and arcs:
                start = 0. if block == 0 else np.pi
                nodes = np.arange(8)+(.5 if balanced_arcs else 0.)
                free(np.exp(1j*chirality*(start+np.pi*np.sin(np.pi*nodes/16)**2)))
            if block == 1 and compensation:
                count = compensation_samples(repeat,repeats,balanced_arcs)
                free(-np.ones(count))
    free(np.cos(np.pi*np.arange(4)/8)**2)
    # Remove the qubit-only endpoint, including any irrelevant global sign.
    Q0 = C[0].copy()
    C = np.asarray([Q0.conj().T@coefficient for coefficient in C])
    A, B, E, F = C[1:5]
    L = np.asarray([A, B-A@A/2,
                    E-(A@B+B@A)/2+A@A@A/3,
                    F-(A@E+E@A+B@B)/2+
                    (A@A@B+A@B@A+B@A@A)/3-A@A@A@A/4])
    a = np.diag(np.sqrt(np.arange(1,levels)),1)
    G = np.kron(a@a-a.T@a.T, np.diag([1.,-1.]))
    Z = 1j*np.diag(space.z)
    low = np.arange(8)  # Include n=3: both a**3 and adag**3 act nontrivially.
    NZ = 1j*np.diag(space.n*space.z)
    targets = np.stack([G[:,low].ravel(), Z[:,low].ravel(), NZ[:,low].ravel(),
                        (1j*I)[:,low].ravel()],axis=1)
    coefficient = np.linalg.lstsq(targets, L[1][:,low].ravel(),rcond=None)[0]
    remainder = L[1][:,low].ravel()-targets@coefficient
    return dict(first_norm=float(np.linalg.norm(L[0][:,low])),
                second_squeeze=float(coefficient[0].real),
                second_qubit_Z=float(coefficient[1].real),
                second_chi=float(coefficient[2].real),
                second_unwanted_norm=float(np.linalg.norm(remainder)),
                cubic_norm=float(np.linalg.norm(L[2][:,low])),
                quartic_norm=float(np.linalg.norm(L[3][:,low]))), C, L


def magnus_test():
    rows = []
    for index, entry in enumerate(catalogue()[:4]):
        for arcs in [False, True]:
            for compensation in [False, True]:
                result, C, L = series(entry['word'], 1, compensation, arcs)
                row = dict(block_index=index, word=entry['word'], arcs=arcs,
                           compensation=compensation, **result)
                rows.append(row)
                print('PHASE-CYCLE MAGNUS', row, flush=True)
    assert len(rows) == 16
    (OUT/'phase_cycle_magnus.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('MAGNUS COMPARISON:',len(rows),'complete native-waveform series',flush=True)


def chirality_test(balanced_arcs=False):
    rows=[]
    for index,entry in enumerate(catalogue()[:4]):
        for chirality_cycle in [False, True]:
            result,C,L=series(entry['word'],2,True,True,chirality_cycle,balanced_arcs)
            row=dict(block_index=index,word=entry['word'],repeats=2,
                     chirality_cycle=chirality_cycle,**result)
            rows.append(row)
            print('ARC-CHIRALITY MAGNUS',row,flush=True)
    assert len(rows)==8
    (OUT/('balanced_magnus.json' if balanced_arcs else 'chirality_magnus.json')).write_text(json.dumps(rows,indent=2)+'\n')
    print('COMPLETED ARC-CHIRALITY:',len(rows),'full-waveform series; n0..3',flush=True)


def native_fidelity(word, repeats, g, levels=96, compensation=True, chirality_cycle=False,
                    balanced_arcs=False):
    space = cd.Space(levels, 9)
    inputs = np.eye(2*levels,dtype=complex)[:,space.inputs]
    target = cd.target_columns(space,'squeeze',cd.R_8DB)
    q, s = waveform(word, repeats, g, compensation, True, chirality_cycle,balanced_arcs)
    output = hw.propagate(q,s,inputs,cd.DT,order=12,fast=True)
    zg, ze = np.vdot(target[0::2],output[0::2]), np.vdot(target[1::2],output[1::2])
    angle = np.angle(ze)-np.angle(zg)
    frame = np.exp(1j*angle*space.z/2)[:,None]
    overlap = np.sum(target.conj()*frame*output,axis=0)
    return dict(fidelity=float(abs(np.sum(overlap)/18)**2),qubit_frame_Rz=float(-angle),
                worst_input_fidelity=float(np.min(abs(overlap)**2)),
                input_columns=18,time_us=len(q)*cd.DT,
                top_level_population=float(np.max(np.sum(abs(output[-8:])**2,axis=0))))


def screen(chirality_cycle=False,balanced_arcs=False):
    rows=[]; started=time.monotonic()
    for index, entry in enumerate(catalogue()[:4]):
        c2=4*entry['squeeze_second_coefficient']
        for repeats in ([2,4] if chirality_cycle else [2,3,4,5]):
            g0=np.sqrt((cd.R_8DB/2)/(c2*repeats))
            history=[]
            def cost(g):
                result=native_fidelity(entry['word'],repeats,g,chirality_cycle=chirality_cycle,
                                       balanced_arcs=balanced_arcs)
                history.append(dict(g=float(g),**result))
                return 1-result['fidelity']
            minimize_scalar(cost,bounds=(g0*.85,min(g0*1.15,MAX_G*.985)),
                            method='bounded',options=dict(maxiter=22,xatol=1e-6))
            best=max(history,key=lambda row:row['fidelity'])
            row=dict(name=f'squeeze_sign_cycle_c{int(chirality_cycle)}_b{index}_r{repeats}',block_index=index,
                     word=entry['word'],repeats=repeats,cells=32*repeats,
                     chirality_cycle=chirality_cycle,
                     balanced_arcs=balanced_arcs,
                     scalar_evaluations=len(history),history=history,**best)
            rows.append(row)
            filename = 'balanced_screen.json' if balanced_arcs else ('chirality_screen.json' if chirality_cycle else 'phase_cycle_screen.json')
            (OUT/filename).write_text(json.dumps(rows,indent=2)+'\n')
            print('PHASE-CYCLE NATIVE',row['name'],'F',row['fidelity'],'T',row['time_us'],
                  'g',row['g'],'evals',len(history),flush=True)
    assert len(rows)==(8 if chirality_cycle else 16)
    print('COMPLETED PHASE-CYCLE:',len(rows),'gates;',18*sum(r['scalar_evaluations'] for r in rows),
          'inputcolumns; seconds',time.monotonic()-started,flush=True)
    best=max(rows,key=lambda row:row['fidelity'])
    q,s=waveform(best['word'],best['repeats'],best['g'],chirality_cycle=chirality_cycle,
                 balanced_arcs=balanced_arcs)
    filename = 'balanced_best_raw.npz' if balanced_arcs else ('chirality_best_raw.npz' if chirality_cycle else 'phase_cycle_best_raw.npz')
    np.savez_compressed(OUT/filename,qubit=q,sideband=s,dt=cd.DT)


def mixture_screen():
    entries=catalogue()[:4]
    # Each individual B occurs once at +g and once at -g, cancelling its L3.
    word=[entry['word'] for entry in entries for _ in range(2)]
    rows=[]
    for repeats in [2,4]:
        c2=4*entries[0]['squeeze_second_coefficient']
        g0=np.sqrt((cd.R_8DB/2)/(c2*repeats));history=[]
        def cost(g):
            result=native_fidelity(word,repeats,g,chirality_cycle=True,balanced_arcs=True)
            history.append(dict(g=float(g),**result))
            return 1-result['fidelity']
        minimize_scalar(cost,bounds=(g0*.85,min(g0*1.15,MAX_G*.985)),method='bounded',
                        options=dict(maxiter=22,xatol=1e-6))
        best=max(history,key=lambda row:row['fidelity'])
        result=dict(name=f'squeeze_quartic_mixture_r{repeats}',words=word,
                    repeats=repeats,scalar_evaluations=len(history),history=history,**best)
        rows.append(result)
        print('QUARTIC-MIXTURE NATIVE',result['name'],'F',result['fidelity'],
              'T',result['time_us'],'g',result['g'],flush=True)
    assert len(rows)==2
    (OUT/'mixture_screen.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('COMPLETED QUARTIC-MIXTURE:2gates;',18*sum(r['scalar_evaluations'] for r in rows),
          'inputcolumns',flush=True)


def convergence_screen():
    entry=catalogue()[0];rows=[]
    for repeats in [8,12,16]:
        g0=np.sqrt((cd.R_8DB/2)/(4*entry['squeeze_second_coefficient']*repeats));history=[]
        def cost(g):
            result=native_fidelity(entry['word'],repeats,g,chirality_cycle=True,
                                   balanced_arcs=True)
            history.append(dict(g=float(g),**result))
            return 1-result['fidelity']
        minimize_scalar(cost,bounds=(g0*.9,g0*1.1),method='bounded',
                        options=dict(maxiter=20,xatol=1e-6))
        best=max(history,key=lambda row:row['fidelity'])
        result=dict(name=f'squeeze_balanced_convergence_r{repeats}',word=entry['word'],
                    repeats=repeats,scalar_evaluations=len(history),history=history,**best)
        rows.append(result)
        print('BALANCED-CONVERGENCE',repeats,'F',result['fidelity'],
              'T',result['time_us'],'swapns',1000*np.pi/(2*result['g']),flush=True)
    assert len(rows)==3
    (OUT/'convergence_screen.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('COMPLETED BALANCED-CONVERGENCE:3gates;',18*sum(r['scalar_evaluations'] for r in rows),
          'inputcolumns',flush=True)


def export_rejected():
    rows=json.loads((OUT/'balanced_screen.json').read_text())
    best=max(rows,key=lambda row:row['fidelity'])
    q,s=waveform(best['word'],best['repeats'],best['g'],chirality_cycle=True,balanced_arcs=True)
    pulses=[];arcs=[];position=4
    for repeat in range(best['repeats']):
        chirality=-1 if repeat%2 else 1
        for block in range(4):
            for axis in best['word']:
                pulses.append(dict(start_sample=position,samples=24,
                                   angle_fraction=1.,phase_rad=axis*np.pi/2))
                position+=24
            if block in (0,2):
                arcs.append(dict(start_sample=position,samples=8,
                                 center_sample=position+4,chirality=chirality))
                assert np.max(abs(q[position:position+8]))==0
                position+=8
            if block==1:
                position+=compensation_samples(repeat,best['repeats'],True)
    assert position+4==len(q) and len(pulses)==32*best['repeats']
    statistics,_,_=series(best['word'],best['repeats'],True,True,True,True)
    filtered=hw.rfdc_interpolate(s)
    delay=int(np.argmax(abs(hw.rfdc_interpolate(np.r_[1.,np.zeros(40)]))))
    arc_ratios=[]
    for arc in arcs:
        start=8*arc['start_sample']+delay
        arc_ratios.extend((abs(filtered[start:start+8*arc['samples']])/best['g']).tolist())
    report=dict(gate='squeeze',strength=cd.R_8DB,sideband_mode='phase_arc',
                target='exp[(r/2) sigma_z (a^2-adag^2)]',input_photons_max=8,
                pulses=pulses,phase_arcs=arcs,dt=cd.DT,g=best['g'],
                samples=len(q),native_search_result=best,derivation_terms=statistics,
                interpretation='Rejected: exact first-moment and arc chi cancellation does not ensure 99% at finite coupling.',
                expected_removed_terms=['First-order JC average','Second-order arc sigma_z*n',
                                        'Ideal block odd-in-g cubic contribution'],
                retained_terms=['Finite-arc cubic terms','Quartic and higher JC terms'],
                rfdc_arc_magnitude_ratio_min=float(min(arc_ratios)),
                rfdc_arc_magnitude_ratio_max=float(max(arc_ratios)),
                rfdc_peak_g=float(max(abs(filtered))),
                phase_convention='phase_rad is the qubit Hamiltonian phase; physical DAC qubit IQ is conjugated.',
                arc_sampling='Eight midpoint samples of a sin^2 phase ramp in a 10 ns allocated interval; full RFDC response must be replayed.')
    np.savez_compressed(OUT/'rejected_balanced_raw.npz',qubit=q,sideband=s,dt=cd.DT)
    (OUT/'rejected_balanced.json').write_text(json.dumps(report,indent=2)+'\n')
    print('REJECTED WAVEFORM EXPORT:',len(pulses),'30ns pi pulses;',len(arcs),
          '10ns qubit-off phase arcs;',len(q),'samples;nativeF',best['fidelity'],
          'RFDCgpeak',report['rfdc_peak_g'],flush=True)


def certify_convergence():
    """Certify the slow, converged structured reference; no control refitting."""
    results=json.loads((OUT/'convergence_screen.json').read_text())
    result=max(results,key=lambda row:row['fidelity'])
    q,s=waveform(result['word'],result['repeats'],result['g'],
                 chirality_cycle=True,balanced_arcs=True)
    swap=np.pi/(2*result['g']);qscale,sscale=hw.dac_scales(swap)
    qw,qr=hw.to_words_and_back(q,np.pi/cd.PI_TIME,qscale,qubit=True)
    sw,sr=hw.to_words_and_back(s,result['g'],sscale)
    qp,sp=hw.rfdc_interpolate(qr),hw.rfdc_interpolate(sr)
    assert max(abs(sp))<=MAX_G
    previous=None;rows=[];comparison_count=0
    for levels in range(128,577,64):
        print('START STRUCTURED CERTIFICATE:levels',levels,'18inputcolumns',len(qp),
              'RFDCsamples;SDKwords',qw.size+sw.size,flush=True)
        space=cd.Space(levels,9)
        inputs=np.eye(2*levels,dtype=complex)[:,space.inputs]
        target=cd.target_columns(space,'squeeze',cd.R_8DB)
        output=hw.propagate(qp,sp,inputs,cd.DT/8,order=8,fast=True)
        frame=np.exp(-.5j*result['qubit_frame_Rz']*space.z)[:,None]
        overlap=np.sum(target.conj()*frame*output,axis=0)
        row=dict(levels=levels,input_columns=18,
                 fidelity=float(abs(np.sum(overlap)/18)**2),
                 worst_input_fidelity=float(np.min(abs(overlap)**2)))
        if previous is not None:
            old_output,old_target=previous
            pad=((0,2*levels-len(old_output)),(0,0))
            row['output_column_norm_change']=float(np.max(np.linalg.norm(output-np.pad(old_output,pad),axis=0)))
            row['target_column_norm_change']=float(np.max(np.linalg.norm(target-np.pad(old_target,pad),axis=0)))
            comparison_count+=18
        rows.append(row)
        print('STRUCTURED CUTOFF COMPARISON',row,flush=True)
        if previous is not None and max(row['output_column_norm_change'],row['target_column_norm_change'])<1e-9:
            break
        previous=(output,target)
    else:
        raise AssertionError('Structured target/output cutoff failed to converge by576levels.')
    independent=hw.propagate(qp,sp,inputs,cd.DT/8,order=16,fast=False)
    error=float(np.max(abs(independent-output)))
    assert error<1e-10
    report=dict(gate='squeeze',strength=cd.R_8DB,result=result,
                programmed_samples=len(q),played_samples=len(qp),
                sdk_words_compared=int(qw.size+sw.size),cutoff_input_column_comparisons=comparison_count,
                independent_amplitudes_compared=int(output.size),
                independent_max_amplitude_error=error,cutoff_results=rows,
                qubit_frame_Rz_fixed=result['qubit_frame_Rz'],
                max_programmed_g=float(max(abs(s))),max_played_g=float(max(abs(sp))),
                interpretation='Converged >=99% structured gate, slower than the existing4.81us reference.')
    np.savez_compressed(OUT/'converged_reference_raw.npz',qubit=q,sideband=s,dt=cd.DT,
                        qubit_words=qw,sideband_words=sw)
    (OUT/'converged_reference_certificate.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETED STRUCTURED CERTIFICATE:SDKwords',report['sdk_words_compared'],
          'cutoff18columncomparisons',comparison_count,'independentamplitudes',output.size,
          'maxerror',error,'F',rows[-1]['fidelity'],flush=True)


def partial_pulse_coefficients(denominator):
    """Cauchy series of a fixed-duration pi/N pulse, through fourth order."""
    path=OUT/f'partial_pulse_series_N{denominator}.npz'
    if path.exists():
        return np.load(path)['coefficients']
    space=cd.Space(8,4);J=space.hamiltonian(1.,0.);X=space.hamiltonian(0.,1.)
    radius=8.;count=48;circle=[]
    for index in range(count):
        g=radius*np.exp(2j*np.pi*index/count);U=np.eye(16,dtype=complex)
        for omega in cd.pi_samples()/denominator:
            U=expm(-1j*cd.DT*(omega*X+g*J))@U
        circle.append(U)
    transformed=np.fft.fft(np.asarray(circle),axis=0)/count
    coefficients=np.asarray([transformed[k]/radius**k for k in range(5)])
    np.savez_compressed(path,coefficients=coefficients)
    return coefficients


def composite_sequence(word,denominator,outer_signs):
    """Same-sign perpendicular sandwiches have exact logical Pauli endpoints.

    R_y(alpha) R_x(pi) R_y(alpha)=R_x(pi); negative logical pulses use the
    inverse physical sandwich. The path, and therefore JC errors, changes.
    """
    pulses=[]
    for axis in word:
        is_x=axis%2==0;negative=axis>=2
        outer=(np.pi/2 if is_x else 0.)
        outer+=np.pi*(outer_signs[0 if is_x else 1]<0)
        outer+=np.pi*negative
        pulses.extend([(1/denominator,outer),(1.,axis*np.pi/2),(1/denominator,outer)])
    assert len(pulses)==3*len(word)
    return pulses


def composite_polynomial(pulses,coefficients):
    space=cd.Space(8,4);C=np.zeros((5,16,16),complex);C[0]=np.eye(16)
    for fraction,phase in pulses:
        D=np.exp(1j*phase*space.k)
        base=coefficients[round(1/fraction)]
        E=D[None,:,None]*base*D.conj()[None,None,:]
        C=pproduct(E,C)
    C=np.asarray([C[0].conj().T@term for term in C])
    A,B,E,F=C[1:]
    L=np.asarray([A,B-A@A/2,E-(A@B+B@A)/2+A@A@A/3,
                  F-(A@E+E@A+B@B)/2+(A@A@B+A@B@A+B@A@A)/3-A@A@A@A/4])
    a=np.diag(np.sqrt(np.arange(1,8)),1);G=np.kron(a@a-a.T@a.T,np.diag([1.,-1.]))
    basis=np.stack([G[:,:8].ravel(),(1j*np.diag(space.z))[:,:8].ravel(),
                    (1j*np.eye(16))[:,:8].ravel(),
                    (1j*np.diag(space.n*space.z))[:,:8].ravel()],axis=1)
    projected=np.linalg.lstsq(basis,L[1][:,:8].ravel(),rcond=None)[0]
    residual=L[1][:,:8].ravel()-basis@projected
    return dict(first_norm=float(np.linalg.norm(L[0][:,:8])),
                second_squeeze=float(projected[0].real),
                second_chi=float(projected[3].real),
                second_unwanted_norm=float(np.linalg.norm(residual)),
                cubic_norm=float(np.linalg.norm(L[2][:,:8])),
                quartic_norm=float(np.linalg.norm(L[3][:,:8])))


def composite_catalogue():
    pi_coefficients=partial_pulse_coefficients(1);rows=[]
    for denominator in [2,3,4]:
        coefficients={1:pi_coefficients,denominator:partial_pulse_coefficients(denominator)}
        for signs in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            for index,entry in enumerate(catalogue()):
                pulses=composite_sequence(entry['word'],denominator,signs)
                statistics=composite_polynomial(pulses,coefficients)
                row=dict(denominator=denominator,outer_signs=signs,word_index=index,
                         logical_word=entry['word'],physical_pulses=pulses,
                         native_pi_count=8,physical_pulse_count=24,**statistics)
                row['proper_target']=(statistics['first_norm']<1e-10 and
                                      statistics['second_unwanted_norm']<1e-10 and
                                      abs(statistics['second_chi'])<1e-10 and
                                      statistics['second_squeeze']>1e-8)
                rows.append(row)
        selected=[row for row in rows if row['denominator']==denominator and row['proper_target']]
        print('COMPOSITE SANDWICH FILTER:N',denominator,'64blocks;',len(selected),
              'properpositiveSQblocks;minimumfirstnorm',min(row['first_norm'] for row in rows if row['denominator']==denominator),flush=True)
    assert len(rows)==192
    (OUT/'composite_catalogue.json').write_text(json.dumps(rows,indent=2)+'\n')
    good=[row for row in rows if row['proper_target']]
    good.sort(key=lambda row:row['cubic_norm']/row['second_squeeze']**1.5)
    print('COMPLETED COMPOSITE CATALOGUE:192blocks;',len(good),'properpositiveSQblocks',flush=True)
    for row in good[:4]:
        print('COMPOSITE PROPER CANDIDATE',row['denominator'],row['outer_signs'],
              row['word_index'],{k:row[k] for k in ['second_squeeze','cubic_norm','quartic_norm']},flush=True)


def composite_native(pulses,repeats,g,levels=96):
    q=np.r_[np.zeros(4),np.concatenate([cd.pi_samples()*fraction*np.exp(1j*phase)
                                       for fraction,phase in pulses]*repeats),np.zeros(4)]
    s=g*np.r_[np.sin(np.pi*np.arange(4)/8)**2,np.ones(len(q)-8),
              np.cos(np.pi*np.arange(4)/8)**2]
    space=cd.Space(levels,9);inputs=np.eye(2*levels,dtype=complex)[:,space.inputs]
    target=cd.target_columns(space,'squeeze',cd.R_8DB)
    output=hw.propagate(q,s,inputs,cd.DT,order=12,fast=True)
    angle=np.angle(np.vdot(target[1::2],output[1::2]))-np.angle(np.vdot(target[0::2],output[0::2]))
    overlap=np.sum(target.conj()*np.exp(.5j*angle*space.z[:,None])*output,axis=0)
    return dict(fidelity=float(abs(np.sum(overlap)/18)**2),qubit_frame_Rz=float(-angle),
                time_us=len(q)*cd.DT,g=float(g),input_columns=18)


def composite_screen():
    rows=json.loads((OUT/'composite_catalogue.json').read_text())
    good=[row for row in rows if row['proper_target']]
    chosen=[]
    for denominator in [2,3,4]:
        group=[row for row in good if row['denominator']==denominator]
        group.sort(key=lambda row:row['cubic_norm']/row['second_squeeze']**1.5)
        chosen.extend(group[:2])
    results=[]
    for index,entry in enumerate(chosen):
        minimum=max(1,int(np.ceil((cd.R_8DB/2)/(entry['second_squeeze']*(MAX_G*.985)**2))))
        for repeats in range(minimum,7):
            g0=np.sqrt((cd.R_8DB/2)/(entry['second_squeeze']*repeats));history=[]
            def cost(g):
                result=composite_native(entry['physical_pulses'],repeats,g)
                history.append(result);return 1-result['fidelity']
            minimize_scalar(cost,bounds=(g0*.85,min(g0*1.15,MAX_G*.985)),method='bounded',
                            options=dict(maxiter=18,xatol=1e-6))
            best=max(history,key=lambda row:row['fidelity'])
            result=dict(candidate=index,repeats=repeats,source=entry,
                        scalar_evaluations=len(history),**best)
            results.append(result)
            print('COMPOSITE NATIVE',entry['denominator'],entry['outer_signs'],
                  entry['word_index'],'R',repeats,'F',result['fidelity'],'T',result['time_us'],flush=True)
    (OUT/'composite_screen.json').write_text(json.dumps(results,indent=2)+'\n')
    print('COMPLETED COMPOSITE SCREEN:',len(results),'gates;',
          18*sum(row['scalar_evaluations'] for row in results),'inputcolumns',flush=True)


def composite_pair_catalogue():
    pi_coefficients=partial_pulse_coefficients(1);rows=[]
    for denominator in [2,3,4]:
        coefficients={1:pi_coefficients,denominator:partial_pulse_coefficients(denominator)}
        for ordering in [1,-1]:
            for index,entry in enumerate(catalogue()):
                pulses=(composite_sequence(entry['word'],denominator,[ordering,ordering])+
                        composite_sequence(entry['word'],denominator,[-ordering,-ordering]))
                statistics=composite_polynomial(pulses,coefficients)
                row=dict(denominator=denominator,ordering=ordering,word_index=index,
                         logical_word=entry['word'],physical_pulses=pulses,
                         physical_pulse_count=48,**statistics)
                row['proper_target']=(statistics['first_norm']<1e-10 and
                                      statistics['second_unwanted_norm']<1e-10 and
                                      abs(statistics['second_chi'])<1e-10 and
                                      statistics['second_squeeze']>1e-8)
                rows.append(row)
    assert len(rows)==96
    (OUT/'composite_pair_catalogue.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('COMPLETED COMPOSITE CHIRALITY-PAIR:96blocks;',sum(row['proper_target'] for row in rows),
          'properpositiveSQblocks',flush=True)
    for denominator in [2,3,4]:
        good=[row for row in rows if row['denominator']==denominator and row['proper_target']]
        good.sort(key=lambda row:row['cubic_norm']/row['second_squeeze']**1.5)
        if good:
            row=good[0]
            print('COMPOSITE PAIR',denominator,'word',row['word_index'],'ordering',row['ordering'],
                  {k:row[k] for k in ['first_norm','second_squeeze','second_chi','second_unwanted_norm','cubic_norm','quartic_norm']},flush=True)


def composite_pair_screen():
    rows=json.loads((OUT/'composite_pair_catalogue.json').read_text());chosen=[]
    for denominator in [2,3,4]:
        good=[row for row in rows if row['denominator']==denominator and row['proper_target']]
        good.sort(key=lambda row:row['cubic_norm']/row['second_squeeze']**1.5)
        chosen.extend(good[:2])
    results=[]
    for index,entry in enumerate(chosen):
        minimum=max(1,int(np.ceil((cd.R_8DB/2)/(entry['second_squeeze']*(MAX_G*.985)**2))))
        for repeats in range(minimum,4):
            g0=np.sqrt((cd.R_8DB/2)/(entry['second_squeeze']*repeats));history=[]
            def cost(g):
                result=composite_native(entry['physical_pulses'],repeats,g)
                history.append(result);return 1-result['fidelity']
            minimize_scalar(cost,bounds=(g0*.8,min(g0*1.2,MAX_G*.985)),method='bounded',
                            options=dict(maxiter=20,xatol=1e-6))
            best=max(history,key=lambda row:row['fidelity'])
            result=dict(candidate=index,repeats=repeats,source=entry,
                        scalar_evaluations=len(history),**best)
            results.append(result)
            (OUT/'composite_pair_screen.json').write_text(json.dumps(results,indent=2)+'\n')
            print('COMPOSITE-PAIR NATIVE',entry['denominator'],'word',entry['word_index'],
                  'ordering',entry['ordering'],'R',repeats,'F',result['fidelity'],'T',result['time_us'],flush=True)
    print('COMPLETED COMPOSITE-PAIR SCREEN:',len(results),'gates;',
          18*sum(row['scalar_evaluations'] for row in results),'inputcolumns',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['magnus','screen','chirality','chirality_screen',
                                       'balanced','balanced_screen','mixture','convergence','export','certify',
                                       'composite_catalogue','composite_screen','composite_pair_catalogue',
                                       'composite_pair_screen'])
    args=parser.parse_args()
    if args.stage=='magnus':magnus_test()
    elif args.stage=='screen':screen()
    elif args.stage=='chirality':chirality_test()
    elif args.stage=='chirality_screen':screen(True)
    elif args.stage=='balanced':chirality_test(True)
    elif args.stage=='balanced_screen':screen(True,True)
    elif args.stage=='mixture':mixture_screen()
    elif args.stage=='convergence':convergence_screen()
    elif args.stage=='export':export_rejected()
    elif args.stage=='certify':certify_convergence()
    elif args.stage=='composite_catalogue':composite_catalogue()
    elif args.stage=='composite_screen':composite_screen()
    elif args.stage=='composite_pair_catalogue':composite_pair_catalogue()
    else:composite_pair_screen()
