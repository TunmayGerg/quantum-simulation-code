"""Identify the finite-coupling error, separately from formal Magnus terms.

Compute the complete truncated propagator, remove the exact CD and declared
qubit frame, and take the principal matrix logarithm. Check low-input error
columns at two larger oscillator cutoffs before interpreting fitted operators.
A polynomial fit is diagnostic and reports its unexplained residual.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time
import numpy as np
from scipy.linalg import expm,logm
from project_paths import ROOT
from notebook_figures import certificate,drives
import cd_trains as cd
import hardware_check as hw

OUT=ROOT/'runs/phase_arc_cd'


def analyse(name,levels):
    started=time.time();r=certificate(name);q,s,qf,sf,w=drives(name)
    sp=cd.Space(levels,9);size=2*levels
    V=hw.propagate(qf,sf,np.eye(size,dtype=complex),cd.DT/8,order=12,fast=True)
    V*=np.exp(-1j*r['qubit_frame_Rz']*sp.z/2)[:,None]
    a=np.diag(np.sqrt(np.arange(1,levels)),1);ad=a.T;I=np.eye(levels);n=ad@a
    D=expm(ad-a);plus=(D+D.T)/2;minus=(D-D.T)/2
    target=np.zeros((size,size),complex)
    target[0::2,0::2]=target[1::2,1::2]=plus
    target[0::2,1::2]=target[1::2,0::2]=minus
    E=target.conj().T@V;phase=np.angle(np.trace(E[np.ix_(sp.inputs,sp.inputs)]));E*=np.exp(-1j*phase)
    G=1j*logm(E);unitary=float(np.linalg.norm(E.conj().T@E-np.eye(size)))
    assert unitary<1e-9 and np.max(abs(G-G.conj().T))<1e-8
    polynomials=[('I',I),('X',a+ad),('P',1j*(ad-a)),('n',n),
        ('a2+adag2',a@a+ad@ad),('i(adag2-a2)',1j*(ad@ad-a@a)),
        ('{n+1/2,X}/2',((n+.5*I)@(a+ad)+(a+ad)@(n+.5*I))/2),
        ('{n+1/2,P}/2',((n+.5*I)@(1j*(ad-a))+(1j*(ad-a))@(n+.5*I))/2),
        ('a3+adag3',a@a@a+ad@ad@ad),('i(adag3-a3)',1j*(ad@ad@ad-a@a@a))]
    paulis=[('I',np.eye(2)),('sigma_x',np.array([[0,1],[1,0]])),
        ('sigma_y',np.array([[0,-1j],[1j,0]])),('sigma_z',np.diag([1,-1]))]
    names=[];matrices=[]
    for qname,Q in paulis:
        for oname,O in polynomials:names.append(qname+' * '+oname);matrices.append(np.kron(O,Q)[:,sp.inputs])
    B=np.stack([M.ravel() for M in matrices],axis=1);g=G[:,sp.inputs].ravel()
    # Real coefficients enforce a Hermitian polynomial generator.
    A=np.r_[B.real,B.imag];v=np.r_[g.real,g.imag];coef=np.linalg.lstsq(A,v,rcond=None)[0]
    residual=np.linalg.norm(A@coef-v);total=np.linalg.norm(v)
    terms=[]
    for j,c in enumerate(coef):
        reduced=np.delete(A,j,axis=1);fit=np.linalg.lstsq(reduced,v,rcond=None)[0]
        unique=max(0,np.linalg.norm(reduced@fit-v)**2-residual**2)**.5
        terms.append(dict(operator=names[j],coefficient=float(c),unique_generator_norm=float(unique),
            contribution_norm=float(abs(c)*np.linalg.norm(B[:,j]))))
    terms.sort(key=lambda x:-x['unique_generator_norm'])
    print('FINITE GATE ERROR',name,levels,'fullcolumns',size,'amplitudes',V.size,
          'unitarity',unitary,'fitresidualfraction',residual/total,'seconds',time.time()-started,flush=True)
    return dict(name=name,levels=levels,full_columns=size,amplitudes=V.size,unitarity_error=unitary,
        fit_relative_residual=float(residual/total),terms=terms),G[:,sp.inputs]


if __name__=='__main__':
    results=[]
    for name in ['cd_constant_reference','arc_lifted_n12']:
        previous=None;previous_record=None
        for levels in [64,96,128]:
            r,G=analyse(name,levels)
            if previous is not None:
                pad=np.zeros_like(G);pad[:len(previous)]=previous
                difference=float(np.max(abs(G-pad)));r['low_input_logarithm_cutoff_change']=difference
                print('ERROR LOG CUTOFF',name,'maxdifference',difference,flush=True)
                r['logarithm_stable']=bool(difference<1e-7)
                old={term['operator']:term['coefficient'] for term in previous_record['terms']}
                change=max(abs(term['coefficient']-old[term['operator']]) for term in r['terms'])
                r['polynomial_coefficient_cutoff_change']=float(change)
                r['polynomial_fit_stable']=bool(change<1e-8 and
                    abs(r['fit_relative_residual']-previous_record['fit_relative_residual'])<1e-7)
                if levels==128:assert r['polynomial_fit_stable']
            results.append(r);previous=G;previous_record=r
    assert len(results)==6
    (OUT/'finite_gate_error.json').write_text(json.dumps(results,indent=2)+'\n')
    print('FINITE ERROR ANALYSIS:6fullpropagators;',sum(r['full_columns'] for r in results),'columns;',
          sum(r['amplitudes'] for r in results),'complex amplitudes',flush=True)
