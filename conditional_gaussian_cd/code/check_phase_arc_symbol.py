"""Compare the principal-symbol design model with full quantum derivatives."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json
import numpy as np
from scipy.linalg import expm
from project_paths import ROOT
from phase_arc_cd import ArcCD,symbol,OUT
import cd_trains as cd


def check(p,model,require_symmetric=True):
    q,s=model.raw(p);s=s/np.exp(p[-1]);levels=8;space=cd.Space(levels,4)
    a=np.diag(np.sqrt(np.arange(1,levels)),1);X=a+a.T;P=1j*(a.T-a)
    sx=np.array([[0,1],[1,0]]);sy=np.array([[0,-1j],[1j,0]]);sz=np.diag([1,-1])
    SP=(sx-1j*sy)/2;d=2*levels
    matrices=[];radius=2.;count=40
    for k in range(count):
        g=radius*np.exp(2j*np.pi*k/count);U=np.eye(d,dtype=complex);cache={}
        for drive,pump in zip(q,s):
            key=(drive,pump)
            if key not in cache:
                Hq=np.kron(np.eye(levels),drive*SP+drive.conjugate()*SP.conj().T)
                J=np.kron(a,pump*SP)+np.kron(a.T,pump.conjugate()*SP.conj().T)
                cache[key]=expm(-1j*cd.DT*(Hq+g*J))
            U=cache[key]@U
        matrices.append(U)
    C=np.fft.fft(np.array(matrices),axis=0)/count
    C=np.array([C[k]/radius**k for k in range(4)])
    C=np.array([C[0].conj().T@z for z in C]);A,B,E=C[1:]
    L1=A;L2=B-A@A/2;L3=E-(A@B+B@A)/2+A@A@A/3
    polys=[X@X@X,(X@X@P+X@P@X+P@X@X)/3,
           (X@P@P+P@X@P+P@P@X)/3,P@P@P,X,P]
    low=np.arange(8)
    # Oscillator commutators also produce unconditional linear displacements.
    # Keep the identity-qubit terms; otherwise a correct principal-symbol
    # match would leave a misleading unexplained residual in the full fit.
    basis=np.stack([(-1j*np.kron(poly,pauli))[:,low].ravel() for poly in polys for pauli in [np.eye(2),sx,sy,sz]],axis=1)
    coefficients=np.linalg.lstsq(basis,L3[:,low].ravel(),rcond=None)[0].reshape(6,4)
    residual=np.linalg.norm(basis@coefficients.ravel()-L3[:,low].ravel())
    first,second,cubic=symbol(q,s,32)
    predicted=-1j*sum(np.kron(poly,sum(c*Q for c,Q in zip(coef,[sx,sy,sz]))) for poly,coef in zip([X,P],first))
    first_error=float(np.max(abs((L1-predicted)[:,low])))
    third_error=float(np.max(abs(coefficients[:4,1:]-cubic)))
    print('CHECK DETAILS',residual,first_error,third_error,flush=True)
    assert residual<1e-11 and first_error<1e-5 and third_error<1e-8
    if require_symmetric:assert np.linalg.norm(L2[:,low])<1e-11
    return dict(first_amplitudes=d*len(low),third_operator_coefficients=12,
        first_max_error=first_error,third_max_error=third_error,operator_fit_residual=float(residual),
        exact_second_norm=float(np.linalg.norm(L2[:,low])),principal_third_quantum=coefficients[:4,1:].real.tolist(),
        unconditional_linear_third=coefficients[4:,0].real.tolist(),
        principal_third_symbol=cubic.tolist())


if __name__=='__main__':
    model=ArcCD(8,24,True);m=model.nfree
    records=[]
    for phase_offsets in [np.zeros(m),np.array([.4,-.3,.7,-.2])]:
        p=np.r_[[0,0,np.pi,np.pi],-np.pi/2+phase_offsets,np.log(6.)]
        row=check(p,model);records.append(row)
        print('SYMBOL QUANTUM CHECK:128 first amplitudes;12 cubic coefficients;firsterror',row['first_max_error'],
              'cubicerror',row['third_max_error'],'secondnorm',row['exact_second_norm'],flush=True)
    assert len(records)==2
    (OUT/'symbol_quantum_validation.json').write_text(json.dumps(records,indent=2)+'\n')
    print('SYMBOL VALIDATION:2 waveforms;256 first amplitudes;24 cubic coefficients',flush=True)
