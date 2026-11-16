"""Test a bridge-free, fixed-sideband symmetry suggested in the research debate.

An open qubit path ends at Z. Playing it twice conjugates the second block's
interaction error by Z. This can cancel qubit-X/Y cubic terms, but not the
qubit-Z cubic terms. Require zero first JC moment before claiming cancellation.
Meet-in-the-middle searches exact finite 30 ns pi/pi-over-two pulse paths.
There are no per-sample optimization variables and no sideband phase changes.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import itertools
import json
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import expm
from project_paths import ROOT
import strategic_cd as controls
from strategic_squeezing import partial_pulse_coefficients,pproduct
import cd_trains as cd

OUT=ROOT/'runs/open_z_squeezing';OUT.mkdir(exist_ok=True)
PAULIS=[controls.X,controls.Y,controls.Z]
LETTERS=[(phase,fraction) for fraction in (1.,.5) for phase in np.arange(4)*np.pi/2]


def frame_key(Q):
    R=np.array([[np.trace(P@Q.conj().T@S@Q).real/2 for S in PAULIS] for P in PAULIS])
    assert np.max(abs(R-np.round(R)))<1e-12
    return tuple(np.round(R).astype(int).ravel())


def vector(M):return np.r_[M.real.ravel(),M.imag.ravel()]


def moments():
    result=[]
    for phase,fraction in LETTERS:
        Q,M=controls.native_qubit_moment(fraction)
        D=np.diag(np.exp(np.array([-1,1])*1j*phase/2))
        result.append((D@Q@D.conj().T,np.exp(-1j*phase)*D@M@D.conj().T))
    return result


def path_moment(word,pulses):
    Q=np.eye(2,dtype=complex);M=np.zeros((2,2),complex)
    for letter in word:
        q,m=pulses[letter];M+=Q.conj().T@m@Q;Q=q@Q
    return Q,M


def logarithm(C):
    # Remove the qubit-only endpoint; expansion variable is constant JC g.
    C=np.asarray([C[0].conj().T@term for term in C])
    A,B,E,F=C[1:]
    return np.asarray([A,B-A@A/2,E-(A@B+B@A)/2+A@A@A/3,
        F-(A@E+E@A+B@B)/2+(A@A@B+A@B@A+B@A@A)/3-A@A@A@A/4])


def polynomial(word):
    coefficients={n:partial_pulse_coefficients(n) for n in (1,2)}
    space=cd.Space(8,4);C=np.zeros((5,16,16),complex);C[0]=np.eye(16)
    for index in word:
        phase,fraction=LETTERS[index];D=np.exp(1j*phase*space.k)
        E=D[None,:,None]*coefficients[round(1/fraction)]*D.conj()[None,None,:]
        C=pproduct(E,C)
    return C


def main(half_length):
    started=time.time();pulses=moments()
    words=list(itertools.product(range(8),repeat=half_length));groups={}
    assert len(words)==8**half_length
    frames=[];first=[]
    for index,word in enumerate(words):
        Q,M=path_moment(word,pulses);frames.append(Q);first.append(M)
        groups.setdefault(frame_key(Q),[]).append(index)
    trees={key:(ids,cKDTree(np.array([vector(first[i]) for i in ids]))) for key,ids in groups.items()}
    selected=[];residuals=[];endpoint_pairs=0;queries=0;nearest=float('inf')
    for i,(Q,M) in enumerate(zip(frames,first)):
        key=frame_key(controls.Z@Q.conj().T)
        if key not in trees:continue
        ids,tree=trees[key]
        endpoint_pairs+=len(ids);queries+=1
        required=-Q@M@Q.conj().T
        nearest=min(nearest,float(tree.query(vector(required))[0]))
        for k in tree.query_ball_point(vector(required),r=1e-10):
            j=ids[k];full=words[i]+words[j]
            endpoint,moment=path_moment(full,pulses)
            assert frame_key(endpoint)==frame_key(controls.Z)
            residual=float(np.linalg.norm(moment));assert residual<1e-10
            selected.append(full);residuals.append(residual)
    assert endpoint_pairs>0 and queries>0
    print('OPEN-Z MOMENT SEARCH:',8**(2*half_length),'implicit words;',len(words),'half-paths;',
          endpoint_pairs,'Z-endpoint pairs;',queries,'exact nearest-neighbor queries;',
          len(selected),'zero-first-average Z endpoints;minimum residual',nearest,flush=True)
    a=np.diag(np.sqrt(np.arange(1,8)),1);G=np.kron(a@a-a.T@a.T,controls.Z)
    z=1j*np.kron(np.eye(8),controls.Z);I=1j*np.eye(16);low=np.arange(8)
    basis=np.stack([g[:,low].ravel() for g in [G,z,I]],axis=1);pinv=np.linalg.pinv(basis)
    # Independent formulations: time-integrated qubit moment versus full
    # oscillator/qubit Cauchy coefficient, for explicitly counted test paths.
    errors=[]
    rng=np.random.default_rng(115)
    for _ in range(16):
        word=tuple(rng.integers(0,8,2*half_length));_,M=path_moment(word,pulses)
        expected=-1j*(np.kron(a,M)+np.kron(a.T,M.conj().T))
        actual=logarithm(polynomial(word))[0]
        errors.append(float(np.max(abs((expected-actual)[:,low]))))
    assert len(errors)==16 and max(errors)<1e-11
    print('OPEN-Z FIRST-MOMENT VALIDATION:16 paths;',16*16*8,
          'complex coefficient amplitudes;max error',max(errors),flush=True)
    rows=[]
    # Verify all proper paths, rather than a fitted or cherry-picked subset.
    for index,word in enumerate(selected):
        C=polynomial(word);L=logarithm(C);pair=logarithm(pproduct(C,C))
        coeff=pinv@pair[1][:,low].ravel()
        unwanted=np.linalg.norm(pair[1][:,low].ravel()-basis@coeff)
        Z=np.kron(np.eye(8),controls.Z)
        even=(L[2]+Z@L[2]@Z)/2
        check=float(np.max(abs((pair[2]-2*even)[:,low])))
        assert np.linalg.norm(L[0][:,low])<1e-9 and check<1e-10
        row=dict(word=list(word),first_norm=float(np.linalg.norm(L[0][:,low])),
            squeeze_second=float(coeff[0].real),second_unwanted=float(unwanted),
            single_cubic=float(np.linalg.norm(L[2][:,low])),paired_cubic=float(np.linalg.norm(pair[2][:,low])),
            cubic_symmetry_error=check,paired_quartic=float(np.linalg.norm(pair[3][:,low])),
            proper_squeeze=bool(coeff[0].real>1e-8 and unwanted<1e-9 and abs(coeff[0].imag)<1e-9))
        rows.append(row)
        if (index+1)%500==0:print('OPEN-Z POLYNOMIAL:',index+1,'/',len(selected),flush=True)
    report=dict(implicit_words=8**(2*half_length),half_paths=len(words),pulse_ns=30,
        fractions=[1.,.5],sideband='constant magnitude and phase; no outer ramps in primitive',
        zero_first_Z_endpoints=len(selected),max_first_residual=max(residuals,default=None),
        Z_endpoint_pairs=endpoint_pairs,nearest_queries=queries,minimum_first_moment_norm=nearest,
        independent_moment_test_paths=16,independent_moment_amplitudes=2048,
        independent_moment_max_error=max(errors),
        proper_squeeze_pairs=sum(r['proper_squeeze'] for r in rows),results=rows,seconds=time.time()-started,
        scope='Exact finite catalogue at fixed length. No claim about other angles, axes, timings, or global speed limits.')
    (OUT/f'length{2*half_length}.json').write_text(json.dumps(report,indent=2)+'\n')
    print('OPEN-Z RESULT:',report['implicit_words'],'words;',len(rows),'polynomial checks;',
          report['proper_squeeze_pairs'],'proper squeezing pairs;',report['seconds'],'seconds',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--half-length',type=int,default=4,choices=[3,4,5])
    main(parser.parse_args().half_length)
