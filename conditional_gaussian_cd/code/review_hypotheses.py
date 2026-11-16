"""Independent finite-pulse checks of the structured-gate hypotheses.

This file never fits a target or changes a pulse recipe. It derives the CD
first moment by an exact qubit integral, and independently extracts a native
block logarithm by Cauchy differentiation of matrix products.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import numpy as np
from scipy.linalg import expm
from project_paths import ROOT
import cd_trains as cd

OUT = ROOT / 'runs' / 'independent_review'
PAULI = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]], complex)


def cd_word(n, psi):
    """Four-pulse inverse loop, X, five-ns dwell, same loop, X."""
    block = [(0., 1/n), (psi, 1.), (psi+np.pi, 1.), (np.pi, 1/n)]
    return block + [(0.,1.), (0.,0.)] + block + [(0.,1.)]


def exact_qubit_moment(word):
    """Integrate Q(t)^dagger sigma_j Q(t) exactly on every DAC sample.

    The Rodrigues integral below is independent of the Gauss quadrature in
    the synthesis code. The final pair contains coefficients multiplying X
    and P in H_JC/g=(sigma_x X+sigma_y P)/2.
    """
    q = np.eye(2, dtype=complex)
    moments = np.zeros((2,2,2), complex)
    # Both outer ramps see the scalar final qubit endpoint. Their combined
    # sideband area is exactly five ns for the supplied four-sample ramps.
    moments[0] += .005*PAULI[0]/2
    moments[1] += .005*PAULI[1]/2
    for phase, fraction in word:
        if fraction == 0:
            moments += .005*np.array([q.conj().T@p@q for p in PAULI[:2]])/2
            continue
        axis = np.array([np.cos(phase), np.sin(phase), 0.])
        generator = np.einsum('i,ijk->jk',axis,PAULI)
        for omega in cd.pi_samples()*fraction:
            w = 2*omega
            for j in range(2):
                v = np.eye(3)[j]
                if abs(w) < 1e-12:
                    integral = cd.DT*v
                else:
                    sine = np.sin(w*cd.DT)/w
                    integral = (sine*v - (1-np.cos(w*cd.DT))/w*np.cross(axis,v)
                                +(cd.DT-sine)*axis*np.dot(axis,v))
                integrated = np.einsum('i,ijk->jk',integral,PAULI)
                moments[j] += q.conj().T@integrated@q/2
            q = expm(-1j*omega*cd.DT*generator)@q
    coefficients = np.array([[np.trace(p@m).real/2 for p in PAULI] for m in moments])
    return q, moments, coefficients


def native_series(word, levels=8, count=32, radius=2., ramps=True):
    """Cauchy derivatives of the whole native word, with no BCH operations."""
    space = cd.Space(levels,3)
    J = space.hamiltonian(1.,0.)
    drive = space.hamiltonian(0.,1.)
    result = []
    for k in range(count):
        g = radius*np.exp(2j*np.pi*k/count)
        if ramps:
            # Since each ramp has only JC on, its sample exponentials commute.
            U = expm(-1j*g*J*.001875)
        else:
            U = np.eye(2*levels,dtype=complex)
        cache = {}
        for phase, fraction in word:
            if fraction == 0:
                U = expm(-1j*g*J*.005)@U
                continue
            if fraction not in cache:
                P = np.eye(2*levels,dtype=complex)
                for omega in cd.pi_samples()*fraction:
                    P = expm(-1j*cd.DT*(g*J+omega*drive))@P
                cache[fraction] = P
            # Joint cavity/qubit phase rotates the qubit drive while preserving
            # JC. This identity is independently checked by native products.
            D = np.exp(1j*phase*space.k)
            P = D[:,None]*cache[fraction]*D.conj()[None,:]
            U = P@U
        if ramps:
            U = expm(-1j*g*J*.003125)@U
        result.append(U)
    C = np.fft.fft(np.array(result),axis=0)/count
    C = np.array([C[k]/radius**k for k in range(5)])
    Q = C[0]
    V = np.array([Q.conj().T@v for v in C])
    A,B,C3 = V[1:4]
    logs = np.array([A, B-A@A/2, C3-(A@B+B@A)/2+A@A@A/3])
    return space, Q, V, logs


def test_cd():
    rows=[]
    for n in [1,2,3,4]:
        for psi in [0.,np.pi/4]:
            word=cd_word(n,psi)
            Q,M,c=exact_qubit_moment(word)
            space,Qnative,V,L=native_series(word)
            a=np.diag(np.sqrt(np.arange(1,space.levels)),1)
            quadratures=[a+a.T,1j*(a.T-a)]
            expected=-1j*sum(np.kron(O,m) for O,m in zip(quadratures,M))
            low=np.arange(6)
            residual=float(np.linalg.norm(c[:,1:]))
            first_error=float(np.max(abs(L[0]-expected)))
            closure=float(np.max(abs(Q+np.eye(2))))
            closure_native=float(np.max(abs(Qnative+np.eye(2*space.levels))))
            # A common qubit-Z frame is allowed, but it cannot hide oscillator
            # dependent number rotations or mixed quadrature errors.
            allowed=1j*np.kron(np.eye(space.levels),PAULI[2])
            v=L[1][:,low].ravel();b=allowed[:,low].ravel()
            second_residual=float(np.linalg.norm(v-b*np.vdot(b,v)/np.vdot(b,b)))
            row=dict(N=n,psi=psi,qubit_closure_error=closure,
                     native_closure_error=closure_native,first_coefficients=c.tolist(),
                     unwanted_first_force_norm=residual,first_native_max_error=first_error,
                     useful_first_force=float(np.linalg.norm(c[:,0])),
                     beta1_leading_swap_ns=float(np.pi*np.linalg.norm(c[:,0])/2*1000),
                     second_unwanted_norm=second_residual,
                     native_amplitudes_compared=int(expected.size))
            assert max(closure,closure_native)<1e-12
            assert residual<1e-12 and first_error<1e-12
            rows.append(row)
            print('CD MOMENT',json.dumps(row),flush=True)
    assert len(rows)==8
    payload=dict(gates_compared=len(rows),native_amplitudes_compared=sum(r['native_amplitudes_compared'] for r in rows),results=rows)
    (OUT/'cd_moments.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('CD COMPARISONS',len(rows),'gates;',payload['native_amplitudes_compared'],'complex first-derivative amplitudes',flush=True)


def operator_fit(matrix, levels, degree):
    """Resolve the error into a named Hermitian polynomial operator basis."""
    a=np.diag(np.sqrt(np.arange(1,levels)),1);ad=a.T;n=ad@a
    X=a+ad;P=1j*(ad-a);number=n+np.eye(levels)/2
    if degree==2:
        oscillators=[('I',np.eye(levels)),('n+1/2',number),('a2+adag2',a@a+ad@ad),
                     ('i(adag2-a2)',1j*(ad@ad-a@a))]
    else:
        oscillators=[('X',X),('P',P),('a3+adag3',a@a@a+ad@ad@ad),
                     ('i(adag3-a3)',1j*(ad@ad@ad-a@a@a)),
                     ('{n+1/2,X}/2',(number@X+X@number)/2),
                     ('{n+1/2,P}/2',(number@P+P@number)/2)]
    names=[];basis=[]
    for qubit,p in zip(['I','sigma_x','sigma_y','sigma_z'],[np.eye(2),*PAULI]):
        for name,O in oscillators:
            names.append(qubit+' * '+name);basis.append(-1j*np.kron(O,p))
    # Include n=3 for cubic polynomials: otherwise a^3 annihilates every
    # probed input, making the a^3/adag^3 coefficients non-identifiable.
    low=np.arange(6 if degree==2 else 8);B=np.array([b[:,low].ravel() for b in basis]).T
    v=matrix[:,low].ravel();coefficients=np.linalg.lstsq(B,v,rcond=None)[0]
    residual=float(np.linalg.norm(v-B@coefficients))
    rows=[dict(operator=name,coefficient_real=float(c.real),coefficient_imag=float(c.imag))
          for name,c in zip(names,coefficients) if abs(c)>1e-10]
    rows.sort(key=lambda r:-abs(r['coefficient_real']))
    return dict(residual_norm=residual,terms=rows)


def test_cd_errors():
    rows=[]
    words=[('ten_pulse_twirl',cd_word(1,np.pi/4)),
           ('eight_pulse_same_axis',[(p,1.) for p in [0,0,np.pi,np.pi,0,0,np.pi,np.pi]])]
    for name,word in words:
        space,Q,V,L=native_series(word,ramps=False)
        row=dict(name=name,second=operator_fit(L[1],space.levels,2),
                 third=operator_fit(L[2],space.levels,3),
                 second_norm=float(np.linalg.norm(L[1][:,:6])),
                 third_norm=float(np.linalg.norm(L[2][:,:6])))
        rows.append(row);print('CD ERROR OPERATORS',json.dumps(row),flush=True)
    assert len(rows)==2
    (OUT/'cd_error_operators.json').write_text(json.dumps(rows,indent=2)+'\n')


def test_same_axis_cubic():
    """A sign-definite cubic term explains why another X twirl cannot help.

    The principal oscillator polynomial can treat X,P as commuting symbols:
    their commutator only contributes lower-degree polynomials. With
    b=(cos(theta),-sin(theta)) and integral b=0, the Magnus triple integral
    multiplying sigma_x X P^2 reduces to -2 integral |integral b|^2.
    The factors (g/2)^3 give the coefficient -integral|F|^2/4 in log(U)/g^3.
    """
    word=[(p,1.) for p in [0,0,np.pi,np.pi,0,0,np.pi,np.pi]]
    nodes,weights=np.polynomial.legendre.leggauss(16)
    times=(nodes+1)*cd.DT/2
    theta=0.;F=0j;integral=0.
    for phase,fraction in word:
        sign=1. if abs(np.cos(phase)-1)<1e-12 else -1.
        for omega in cd.pi_samples()*sign*fraction:
            w=2*omega
            if abs(w)<1e-12:
                increments=np.exp(-1j*theta)*times
                final=np.exp(-1j*theta)*cd.DT
            else:
                increments=np.exp(-1j*theta)*(-np.expm1(-1j*w*times))/(1j*w)
                final=np.exp(-1j*theta)*(-np.expm1(-1j*w*cd.DT))/(1j*w)
            integral+=float(np.dot(weights,abs(F+increments)**2)*cd.DT/2)
            F+=final;theta+=w*cd.DT
    space,Q,V,L=native_series(word,ramps=False)
    decomposition=operator_fit(L[2],space.levels,3)
    c=next(t['coefficient_real'] for t in decomposition['terms']
           if t['operator']=='sigma_x * {n+1/2,X}/2')
    # W(X P^2)={n+1/2,X}/2-(a^3+adag^3). The two native
    # coefficients must be opposites if this is the principal cubic error.
    opposite=next(t['coefficient_real'] for t in decomposition['terms']
                  if t['operator']=='sigma_x * a3+adag3')
    predicted=-integral/4
    error=abs(c-predicted)
    assert abs(F)<1e-13 and error<1e-14 and abs(c+opposite)<1e-14
    payload=dict(word=['X','X','-X','-X']*2,first_transverse_integral_abs=float(abs(F)),
                 integral_abs_F_squared_us3=integral,
                 predicted_sigma_x_Weyl_XP2_coefficient_us3=predicted,
                 native_coefficient_us3=c,coefficient_error=error,
                 coefficient_comparisons=2,
                 conclusion='The longitudinal principal cubic coefficient is strictly negative for finite X-only control; another X twirl cannot cancel it.')
    (OUT/'same_axis_cubic.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('SAME AXIS CUBIC',json.dumps(payload),flush=True)


def test_bch():
    terms=np.load(ROOT/'runs'/'fixed_sideband'/'proper_block_terms.npz')
    # Positive-squeeze and a generic zero-first-moment word probe independent
    # pieces of the catalogue, without fitting any target coefficient here.
    selected=[int(np.flatnonzero((terms['projection'][:,0].real>1e-6)&(terms['residual']<1e-9))[0]),0]
    rows=[]
    for index in selected:
        word=terms['words'][index]
        phases=np.arange(4)*np.pi/2
        _,Q,V,L=native_series([(phases[p],1.) for p in word],ramps=False)
        low=np.arange(6)
        errors=[float(np.max(abs(L[k][:,low]-terms[name][index][:,low])))
                for k,name in enumerate(['first','second','third'])]
        count=3*L.shape[1]*len(low)
        assert max(errors)<1e-11,errors
        row=dict(word=word.tolist(),index=index,first_second_third_max_errors=errors,
                 coefficient_amplitudes_compared=count)
        rows.append(row)
        print('BCH NATIVE',json.dumps(row),flush=True)
    assert len(rows)==2
    payload=dict(gates_compared=2,coefficient_amplitudes_compared=sum(r['coefficient_amplitudes_compared'] for r in rows),results=rows)
    (OUT/'bch_native.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('BCH COMPARISONS',payload['coefficient_amplitudes_compared'],'complex coefficient amplitudes',flush=True)


def test_phase_arc():
    import hardware_check as hw
    g=np.pi/(2*.200)
    before=64; ramp=8; after=64
    u=np.arange(ramp+1)/ramp
    phase=np.r_[np.zeros(before),np.pi*(10*u**3-15*u**4+6*u**5),np.full(after-1,np.pi)]
    envelope=np.ones(len(phase));envelope[:4]=np.sin(np.pi*np.arange(4)/8)**2
    envelope[-4:]=np.cos(np.pi*np.arange(4)/8)**2
    s=g*envelope*np.exp(1j*phase)
    qs,ss=hw.dac_scales(.200)
    words,decoded=hw.to_words_and_back(s,g,ss)
    played=hw.rfdc_interpolate(decoded)
    flat=hw.rfdc_interpolate(g*envelope)
    delay=145 # fine samples; 22.65625 ns common filter group delay
    region=slice(before*8+delay-80,(before+ramp)*8+delay+80)
    amp=abs(played[region])/g
    raw=abs(s[4:-4])/g
    restored=abs(decoded[4:-4])/g
    # Comparing the same central fine samples eliminates outer-startup tails.
    expected=abs(flat[region])/g
    ratio=amp/expected
    payload=dict(programmed_samples=len(s),fine_samples_compared=len(ratio),
                 phase_change_ns=10.,programmed_radius_error=float(np.max(abs(raw-1))),
                 int16_radius_min=float(min(restored)),int16_radius_max=float(max(restored)),
                 played_radius_min=float(min(amp)),played_radius_max=float(max(amp)),
                 radius_relative_to_flat_min=float(min(ratio)),radius_relative_to_flat_max=float(max(ratio)),
                 maximum_played_g=float(np.max(abs(played))),cap_g=g,
                 maximum_global_ratio=float(np.max(abs(played))/g),
                 cap_safe_nominal_swap_ns=float(200*np.max(abs(played))/g),
                 assertion='Constant radius before the FIR does not imply constant radius after the FIR.')
    assert len(s)==136 and len(ratio)==224
    assert payload['programmed_radius_error']<1e-14
    (OUT/'phase_arc.json').write_text(json.dumps(payload,indent=2)+'\n')
    np.savez_compressed(OUT/'phase_arc_arrays.npz',phase=phase,programmed=s,decoded=decoded,played=played,flat=flat)
    print('PHASE ARC',json.dumps(payload),flush=True)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cd','errors','cubic','bch','arc','all'])
    stage=parser.parse_args().stage
    if stage in ['cd','all']:test_cd()
    if stage in ['errors','all']:test_cd_errors()
    if stage in ['cubic','all']:test_same_axis_cubic()
    if stage in ['bch','all']:test_bch()
    if stage in ['arc','all']:test_phase_arc()
