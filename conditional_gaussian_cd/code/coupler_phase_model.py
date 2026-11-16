"""Actual-IQ transition shaping and an explicitly driven cosine LINC model.

The measured 6.8 GHz linear-point frequency and approximate 4 GHz tuning span
constrain EL*EC and 2EJ/EL. EC, normal-mode participations, array count and the
analog flux transfer are still assumptions; they are never labelled measured.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import ctypes,hashlib,subprocess,math,json,time
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.special import j1,eval_genlaguerre,gammaln
from scipy.optimize import brentq
from acadia.sample_arithmetic import sample_to_complex
import cd_trains as cd
import hardware_check as hw
from scaled_cd import target


def smooth_motif(parameters,repeats,gain=1.,shape='phase',transition_ns=10.):
    """Smooth each pump boundary inside its existing 30 ns cells.

    A ten-ns boundary uses 4 interface samples before and 4 after it. No
    extra delay is inserted. Qubit pi envelopes are unchanged. `zero` ramps
    down for5ns, changes phase at zero amplitude, then ramps up for5ns.
    """
    p=np.asarray(parameters,float);qh=np.array([0.,0.,np.pi,np.pi])+p[:4]
    qphase=np.tile(np.r_[qh,qh[::-1]+np.pi],repeats)
    ph=-np.pi/2+p[4:8];phase=np.tile(np.r_[ph,ph[::-1]],repeats)
    q=np.r_[np.zeros(4),np.concatenate([cd.pi_samples()*np.exp(1j*v) for v in qphase]),np.zeros(4)]
    s=np.r_[np.sin(np.pi*np.arange(4)/8)**2*np.exp(1j*phase[0]),
            np.repeat(np.exp(1j*phase),24),np.cos(np.pi*np.arange(4)/8)**2*np.exp(1j*phase[-1])]
    half=int(round(transition_ns/2/(cd.DT*1000)))
    assert abs(2*half*cd.DT*1000-transition_ns)<1e-12 and half<=12
    if shape!='abrupt':
        u=np.arange(2*half)/(2*half);h=10*u**3-15*u**4+6*u**5
        for j in range(1,len(phase)):
            left,right=np.exp(1j*phase[j-1]),np.exp(1j*phase[j])
            if abs(left-right)<1e-12:continue
            center=4+24*j
            if shape=='phase':
                delta=np.angle(right/left);values=left*np.exp(1j*delta*h)
            elif shape=='cartesian':values=(1-h)*left+h*right
            elif shape=='zero':
                values=np.r_[left*np.cos(np.pi*np.arange(half)/(2*half))**2,
                             right*np.sin(np.pi*np.arange(half)/(2*half))**2]
            else:raise ValueError(shape)
            s[center-half:center+half]=values
    qs,ss=hw.dac_scales(.3);g=np.pi/(2*.3)
    qw,_=hw.to_words_and_back(q,np.pi/cd.PI_TIME,qs,qubit=True)
    sw,_=hw.to_words_and_back(s*g*p[8]*gain,g,ss)
    return dict(qubit=qw,sideband=sw)


def effective_score(parameters,repeats,gain,shape='phase',levels=96,outputs=False):
    words=smooth_motif(parameters,repeats,gain,shape)
    qs,ss=hw.dac_scales(.3)
    q=hw.rfdc_interpolate(sample_to_complex(words['qubit'])).conj()/qs*(np.pi/cd.PI_TIME)
    s=hw.rfdc_interpolate(sample_to_complex(words['sideband']))/ss*(np.pi/(2*.3))
    space,initial,T=target(.4*repeats*gain,levels)
    out=hw.propagate(q,s,initial,cd.DT/8,order=8,fast=True)
    overlaps=np.sum(T.conj()*out,axis=0);subset=np.r_[np.arange(9),11+np.arange(9)]
    r=dict(shape=shape,beta=.4*repeats*gain,repeats=repeats,gain=gain,
           fidelity=float(abs(overlaps.mean())**2),fidelity_nmax8=float(abs(overlaps[subset].mean())**2),
           worst_basis_fidelity=float(min(abs(overlaps)**2)),levels=levels,columns=22,
           command_us=len(words['qubit'])*cd.DT,RFDC_window_us=len(q)*cd.DT/8)
    return (r,out,T,words) if outputs else r


@dataclass(frozen=True)
class Circuit:
    f_coupler_GHz:float=6.8
    full_tuning_GHz:float=4.
    EC_GHz:float=.100          # illustrative, from paper, not measured here
    loops:int=1               # array size is not determined by the frequency
    f_qubit_GHz:float=4.630267134
    f_cavity_GHz:float=5.035155906
    coupling_qc_GHz:float=.120 # illustrative bare linear couplings from paper
    coupling_ac_GHz:float=.050
    pump_drive_GHz:float|None=None
    qubit_drive_GHz:float|None=None
    flux_per_DAC_override:float|None=None

    def parameters(self):
        d=self.full_tuning_GHz/self.f_coupler_GHz
        ratio=d*np.sqrt(1-d*d/4)
        EL=self.f_coupler_GHz**2/(8*self.EC_GHz);EJ=ratio*EL/2
        lam=(2*self.EC_GHz/EL)**.25
        lq=lam*self.coupling_qc_GHz/(self.f_coupler_GHz-self.f_qubit_GHz)
        la=lam*self.coupling_ac_GHz/(self.f_coupler_GHz-self.f_cavity_GHz)
        # Keep the full sin(flux) drive. Calibrate its leading resonant JC
        # coefficient with 2*J1(A), rather than assuming sin(A cos)=A cos.
        debye=np.exp(-(lam*lam+lq*lq+la*la)/(2*self.loops**2))
        coefficient=2*(2*np.pi*EJ)*lq*la*debye # rad/ns times J1(flux)
        g600=np.pi/(2*600.)
        amplitude=brentq(lambda a:coefficient*j1(a)-g600,1e-9,1.)
        return dict(EL_GHz=EL,EJ_GHz=EJ,ratio_2EJ_EL=ratio,lambda_c=lam,
                    lambda_q=lq,lambda_a=la,flux_at_DAC_0045=amplitude,
                    flux_per_DAC=amplitude/.045 if self.flux_per_DAC_override is None else self.flux_per_DAC_override,
                    inferred_fmin_GHz=self.f_coupler_GHz*np.sqrt(1-ratio),
                    inferred_fmax_GHz=self.f_coupler_GHz*np.sqrt(1+ratio))


@lru_cache(maxsize=12)
def operators(circuit,na,nc):
    """Full cosine, including higher powers and the dynamical coupler mode.

    Basis (cavity photon, coupler photon, g/e). Linear-mode mixing is encoded
    in assumed participations. The qubit's displacement operator is projected
    from a harmonic normal mode onto g/e, retaining its distinct diagonals.
    """
    p=circuit.parameters();M=circuit.loops
    def trig(levels,lam):
        # Project the INFINITE-oscillator cosine, rather than taking cosine
        # after truncating X. Exact Laguerre elements avoid artificial boundary
        # terms at the top of the coupler ladder and converge much faster.
        displacement=np.empty((levels,levels),complex)
        for i in range(levels):
            for j in range(levels):
                lo,hi=min(i,j),max(i,j);d=hi-lo
                displacement[i,j]=np.exp(-lam*lam/2+(gammaln(lo+1)-gammaln(hi+1))/2)*(1j*lam)**d*eval_genlaguerre(lo,d,lam*lam)
        return displacement.real,displacement.imag
    Ca,Sa=trig(na,p['lambda_a']/M);Cc,Sc=trig(nc,p['lambda_c']/M)
    cosine=np.kron(Ca,Cc)-np.kron(Sa,Sc)
    sine=np.kron(Sa,Cc)+np.kron(Ca,Sc)
    q=p['lambda_q']/M
    Cq=np.exp(-q*q/2)*np.diag([1,1-q*q]);Sq=np.exp(-q*q/2)*q*np.array([[0,1],[1,0]])
    matrix=np.kron(cosine,Cq)-np.kron(sine,Sq)
    # Use the infinite-oscillator vacuum expectation, so the removed global
    # scalar is identical when either Fock cutoff is enlarged.
    ground=np.exp(-(p['lambda_q']**2+p['lambda_a']**2+p['lambda_c']**2)/(2*M*M))
    matrix-=ground*np.eye(len(matrix))
    sparse=matrix.copy();sparse[abs(sparse)<1e-15]=0
    omitted=float(np.max(np.sum(abs(matrix-sparse),axis=1)))
    C=csr_matrix(sparse)
    energy=(np.arange(na)[:,None,None]*circuit.f_cavity_GHz+
            np.arange(nc)[None,:,None]*circuit.f_coupler_GHz+
            np.arange(2)[None,None,:]*circuit.f_qubit_GHz).ravel()*2*np.pi
    energy-=.5*(energy.min()+energy.max()) # only a global phase changes
    return p,C,energy,omitted


def microscopic(words,circuit=Circuit(),na=16,nc=6,dt_ns=.01,inputs=None,
                initial_state=None,rf_samples=None,duration_ns=None,method='chebyshev'):
    """No rotating-wave or adiabatic elimination of the coupler drive.

    A continuous carrier multiplies linearly reconstructed RFDC output samples.
    This is an assumed ideal analog reconstruction, NOT a measured flux-line
    transfer function. Timestep convergence and sparse-discard bounds are explicit.
    """
    started=time.monotonic();p,C,energy,omitted=operators(circuit,na,nc)
    if rf_samples is None:
        pump=np.ascontiguousarray(hw.rfdc_interpolate(sample_to_complex(words['sideband'])),complex)
        qubit=np.ascontiguousarray(hw.rfdc_interpolate(sample_to_complex(words['qubit'])),complex)
    else:
        pump=np.ascontiguousarray(rf_samples[0],complex);qubit=np.ascontiguousarray(rf_samples[1],complex)
    sample_dt=cd.DT*1000/8;T=len(pump)*sample_dt
    if duration_ns is not None:T=duration_ns;sample_dt=T/len(pump)
    steps=int(np.ceil(T/dt_ns));dt=T/steps
    if inputs is None:inputs=[(q,n) for q in range(2) for n in [0,4,8]]
    if initial_state is None:
        initial=np.zeros((2*na*nc,len(inputs)),complex)
        for col,(q,n) in enumerate(inputs):initial[(n*nc)*2+q,col]=1
    else:initial=np.ascontiguousarray(initial_state,complex)
    columns=initial.shape[1]
    drive_factor=2*circuit.loops**2*2*np.pi*p['EJ_GHz']
    qs,_=hw.dac_scales(.3);qgain=(np.pi/(cd.PI_TIME*1000))/qs
    fmax=drive_factor*min(1.,p['flux_per_DAC']*max(abs(pump)))
    rowbound=float(np.max(np.asarray(abs(C).sum(axis=1))))
    # Coefficients in each CF4 exponential have |a1|+|a2|=1/sqrt(3).
    bound=dt*(.5*max(abs(energy))+fmax*rowbound/np.sqrt(3)+2*qgain*max(abs(qubit))/np.sqrt(3))
    order=12
    if method=='taylor':
        while 2*steps*math.exp(bound)*bound**(order+1)/math.factorial(order+1)>1e-11:order+=1
    else:
        assert method=='chebyshev'
        # |J_k(r)| <= (r/2)^k/k! exp[r²/(4(k+1))]. Bound the remaining
        # geometric tail of the Chebyshev/Bessel expansion on [-1,1].
        def remainder(k):
            return 2*np.exp(bound*bound/(4*(k+2)))*(bound/2)**(k+1)/math.factorial(k+1)/(1-bound/(2*(k+2)))
        while bound>=2*(order+2) or 2*steps*remainder(order)>1e-11:order+=1
    assert order<=48,(order,bound)
    cpp=CODE / 'coupler_replay.cpp';sha=hashlib.sha256(cpp.read_bytes()).hexdigest()[:16]
    lib=Path('/tmp')/f'linc_replay_{sha}.so'
    if not lib.exists():subprocess.run(['g++','-O3','-std=c++17','-march=native','-shared','-fPIC',str(cpp),'-o',str(lib)],check=True)
    fn=getattr(ctypes.CDLL(str(lib)),'replay_linc_chebyshev' if method=='chebyshev' else 'replay_linc')
    fn.argtypes=[ctypes.c_int]*5+[ctypes.c_double]*7+[ctypes.c_void_p]*8
    energy=np.ascontiguousarray(energy,float);rowptr=np.ascontiguousarray(C.indptr,np.int32)
    colidx=np.ascontiguousarray(C.indices,np.int32);values=np.ascontiguousarray(C.data,float)
    out=np.empty_like(initial)
    pump_frequency=circuit.f_cavity_GHz-circuit.f_qubit_GHz if circuit.pump_drive_GHz is None else circuit.pump_drive_GHz
    qubit_frequency=circuit.f_qubit_GHz if circuit.qubit_drive_GHz is None else circuit.qubit_drive_GHz
    fn(len(energy),columns,len(pump),steps,order,sample_dt,dt,
       2*np.pi*pump_frequency,2*np.pi*qubit_frequency,
       p['flux_per_DAC'],drive_factor,qgain,energy.ctypes.data,rowptr.ctypes.data,
       colidx.ctypes.data,values.ctypes.data,pump.ctypes.data,qubit.ctypes.data,
       initial.ctypes.data,out.ctypes.data)
    norm=float(np.max(abs(np.sum(abs(out)**2,axis=0)-1)));assert norm<1e-9
    # Transform back to the static-mode rotating frame. Global scalar shifts
    # in H0 and the cosine do not affect any coherent process fidelity.
    out*=np.exp(1j*energy*T)[:,None]
    tensor=out.reshape(na,nc,2,columns)
    coupler_population=np.sum(abs(tensor[:,1:])**2,axis=(0,1,2))
    result=dict(circuit=circuit.__dict__,inferred=p,cavity_levels=na,coupler_levels=nc,
                cosine_projection='infinite_oscillator_projected',
                duration_ns=T,timestep_ns=dt,steps=steps,taylor_order=order,columns=columns,
                polynomial_solver=method,polynomial_order=order,
                sparse_entries=C.nnz,sparse_discard_amplitude_bound=T*fmax*omitted,
                norm_error=norm,final_coupler_excitation=coupler_population.tolist(),
                max_final_coupler_excitation=float(max(coupler_population)),elapsed_seconds=time.monotonic()-started)
    return result,out


def boundary_words(delta=np.pi/6,shape='phase',scale=.06):
    """60ns isolated phase-boundary test; central transition always lasts10ns."""
    count=48;z=np.ones(count,complex);z[24:]=np.exp(1j*delta)
    if shape!='abrupt':
        u=np.arange(8)/8;h=10*u**3-15*u**4+6*u**5
        if shape=='phase':v=np.exp(1j*delta*h)
        elif shape=='cartesian':v=1-h+h*np.exp(1j*delta)
        elif shape=='zero':v=np.r_[np.cos(np.pi*np.arange(4)/8)**2,np.exp(1j*delta)*np.sin(np.pi*np.arange(4)/8)**2]
        else:raise ValueError(shape)
        z[20:28]=v
    z[:4]*=np.sin(np.pi*np.arange(4)/8)**2;z[-4:]*=np.cos(np.pi*np.arange(4)/8)**2
    program=json.loads(hw.PROGRAM.read_text())
    qw,_=hw.sdk_load(np.zeros(count,complex),'DAC7',program['idle_pi']['name'])
    sw,_=hw.sdk_load(z*scale,'DAC10',program['pump']['name'])
    return dict(qubit=qw,sideband=sw)


def optimise_transition_family(kind,maxeval=650):
    """Refine shared echo parameters, not arbitrary waveform samples.

    `constant` keeps the pump at -pi/2, a fair fixed-pump comparison for real
    CD(beta). The historical CD qubit-only grid fixed it at0 instead.
    """
    from scipy.optimize import minimize
    base=np.array(json.loads((ROOT/'runs/scalable_cd/recipe.json').read_text())['parameters'])
    constant=kind=='constant'
    if constant:
        base[4:8]=0.;initial=np.r_[base[:4],.61]
    else:
        base[8]=json.loads((ROOT/'runs/audit/smooth_zero.json').read_text())['parameters'][8]
        initial=base.copy()
    history=[];best=[np.inf,None,None]
    def decode(x):
        p=base.copy()
        if constant:p[:4]=x[:4];p[8]=x[4]
        else:p=x.copy()
        return p
    def cost(x):
        p=decode(x)
        if not .2<p[8]<1.2:return 1+abs(p[8]-.7)
        shape='phase' if constant else 'zero'
        scores=[effective_score(p,m,s,shape,64) for m,s in [(1,.5),(1,1),(3,5/6),(5,1)]]
        errors=np.array([1-r['fidelity'] for r in scores]);value=float(errors.mean()+errors.max())
        history.append(dict(parameters=p.tolist(),scores=scores,objective=value))
        if value<best[0]:best[:]=[value,p.copy(),scores]
        if len(history)%50==0:print('TRANSITION FIT',kind,len(history),'best',best[0],
               'F',[round(s['fidelity'],8) for s in best[2]],flush=True)
        return value
    simplex=np.vstack([initial,initial+np.diag(np.r_[np.full(len(initial)-1,.05),.02])])
    fit=minimize(cost,initial,method='Nelder-Mead',options=dict(maxfev=maxeval,initial_simplex=simplex,
                  xatol=1e-5,fatol=1e-10))
    result=dict(kind=kind,parameters=best[1].tolist(),objective=best[0],scores=best[2],
                waveform_evaluations=4*len(history),input_columns=88*len(history),
                optimizer_message=str(fit.message),history=history)
    (ROOT/'runs/audit'/f'transition_fit_{kind}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FINISHED TRANSITION FIT',kind,'F',[r['fidelity'] for r in best[2]],
          'waveforms',result['waveform_evaluations'],'inputcolumns',result['input_columns'],flush=True)
    return result
