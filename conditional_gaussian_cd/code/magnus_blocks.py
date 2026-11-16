"""Select proper fixed-sideband squeezing blocks by their Magnus terms.

Every physical pulse is a 30 ns pi rotation about +X,+Y,-X or -Y. We first
require zero first-order JC average and the correct second-order squeeze
generator. A qubit-only phase cycle can remove error terms odd under joint
qubit/cavity parity, without changing sideband phase or amplitude.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time
import numpy as np
from scipy.linalg import expm
from project_paths import ROOT,CODE
import cd_trains as cd
from fixed_sideband import OUT,MAX_G

PHASES=np.arange(4)*np.pi/2


def pulse_series(levels=8,order=4):
    """Cauchy derivatives of the exact sampled pulse, not a small-g fit."""
    space=cd.Space(levels,3);d=2*levels;radius=8.;count=48
    J=space.hamiltonian(1.,0.);X=space.hamiltonian(0.,1.)
    circle=[]
    for j in range(count):
        g=radius*np.exp(2j*np.pi*j/count);U=np.eye(d,dtype=complex)
        for omega in cd.pi_samples():U=expm(-1j*cd.DT*(omega*X+g*J))@U
        circle.append(U)
    coefficients=np.fft.fft(np.asarray(circle),axis=0)/count
    coefficients=np.asarray([coefficients[k]/radius**k for k in range(order+1)])
    paulis=[np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    frames=[np.kron(np.eye(levels),P) for P in paulis]
    V=np.empty((4,4,order+1,d,d),complex)
    for phase_id,phase in enumerate(PHASES):
        D=np.exp(1j*phase*space.k)
        C=D[None,:,None]*coefficients*D.conj()[None,None,:]
        for frame,P in enumerate(frames):
            for k in range(order+1):V[frame,phase_id,k]=P.conj().T@C[0].conj().T@C[k]@P
    assert max(abs(V[:,:,0]-np.eye(d)).ravel())<1e-13
    logs=np.zeros((4,4,3,d,d),complex)
    for f in range(4):
        for p in range(4):
            A,B,C=V[f,p,1:4]
            logs[f,p,0]=A
            logs[f,p,1]=B-A@A/2
            logs[f,p,2]=C-(A@B+B@A)/2+A@A@A/3
    # Independent check: Taylor coefficients reproduce an actual native pulse.
    testg=.3;native=np.eye(d,dtype=complex)
    for omega in cd.pi_samples():native=expm(-1j*cd.DT*(omega*X+testg*J))@native
    reconstructed=sum(coefficients[k]*testg**k for k in range(order+1))
    error=float(np.max(abs(reconstructed-native)))
    assert error<1e-9,(error,levels)
    print('MAGNUS PULSE SERIES:',d*d,'amplitudes;nativecomparisonerror',error,flush=True)
    return space,V,logs


def comm(A,B):return A@B-B@A


def catalogue():
    space,V,logs=pulse_series();d=2*space.levels
    a=np.diag(np.sqrt(np.arange(1,space.levels)),1);X=a+a.T;P=1j*(a.T-a)
    paulis=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    basis=np.asarray([np.kron(O,Q) for Q in paulis for O in [X,P]])
    low=np.arange(6)
    B=basis[:,:,low].reshape(6,-1).T
    first=np.empty((4,4,6),complex)
    for f in range(4):
        for p in range(4):first[f,p]=np.linalg.lstsq(B,logs[f,p,0][:,low].ravel(),rcond=None)[0]
    words=((np.arange(4**8)[:,None]>>(2*np.arange(8)))&3).astype(int)
    frame=np.zeros(len(words),int);moment=np.zeros((len(words),6),complex)
    for j in range(8):
        p=words[:,j];moment+=first[frame,p];frame^=np.where(p%2==0,1,2)
    keep=(frame==0)&(np.linalg.norm(moment,axis=1)<1e-10)
    proper=words[keep]
    print('MAGNUS FIRST-ORDER FILTER:',len(words),'words;',len(proper),'zero-averageclosedwords',flush=True)
    assert len(proper)>0
    # BCH combines exact pulse logarithms through cubic order in JC strength.
    count=len(proper);O1=np.zeros((count,d,d),complex);O2=O1.copy();O3=O1.copy()
    frame=np.zeros(count,int)
    for j in range(8):
        A=logs[frame,proper[:,j],0];B2=logs[frame,proper[:,j],1];C3=logs[frame,proper[:,j],2]
        AB=comm(A,O1)
        O3=O3+C3+(comm(A,O2)+comm(B2,O1))/2+(comm(A,AB)+comm(O1,-AB))/12
        O2=O2+B2+AB/2;O1=O1+A
        frame^=np.where(proper[:,j]%2==0,1,2)
    G=np.kron(a@a-a.T@a.T,np.diag([1,-1]))
    Z=1j*np.kron(np.eye(space.levels),np.diag([1,-1]))
    I=1j*np.eye(d)
    targets=np.stack([G[:,low].ravel(),Z[:,low].ravel(),I[:,low].ravel()],axis=1)
    pinv=np.linalg.pinv(targets)
    projection=O2[:,:,low].reshape(count,-1)@pinv.T
    residual=O2[:,:,low].reshape(count,-1)-projection@targets.T
    residual_norm=np.linalg.norm(residual,axis=1)
    good=(projection[:,0].real>1e-8)&(residual_norm<1e-9)&(abs(projection[:,0].imag)<1e-9)
    np.savez_compressed(OUT/'proper_block_terms.npz',words=proper,first=O1,second=O2,third=O3,
                        projection=projection,residual=residual_norm)
    D=np.exp(1j*np.pi*space.k)
    transformed=D[None,:,None]*O3*D.conj()[None,None,:]
    even=(O3+transformed)/2;odd=(O3-transformed)/2
    rows=[]
    for i in np.flatnonzero(good):
        c=float(projection[i,0].real)
        rows.append(dict(word=proper[i].tolist(),squeeze_second_coefficient=c,
                         first_norm=float(np.linalg.norm(O1[i][:,low])),
                         second_unwanted_norm=float(residual_norm[i]),
                         cubic_norm=float(np.linalg.norm(O3[i][:,low])),
                         cubic_even_norm=float(np.linalg.norm(even[i][:,low])),
                         cubic_odd_norm=float(np.linalg.norm(odd[i][:,low])),
                         qubit_Z_second_coefficient=float(projection[i,1].real)))
    assert rows
    # Scale by useful squeezing strength, not an arbitrary small coefficient.
    rows.sort(key=lambda r:r['cubic_even_norm']/r['squeeze_second_coefficient']**1.5)
    payload=dict(enumerated_words=len(words),zero_first_order_words=len(proper),
                 pure_squeeze_words=len(rows),pulse_ns=30,sideband_fixed=True,results=rows)
    (OUT/'magnus_catalogue.json').write_text(json.dumps(payload,indent=2)+'\n')
    np.savez_compressed(OUT/'pulse_series.npz',V=V,logs=logs)
    print('MAGNUS TARGET FILTER:',len(rows),'pure-squeezeclosedwords',flush=True)
    for r in rows[:4]:print('MAGNUS BLOCK',r,flush=True)


def screen():
    catalogue=json.loads((OUT/'magnus_catalogue.json').read_text())
    # Deduplicate identical patterns, while preserving the documented criterion.
    chosen=[];seen=set()
    for r in catalogue['results']:
        key=tuple(r['word'])
        if key in seen:continue
        seen.add(key);chosen.append(r)
        if len(chosen)==12:break
    from constant_pump_gaussian import FixedPumpTrain
    rows=[];pulse_cache={}
    for index,r in enumerate(chosen):
        for paired in [False,True]:
            phases=PHASES[np.asarray(r['word'])]
            if paired:phases=np.r_[phases,phases+np.pi]
            c2=r['squeeze_second_coefficient']*(2 if paired else 1)
            minimum=max(1,int(np.ceil((cd.R_8DB/2)/(c2*(MAX_G*.985)**2))))
            for repeats in sorted(set([minimum,minimum+1,minimum+2,minimum+4,minimum+8,minimum+12])):
                n=len(phases)*repeats
                if n>256:continue
                g=float(np.sqrt((cd.R_8DB/2)/(c2*repeats)))
                key=round(g,10)
                if key not in pulse_cache:
                    tr=FixedPumpTrain('squeeze',cd.R_8DB,np.pi/(2*g),1,0.,64);tr.pump_phase=0.
                    pulse_cache[key]=tr
                tr=pulse_cache[key];state=tr.up@tr.inputs
                U=np.eye(128,dtype=complex)
                for phi in phases:
                    D=np.exp(1j*phi*tr.space.k)
                    U=(D[:,None]*tr.U0*D.conj()[None,:])@U
                state=tr.down@(np.linalg.matrix_power(U,repeats)@state)
                frame,angle=tr.qubit_frame(state)
                F=float(abs(np.vdot(tr.target,frame*state)/18)**2)
                row=dict(name=f'magnus_sq_b{index}_pair{int(paired)}_r{repeats}',cells=n,
                         block_phases=phases.tolist(),qubit_phases=np.tile(phases,repeats).tolist(),
                         repeats=repeats,g=g,cell_fidelity=F,qubit_frame_Rz=-angle,
                         time_us=n*.03+.010,magnus_source=r,paired=paired)
                rows.append(row)
    assert rows
    rows.sort(key=lambda r:(r['time_us'],-r['cell_fidelity']))
    (OUT/'magnus_screen.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('MAGNUS EXACT CELL SCREEN:',len(rows),'gates;',18*len(rows),'inputcolumns',flush=True)
    for r in sorted(rows,key=lambda r:-r['cell_fidelity'])[:8]:
        print('MAGNUS SCREEN',r['name'],'N',r['cells'],'F',r['cell_fidelity'],'g',r['g'],flush=True)


def dcg_blocks():
    """Insert a zero-average Eulerian bridge that conjugates a block by Z.

    B, bridge-prefix, B, bridge-suffix: the prefix ends in qubit Z. The
    second B therefore has its error conjugated by Z while desired sigma_z
    squeezing is preserved. All eight bridge pulses are included in Magnus.
    """
    data=np.load(OUT/'proper_block_terms.npz');words=data['words'];logs=np.load(OUT/'pulse_series.npz')['logs']
    projection=data['projection'];residual=data['residual'];d=logs.shape[-1]
    eligible=(residual<1e-9)&(abs(projection[:,0].imag)<1e-9)
    indices=np.flatnonzero(eligible)
    bases=np.flatnonzero(eligible&(projection[:,0].real>1e-8))
    Z=np.diag(np.tile([1.,-1.],d//2));low=np.arange(6);rows=[]
    for bi in bases[:8]:
        B2=data['second'][bi];B3=data['third'][bi]
        for cut in [2,4,6]:
            prefix=np.zeros(len(words),int)
            for j in range(cut):prefix^=np.where(words[:,j]%2==0,1,2)
            selected=indices[prefix[indices]==3]
            if not len(selected):continue
            E=words[selected];count=len(E);frame=np.zeros(count,int)
            O1=np.zeros((count,d,d),complex)
            O2=np.broadcast_to(B2,(count,d,d)).copy();O3=np.broadcast_to(B3,(count,d,d)).copy()
            for j in range(8):
                A=logs[frame,E[:,j],0];B=logs[frame,E[:,j],1];C=logs[frame,E[:,j],2]
                AB=comm(A,O1)
                O3=O3+C+(comm(A,O2)+comm(B,O1))/2+(comm(A,AB)+comm(O1,-AB))/12
                O2=O2+B+AB/2;O1=O1+A
                frame^=np.where(E[:,j]%2==0,1,2)
                if j+1==cut:
                    C2=Z@B2@Z;C3=Z@B3@Z
                    O3=O3+C3+comm(C2,O1)/2;O2=O2+C2
            for i,ei in enumerate(selected):
                c2=float(2*projection[bi,0].real+projection[ei,0].real)
                if c2<=1e-8:continue
                word=np.r_[words[bi],words[ei,:cut],words[bi],words[ei,cut:]]
                rows.append(dict(word=word.tolist(),squeeze_second_coefficient=c2,
                                 first_norm=float(np.linalg.norm(O1[i][:,low])),
                                 cubic_norm=float(np.linalg.norm(O3[i][:,low])),
                                 base_index=int(bi),bridge_index=int(ei),bridge_split=cut))
    assert rows
    rows.sort(key=lambda r:r['cubic_norm']/r['squeeze_second_coefficient']**1.5)
    (OUT/'dcg_block_catalogue.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('DCG MAGNUS BRIDGE CATALOGUE:',len(rows),'proper24pulseconstructions',flush=True)
    for r in rows[:5]:print('DCG MAGNUS',r,flush=True)


def dcg_screen():
    from scipy.optimize import minimize_scalar
    rows=json.loads((OUT/'dcg_block_catalogue.json').read_text());chosen=[];seen=set()
    for r in rows:
        key=tuple(r['word'])
        if key in seen:continue
        seen.add(key);chosen.append(r)
        if len(chosen)==6:break
    results=[]
    for index,row in enumerate(chosen):
        phi=PHASES[np.asarray(row['word'])];c2=row['squeeze_second_coefficient']
        minimum=max(1,int(np.ceil((cd.R_8DB/2)/(c2*(MAX_G*.985)**2))))
        for repeats in range(minimum,7):
            n=24*repeats;g0=float(np.sqrt((cd.R_8DB/2)/(c2*repeats)));history=[]
            def cost(g):
                tr=cd.Train('squeeze',cd.R_8DB,np.pi/(2*g),24,0.,64)
                U=np.eye(128,dtype=complex)
                for phase in phi:
                    D=np.exp(1j*phase*tr.space.k)
                    U=(D[:,None]*tr.U0*D.conj()[None,:])@U
                state=tr.down@(np.linalg.matrix_power(U,repeats)@(tr.up@tr.inputs))
                frame,angle=tr.qubit_frame(state)
                F=float(abs(np.vdot(tr.target,frame*state)/18)**2)
                history.append(dict(g=float(g),fidelity=F,qubit_frame_Rz=-angle))
                return 1-F
            fitted=minimize_scalar(cost,bounds=(g0*.88,min(g0*1.12,MAX_G*.985)),method='bounded',options=dict(maxiter=22,xatol=1e-7))
            best=max(history,key=lambda r:r['fidelity'])
            result=dict(name=f'squeeze_dcg_b{index}_r{repeats}',cells=n,qubit_phases=np.tile(phi,repeats).tolist(),
                        block_phases=phi.tolist(),repeats=repeats,g=best['g'],cell_fidelity=best['fidelity'],
                        qubit_frame_Rz=best['qubit_frame_Rz'],time_us=.03*n+.010,magnus=row,scalar_evaluations=len(history))
            results.append(result)
            print('DCG SCALAR SCREEN',result['name'],'N',n,'F',result['cell_fidelity'],'g',result['g'],flush=True)
    assert results
    (OUT/'dcg_screen.json').write_text(json.dumps(results,indent=2)+'\n')
    print('COMPLETED DCG SCALAR SCREEN:',len(results),'gates;',18*sum(r['scalar_evaluations'] for r in results),'inputcolumns',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['catalogue','screen','dcg','dcg_screen']);args=parser.parse_args()
    if args.stage=='catalogue':catalogue()
    elif args.stage=='screen':screen()
    elif args.stage=='dcg':dcg_blocks()
    else:dcg_screen()
