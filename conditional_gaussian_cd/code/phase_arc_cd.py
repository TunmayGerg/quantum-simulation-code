"""CD with smooth constant-radius JC phase changes and finite qubit pulses.

First design family: q(T-t)=-q(t), s(T-t)=s(t). The qubit-only path closes and
its interaction Hamiltonian is time symmetric, removing even Magnus orders.
JC phases vary between pulse cells along 10 ns quintic circular IQ arcs.
Each arc center is an exact programmed qubit-envelope zero. The sideband
plateau magnitude is constant. Final scores use SDK words and the full RFDC
response, never a Cartesian IQ blend or an ideal instantaneous phase jump.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,ctypes,hashlib,json,subprocess,time,itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import least_squares,minimize
from scipy.signal import fftconvolve
from project_paths import ROOT,CODE
from shaped_pulse_optimisation import PlayedGate
import cd_trains as cd
import hardware_check as hw

OUT=ROOT/'runs/phase_arc_cd';OUT.mkdir(exist_ok=True)
MAX_G=np.pi/(2*.200)


class ArcCD(PlayedGate):
    def __init__(self,cells=10,levels=32,symmetric=True,fractions=None,gaps=None):
        super().__init__('cd',cells,levels,True,False)
        self.gaps=np.zeros(cells,int) if gaps is None else np.asarray(gaps,int)
        assert len(self.gaps)==cells and self.gaps[0]==0 and np.all(self.gaps>=0)
        assert sum(self.gaps)%4==0
        if symmetric:assert np.all(self.gaps[1:]==self.gaps[1:][::-1])
        self.starts=4+24*np.arange(cells)+np.cumsum(self.gaps)
        self.symmetric=symmetric;self.count=8+24*cells+sum(self.gaps)
        self.qshape=np.zeros(self.count,complex);self.qcell=np.full(self.count,-1,int)
        half=cells//2;assert not symmetric or cells%2==0
        self.rotation_fractions=np.ones(cells) if fractions is None else np.asarray(fractions)
        assert len(self.rotation_fractions)==cells
        if symmetric:assert np.all(self.rotation_fractions==self.rotation_fractions[::-1])
        for j in range(cells):
            shape=cd.pi_samples().copy()*self.rotation_fractions[j]
            if symmetric and j>=half:shape=shape[::-1]
            selection=slice(self.starts[j],self.starts[j]+24)
            self.qshape[selection]=shape;self.qcell[selection]=j
        phase_W=np.zeros((self.count,cells))
        for j in range(cells):phase_W[self.starts[j]:self.starts[j+1] if j+1<cells else self.count-4,j]=1
        phase_W[:4,0]=1;phase_W[-4:,-1]=1
        self.arc_centers=[]
        for j in range(1,cells):
            center=self.starts[j]-(self.gaps[j]+1 if symmetric and j>half else 0)
            u=np.arange(9)/8;blend=10*u**3-15*u**4+6*u**5
            selection=slice(center-4,center+5)
            phase_W[selection]=0;phase_W[selection,j-1]=1-blend;phase_W[selection,j]=blend
            if symmetric and j==half:continue  # adjacent pump phases are identical
            assert self.qshape[center]==0
            self.arc_centers.append(center)
        self.phase_W=phase_W
        self.envelope=np.ones(self.count)
        ramp=np.sin(np.pi*(np.arange(4)+(.5 if symmetric else 0))/8)**2
        self.envelope[:4]=ramp
        self.envelope[-4:]=ramp[::-1] if symmetric else np.cos(np.pi*np.arange(4)/8)**2
        self.maximum_g=MAX_G*.975  # conservative; actual filtered cap is also asserted
        self.nfree=half if symmetric else cells

    def expand(self,p):
        m=self.nfree;phi=p[:m];psi=p[m:2*m]
        if self.symmetric:phi=np.r_[phi,phi[::-1]+np.pi];psi=np.r_[psi,psi[::-1]]
        return phi,psi,float(np.exp(p[-1]))

    def raw(self,p):
        phi,psi,g=self.expand(p);q=self.qshape.copy();keep=self.qcell>=0
        q[keep]*=np.exp(1j*phi[self.qcell[keep]])
        s=g*self.envelope*np.exp(1j*(self.phase_W@psi))
        assert max(abs(abs(s[4:-4])-g))<1e-12
        if self.symmetric:
            assert max(abs(q+q[::-1]))<1e-12 and max(abs(s-s[::-1]))<1e-12
        return q,s

    def arrays(self,p,quantized=True):
        q,s=self.raw(p);words=None
        if quantized:
            qs,ss=hw.dac_scales(.6)
            qw,q=hw.to_words_and_back(q,np.pi/cd.PI_TIME,qs,qubit=True)
            sw,s=hw.to_words_and_back(s,np.pi/(2*.6),ss);words=dict(qubit=qw,sideband=sw)
        qf,sf=hw.rfdc_interpolate(q),hw.rfdc_interpolate(s)
        assert max(abs(sf))<=MAX_G,(max(abs(sf)),p)
        return q,s,qf,sf,words

    def evaluate(self,p,gradient=True,quantized=True,outputs=False):
        q,s,qf,sf,words=self.arrays(p,quantized)
        V=hw.propagate(qf,sf,self.initial,cd.DT/8,order=8,fast=True)
        angle=0.
        if self.frame:
            angle=float(np.angle(np.vdot(self.target[0::2],V[0::2]))-np.angle(np.vdot(self.target[1::2],V[1::2])))
        frame=np.exp(-1j*angle*self.z/2)[:,None]
        trace=np.vdot(self.target,frame*V);cost=float(1-abs(trace/18)**2)
        if outputs:return dict(fidelity=1-cost,qubit_frame_Rz=angle),frame*V,self.target,words
        if not gradient:return cost
        adjoint=np.ascontiguousarray(frame.conj()*self.target);grad=np.zeros((len(qf),4),float)
        self.backward(self.levels,18,len(qf),8,cd.DT/8,float(trace.real),float(trace.imag),
            qf.ctypes.data,sf.ctypes.data,V.ctypes.data,adjoint.ctypes.data,grad.ctypes.data)
        Gq=fftconvolve(grad[:,0]+1j*grad[:,1],self.filter[::-1],mode='valid')[::8]
        Gs=fftconvolve(grad[:,2]+1j*grad[:,3],self.filter[::-1],mode='valid')[::8]
        qb,sb=self.raw(p);keep=self.qcell>=0
        dq=np.bincount(self.qcell[keep],weights=np.real(Gq[keep].conj()*1j*qb[keep]),minlength=self.cells)
        ds=self.phase_W.T@np.real(Gs.conj()*1j*sb)
        dg=float(np.real(np.vdot(Gs,sb)))
        dq,ds=self.reduce_gradient(dq,ds)
        return cost,np.r_[dq,ds,dg]

    def reduce_gradient(self,dq,ds):
        if self.symmetric:
            m=self.nfree;dq=dq[:m]+dq[m:][::-1];ds=ds[:m]+ds[m:][::-1]
        return dq,ds


def symbol(q,s,steps=4,dt=cd.DT):
    source=CODE/'cd_magnus_symbol.cpp'
    library=Path('/tmp')/('cd_magnus_symbol_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+'.so')
    if not library.exists():subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    function=ctypes.CDLL(str(library)).cd_magnus_symbol
    function.argtypes=[ctypes.c_int,ctypes.c_double,ctypes.c_int]+[ctypes.c_void_p]*3
    q=np.ascontiguousarray(q);s=np.ascontiguousarray(s);terms=np.zeros((4,3,3))
    function(len(q),dt,steps,q.ctypes.data,s.ctypes.data,terms.ctypes.data)
    A=terms[:2,0];B=terms[:,1];C=terms[:,2]
    cubic=np.array([C[0],(C[2]-C[3])/2-C[1],(C[2]+C[3])/2-C[0],C[1]])
    return A,B,cubic


def check():
    rng=np.random.default_rng(1511);errors=[];sym=[]
    for symmetric in [True,False]:
        model=ArcCD(8,24,symmetric);m=model.nfree
        p=np.r_[rng.normal(0,.2,m),-np.pi/2+rng.normal(0,.3,m),np.log(6.)]
        value,gradient=model.evaluate(p,quantized=False)
        for _ in range(4):
            direction=rng.normal(size=len(p));direction/=np.linalg.norm(direction);epsilon=1e-5
            difference=(model.evaluate(p+epsilon*direction,False,False)-model.evaluate(p-epsilon*direction,False,False))/(2*epsilon)
            errors.append(abs(difference-gradient@direction))
        q,s=model.raw(p);A,B,C=symbol(q,s/np.exp(p[-1]),16)
        if symmetric:assert max(abs(B.ravel()))<1e-13
        sym.append(dict(symmetric=symmetric,first=A.tolist(),second_norm=float(np.linalg.norm(B)),cubic=C.tolist()))
    assert len(errors)==8 and max(errors)<2e-5,errors
    report=dict(gradient_directions=8,gradient_errors=errors,max_gradient_error=max(errors),symbol_examples=sym)
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ARC CD VALIDATION:8 actual-filter gradient directions;max error',max(errors),flush=True)


def seed_search(cells=10):
    model=ArcCD(cells,24,True);m=model.nfree;rng=np.random.default_rng(718)
    rows=[];start=time.time()
    signs=[np.zeros(m),np.resize([0.,0.,np.pi,np.pi],m),np.resize([0.,np.pi],m),
           np.resize([0.,np.pi,np.pi,0.],m)]
    profiles=[np.zeros(m),.5*np.cos(np.linspace(0,np.pi,m)),.8*np.sin(np.linspace(0,2*np.pi,m)),
              np.resize([.8,-.8],m),np.resize([1.4,0.,-1.4,0.],m)]
    count=0
    for mode in ['phase_only','mixed_axes']:
        for i,(qseed,profile) in enumerate(itertools.product(signs,profiles)):
            initial_g=min(2/(.030*cells+.005)*1.04,model.maximum_g)
            initial=np.r_[qseed,-np.pi/2+profile,np.log(initial_g)]
            if mode=='mixed_axes':initial[:m]+=.15*np.sin(np.arange(m)+i)
            def residual(x):
                p=np.r_[qseed,x] if mode=='phase_only' else x
                q,s=model.raw(p);g=np.exp(p[-1]);A,B,C=symbol(q,s/g)
                desired=np.zeros((2,3));desired[1,0]=1
                return np.r_[4*(g*A-desired).ravel(),20*g**3*C.ravel()]
            initial=initial[m:] if mode=='phase_only' else initial
            lower=np.r_[np.full(m,-3*np.pi)] if mode=='phase_only' else np.full(2*m,-3*np.pi)
            upper=np.r_[np.full(m,3*np.pi)] if mode=='phase_only' else np.full(2*m,3*np.pi)
            lower=np.r_[lower,np.log(2.)];upper=np.r_[upper,np.log(model.maximum_g)]
            fit=least_squares(residual,initial,bounds=(lower,upper),max_nfev=180,
                              ftol=1e-9,xtol=1e-9,gtol=1e-9)
            p=np.r_[qseed,fit.x] if mode=='phase_only' else fit.x
            q,s=model.raw(p);g=np.exp(p[-1]);A,B,C=symbol(q,s/g,16)
            F=1-model.evaluate(p,False)
            row=dict(name=f'arc_cd_n{cells}_{mode}_{i}',cells=cells,symmetric=True,mode=mode,
                parameters=p.tolist(),seed_phi=qseed.tolist(),seed_phase_offset=profile.tolist(),
                first=A.tolist(),first_target_residual=float(np.linalg.norm(g*A-np.array([[0,0,0],[1,0,0]]))),
                second_norm=float(np.linalg.norm(B)),cubic=C.tolist(),scaled_cubic_norm=float(g**3*np.linalg.norm(C)),
                fidelity=F,evaluations=fit.nfev,command_us=model.count*cd.DT)
            rows.append(row);count+=1
            (OUT/f'seeds_n{cells}.json').write_text(json.dumps(rows,indent=2)+'\n')
            print('ARC SYMBOL SEED',cells,mode,i,'F',F,'firsterror',row['first_target_residual'],
                  'cubic',row['scaled_cubic_norm'],'eval',fit.nfev,flush=True)
    assert len(rows)==40
    print('ARC SEED SEARCH:40 structured fits;',40*18,'actual-filter input columns;',time.time()-start,'seconds',flush=True)


def export(model,p,name,extra):
    q,s=model.raw(p);phi,psi,g=model.expand(p)
    controls=OUT/(name+'_controls.npz');np.savez_compressed(controls,q=q,s=s)
    arcs=[]
    for j in range(1,model.cells):
        if abs(psi[j]-psi[j-1])<1e-10:continue
        center=model.starts[j]-(model.gaps[j]+1 if model.symmetric and j>model.cells//2 else 0)
        arcs.append(dict(center_sample=int(center),start_phase_rad=float(psi[j-1]),
            end_phase_rad=float(psi[j]),duration_ns=10.,shape='quintic constant-radius'))
    meta=dict(name=name,gate='cd',strength=1.,sideband_mode='phase_arc' if arcs else 'fixed',
        controls_file=str(controls.relative_to(ROOT)),qubit_frame_Rz=extra.get('qubit_frame_Rz',0.),
        magnus_justification=('Mirrored qubit inverse and even pump phase cancel second order; phase geometry targets principal cubic errors.'
            if model.symmetric else 'Refinement of mirrored/rotary first-force seeds with phase-only JC control to change cubic errors.'),
        symmetric=model.symmetric,phase_arcs=arcs,pump_phases=psi.tolist(),g_nominal=g,
        pulses=[dict(start_sample=int(model.starts[j]),samples=24,angle_fraction=float(model.rotation_fractions[j]),
            phase_rad=float(phi[j]),reversed_envelope=bool(model.symmetric and j>=model.cells//2)) for j in range(model.cells) if model.rotation_fractions[j]>0],
        research_fit=extra)
    path=OUT/(name+'_recipe.json');path.write_text(json.dumps(meta,indent=2)+'\n');return path


def fit(job):
    started=time.time();n=job['cells'];symmetric=job.get('symmetric',True)
    model=ArcCD(n,job.get('levels',32),symmetric,job.get('fractions'),job.get('gaps'))
    model.frame=job.get('frame',False)
    initial=np.asarray(job['parameters']);assert len(initial)==2*model.nfree+1
    history=[];best=[float('inf'),initial.copy()]
    def objective(p):
        value,gradient=model.evaluate(p);history.append(value)
        if value<best[0]:best[:]=[value,p.copy()]
        if len(history)%40==0:print('ARC FULL FIT',job['name'],len(history),'bestF',1-best[0],flush=True)
        return value,gradient
    def callback(p):
        if best[0]<job.get('stop',.006) or len(history)>=job.get('budget',260):raise StopIteration
    m=model.nfree
    lower=np.asarray(job.get('phase_lower',np.full(m,-np.pi/2-2.5)))
    upper=np.asarray(job.get('phase_upper',np.full(m,-np.pi/2+2.5)))
    bounds=[(None,None)]*m+list(zip(lower,upper))+[(np.log(2.),np.log(model.maximum_g))]
    initial[m:2*m]=np.clip(initial[m:2*m],lower+1e-6,upper-1e-6)
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,callback=callback,
        options=dict(maxiter=400,maxcor=30,maxls=25,ftol=1e-13,gtol=1e-8))
    p=best[1];report,V,T,words=model.evaluate(p,False,outputs=True)
    q,s,qf,sf,_=model.arrays(p)
    A,B,C=symbol(*[v for v in model.raw(p)],steps=16)
    overlaps=np.sum(T.conj()*V,axis=0)
    report.update(name=job['name'],job=job,parameters=p.tolist(),initial_fidelity=1-history[0],
        worst_basis_fidelity=float(min(abs(overlaps)**2)),evaluations=len(history),history=history,
        command_us=model.count*cd.DT,peak_g=float(max(abs(sf))),peak_swap_ns=float(1000*np.pi/(2*max(abs(sf)))),
        seconds=time.time()-started,optimizer_message=str(result.message),
        magnus_first=A.tolist(),magnus_second_norm=float(np.linalg.norm(B)),magnus_third_principal=C.tolist())
    # Save the counted fit before producing optional export artifacts too.
    (OUT/(job['name']+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    recipe=export(model,p,job['name'],dict(initial_fidelity=report['initial_fidelity'],evaluations=len(history),
        final_fidelity=report['fidelity'],seed_reason=job['seed_reason'],qubit_frame_Rz=report['qubit_frame_Rz']))
    report['recipe_file']=str(recipe.relative_to(ROOT))
    (OUT/(job['name']+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('ARC FULL RESULT',job['name'],'F',report['fidelity'],'Tns',1000*report['command_us'],
          'peak_swap',report['peak_swap_ns'],'eval',len(history),flush=True)
    return report


def refine(cells=10):
    seeds=json.loads((OUT/f'seeds_n{cells}.json').read_text());jobs=[];selected=[]
    for row in sorted(seeds,key=lambda r:-r['fidelity']):
        if any(np.linalg.norm(np.array(row['parameters'])-np.array(other['parameters']))<.2 for other in selected):continue
        selected.append(row)
        if len(selected)==6:break
    for i,row in enumerate(selected):
        jobs.append(dict(name=f'arc_sym_n{cells}_magnus{i}',cells=cells,symmetric=True,parameters=row['parameters'],
            seed_reason='First-force and principal-cubic least-squares design with exact interaction-frame time symmetry.'))
    m=cells//2
    for i,half_phi in enumerate([np.resize([0,0,np.pi,np.pi],m),np.zeros(m)]):
        jobs.append(dict(name=f'arc_sym_n{cells}_rotary{i}',cells=cells,symmetric=True,
            parameters=np.r_[half_phi,np.full(m,-np.pi/2),np.log(min(2/(cells*.03+.005),MAX_G*.975))].tolist(),
            seed_reason='Mirrored finite-pulse rotary path; fixed pump phase initially.'))
    for i,qphase in enumerate([np.resize([0,0,np.pi,np.pi],cells),np.zeros(cells)]):
        jobs.append(dict(name=f'arc_free_n{cells}_rotary{i}',cells=cells,symmetric=False,
            parameters=np.r_[qphase,np.full(cells,-np.pi/2),np.log(min(2/(cells*.03+.005),MAX_G*.975))].tolist(),
            seed_reason='Known rotary finite-pulse cancellation; allow smooth JC phase correction at its qubit zeros.'))
    results=[fit(job) for job in jobs]
    assert len(results)==10
    (OUT/f'refine_n{cells}.json').write_text(json.dumps(results,indent=2)+'\n')
    print('ARC FULL COMPARISONS:10 fits;',18*sum(r['evaluations'] for r in results),'18-input-column evaluations',flush=True)


def pattern_study(workers=3):
    jobs=[];n=10;m=5
    refined=json.loads((OUT/'refine_n10.json').read_text())
    best=max((r for r in refined if r['job']['symmetric']),key=lambda r:r['fidelity'])
    p0=np.asarray(best['parameters'])
    # Half-angle endpoints change the qubit trajectory while retaining the
    # exact inverse closure and even-Magnus cancellation of the mirrored gate.
    for label,fractions in [('outerhalf',[.5]+[1.]*8+[.5]),
                             ('fourhalf',[.5,.5]+[1.]*6+[.5,.5]),
                             ('middlehalf',[1.]*4+[.5,.5]+[1.]*4)]:
        for seed in [0,1]:
            p=p0.copy()
            if seed:p[:m]=np.resize([0.,0.,np.pi,np.pi],m);p[m:2*m]=-np.pi/2+.4*np.cos(np.linspace(0,np.pi,m))
            jobs.append(dict(name=f'arc_sym_n10_{label}_{seed}',cells=n,symmetric=True,fractions=fractions,
                parameters=p.tolist(),budget=220,seed_reason='Fractional physical endpoints reshape the cubic path; mirrored inverse cancels second order.'))
    source=max((r for r in refined if not r['job']['symmetric']),key=lambda r:r['fidelity'])
    base=np.asarray(source['parameters'])
    for label,gaps in [('all5',[0]+[4]*9),('even5',[0]+[4 if j%2==0 else 0 for j in range(1,10)]),
                        ('center5',[0]*5+[4]+[0]*4),('zero',[0]*10)]:
        for seed in [0,1]:
            p=base.copy();p[-1]+=np.log(.305/(.305+sum(gaps)*cd.DT))
            # Opposed qubit-axis and pump-phase harmonics create a distinct
            # first/cubic geometry, rather than a random waveform start.
            if seed:
                harmonic=np.sin(2*np.pi*(np.arange(n)+.5)/n)
                p[:n]+=.65*harmonic;p[n:2*n]+=.6*harmonic
            jobs.append(dict(name=f'arc_free_n10_{label}_{seed}',cells=10,symmetric=False,gaps=gaps,
                parameters=p.tolist(),budget=260,seed_reason='Rotary cancellation with pump phase arcs at qubit zeros; optional physical waits and correlated axis harmonics.'))
    # Nine pulses become qubit-closed rotary seeds by using two pi/2
    # envelopes; all envelopes still take30ns. Target remains beta1.
    for label,indices in [('ends',[0,8]),('center',[3,4]),('quarter',[1,7])]:
        fractions=np.ones(9);fractions[indices]=.5
        for seed in [0,1]:
            phi=np.resize([0.,0.,np.pi,np.pi],9)
            if seed:phi+=.45*np.sin(2*np.pi*(np.arange(9)+.5)/9)
            psi=np.full(9,-np.pi/2)+(.5 if seed else .2)*np.cos(2*np.pi*np.arange(9)/9)
            jobs.append(dict(name=f'arc_free_n9_{label}_{seed}',cells=9,symmetric=False,
                fractions=fractions.tolist(),parameters=np.r_[phi,psi,np.log(MAX_G*.974)].tolist(),budget=260,
                seed_reason='Two pi/2 envelopes make a closed nine-pulse rotary seed near the first-force speed bound.'))
    assert len(jobs)==20
    print('START ARC PATTERNS:20 physically motivated fits;all18inputs;workers',workers,flush=True)
    with ProcessPoolExecutor(workers) as pool:results=list(pool.map(fit,jobs))
    assert len(results)==20
    (OUT/'pattern_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('ARC PATTERN COMPARISONS:20 fits;',18*sum(r['evaluations'] for r in results),'input-column evaluations;',
          sum(r['fidelity']>.99 for r in results),'design scores above99%',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['check','seeds','refine','patterns']);parser.add_argument('--cells',type=int,default=10)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.stage=='check':check()
    elif args.stage=='seeds':seed_search(args.cells)
    elif args.stage=='refine':refine(args.cells)
    else:pattern_study(args.workers)
