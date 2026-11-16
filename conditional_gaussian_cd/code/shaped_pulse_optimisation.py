"""Fit pulse phases and one constant pump strength through smooth actual playback.

Controls are 30 ns pi-pulse axes and plateau sideband phases. Every pump
transition uses a 10 ns quintic Cartesian I/Q interpolation. The objective
packs SDK words and propagates RFDC output on EVERY call. The derivative uses
the continuous-control chain through those filters (ignoring quantizer steps)
and a checked endpoint approximation to each local Frechet integral.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,ctypes,hashlib,json,subprocess,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.signal import fftconvolve
import cd_trains as cd
import hardware_check as hw


OUT=ROOT/'runs/constant_pump_gaussian'


class PlayedGate:
    def __init__(self,gate,cells,levels=64,pump_control=True,frame=True,angle_fractions=None):
        self.gate=gate;self.cells=cells;self.levels=levels
        self.pump_control=pump_control;self.frame=frame
        self.strength={'cd':1.,'squeeze':cd.R_8DB,'rotation':np.pi/2}[gate]
        space=cd.Space(levels,9);self.z=space.z
        self.initial=np.eye(2*levels,dtype=complex)[:,space.inputs]
        self.target=cd.target_columns(space,gate,self.strength)
        fractions=np.ones(cells) if angle_fractions is None else np.asarray(angle_fractions)
        assert fractions.shape==(cells,) and np.all((fractions>0)&(fractions<=1))
        count=8+24*cells;self.qshape=np.zeros(count,complex)
        self.qcell=np.full(count,-1,int)
        W=np.zeros((count,cells))
        for j in range(cells):
            selection=slice(4+24*j,4+24*(j+1))
            self.qshape[selection]=cd.pi_samples()*fractions[j];self.qcell[selection]=j
            W[selection,j]=1.
        W[:4,0]=np.sin(np.pi*np.arange(4)/8)**2
        W[-4:,-1]=np.cos(np.pi*np.arange(4)/8)**2
        u=np.arange(8)/8;h=10*u**3-15*u**4+6*u**5
        for j in range(1,cells):
            selection=slice(24*j,24*j+8)
            W[selection,:]=0;W[selection,j-1]=1-h;W[selection,j]=h
        self.W=csr_matrix(W);self.filter=hw.rfdc_interpolate(np.array([1.])).real
        cpp=CODE / 'waveform_adjoint.cpp'
        lib=Path('/tmp')/('waveform_adjoint_'+hashlib.sha256(cpp.read_bytes()).hexdigest()[:16]+'.so')
        if not lib.exists():subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(cpp),'-o',str(lib)],check=True)
        self.backward=ctypes.CDLL(str(lib)).backward_waveform
        self.backward.argtypes=[ctypes.c_int]*4+[ctypes.c_double]*3+[ctypes.c_void_p]*5

    def arrays(self,parameters,quantized=True):
        n=self.cells;phi=parameters[:n]
        psi=parameters[n:2*n] if self.pump_control else np.full(n,-np.pi/2 if self.gate=='cd' else 0.)
        g=np.exp(parameters[-1])
        q=self.qshape.copy();keep=self.qcell>=0
        q[keep]*=np.exp(1j*phi[self.qcell[keep]])
        s=g*(self.W@np.exp(1j*psi))
        words=None
        if quantized:
            qs,ss=hw.dac_scales(.6)
            qw,q=hw.to_words_and_back(q,np.pi/cd.PI_TIME,qs,qubit=True)
            sw,s=hw.to_words_and_back(s,np.pi/(2*.6),ss)
            words=dict(qubit=qw,sideband=sw)
        return q,s,hw.rfdc_interpolate(q),hw.rfdc_interpolate(s),words

    def evaluate(self,parameters,gradient=True,quantized=True,outputs=False):
        q,s,qf,sf,words=self.arrays(parameters,quantized)
        output=hw.propagate(qf,sf,self.initial,cd.DT/8,order=8,fast=True)
        zeta=0.
        if self.frame:
            zg=np.vdot(self.target[0::2],output[0::2]);ze=np.vdot(self.target[1::2],output[1::2])
            zeta=float(np.angle(zg)-np.angle(ze))
        frame=np.exp(-1j*zeta*self.z/2)[:,None]
        trace=np.vdot(self.target,frame*output);cost=float(1-abs(trace/18)**2)
        if outputs:return dict(fidelity=1-cost,qubit_frame_Rz=zeta),frame*output,self.target,words
        if not gradient:return cost
        adjoint=np.ascontiguousarray(frame.conj()*self.target)
        grad=np.zeros((len(qf),4),float)
        self.backward(self.levels,18,len(qf),8,cd.DT/8,float(trace.real),float(trace.imag),
                      qf.ctypes.data,sf.ctypes.data,output.ctypes.data,adjoint.ctypes.data,grad.ctypes.data)
        Gq=fftconvolve(grad[:,0]+1j*grad[:,1],self.filter[::-1],mode='valid')[::8]
        Gs=fftconvolve(grad[:,2]+1j*grad[:,3],self.filter[::-1],mode='valid')[::8]
        keep=self.qcell>=0
        # Use the continuous parameter derivative; forward controls still use
        # the signed SDK words. This is not a derivative of the staircase map.
        qb=self.qshape[keep]*np.exp(1j*parameters[:self.cells][self.qcell[keep]])
        dphi=np.bincount(self.qcell[keep],weights=np.real(Gq[keep].conj()*1j*qb),minlength=self.cells)
        if self.pump_control:
            psi=parameters[self.cells:2*self.cells]
            dpsi=np.real((self.W.T@Gs).conj()*1j*np.exp(1j*psi))*np.exp(parameters[-1])
        else:dpsi=np.array([])
        # Log(g) has derivative s itself.
        sb=np.exp(parameters[-1])*(self.W@np.exp(1j*(psi if self.pump_control else
                         np.full(self.cells,-np.pi/2 if self.gate=='cd' else 0.))))
        dg=float(np.real(np.vdot(Gs,sb)))
        return cost,np.r_[dphi,dpsi,dg]


def gradient_check():
    model=PlayedGate('cd',8,24,True,False)
    rng=np.random.default_rng(104)
    p=np.r_[rng.normal(0,.1,8),-np.pi/2+rng.normal(0,.1,8),np.log(5.8)]
    _,g=model.evaluate(p,quantized=False);errors=[]
    for i in range(4):
        v=rng.normal(size=len(p));v/=np.linalg.norm(v);e=1e-5
        fd=(model.evaluate(p+e*v,False,False)-model.evaluate(p-e*v,False,False))/(2*e)
        errors.append(float(abs(fd-g@v)))
    assert len(errors)==4 and max(errors)<2e-5,(errors,g)
    print('PLAYED-WAVEFORM GRADIENT:4directions;maxfinite-differenceerror',max(errors),flush=True)
    (OUT/'waveform_gradient_check.json').write_text(json.dumps(dict(directions=4,max_error=max(errors),errors=errors),indent=2)+'\n')


def optimise(job):
    start=time.monotonic();n=job['cells'];pump=job.get('pump_control',True)
    design_seed=None
    if job.get('cell_seed',False):
        # Cheap neighboring-strength continuation supplies the starting axes;
        # its score is never substituted for actual 10ns SDK/RFDC playback.
        from optimise_trains import optimise as fit_cells
        design_seed=fit_cells(dict(gate=job['gate'],swap=np.pi/(2*job['g']),cells=n,
                                  gap=0.,kind='free',start='given',
                                  p0=list(job['qubit_phases'])+list(job['pump_phases']),
                                  levels=job.get('levels',64),maxiter=1000,maxcor=30))
        job=dict(job,qubit_phases=design_seed['qubit_phases'],pump_phases=design_seed['pump_phases'])
        print('CELL CONTINUATION SEED',job['name'],'F',design_seed['fidelity'],flush=True)
    model=PlayedGate(job['gate'],n,job.get('levels',64),pump,job.get('frame',True),job.get('angle_fractions'))
    block=job.get('block',n);assert n%block==0
    initial=np.r_[job['qubit_phases'][:block],job['pump_phases'][:block] if pump else [],np.log(job['g'])]
    def expand(p):
        return np.r_[np.tile(p[:block],n//block),np.tile(p[block:2*block],n//block) if pump else [],p[-1]]
    history=[];best=[np.inf,None]
    def cost(p):
        value,gradient=model.evaluate(expand(p));history.append(value)
        gradient=np.r_[gradient[:n].reshape(-1,block).sum(axis=0),
                       gradient[n:2*n].reshape(-1,block).sum(axis=0) if pump else [],gradient[-1]]
        if value<best[0]:best[:]=[value,p.copy()]
        if len(history)%40==0:print('PLAYBACK FIT',job['name'],'eval',len(history),'bestF',1-best[0],flush=True)
        if len(history)%10==0:
            (OUT/(job['name']+'_checkpoint.json')).write_text(json.dumps(dict(job=job,parameters=expand(best[1]).tolist(),fidelity=1-best[0],evaluations=len(history)))+'\n')
        return value,gradient
    def callback(p):
        if best[0]<job.get('stop_infidelity',.008):raise StopIteration
        if len(history)>=job.get('evaluation_budget',100000):raise StopIteration
        if time.monotonic()-start>job.get('time_budget_seconds',3600):raise StopIteration
    bounds=[(None,None)]*(len(initial)-1)+[(np.log(2.),np.log(16.))]
    if job.get('fixed_g',False):bounds[-1]=(initial[-1],initial[-1])
    result=minimize(cost,initial,jac=True,method='L-BFGS-B',bounds=bounds,callback=callback,
                    options=dict(maxiter=job.get('maxiter',500),maxcor=30,maxls=30,ftol=1e-12,gtol=1e-8))
    p=expand(best[1]);report,output,target,words=model.evaluate(p,False,outputs=True)
    overlaps=np.sum(target.conj()*output,axis=0)
    report.update(job=job,parameters=p.tolist(),pattern_parameters=best[1].tolist(),initial_fidelity=1-history[0],
                  worst_basis_fidelity=float(min(abs(overlaps)**2)),history=history,
                  evaluations=len(history),seconds=time.monotonic()-start,
                  command_us=(24*n+8)*cd.DT,RFDC_window_us=len(model.arrays(p)[2])*cd.DT/8,
                  g=float(np.exp(p[-1])),equivalent_swap_ns=float(1000*np.pi/(2*np.exp(p[-1]))),
                  optimizer_message=str(result.message))
    if design_seed is not None:
        report['cell_seed_fidelity']=design_seed['fidelity']
        report['cell_seed_evaluations']=design_seed['evaluations']
    (OUT/(job['name']+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('PLAYBACK RESULT',job['name'],'F',report['fidelity'],'command',report['command_us'],
          'RFDCwindow',report['RFDC_window_us'],'evaluations',len(history),'seconds',report['seconds'],flush=True)
    return report


def jobs(stage):
    if stage=='shorter':
        from constant_pump_gaussian import resample
        result=[]
        for gate,basefile,points in [
            ('squeeze','strength_squeeze_swap300.json',[(96,200),(96,225),(96,250)]),
            ('rotation','strength_rotation_swap150.json',[(80,125),(80,150),(64,100),(64,125)])]:
            base=json.loads((OUT/basefile).read_text());oldn=base['job']['cells'];p=np.asarray(base['parameters'])
            for n,swap_ns in points:
                result.append(dict(name=f'shorter_{gate}_n{n}_swap{swap_ns}',gate=gate,cells=n,
                                   qubit_phases=resample(p[:oldn],n).tolist(),
                                   pump_phases=resample(p[oldn:2*oldn],n).tolist(),
                                   g=float(np.pi/(2*swap_ns*.001)),fixed_g=True,pump_control=True,frame=True,
                                   levels=64 if gate=='squeeze' else 48,cell_seed=True,
                                   maxiter=60,evaluation_budget=35,time_budget_seconds=180,stop_infidelity=.007))
        return result
    if stage=='gaussian_strength':
        result=[]
        for gate in ['squeeze','rotation']:
            r=json.loads((OUT/f'played_{gate}_smooth.json').read_text());n=r['job']['cells'];p=np.asarray(r['parameters'])
            for swap_ns in [150,200,250,300,400,600]:
                result.append(dict(name=f'strength_{gate}_swap{swap_ns}',gate=gate,cells=n,
                                   qubit_phases=p[:n].tolist(),pump_phases=p[n:2*n].tolist(),
                                   g=float(np.pi/(2*swap_ns*.001)),fixed_g=True,pump_control=True,frame=True,
                                   levels=64 if gate=='squeeze' else 48,maxiter=60,evaluation_budget=42,
                                   time_budget_seconds=240,stop_infidelity=.005,cell_seed=True))
        return result
    if stage=='angles':
        result=[];library=json.loads((OUT/'binary_cd_summary.json').read_text())['results']
        for n in [8,10]:
            base=max([r for r in library if r['job']['cells']==n],key=lambda r:r['fidelity'])
            for label,indices,fraction in [('outer_half',[0,n-1],.5),('four_half',[0,1,n-2,n-1],.5),
                                           ('middle_half',[n//2-1,n//2],.5),('outer_third',[0,n-1],1/3),
                                           ('four_quarter',[0,1,n-2,n-1],.25)]:
                for seed in [0,1]:
                    rng=np.random.default_rng(581+n+seed);q=np.array(base['qubit_phases'])+rng.normal(0,.25,n)
                    angles=np.ones(n);angles[indices]=fraction
                    result.append(dict(name=f'angle_cd_n{n}_{label}_seed{seed}',gate='cd',cells=n,
                                       qubit_phases=q.tolist(),pump_phases=(-np.pi/2+rng.normal(0,.15,n)).tolist(),
                                       g=np.pi/(2*base['job']['swap'])*(1.15 if seed else 1.),
                                       pump_control=True,frame=True,levels=48,maxiter=200,evaluation_budget=150,
                                       time_budget_seconds=100,angle_fractions=angles.tolist()))
        return result
    if stage=='strength':
        previous=json.loads((OUT/'cd_played_summary.json').read_text());result=[]
        for cells in [10,12]:
            for pump in [False,True]:
                best=max([r for r in previous if r['job']['cells']==cells and r['job']['pump_control']==pump],key=lambda r:r['fidelity'])
                p=np.array(best['parameters']);n=cells
                for swap_ns in [150,175,200,225,250,275,300,350,400,500,600]:
                    result.append(dict(name=f'strength_cd_n{cells}_pump{int(pump)}_swap{swap_ns}',gate='cd',cells=cells,
                                       qubit_phases=p[:n].tolist(),pump_phases=(p[n:2*n].tolist() if pump else np.full(n,-np.pi/2).tolist()),
                                       g=float(np.pi/(2*swap_ns*.001)),fixed_g=True,pump_control=pump,
                                       frame=True,levels=48,maxiter=100,evaluation_budget=90,
                                       time_budget_seconds=90,stop_infidelity=.0005))
        return result
    if stage=='cd':
        library=json.loads((OUT/'binary_cd_summary.json').read_text())['results'];result=[]
        for n in [8,10,12]:
            base=max([r for r in library if r['job']['cells']==n],key=lambda r:r['fidelity'])
            for pump in [False,True]:
                for seed in [0,1]:
                    rng=np.random.default_rng(seed+62)
                    q=np.array(base['qubit_phases'])+rng.normal(0,.06,n)
                    psi=-np.pi/2+rng.normal(0,.03,n) if pump else np.full(n,-np.pi/2)
                    result.append(dict(name=f'played_cd_n{n}_pump{int(pump)}_seed{seed}',gate='cd',cells=n,
                                       qubit_phases=q.tolist(),pump_phases=psi.tolist(),g=np.pi/(2*base['job']['swap']),
                                       pump_control=pump,frame=True,levels=48,maxiter=700))
        return result
    if stage=='pattern':
        result=[]
        for gate,filename in [('squeeze','squeeze_native_s0.3_n128_b32_cubic_pair_3.json'),
                              ('rotation','rotation_native_s0.2_n128_b32_cubic_pair_2.json')]:
            base=json.loads((OUT/filename).read_text())
            target=cd.R_8DB if gate=='squeeze' else np.pi/2
            for n,block in [(96,32),(128,32),(128,64)]:
                q=np.resize(base['qubit_phases'][:block],n)
                g=np.pi/(2*base['job']['swap'])*np.sqrt(target/(base['job']['strength']*n/128))
                result.append(dict(name=f'played_{gate}_pattern_n{n}_b{block}',gate=gate,cells=n,block=block,
                                   qubit_phases=q.tolist(),pump_phases=np.zeros(n).tolist(),g=float(g),
                                   pump_control=True,frame=True,levels=96 if gate=='squeeze' else 48,
                                   maxiter=450,trotter_seed=filename))
        return result
    if stage=='cd_seeds':
        result=[];old=json.loads((ROOT/'runs/cd_final.json').read_text())[1]
        from constant_pump_gaussian import resample
        for n in [8,10,12]:
            for seed in ['previous_free','strong_echo','phase_walk']:
                rng=np.random.default_rng(274+n)
                if seed=='previous_free':
                    q=resample(old['qubit_phases'],n);psi=resample(old['pump_phases'],n)
                    g=np.pi/(2*old['job']['swap'])*old['job']['cells']/n
                else:
                    base=max([r for r in json.loads((OUT/'binary_cd_summary.json').read_text())['results'] if r['job']['cells']==n],key=lambda r:r['fidelity'])
                    q=np.array(base['qubit_phases'])+rng.normal(0,.5,n)
                    psi=-np.pi/2+rng.normal(0,.5,n);g=np.pi/(2*base['job']['swap'])*1.35
                    if seed=='phase_walk':psi+=np.linspace(-.8,.8,n)
                result.append(dict(name=f'played_cd_n{n}_{seed}',gate='cd',cells=n,qubit_phases=q.tolist(),
                                   pump_phases=psi.tolist(),g=float(g),pump_control=True,frame=True,levels=48,maxiter=450,seed=seed))
        return result
    result=[]
    for gate,group,index in [('squeeze','round3',42),('rotation','round2',119)]:
        base=json.loads((ROOT/f'runs/{group}.json').read_text())[index]
        result.append(dict(name=f'played_{gate}_smooth',gate=gate,cells=base['job']['cells'],
                           qubit_phases=base['qubit_phases'],pump_phases=base['pump_phases'],
                           g=np.pi/(2*base['job']['swap']),pump_control=True,frame=True,
                           levels=96 if gate=='squeeze' else 48,maxiter=600))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['gradient','cd','gaussian','pattern','cd_seeds','strength','angles','gaussian_strength','shorter'])
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if args.stage=='gradient':gradient_check()
    else:
        work=jobs(args.stage);print('START SHAPED PLAYBACK FITS',len(work),'jobs;10ns transitions;all18inputs',flush=True)
        rows=[]
        with ProcessPoolExecutor(args.workers) as pool:
            for result in pool.map(optimise,work):rows.append(result)
        assert len(rows)==len(work)
        (OUT/(args.stage+'_played_summary.json')).write_text(json.dumps(rows,indent=2)+'\n')
        print('FINAL SHAPED FIT COUNTS',len(rows),'jobs;',sum(r['evaluations'] for r in rows),'actualplaybackevaluations',flush=True)
