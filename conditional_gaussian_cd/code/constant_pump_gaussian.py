"""Short Gaussian gates with an exactly fixed sideband phase.

Only phases of calibrated 30 ns qubit pi pulses are fitted. The cell model
provides a fast design objective; accepted gates are separately replayed
through SDK words and the full RFDC response. No coupler calculation, pump
chirp or per-cell pump phase is used.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.optimize import minimize
import cd_trains as cd


OUT = ROOT / 'runs/constant_pump_gaussian'
OUT.mkdir(exist_ok=True, parents=True)
STRENGTH = dict(squeeze=cd.R_8DB, rotation=np.pi/2, cd=1.)


class FixedPumpTrain(cd.Train):
    """The existing physical cell, with a cheaper qubit-only adjoint gradient."""
    def qubit_frame(self, output):
        if getattr(self,'fixed_frame',False):
            return np.ones((len(output),1),complex),0.
        return super().qubit_frame(output)

    def objective(self, phases, gradient=True):
        pump=getattr(self,'pump_phase',0.)
        controls = np.r_[phases, np.full(self.cells,pump)]
        output, states = self.run(controls, keep_states=gradient)
        frame, _ = self.qubit_frame(output)
        trace = np.vdot(self.target, frame*output)
        cost = 1-abs(trace/self.d)**2
        if not gradient:
            return float(cost)
        back = self.down.conj().T @ (frame.conj()*self.target)
        k = self.space.k[:, None]
        derivative = np.empty(self.cells)
        for j in range(self.cells-1, -1, -1):
            diagonal = np.exp(1j*(phases[j]*k-pump*self.space.n[:,None]))
            before_back = diagonal*(self.U0.conj().T @ (diagonal.conj()*back))
            # dC/dphi = i[k,C]. Reusing C† back avoids another matrix action.
            slope = 1j*(np.vdot(back, k*states[j+1])-
                        np.vdot(before_back, k*states[j]))
            derivative[j] = -2*np.real(trace.conjugate()*slope)/self.d**2
            back = before_back
        return float(cost), derivative


def phase_map(cells, block, family):
    """An explicit repeated block, optionally with one qubit phase advance."""
    assert cells % block == 0
    if family == 'echo':
        assert block % 2 == 0
        source = np.r_[np.arange(block//2), np.arange(block//2)[::-1]]
        offset = np.tile(np.r_[np.zeros(block//2), np.full(block//2, np.pi)], cells//block)
        parameters = block//2
    else:
        source = np.arange(block)
        offset = np.zeros(cells)
        parameters = block
    mapping = np.zeros((cells, parameters+(family == 'advance')))
    mapping[np.arange(cells), np.tile(source, cells//block)] = 1
    if family == 'advance':
        mapping[:, -1] = np.arange(cells)//block
    return mapping, offset


def resample(phases, cells):
    old = np.unwrap(np.asarray(phases))
    return np.interp((np.arange(cells)+.5)/cells, (np.arange(len(old))+.5)/len(old), old)


def starting_phases(job):
    """Record the physical motivation for each reproducible initial guess."""
    cells = job['cells']
    if 'initial_phases' in job:
        return np.asarray(job['initial_phases'])
    if job['seed'] in ['native_commutator','cubic_pair']:
        if job['gate']=='squeeze':
            # Frames I,Y,Z,X give H, -XHX, -H, XHX: a squeeze commutator.
            base=([np.pi/2,0,np.pi/2,0] if job['seed']=='native_commutator' else
                  [np.pi/2,0,np.pi/2,np.pi/2,3*np.pi/2,np.pi,3*np.pi/2,3*np.pi/2])
        else:
            # Two pi pulses synthesize a qubit Z increment. In the zero-width
            # seed this rotates JC by -pi/2, without changing pump phase.
            signs=([1,1,1,1] if job['seed']=='native_commutator' else
                   [1,1,1,-1,1,1,1,-1])
            base=np.array([[0,-sign*np.pi/4] for sign in signs]).ravel()
        return np.resize(base,cells)
    if job['seed'] == 'commutator':
        # Opposed pairs around two noncommuting qubit axes. This is a seed,
        # not an assertion that finite pulses implement exact inverses.
        delta = np.pi/2 if job['gate'] == 'squeeze' else np.pi/4
        pair_axes = [0, delta, np.pi, delta+np.pi,
                     np.pi, delta+np.pi, 0, delta]
        return np.resize(np.repeat(pair_axes, 2), cells)
    if job['seed'] == 'xy8':
        return np.resize([0, np.pi/2, 0, np.pi/2, np.pi/2, 0, np.pi/2, 0], cells)
    if job['seed'] == 'random':
        return np.random.default_rng(job.get('random_seed', 52)).uniform(-np.pi, np.pi, cells)
    old = json.loads((ROOT/'runs/final.json').read_text())
    index = 2 if job['gate'] == 'squeeze' else 6
    phases = resample(old[index]['qubit_phases'], cells)
    # An unwrapped least-squares block projection is a continuation seed.
    # A second, independently motivated force-loop seed is always tested.
    return phases


def fit(job):
    started = time.monotonic()
    levels = job.get('levels', 64 if job['gate'] == 'squeeze' else 40)
    train = FixedPumpTrain(job['gate'], job.get('strength',STRENGTH[job['gate']]), job['swap'],
                           job['cells'], job.get('gap', 0.), levels)
    train.pump_phase=job.get('pump_phase',0.)
    train.fixed_frame=job.get('fixed_frame',False)
    if train.pump_phase:
        # The pump ramps have the SAME phase as its plateau, including CD.
        phase=np.exp(-1j*train.pump_phase*train.space.n)
        train.up=phase[:,None]*train.up*phase.conj()[None,:]
        train.down=phase[:,None]*train.down*phase.conj()[None,:]
    mapping, offset = phase_map(job['cells'], job['block'], job['family'])
    seed = starting_phases(job)
    initial = np.linalg.lstsq(mapping, seed-offset, rcond=None)[0]
    initial_cost = train.objective(offset+mapping@initial, False)
    history = []
    best = [initial_cost, initial.copy()]
    def objective(parameters):
        cost, derivative = train.objective(offset+mapping@parameters)
        history.append(cost)
        if cost < best[0]:
            best[:] = [cost, parameters.copy()]
        return cost, mapping.T@derivative
    def callback(parameters):
        if best[0] < job.get('stop_infidelity', .006):
            raise StopIteration
    fitted = minimize(objective, initial, method='L-BFGS-B', jac=True, callback=callback,
                      options=dict(maxiter=job.get('maxiter', 1800), maxcor=30,
                                   maxls=35, ftol=1e-13, gtol=2e-9))
    phases = offset+mapping@best[1]
    result = dict(job={k:v for k,v in job.items() if k!='initial_phases'},
                  **train.report(np.r_[phases, np.full(train.cells,train.pump_phase)]),
                  initial_fidelity=1-initial_cost, parameters=best[1].tolist(),
                  qubit_phases=phases.tolist(), pump_phases=np.full(train.cells,train.pump_phase).tolist(),
                  evaluations=len(history), seconds=time.monotonic()-started,
                  history=history[::max(1,len(history)//100)], optimizer_message=str(fitted.message))
    path = OUT/(job['name']+'.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print('FIXED PUMP FIT', job['name'], 'T', result['time_us'], 'initialF',
          result['initial_fidelity'], 'finalF', result['fidelity'], 'variables', len(initial),
          'evaluations', len(history), 'seconds', result['seconds'], flush=True)
    return result


def gradient_check():
    train = FixedPumpTrain('squeeze', .3, .3, 16, .005, 20)
    phases = np.random.default_rng(71).normal(size=16)
    cost, gradient = train.objective(phases)
    reference, full_gradient = train.infidelity(np.r_[phases,np.zeros(16)])
    assert abs(cost-reference)<1e-14 and max(abs(gradient-full_gradient[:16]))<1e-12
    errors=[]
    for seed in range(3):
        direction=np.random.default_rng(seed).normal(size=16);direction/=np.linalg.norm(direction)
        eps=1e-5
        difference=(train.objective(phases+eps*direction,False)-train.objective(phases-eps*direction,False))/(2*eps)
        errors.append(abs(difference-gradient@direction))
    assert len(errors)==3 and max(errors)<1e-8
    print('FIXED PUMP GRADIENT:16components compared with existing gradient;3finite-difference directions;maxerror',max(errors),flush=True)


def jobs_for(stage):
    jobs=[]
    for gate in ['squeeze','rotation']:
        if stage == 'structured':
            points=[(.3,160,32),(.2,128,32),(.15,96,32),(.2,128,64)]
            for swap,cells,block in points:
                for family,seed in [('motif','commutator'),('advance','commutator'),
                                    ('motif','continuation')]:
                    name=f'{gate}_{swap:g}_{cells}_b{block}_{family}_{seed}'
                    jobs.append(dict(name=name,gate=gate,swap=swap,cells=cells,block=block,
                                     family=family,seed=seed,maxiter=1800))
        elif stage == 'continuation':
            for swap,cells,gap in [(.3,160,0),(.2,128,0),(.15,96,0),(.6,128,.0175)]:
                for seed in ['continuation','commutator']:
                    name=f'{gate}_{swap:g}_{cells}_free_{seed}'
                    jobs.append(dict(name=name,gate=gate,swap=swap,cells=cells,gap=gap,
                                     block=cells,family='motif',seed=seed,maxiter=2000))
    return jobs


def run(jobs, workers=8):
    print('START FIXED-SIDEBAND STUDY',len(jobs),'fits;all18inputs;workers',workers,flush=True)
    results=[]
    with ProcessPoolExecutor(workers) as pool:
        futures=[pool.submit(fit,job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            print('COMPLETED FITS',len(results),'/',len(jobs),flush=True)
    assert len(results)==len(jobs)
    return results


def compression_path(gate):
    """Preserve g times cell length while reducing idle time, then increase g.

    The old solution supplies a high-fidelity starting basin. Each neighboring
    problem changes the finite-pulse correction smoothly; it does not randomize
    the whole phase sequence again.
    """
    old=json.loads((ROOT/'runs/final.json').read_text())[2 if gate=='squeeze' else 6]
    phases=np.array(old['qubit_phases']);cells=old['job']['cells']
    gap0=old['job']['gap'];swap0=old['job']['swap'];rows=[]
    gaps=np.arange(gap0-.0025,-1e-10,-.0025)
    schedule=[(max(0,float(gap)),swap0*(cd.PI_TIME+gap)/(cd.PI_TIME+gap0)) for gap in gaps]
    lastswap=schedule[-1][1]
    schedule += [(0.,float(swap)) for swap in np.arange(lastswap-.025,.149,-.025)]
    for step,(gap,swap) in enumerate(schedule):
        job=dict(name=f'{gate}_compression_{step:02d}',gate=gate,swap=swap,cells=cells,
                 gap=gap,block=cells,family='motif',seed='neighboring high-fidelity solution',
                 initial_phases=phases.tolist(),levels=96 if gate=='squeeze' else 48,
                 maxiter=1800,stop_infidelity=.007)
        r=fit(job);rows.append(r);phases=np.asarray(r['qubit_phases'])
        if r['fidelity']<.985:
            print('COMPRESSION STOP',gate,'phase-only continuation below98.5%',flush=True)
            break
    (OUT/(gate+'_compression_path.json')).write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def strength_path(spec):
    """Turn a decoupling/commutator block into a weak Gaussian generator.

    Continue the target strength from zero to the final value while keeping
    every repeat tied. This avoids asking a local fit to jump directly from
    a poor commutator approximation to a large 8 dB gate.
    """
    gate,swap,cells,block,seed=spec
    phases=None;rows=[]
    for step,fraction in enumerate([0.,.125,.25,.5,.75,1.]):
        job=dict(name=f'{gate}_strength_s{swap:g}_b{block}_{seed}_{step}',gate=gate,
                 strength=fraction*STRENGTH[gate],swap=swap,cells=cells,gap=0.,
                 block=block,family='motif',seed=seed,maxiter=1600,
                 stop_infidelity=.001 if fraction<1 else .006)
        if phases is not None:job['initial_phases']=phases.tolist()
        r=fit(job);rows.append(r);phases=np.asarray(r['qubit_phases'])
        if r['fidelity']<.94:
            print('STRENGTH PATH STOP',gate,swap,block,seed,'fraction',fraction,flush=True)
            break
    (OUT/f'{gate}_strength_s{swap:g}_b{block}_{seed}_path.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def native_path(spec):
    """Coupling/target homotopy from a weak native-JC commutator seed.

    The quadratic generator scales as g² at leading order. Increase both the
    coupling and requested strength together, then refit the tied qubit axes
    to cancel finite-pulse and higher-order errors at each neighboring point.
    """
    gate,swap,cells,block,seed=spec;phases=None;rows=[]
    for step,factor in enumerate([4.,3.,2.,1.5,1.2,1.]):
        job=dict(name=f'{gate}_native_s{swap:g}_n{cells}_b{block}_{seed}_{step}',gate=gate,
                 strength=STRENGTH[gate]/factor**2,swap=swap*factor,cells=cells,
                 gap=0.,block=block,family='motif',seed=seed,maxiter=2000,
                 stop_infidelity=.002 if factor>1 else .006)
        if phases is not None:job['initial_phases']=phases.tolist()
        r=fit(job);rows.append(r);phases=np.asarray(r['qubit_phases'])
        if r['fidelity']<.94:break
    (OUT/f'{gate}_native_s{swap:g}_n{cells}_b{block}_{seed}_path.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def fast_cd_jobs():
    """400ns budget:8or12pi pulses, full echo or individually corrected axes."""
    recipe=json.loads((ROOT/'runs/scalable_cd/recipe.json').read_text())
    p=np.array(recipe['parameters'])
    half=np.array([0,0,np.pi,np.pi])+p[:4]
    jobs=[]
    for cells in [8,12]:
        first=resample(half,cells//2)
        native=np.r_[first,first[::-1]+np.pi]
        for swap in [.25,.225,.2,.175,.15,.125,.1]:
            for family in ['echo','motif']:
                jobs.append(dict(name=f'cd_fast_n{cells}_s{swap:g}_{family}',gate='cd',strength=1.,
                                 swap=swap,cells=cells,gap=0.,block=cells,family=family,
                                 seed='previous shared-CD rotary echo',initial_phases=native.tolist(),
                                 pump_phase=-np.pi/2,fixed_frame=True,levels=48,
                                 maxiter=2500,stop_infidelity=.004))
    return jobs


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['gradient','structured','continuation','compression','strength','native','fast_cd'])
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.stage=='gradient':
        gradient_check()
    elif args.stage=='compression':
        print('START two continuation paths:preserve integrated sideband then shorten;all18inputs',flush=True)
        with ProcessPoolExecutor(2) as pool:rows=list(pool.map(compression_path,['squeeze','rotation']))
        print('COMPRESSION COUNTS',len(rows),'paths;',sum(len(r) for r in rows),'fits',flush=True)
    elif args.stage=='strength':
        specs=[(gate,swap,128,block,seed) for gate in ['squeeze','rotation']
               for swap in [.3,.2,.15] for block in [16,32,64] for seed in ['xy8','commutator']]
        print('START',len(specs),'strength-continuation paths; tied repeated qubit blocks',flush=True)
        with ProcessPoolExecutor(args.workers) as pool:rows=list(pool.map(strength_path,specs))
        print('STRENGTH COUNTS',len(rows),'paths;',sum(len(r) for r in rows),'fits',flush=True)
    elif args.stage=='native':
        specs=[(gate,swap,cells,block,seed) for gate in ['squeeze','rotation']
               for swap,cells,block in [(.3,128,16),(.3,128,32),(.2,128,32),
                                        (.2,128,64),(.3,192,64)]
               for seed in ['native_commutator','cubic_pair']]
        print('START',len(specs),'native commutator continuation paths;fixed sideband phase',flush=True)
        with ProcessPoolExecutor(args.workers) as pool:rows=list(pool.map(native_path,specs))
        print('NATIVE PATH COUNTS',len(rows),'paths;',sum(len(r) for r in rows),'fits',flush=True)
    elif args.stage=='fast_cd':
        rows=run(fast_cd_jobs(),args.workers)
        (OUT/'fast_cd_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('FAST CD COUNTS',len(rows),'fits;',sum(r['evaluations'] for r in rows),'evaluations',flush=True)
    else:
        rows=run(jobs_for(args.stage),args.workers)
        (OUT/(args.stage+'_summary.json')).write_text(json.dumps(rows,indent=2)+'\n')
        print('FINAL FIT COUNTS',len(rows),'fits;',sum(r['evaluations'] for r in rows),'objective evaluations',flush=True)
