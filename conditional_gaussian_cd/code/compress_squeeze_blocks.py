"""Shorten an accepted squeeze by deleting the least important pulse pair.

At each step the full 18-input coherent trace chooses the pair. The JC
commutator area suggests g proportional to N**(-1/2) as pulses are removed.
An exact cell adjoint then corrects the phases. These cell scores are seeds,
not final fidelity claims; the separate played-waveform fit follows.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import cd_trains as cd
from higher_order_cd import OUT, ROOT
from shaped_pulse_optimisation import optimise


def delete_pair(train, x):
    n = train.cells; phi, psi = x[:n], x[n:]
    D = np.exp(1j*(phi[:, None]*train.space.k-psi[:, None]*train.space.n))
    forward = [train.up @ train.inputs]
    for j in range(n):
        forward.append(D[j, :, None]*(train.U0 @ (D[j, :, None].conj()*forward[-1])))
    targetg = train.target.copy(); targetg[1::2] = 0
    targete = train.target.copy(); targete[0::2] = 0
    backwardg = [None]*(n+1); backwarde = [None]*(n+1)
    backwardg[n] = train.down.conj().T @ targetg
    backwarde[n] = train.down.conj().T @ targete
    for j in range(n-1, -1, -1):
        for backward in [backwardg, backwarde]:
            backward[j] = D[j, :, None]*(train.U0.conj().T @ (D[j, :, None].conj()*backward[j+1]))
    scores = [(abs(np.vdot(backwardg[j+2], forward[j]))+
               abs(np.vdot(backwarde[j+2], forward[j])))**2/18**2 for j in range(n-1)]
    choice = int(np.argmax(scores))
    keep = np.r_[np.arange(choice), np.arange(choice+2, n)]
    return np.r_[phi[keep], psi[keep]], choice, float(scores[choice])


def main():
    source = json.loads((ROOT/'runs/constant_pump_gaussian/strength_squeeze_swap300.json').read_text())
    n0 = source['job']['cells']; p = np.asarray(source['parameters'])
    x = p[:-1].copy(); g0 = np.exp(p[-1]); history = []
    train = cd.Train('squeeze', cd.R_8DB, np.pi/(2*g0), n0, 0., 64)
    for n in range(n0-2, 111, -2):
        x, deleted, uncorrected = delete_pair(train, x)
        g = g0*np.sqrt(n0/n)
        train = cd.Train('squeeze', cd.R_8DB, np.pi/(2*g), n, 0., 64)
        best = [np.inf, x.copy()]; evaluations = [0]
        def cost(y):
            value, gradient = train.infidelity(y)
            evaluations[0] += 1
            if value < best[0]: best[:] = [value, y.copy()]
            return value, gradient
        minimize(cost, x, jac=True, method='L-BFGS-B',
                 options=dict(maxiter=140, maxcor=25, maxls=25, ftol=1e-12, gtol=1e-8))
        x = best[1]
        row = dict(cells=n, deleted_pair_start=deleted, deletion_fidelity=uncorrected,
                   corrected_cell_fidelity=1-best[0], evaluations=evaluations[0],
                   g=g, qubit_phases=x[:n].tolist(), pump_phases=x[n:].tolist())
        history.append(row)
        print('SQUEEZE PAIR REMOVAL', n, 'deleted', deleted, 'before', uncorrected,
              'corrected', 1-best[0], 'evaluations', evaluations[0], flush=True)
        (OUT/'squeeze_pair_removal.json').write_text(json.dumps(history, indent=2)+'\n')
    # Correct the finite 10 ns pump transitions after the cell construction.
    job = dict(name='compressed_squeeze_n112', gate='squeeze', cells=n,
               qubit_phases=x[:n].tolist(), pump_phases=x[n:].tolist(), g=float(g),
               pump_control=True, frame=True, levels=64, maxiter=200,
               evaluation_budget=90, time_budget_seconds=360, stop_infidelity=.004)
    result = optimise(job)
    (OUT/'compressed_squeeze_n112.json').write_text(json.dumps(result, indent=2)+'\n')
    print('COMPLETED PAIR REMOVAL', len(history), 'pair comparisons/corrections;',
          'final played F', result['fidelity'], flush=True)


def finish_nearby(n):
    rows=json.loads((OUT/'squeeze_pair_removal.json').read_text())
    row=next(r for r in rows if r['cells']==n)
    job=dict(name=f'compressed_squeeze_n{n}',gate='squeeze',cells=n,
             qubit_phases=row['qubit_phases'],pump_phases=row['pump_phases'],g=row['g'],
             pump_control=True,frame=True,levels=64,maxiter=150,
             evaluation_budget=85,time_budget_seconds=320,stop_infidelity=.0065)
    result=optimise(job)
    (OUT/(job['name']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    if '--nearby' in sys.argv:
        print('START NEARBY SQUEEZE CORRECTIONS:2fits;all18inputs',flush=True)
        with ProcessPoolExecutor(2) as pool:rows=list(pool.map(finish_nearby,[120,124]))
        assert len(rows)==2
        (OUT/'nearby_squeeze_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('COMPLETED NEARBY SQUEEZE CORRECTIONS',len(rows),'fits;',18*sum(r['evaluations'] for r in rows),'inputcolumns',flush=True)
    else:main()
