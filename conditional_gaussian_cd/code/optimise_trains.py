"""Optimise the phases of a pi-pulse train for conditional squeezing or conditional rotation.

    $ACADIA_ENV/bin/python optimise_trains.py sweep      # every (gate, swap, cells, structure, start) in SWEEP -> runs/*.json

One optimisation = L-BFGS-B on 1 - F with the exact gradient from cd_trains.Train (about 0.1 s per evaluation for
128 cells). Several starting points run in parallel; the best is kept. Results (phases, fidelity, history) are saved
as JSON so the notebook only reads them.

Starting points:
    "cd seed"     the textbook commutator construction: pairs of cells act as conditional displacements (CDs) and
                  4 CDs in a loop A, B, -A, -B make the commutator [A, B]; A, B are chosen so that the commutator is
                  the squeezing (sigma_x x, sigma_y p) or rotation (sigma_x x, -sigma_y x and the same with p)
                  generator, with the loop angle set to give the target strength (needs cells divisible by 16 / 32).
    "random k"    uniformly random phases (seed k)
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import minimize

import cd_trains as cd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = str(ROOT / "runs")
STRENGTH = {"squeeze": cd.R_8DB, "rotation": np.pi / 2}     # 8 dB squeezing; pi relative rotation between g and e


def cd_seed(gate, cells, g, cell_time):
    """Commutator-loop starting phases (qubit phase, pump phase) per cell; two cells make one CD."""
    per_block = 16 if gate == "squeeze" else 32
    if cells % per_block:
        return None
    blocks = cells // per_block
    eps = g * (2 * cell_time) / np.sqrt(2)                           # CD amplitude of one two-cell CD
    nominal = (4 if gate == "squeeze" else 8) * eps ** 2 * blocks    # strength if every loop is fully open
    delta = np.arcsin(min(0.999999, STRENGTH[gate] / nominal))      # open each loop just enough
    pairs = []
    inverse = lambda pair: (pair[0] + np.pi, pair[1] + np.pi)
    for _ in range(blocks):
        for quadrature in (["xp"] if gate == "squeeze" else ["x", "p"]):
            if gate == "squeeze":
                A, B = (0.0, np.pi), (delta, delta + np.pi / 2)
            else:
                base = np.pi if quadrature == "x" else np.pi / 2
                A, B = (0.0, base), (-delta, base - delta)
            for pair in (A, B, inverse(A), inverse(B), inverse(A), inverse(B), A, B):
                pairs += [pair, pair]
    return np.r_[[q for q, _ in pairs], [p for _, p in pairs]]


def optimise(job):
    """job = dict(gate, swap, cells, gap, kind, motif, start, levels, maxiter). Returns the result dict."""
    t0 = time.time()
    train = cd.Train(job["gate"], job.get("strength", STRENGTH.get(job["gate"])), job["swap"], job["cells"], job["gap"],
                     job.get("levels", 128), job.get("n_input", 9))
    A = cd.structure(job["cells"], job["kind"], job.get("motif"))
    offset=np.zeros(2*job['cells'])
    if job['kind']=='qubit only':offset[job['cells']:]=job.get('fixed_pump_phase',0.)
    rng = np.random.default_rng(job.get("seed", 0))
    if job["start"] == "cd seed":
        x0 = cd_seed(job["gate"], job["cells"], train.g, cd.PI_TIME + job["gap"])
        p0 = np.linalg.lstsq(A, x0-offset, rcond=None)[0]              # nearest structured parameters
    elif job["start"] == "native":                                       # today's X X Xbar Xbar train, given offsets
        x0 = np.r_[np.resize([0.0, 0.0, np.pi, np.pi], job["cells"]) + job["offsets"][0], np.full(job["cells"], job["offsets"][1])]
        p0 = np.linalg.lstsq(A, x0-offset, rcond=None)[0]
    elif job["start"] == "given":
        p0 = np.array(job["p0"])
    else:
        p0 = rng.uniform(0, 2 * np.pi, A.shape[1])
    history = []

    def cost(p):
        c, g = train.infidelity(offset+A @ p)
        history.append(c)
        return c, A.T @ g

    best = minimize(cost, p0, jac=True, method="L-BFGS-B",
                    options=dict(maxiter=job.get("maxiter", 1500), maxcor=job.get("maxcor", 10), ftol=1e-13, gtol=1e-9, maxls=40))
    x = offset+A @ best.x
    out = dict(job={k: v for k, v in job.items() if k != "p0"}, **train.report(x),
               qubit_phases=(x[:job["cells"]] % (2 * np.pi)).tolist(), pump_phases=(x[job["cells"]:] % (2 * np.pi)).tolist(),
               parameters=best.x.tolist(), evaluations=len(history), seconds=time.time() - t0,
               history=history[::max(1, len(history) // 200)], message=str(best.message))
    return out


def run_jobs(jobs, workers=24):
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(optimise, jobs))
    assert len(results) == len(jobs)
    return results


def save(name, data):
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, name), "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "sweep":
        # how short can each gate be? every swap time, number of cells, and 4 random starts + the cd seed
        jobs = []
        for gate in ("squeeze", "rotation"):
            for swap in (0.6, 0.3, 0.2):
                for cells in (32, 48, 64, 96, 128, 160, 192, 256):
                    for start in ["cd seed"] + [f"random {s}" for s in range(4)]:
                        if start == "cd seed" and cd_seed(gate, cells, 1.0, 0.03) is None:
                            continue
                        jobs.append(dict(gate=gate, swap=swap, cells=cells, gap=0.0, kind="free", start=start,
                                         seed=int(start.split()[-1]) if start.startswith("random") else 0))
        jobs.sort(key=lambda j: -j["cells"])                             # longest first, for load balance
        t0 = time.time()
        results = run_jobs(jobs)
        save("sweep.json", results)
        print(f"{len(results)} optimisations in {time.time() - t0:.0f} s")
    elif what == "round2":
        # (A) more iterations for the best sweep runs, (B) gaps at fixed time, (C) structured trains
        sweep = json.load(open(os.path.join(RUNS, "sweep.json")))
        levels = {"squeeze": 96, "rotation": 48}                         # cutoffs checked to 1e-8 against 192 levels
        jobs = []
        best = {}
        for r in sweep:
            key = (r["job"]["gate"], r["job"]["swap"], r["cells"])
            if key not in best or r["fidelity"] > best[key]["fidelity"]:
                best[key] = r
        for (gate, swap, cells), r in best.items():                      # (A) continue the best run of each point
            if r["fidelity"] > 0.9:
                jobs.append(dict(study="refine", gate=gate, swap=swap, cells=cells, gap=0.0, kind="free", start="given",
                                 p0=r["parameters"], levels=levels[gate], maxiter=5000))
        for gate in ("squeeze", "rotation"):                             # (B) 600 ns swap, same time, different gaps
            for cells, gap in ((256, 0.0), (192, 0.010), (128, 0.030),
                               (256, 0.0075), (192, 0.020), (160, 0.030),
                               (320, 0.010), (256, 0.020), (160, 0.050)):
                for seed in (0, 1):
                    jobs.append(dict(study="gaps", gate=gate, swap=0.6, cells=cells, gap=gap, kind="free",
                                     start=f"random {seed}", seed=seed, levels=levels[gate], maxiter=5000))
        points = [("squeeze", 0.3, 160, 0.0), ("rotation", 0.2, 128, 0.0), ("squeeze", 0.6, 256, 0.020), ("rotation", 0.6, 256, 0.020)]
        for gate, swap, cells, gap in points:                            # (C) structures
            for kind, motif in (("free", None), ("palindrome", None), ("motif", 16), ("motif", 32), ("qubit only", None)):
                for seed in (0, 1, 2):
                    jobs.append(dict(study="structure", gate=gate, swap=swap, cells=cells, gap=gap, kind=kind, motif=motif,
                                     start=f"random {seed}", seed=seed, levels=levels[gate], maxiter=5000))
        jobs.sort(key=lambda j: -j["cells"] * (2 if j["gate"] == "squeeze" else 1))
        t0 = time.time()
        results = run_jobs(jobs)
        save("round2.json", results)
        print(f"{len(results)} optimisations in {time.time() - t0:.0f} s")
    elif what == "round3":
        # shortest gate with gaps: for each total time, several numbers of pi pulses with the gap that fills the time
        levels = {"squeeze": 96, "rotation": 48}
        jobs = []
        grid = {0.6: [4.81, 5.77, 6.73, 7.69, 8.65], 0.3: [2.89, 3.85, 4.81]}
        for swap, times in grid.items():
            for total in times:
                for cells in (64, 96, 128, 160):
                    gap = round(((total - 0.010) / cells - cd.PI_TIME) / 0.0025) * 0.0025   # whole DAC sample pairs
                    if gap < -1e-12 or gap > 0.100:
                        continue
                    gap = max(gap, 0.0)
                    for gate in ("squeeze", "rotation"):
                        for kind in ("free", "qubit only"):
                            if swap == 0.3 and kind == "qubit only":
                                continue
                            for seed in (0, 1):
                                jobs.append(dict(study="time", gate=gate, swap=swap, cells=cells, gap=gap, kind=kind,
                                                 start=f"random {seed}", seed=seed, levels=levels[gate],
                                                 maxiter=8000, maxcor=50))
        jobs.sort(key=lambda j: -j["cells"] * (2 if j["gate"] == "squeeze" else 1))
        print(len(jobs), "jobs", flush=True)
        t0 = time.time()
        results = run_jobs(jobs)
        save("round3.json", results)
        print(f"{len(results)} optimisations in {time.time() - t0:.0f} s")
    elif what == "no_chi":
        # every earlier optimisation (made with a static dispersive shift chi) re-optimised WITHOUT chi, warm-started
        # from its own phases; same studies, structures and starts (2026-11-14: the model has no static chi)
        old = sys.argv[2]                                                  # folder with the earlier sweep/round2/round3
        levels = {"squeeze": 96, "rotation": 48}
        for name in ("sweep.json", "round2.json", "round3.json"):
            jobs = []
            for r in json.load(open(os.path.join(old, name))):
                job = dict(r["job"], start="given", p0=r["parameters"], levels=levels[r["job"]["gate"]], maxiter=2000, maxcor=50,
                           first_start=r["job"]["start"], study=r["job"].get("study", "sweep"))
                jobs.append(job)
            jobs.sort(key=lambda j: -j["cells"] * (2 if j["gate"] == "squeeze" else 1))
            t0 = time.time()
            results = run_jobs(jobs)
            save(name, results)
            print(f"{name}: {len(results)} optimisations in {time.time() - t0:.0f} s", flush=True)
    elif what == "cd":
        # conditional displacement exp[beta sigma_x (a^dag - a)] on qubit g/e x Fock 0..10, at 300 ns and 600 ns swap:
        # today's X X Xbar Xbar train as a start, and random starts; free, palindromic and qubit-only phases
        jobs = []
        for swap in (0.3, 0.6):
            for beta in (1.0, 2.0):
                rate = np.pi / (2 * swap) / 2                                    # ideal CD rate g/2 (beta per us)
                for cells in (4, 8, 12, 16, 24, 32, 48):
                    for gap in (0.0, 0.010, 0.020, 0.030):
                        T = cells * (cd.PI_TIME + gap) + 0.010
                        if T * rate < 0.8 * beta or T * rate > 4 * beta:         # too short to reach beta, or far too long
                            continue
                        for kind in ("free", "palindrome", "qubit only"):
                            starts = ["native"] + [f"random {s}" for s in range(3)]
                            for start in starts:
                                jobs.append(dict(study="cd", gate="cd", strength=beta, swap=swap, cells=cells, gap=gap,
                                                 kind=kind, start=start, seed=int(start.split()[-1]) if start.startswith("random") else 0,
                                                 fixed_pump_phase=-np.pi/2 if kind=='qubit only' else 0.,
                                                 offsets=(0.0, -np.pi / 2), levels=64, n_input=11, maxiter=4000, maxcor=50))
        jobs.sort(key=lambda j: -j["cells"])
        print(len(jobs), "jobs", flush=True)
        t0 = time.time()
        results = run_jobs(jobs)
        save("cd.json", results)
        print(f"{len(results)} optimisations in {time.time() - t0:.0f} s")
    elif what == "cd_final":
        # the fastest CD gates above 99.9 % and 99 %, at 300 ns and 600 ns, polished and checked on the board waveform
        import hardware_check
        runs = json.load(open(os.path.join(RUNS, "cd.json")))
        jobs = []
        for swap in (0.3, 0.6):
            for beta in (1.0, 2.0):
                for threshold in (0.999, 0.99):
                    ok = [r for r in runs if abs(r["job"]["swap"] - swap) < 1e-9 and r["job"]["strength"] == beta
                          and r["job"]["kind"] == "free" and r["fidelity"] >= threshold]
                    best = min(ok, key=lambda r: (round(r["time_us"], 4), -r["fidelity"]))
                    jobs.append(dict(best["job"], start="given", p0=best["parameters"], maxiter=10000, maxcor=50,
                                     label=f"CD beta = {beta:g}, {int(swap * 1e3)} ns swap, >= {threshold:.1%}",
                                     found_start=best["job"]["start"], found_fidelity=best["fidelity"]))
        polished = run_jobs(jobs, workers=len(jobs))
        for number, r in enumerate(polished):
            job = r["job"]
            x = np.r_[r["qubit_phases"], r["pump_phases"]]
            r["at_128_levels"] = cd.Train("cd", job["strength"], job["swap"], job["cells"], job["gap"], 128, 11).report(x)
            hw = hardware_check.check(r, start_levels=64, step=32)
            r["hardware_check"] = {k: v for k, v in hw.items() if k != "words"}
            r["words_file"] = f"cd_words_{number}.npz"
            np.savez(os.path.join(RUNS, r["words_file"]), qubit=hw["words"]["qubit"], sideband=hw["words"]["sideband"])
            print(job["label"], r["time_us"], r["fidelity"], r["hardware_check"]["fidelity"], flush=True)
        save("cd_final.json", polished)
    elif what == "final":
        # pick the recommended gates from every run, polish them, check them at a high cutoff and on the played waveform
        import hardware_check
        runs = []
        for name in ("sweep.json", "round2.json", "round3.json"):
            for r in json.load(open(os.path.join(RUNS, name))):
                r["file"] = name
                runs.append(r)
        picks = []
        for gate in ("squeeze", "rotation"):
            for swap, kind, threshold in ((0.6, "free", 0.999), (0.6, "free", 0.99), (0.6, "qubit only", 0.99), (0.3, "free", 0.999)):
                ok = [r for r in runs if r["job"]["gate"] == gate and abs(r["job"]["swap"] - swap) < 1e-9
                      and r["job"]["kind"] == kind and r["fidelity"] >= threshold]
                if ok:
                    best = min(ok, key=lambda r: (round(r["time_us"], 4), -r["fidelity"]))
                    picks.append((f"{gate}, {int(swap * 1e3)} ns swap, {kind}, >= {threshold:.1%}", best))
        jobs = []
        for label, r in picks:
            job = dict(r["job"], start="given", p0=r["parameters"], maxiter=20000, maxcor=50, label=label,
                       found_in=r["file"], found_start=r["job"].get("first_start", r["job"]["start"]), found_fidelity=r["fidelity"])
            jobs.append(job)
        polished = run_jobs(jobs, workers=len(jobs))
        for number, r in enumerate(polished):                                    # independent checks of each final gate
            gate, job = r["job"]["gate"], r["job"]
            x = np.r_[r["qubit_phases"], r["pump_phases"]]
            r["at_192_levels"] = cd.Train(gate, STRENGTH[gate], job["swap"], job["cells"], job["gap"], 192).report(x)
            hw = hardware_check.check(r, start_levels=128 if gate == "squeeze" else 64, step=32)
            r["hardware_check"] = {k: v for k, v in hw.items() if k != "words"}
            r["words_file"] = f"words_{number}.npz"                              # the int16 I/Q words of this gate
            np.savez(os.path.join(RUNS, r["words_file"]),
                     qubit=hw["words"]["qubit"], sideband=hw["words"]["sideband"])
            print(r["job"]["label"], r["time_us"], r["fidelity"], r["hardware_check"]["fidelity"], flush=True)
        save("final.json", polished)
    else:
        print(__doc__)
