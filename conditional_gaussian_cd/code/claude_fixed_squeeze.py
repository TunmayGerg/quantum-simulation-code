"""Claude task C5: shortest 8 dB conditional squeeze with a FIXED sideband (amplitude and phase) and only the qubit
pi-pulse phases free, at several sideband strengths up to the cap (200 ns swap).

    ../../acadia_env/bin/python claude_fixed_squeeze.py

Uses optimise_trains.optimise with kind = "qubit only" (pump phase fixed at 0 for every cell, one phase per 30 ns pi
pulse), cd_trains model: two-level qubit, no chi, coherent fidelity on qubit g/e x Fock 0..8. 4 random starts each.
Output runs/claude_research/fixed_squeeze/runs.json.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import optimise_trains as ot
import cd_trains as cd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "claude_research", "fixed_squeeze")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for swap in (0.200, 0.250, 0.300, 0.354):
        for total in (2.89, 3.37, 3.85, 4.33, 4.81):
            for cells in (96, 128, 160, 192):
                gap = round(((total - 0.010) / cells - cd.PI_TIME) / 0.0025) * 0.0025
                if gap < -1e-12 or gap > 0.060:
                    continue
                for seed in range(4):
                    jobs.append(dict(gate="squeeze", swap=swap, cells=cells, gap=max(gap, 0.0), kind="qubit only",
                                     start=f"random {seed}", seed=seed, levels=96, maxiter=6000, maxcor=50))
    jobs.sort(key=lambda j: -j["cells"])
    print(len(jobs), "jobs", flush=True)
    t0 = time.time()
    res = ot.run_jobs(jobs, workers=24)
    json.dump(res, open(os.path.join(OUT, "runs.json"), "w"))
    print(f"{len(res)} optimisations in {time.time() - t0:.0f} s")
