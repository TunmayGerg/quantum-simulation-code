"""How the optimised resonant trains were found: three searches, each against the exact hop on one link.

    $ACADIA_ENV/bin/python optimise_trains.py free        # -> trains/free_search.json
    $ACADIA_ENV/bin/python optimise_trains.py bounded     # -> trains/bounded_search.json   (needs free_search.json)
    $ACADIA_ENV/bin/python optimise_trains.py symmetric   # -> trains/symmetric_trains.json (used by the notebook)
    $ACADIA_ENV/bin/python optimise_trains.py signs       # -> trains/sign_scan.json: every sign pattern at pi/24
    $ACADIA_ENV/bin/python optimise_trains.py patterns    # -> trains/pattern_trains.json: front-runners, every angle

Every search minimises, with L-BFGS-B (a local gradient search),
    cost = (1 - gate fidelity against exp(-i theta Z X_ab) on the 90 link inputs) + 1e-4 x (total pulse area)
The area term is a small push towards short trains. Nothing in the cost asks for conservation or symmetry.

    free        n = 8 / 12 / 16 pulses B A B A ..., every area and phase free, plus three virtual frame phases.
                Start: textbook / paired pattern, and 9 randomly kicked copies.
    bounded     as free, but every area >= A_MIN = g x 5 ns (a fixed sideband scale cannot play less).
                Start: the best free train at theta = pi/24 (16 allowed; pulses below 1e-3 rad dropped and same-leg
                neighbours merged) with all its pulses, or the 2 or 4 smallest removed; and the free pi/48 train.
    symmetric   n = 8 / 12 / 16 pulses, signs + + - - repeated, mirror-symmetric areas (the shape the bounded search
                found), so only n/2 areas are free, each >= A_MIN. Every hop angle the notebook uses, 6 starts.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import minimize

import resonant_trains as rt

AREA_WEIGHT = 1e-4
THETA_12 = np.pi / 24                     # inner hop angle at 12 lattice steps
WORKERS = 32


def save(name, data):
    with open(os.path.join(rt.TRAIN_DIR, name), "w") as f:
        json.dump(data, f, indent=1)


# ------------------------------------------------------------------------------------------------------------------
# free search: areas, phases and three frame phases, no bound
# ------------------------------------------------------------------------------------------------------------------
def free_train(params, n):
    """params = n areas, n phases, frame phase of the transmon, of cavity a, of cavity b."""
    legs = ["B" if i % 2 == 0 else "A" for i in range(n)]
    train = [(legs[i], params[i], params[n + i]) for i in range(n)]
    return train + [(f, 0.0, ph) for f, ph in zip(("frame q", "frame a", "frame b"), params[2 * n:])]


def free_start(theta, n):
    """textbook (8), textbook + textbook at +pi (16, paired), or a 12-pulse mix; area from hop = 2 area^2 per block."""
    signs = rt.W_SYM_SIGNS + [-s for s in rt.W_SYM_SIGNS] + rt.W_SYM_SIGNS
    area = np.sqrt(theta / 2 / (n / 8))
    phases = [(rt.LEG_B_PHASE if i % 2 == 0 else 0.0) + (np.pi if signs[i] < 0 else 0.0) for i in range(n)]
    return np.concatenate([np.full(n, area), phases, [0.0, 0.0, 0.0]])


def free_job(job):
    theta, n, seed = job
    x0 = free_start(theta, n)
    if seed > 0:                                              # randomly kicked copies of the start
        rng = np.random.default_rng(seed)
        x0[:n] *= 1 + 0.25 * rng.standard_normal(n)
        x0[n:2 * n] += 0.4 * rng.standard_normal(n)
    cost = lambda x: rt.link_infidelity(free_train(x, n), theta) + AREA_WEIGHT * np.sum(np.abs(x[:n]))
    best = minimize(cost, x0, method="L-BFGS-B", options={"maxiter": 400, "maxfun": 40000, "ftol": 1e-13, "gtol": 1e-9})
    train = free_train(best.x, n)
    return dict(theta=theta, n_pulses=n, seed=seed, weight=AREA_WEIGHT, infidelity=rt.link_infidelity(train, theta),
                time_ns=1e3 * rt.train_time(train), params=best.x.tolist())


# ------------------------------------------------------------------------------------------------------------------
# bounded search: every area >= A_MIN
# ------------------------------------------------------------------------------------------------------------------
def pulses_only(train):
    return [p for p in train if rt.is_pulse(p[0])]


def bounded_job(job):
    label, start, theta, seed = job
    legs = [leg for leg, _, _ in start]
    n = len(legs)

    def train_of(x):
        train = [(legs[i], x[i], x[n + i]) for i in range(n)]
        return train + [(f, 0.0, ph) for f, ph in zip(("frame q", "frame a", "frame b"), x[2 * n:])]

    x0 = np.concatenate([[max(a, rt.A_MIN) for _, a, _ in start], [ph for _, _, ph in start], [0.0, 0.0, 0.0]])
    if seed > 0:
        rng = np.random.default_rng(seed)
        x0[:n] = np.maximum(x0[:n] * (1 + 0.15 * rng.standard_normal(n)), rt.A_MIN)
        x0[n:2 * n] += 0.2 * rng.standard_normal(n)
    cost = lambda x: rt.link_infidelity(train_of(x), theta) + AREA_WEIGHT * np.sum(x[:n])
    bounds = [(rt.A_MIN, 1.0)] * n + [(None, None)] * (n + 3)
    best = minimize(cost, x0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 600, "maxfun": 60000, "ftol": 1e-14, "gtol": 1e-10})
    train = train_of(best.x)
    return dict(label=label, theta=theta, seed=seed, infidelity=rt.link_infidelity(train, theta),
                time_ns=1e3 * rt.train_time(train), train=[(l, float(a), float(p)) for l, a, p in train])


# ------------------------------------------------------------------------------------------------------------------
# symmetric search: n/2 areas, everything else fixed
# ------------------------------------------------------------------------------------------------------------------
SYMMETRIC_ANGLES = sorted({np.pi / 2 / s for s in (4, 6, 8, 12, 16, 24, 32, 48)} |      # lattice inner and outer
                          {np.pi / 4 / s for s in (4, 6, 8, 12, 16, 24, 32, 48)})       # angles (CSWAP angles included)


def symmetric_job(job):
    n, theta, seed = job
    s = np.sqrt(theta / 2)
    x0 = np.full(n // 2, 0.75 * s)                        # start: flat, with smaller first two pulses
    x0[0] = x0[1] = 0.4 * s
    if seed > 0:
        x0 = x0 * (1 + 0.3 * np.random.default_rng(seed).standard_normal(n // 2))
    x0 = np.maximum(x0, rt.A_MIN)
    cost = lambda x: rt.link_infidelity(rt.symmetric_from_half(x), theta) + AREA_WEIGHT * 2 * np.sum(x)
    best = minimize(cost, x0, method="L-BFGS-B", bounds=[(rt.A_MIN, 1.5)] * (n // 2),
                    options={"maxiter": 800, "ftol": 1e-15, "gtol": 1e-11})
    train = rt.symmetric_from_half(best.x)
    return dict(n=n, theta=theta, seed=seed, infidelity=rt.link_infidelity(train, theta),
                transmon_flip=rt.link_transmon_flip(train), time_ns=1e3 * rt.train_time(train), half_areas=best.x.tolist())


# sign patterns (Section 2b of the notebook): every pattern is scanned at theta = pi/24 ("signs"), then the ones on the
# time-error front get their areas optimised at every angle ("patterns")
PATTERNS = ["++--++++--++", "++--++----++--++", "+++---+--+---+++"]


def pattern_job(job):
    signs, theta, seed = job
    n = len(signs)
    s = np.sqrt(theta / 2)
    rng = np.random.default_rng(seed)
    x0 = np.full(n // 2, 0.75 * s) if seed == 0 else s * (0.3 + 1.2 * rng.random(n // 2))
    x0 = np.maximum(x0, rt.A_MIN)
    cost = lambda x: rt.link_infidelity(rt.mirror_train(x, signs), theta) + AREA_WEIGHT * 2 * np.sum(x)
    best = minimize(cost, x0, method="L-BFGS-B", bounds=[(rt.A_MIN, 1.5)] * (n // 2),
                    options={"maxiter": 800, "ftol": 1e-15, "gtol": 1e-11})
    train = rt.mirror_train(best.x, signs)
    return dict(signs=signs, theta=theta, seed=seed, infidelity=rt.link_infidelity(train, theta),
                transmon_flip=rt.link_transmon_flip(train), time_ns=1e3 * rt.train_time(train), half_areas=best.x.tolist())


def best_of(rows, key):
    best = {}
    for r in rows:
        k = key(r)
        if k not in best or r["infidelity"] < best[k]["infidelity"]:
            best[k] = r
    return best


def run(function, jobs):
    with ProcessPoolExecutor(WORKERS) as pool:
        results = list(pool.map(function, jobs))
    assert len(results) == len(jobs)
    return results


if __name__ == "__main__":
    search = sys.argv[1] if len(sys.argv) > 1 else ""
    if search == "free":
        thetas = [np.pi / 4 / k for k in (2, 3, 4, 6, 8, 12, 16, 24)]
        rows = run(free_job, [(th, n, seed) for th in thetas for n in (8, 12, 16) for seed in range(10)])
        save("free_search.json", rows)
        print(f"{len(rows)} free optimisations; best of 10 starts:")
        for th in thetas:
            for n in (8, 12, 16):
                b = min((r for r in rows if r["theta"] == th and r["n_pulses"] == n), key=lambda r: r["infidelity"])
                print(f"  theta {th:.4f}  {n:2d} pulses  1-F {b['infidelity']:.2e}  {b['time_ns']:.0f} ns")

    elif search == "bounded":
        inner = pulses_only(rt.free_search_train(THETA_12, 16))           # positive areas, neighbours merged
        outer = pulses_only(rt.free_search_train(THETA_12 / 2, 16))
        smallest = sorted(range(len(inner)), key=lambda i: inner[i][1])  # pulse indices, smallest area first
        starts = {}
        for removed in (0, 2, 4):                                         # keep all pulses, or drop the 2 / 4 smallest
            kept = [p for i, p in enumerate(inner) if i not in smallest[:removed]]
            starts[f"inner {len(kept)}, bounded"] = kept
        starts["outer, bounded"] = outer
        jobs = [(label, start, THETA_12 / 2 if label.startswith("outer") else THETA_12, seed)
                for label, start in starts.items() for seed in range(10)]
        best = {}
        for r in run(bounded_job, jobs):
            if r["label"] not in best or r["infidelity"] < best[r["label"]]["infidelity"]:
                best[r["label"]] = r
        save("bounded_search.json", best)
        for label, r in best.items():
            print(f"{label:24s} 1-F {r['infidelity']:.3e}  {r['time_ns']:.0f} ns")

    elif search == "symmetric":
        rows = run(symmetric_job, [(n, th, seed) for n in (8, 12, 16) for th in SYMMETRIC_ANGLES for seed in range(6)])
        best = {}
        for r in rows:
            key = f"{r['n']} {r['theta']:.10f}"
            if key not in best or r["infidelity"] < best[key]["infidelity"]:
                best[key] = r
        save("symmetric_trains.json", best)
        print(f"{len(rows)} optimisations, best of 6 per (n, angle). theta, then 1-F / transmon flip / ns for n = 8, 12, 16")
        for th in SYMMETRIC_ANGLES:
            print(f"{th:.4f}" + "".join(f"   {best[f'{n} {th:.10f}']['infidelity']:.1e} / "
                                         f"{best[f'{n} {th:.10f}']['transmon_flip']:.1e} / "
                                         f"{best[f'{n} {th:.10f}']['time_ns']:4.0f}" for n in (8, 12, 16)))
    elif search == "signs":
        # every sign pattern for 8 and 12 pulses (up to flipping all signs, which gives the same hop) and every
        # mirror-symmetric or mirror-antisymmetric one for 16 pulses; mirror areas optimised, 3 starts, theta = pi/24
        import itertools
        patterns = []
        for n in (8, 12):
            patterns += ["+" + "".join(b) for b in itertools.product("+-", repeat=n - 1)]
        for half in ("+" + "".join(b) for b in itertools.product("+-", repeat=7)):
            flip = half[::-1].translate(str.maketrans("+-", "-+"))
            patterns += [half + half[::-1], half + flip]
        rows = run(pattern_job, [(p, THETA_12, seed) for p in patterns for seed in range(3)])
        best = best_of(rows, lambda r: r["signs"])
        save("sign_scan.json", list(best.values()))
        print(f"{len(patterns)} sign patterns, {len(rows)} optimisations")
    elif search == "patterns":
        rows = run(pattern_job, [(p, th, seed) for p in PATTERNS for th in SYMMETRIC_ANGLES for seed in range(6)])
        save("pattern_trains.json", best_of(rows, lambda r: f"{r['signs']} {r['theta']:.10f}"))
        print(f"{len(rows)} optimisations for {PATTERNS}")
    else:
        print(__doc__)
