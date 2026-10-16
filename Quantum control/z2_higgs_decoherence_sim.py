"""Z2-Higgs on 4 cavities + 3 transmons: the sideband circuit against the ideal model.

WHAT THIS ANSWERS

The notebook (z2_higgs_4cavity_schemes.ipynb) shows the bunching crossover survives
Trotterization and decoherence at ITS assumed coherence times. This file asks the same
question at THIS DEVICE's measured ones, and plots the answer in the same panels the runtime
draws, so a simulated curve and a measured one can be put side by side without re-reading
either axis.

    cavity T1   C1 900 us   C2 130 us   C3 80 us   C4 330 us
    transmon T1 200 us each
    T2 (Ramsey) 150 us for every mode and every transmon

Four curves are produced for each observable:

    ideal          exp(-i H t), no Trotterization, no loss -- the answer the experiment is for
    trotter_ideal  the TARGET's own Trotter step, exp(-i H_loc dt) exp(-i H_L13 dt)
                   exp(-i H_L2 dt), with perfect controlled hops -- what splitting H costs
    trotter        the W_sym sideband circuit, lossless -- what the HARDWARE's Trotter step
                   costs: the splitting above, plus synthesising each hop from sidebands
    lossy          the same circuit with the collapse operators above -- what the device sees

ideal -> trotter_ideal is the step-size error, trotter_ideal -> trotter is the cost of building
the hop out of sideband pulses, and trotter -> lossy is the coherence budget. Keeping them apart
is the point: only the first two are something a different pulse schedule can fix.
"""
import numpy as np
import qutip as qt
from itertools import product
from contextlib import contextmanager


@contextmanager
def one_blas_thread():
    """Run the enclosed linear algebra on ONE BLAS thread.

    Not a micro-optimisation -- on this machine it is the difference between a notebook and an
    overnight job. Every matrix here is a photon-sector BLOCK, at most a few hundred square,
    and OpenBLAS's threads spend longer coordinating over one of those than doing it:

        eigh(175)   0.040 s on 1 thread     4.218 s on 8
        eigh(455)   0.347 s on 1 thread    18.944 s on 8

    a hundredfold the wrong way. The parallelism that pays here is over TRAJECTORIES, which
    run_lossy already takes, and that wants one thread per worker anyway.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:                     # not installed: just run, correctness is unchanged
        yield
        return
    with threadpool_limits(limits=1, user_api="blas"):
        yield

# ---------------------------------------------------------------- device numbers
T1_CAV = {0: 900e-6, 1: 130e-6, 2: 80e-6, 3: 330e-6}
T1_QUBIT = {0: 200e-6, 1: 200e-6, 2: 200e-6}
T2_ALL = 150e-6                      # Ramsey time, cavities and transmons alike

L = 4                                # cavities
NQ = 3                               # transmons, one per link


def pure_dephasing_rate(t1, t2, cavity):
    """The PURE dephasing rate: the total minus the part relaxation already causes.

    The two operator conventions are normalised differently, so the two formulas differ:

        transmon, L = sqrt(g/2) sigma_z   -> the g-e coherence decays at g, so 1/T2 = 1/(2T1) + g
        cavity,   L = sqrt(g) a^dag a     -> the 0-1 coherence decays at g/2 (the rate goes as
                                             (delta n)^2 and delta n = 1), so 1/T2 = 1/(2T1) + g/2

    Hence the factor 2 on the cavity line. Clamped at 0: a T2 longer than 2*T1 is unphysical
    and would ask for a negative rate.
    """
    if cavity:
        return max(2.0 / t2 - 1.0 / t1, 0.0)
    return max(1.0 / t2 - 1.0 / (2.0 * t1), 0.0)


class Lattice:
    """Truncated space and every operator on it.

    A TOTAL-photon-truncated boson basis, not a per-cavity cutoff: the photon budget is shared
    and the bunched states the crossover is about pile many photons onto one site, so a
    per-cavity cutoff would clip exactly the configurations under study.
    """

    def __init__(self, n_cut=4):
        self.n_cut = n_cut
        self.bs = [t for t in product(range(n_cut + 1), repeat=L) if sum(t) <= n_cut]
        self.bidx = {t: i for i, t in enumerate(self.bs)}
        self.nb = len(self.bs)
        self.dq = 2 ** NQ
        self.D = self.dq * self.nb
        dims = [[self.dq, self.nb], [self.dq, self.nb]]

        def boson_a(i):
            m = np.zeros((self.nb, self.nb), complex)
            for b, t in enumerate(self.bs):
                if t[i] > 0:
                    u = list(t); u[i] -= 1
                    m[self.bidx[tuple(u)], b] = np.sqrt(t[i])
            return qt.Qobj(m)

        Iq = qt.qeye(self.dq)
        Ib = qt.qeye(self.nb)
        self.a = [qt.Qobj(np.kron(Iq.full(), boson_a(i).full()), dims=dims) for i in range(L)]
        self.n = [A.dag() * A for A in self.a]

        def qubit_op(op2, l):
            out = np.array([[1.0 + 0j]])
            for k in range(NQ):
                out = np.kron(out, op2 if k == l else np.eye(2))
            return qt.Qobj(np.kron(out, Ib.full()), dims=dims)

        sx = np.array([[0, 1], [1, 0]], complex)
        sz = np.array([[1, 0], [0, -1]], complex)
        sm = np.array([[0, 1], [0, 0]], complex)          # |g><e| with |0> = g
        self.X = [qubit_op(sx, l) for l in range(NQ)]
        self.Z = [qubit_op(sz, l) for l in range(NQ)]
        self.Sm = [qubit_op(sm, l) for l in range(NQ)]
        # the controlled hop on each link
        self.BS = [(self.a[l].dag() * self.a[l + 1] + self.a[l + 1].dag() * self.a[l])
                   for l in range(NQ)]
        self.ZBS = [self.Z[l] * self.BS[l] for l in range(NQ)]

    def H(self, g, J, U):
        """The rotated-frame Hamiltonian the hardware speaks.

        g and J may each be one number or one per link (transmon): the device's links are not
        identical, and a run whose legs have different pulse areas has a different J per link.
        """
        g, J = per_link(g), per_link(J)
        h = 0 * self.n[0]
        for l in range(NQ):
            h = h - g[l] * self.X[l] - J[l] * self.ZBS[l]
        for i in range(L):
            h = h + 0.5 * U * self.n[i] * self.n[i]
        return h

    def gauss_state(self, occ):
        """|occ> with the transmon state Gauss' law forces: X_l = (-1)^(n_0 + ... + n_l)."""
        amp = np.array([1.0 + 0j])
        for l in range(NQ):
            s = (-1) ** int(np.sum(occ[: l + 1]))
            amp = np.kron(amp, np.array([1.0, s], complex) / np.sqrt(2))
        v = np.zeros(self.D, complex)
        b = self.bidx[tuple(occ)]
        for q in range(self.dq):
            v[q * self.nb + b] = amp[q]
        out = qt.Qobj(v.reshape(-1, 1), dims=[[self.dq, self.nb], [1, 1]])
        return out.unit()

    def collapse_ops(self):
        """Photon loss and dephasing on every cavity, relaxation and dephasing on every
        transmon, at the device's measured times."""
        c = []
        for i in range(L):
            c.append(np.sqrt(1.0 / T1_CAV[i]) * self.a[i])
            gphi = pure_dephasing_rate(T1_CAV[i], T2_ALL, cavity=True)
            if gphi > 0:
                c.append(np.sqrt(gphi) * self.n[i])
        for l in range(NQ):
            c.append(np.sqrt(1.0 / T1_QUBIT[l]) * self.Sm[l])
            gphi = pure_dephasing_rate(T1_QUBIT[l], T2_ALL, cavity=False)
            if gphi > 0:
                c.append(np.sqrt(gphi / 2.0) * self.Z[l])
        return c


def per_link(x):
    """A number or a length-NQ sequence -> a length-NQ list (one value per link)."""
    if np.ndim(x) == 0:
        return [float(x)] * NQ
    if len(x) != NQ:
        raise ValueError(f"expected one value or {NQ}, got {len(x)}")
    return [float(v) for v in x]


# ---------------------------------------------------------------- the circuit
#: W_sym in time order: (leg kind, sign). The sign is a pi pump phase and nothing else.
WSYM = (("B", -1), ("A", -1), ("B", +1), ("A", +1),
        ("B", +1), ("A", +1), ("B", -1), ("A", -1))


def sideband_H(lat, link, cavity, phase, g_side):
    """A resonant g1e0 drive: g (e^{i phi} a_cav sigma_+^{(link)} + h.c.).

    The commutator identity that builds the controlled beamsplitter needs only
    [sigma_-, sigma_+] = P_lower - P_upper, so it holds for any two-level ladder, and {g,e} is
    the one this device drives.
    """
    op = np.exp(1j * phase) * lat.a[cavity] * lat.Sm[link].dag()
    return g_side * (op + op.dag())


def step_operators(lat, g, J, U, dt, g_side, eps=None, loc_time=150e-9, pulse_time=None,
                   relative_pump_phase=np.pi / 2):
    """One Trotter step as [(H, duration), ...] in time order: LOC, then L13, then L2.

    LOC is the transmon X rotation with the diagonal U term riding along -- they commute, so
    one exponential does both. L13 drives links 0 and 2 SIMULTANEOUSLY, which is the sum of
    their pulse Hamiltonians because they touch disjoint cavities and disjoint transmons.

    Two ways to give the sideband pulses:
      eps=None or a number   every leg has area eps (default sqrt(J dt / 2)) at coupling g_side,
                             so a pulse lasts eps / g_side
      eps={"A0": ..., ...}   one area per leg, every pulse lasting `pulse_time` -- what a real
                             run plays when the short pulses share a length but not an amplitude
    """
    g = per_link(g)
    if isinstance(eps, dict):
        if pulse_time is None:
            raise ValueError("per-leg eps needs pulse_time, the length every short pulse plays")
        leg_eps, tau = dict(eps), pulse_time
    else:
        if eps is None:
            eps = np.sqrt(min(per_link(J)) * dt / 2.0)   # W_sym gives theta = 2 eps^2
        tau = abs(eps) / g_side                           # one sideband pulse
        leg_eps = {f"{k}{l}": eps for k in "AB" for l in range(NQ)}
    Hloc = (-sum(g[q] * lat.X[q] for q in range(NQ))
            + 0.5 * U * sum(lat.n[i] * lat.n[i] for i in range(L)))
    # The LOC layer wants the rotation exp(-i Hloc dt), but the hardware does it with a pulse
    # that lasts `loc_time` -- a fixed ~150 ns on this device -- not dt. Decoherence acts for
    # the time the pulse really takes, so the segment is the SAME unitary stretched onto its
    # physical duration: a proportionally stronger drive for a proportionally shorter time.
    # Leaving it as (Hloc, dt) would charge the transmons dt of decoherence per step, which for
    # a slow Trotter step is wrong by orders of magnitude and always in the pessimistic
    # direction.
    out = [(Hloc * (dt / loc_time), loc_time)]
    # Leg A's pump phase relative to leg B's: which beamsplitter quadrature the commutator gives.
    # pi/2 is this module's historical default; the runtime's relative_pump_phase_rad is 0.
    base = {"A": float(relative_pump_phase), "B": 0.0}
    for layer, links in (("L13", (0, 2)), ("L2", (1,))):
        for kind, sign in WSYM:
            H = 0 * Hloc
            for link in links:
                cav = link if kind == "A" else link + 1
                H = H + sideband_H(lat, link, cav, base[kind] + (np.pi if sign < 0 else 0.0),
                                   leg_eps[f"{kind}{link}"] / tau)
            out.append((H, tau))
    return out, eps


def ideal_trotter_operators(lat, g, J, U, dt):
    """The TARGET's Trotter step as [(H, dt), ...]: LOC, then L13, then L2, each for dt.

    The same splitting and the same order as the hardware step, but every hop is the exact
    controlled beamsplitter exp(+i J dt Z_l BS_l) rather than one synthesised from eight
    sideband pulses. So against `ideal` it isolates the Trotter (splitting) error, and against
    the sideband circuit it isolates the cost of building the hop out of pulses. No physical
    durations here: this is the mathematical step, so nothing should decohere on it.
    """
    g, J = per_link(g), per_link(J)
    Hloc = (-sum(g[q] * lat.X[q] for q in range(NQ))
            + 0.5 * U * sum(lat.n[i] * lat.n[i] for i in range(L)))
    H13 = -(J[0] * lat.ZBS[0] + J[2] * lat.ZBS[2])   # links 0 and 2 commute: disjoint sites
    H2 = -J[1] * lat.ZBS[1]
    return [(Hloc, dt), (H13, dt), (H2, dt)]


# ---------------------------------------------------------------- propagators
# Every propagator here is BLOCK DIAGONAL in the total photon number, and building it that way
# is what makes the interesting initial states affordable at all.
#
# H conserves the total: the electric term is on the transmons, the hopping moves a photon from
# one cavity to its neighbour, and the U term is diagonal. The effective non-hermitian
# Hamiltonian adds -i/2 sum_k c_k^dag c_k, and every one of those is diagonal in the total too
# -- a_i gives n_i, the cavity dephasing n_i gives n_i^2, sigma_- gives |e><e| and sigma_z
# gives the identity. So no_jump evolution NEVER changes the total photon number; only a jump
# does, and only a cavity loss jump, which lowers it by exactly one.
#
# The saving is large. For |2,2,2,2> the space is 3960 states, and one dense 3960x3960
# propagator is 250 MB with an exponential that costs minutes; the biggest single sector is
# 1320, which is 28 MB and a couple of seconds. Seventeen segments make that the difference
# between running and not.


def gradings(lat):
    """The two conserved quantities available here, as a label per full-space index.

    There is no single one, and using the wrong one silently DELETES physics rather than
    failing, so both are built and each propagator picks the one it actually obeys:

      "cavity"      total photons in the four cavities. The ideal H conserves it -- the
                    electric term only flips transmons and the hopping moves a photon between
                    neighbouring cavities -- but a SIDEBAND does not, because its whole job is
                    to move a photon between a cavity and a transmon.
      "excitation"  cavity photons PLUS excited transmons. Every sideband conserves it, and so
                    does every c^dag c in the effective Hamiltonian (n_i, n_i^2, |e><e|, 1) --
                    but the electric term -g X_l does not, since X flips a transmon.

    Grading the sideband segments by "cavity" was a real bug here: the blocks looked fine, the
    propagators were unitary on each block, and the circuit quietly did nothing like the right
    thing -- a single Trotter step matched its target to 0.9998 while six steps came out at
    0.0003. Hence `BlockProp` VERIFIES the grading instead of trusting it.
    """
    totals = np.array([sum(t) for t in lat.bs])
    cav = np.tile(totals, lat.dq)             # index is q * nb + b, so the pattern repeats
    exc = np.repeat([bin(q).count("1") for q in range(lat.dq)], lat.nb)
    return {"cavity": cav, "excitation": cav + exc}


def sector_index(labels):
    """{label value: array of indices} for one grading."""
    return {int(v): np.flatnonzero(labels == v) for v in np.unique(labels)}


class BlockProp:
    """exp(M), exponentiated block by block in whichever grading M actually respects.

    The grading is CHECKED, not assumed: a matrix with any weight outside its blocks would
    have that weight thrown away, and the result would still be a perfectly plausible-looking
    unitary. So every candidate grading is tested for exact block-diagonality first, and if
    none holds the full matrix is exponentiated -- slower, never wrong.
    """

    def __init__(self, M, lat, grades=None, tol=1e-12):
        from scipy.linalg import expm
        absM = np.abs(M)
        scale = max(float(absM.max()), 1e-300)
        # Test only where M is actually non-zero. Materialising the off-block mask instead --
        # np.where(labels[:,None] != labels[None,:], M, 0) -- is a full D-by-D temporary per
        # grading per segment, which at these dimensions is hundreds of megabytes of pure
        # overhead for a question the sparsity pattern already answers.
        rows, cols = np.nonzero(absM > tol * scale)
        del absM
        # exp(M) for an ANTI-hermitian M -- which is every lossless propagator, M = -i H t --
        # comes from one eigh instead of a Pade exponential with its squaring, and eigh is
        # roughly ten times faster on these blocks. The lossy propagators are not anti-hermitian
        # (the -i/2 c^dag c term is what makes them decay) so those still take expm.
        anti = np.abs(M + M.conj().T).max() <= tol * scale

        def block_exp(sub):
            if anti:
                w, V = np.linalg.eigh(1j * sub)          # 1j * anti-hermitian is hermitian
                return (V * np.exp(-1j * w)) @ V.conj().T
            return expm(sub)

        self.grading = None
        with one_blas_thread():
            for name, labels in (grades or gradings(lat)).items():
                if not np.any(labels[rows] != labels[cols]):
                    self.grading = name
                    self.blocks = [(idx, block_exp(M[np.ix_(idx, idx)]))
                                   for idx in sector_index(labels).values()]
                    break
            if self.grading is None:                   # no grading holds: do it the slow way
                self.grading = "full"
                self.blocks = [(np.arange(M.shape[0]), block_exp(M))]

    def __call__(self, v):
        out = np.zeros_like(v)
        for idx, B in self.blocks:
            out[idx] = B @ v[idx]
        return out


def build_propagators(lat, segs, c_ops=None):
    """One propagator per segment, precomputed once.

    Precomputed because every Trotter step replays the SAME segments: the Hamiltonians and the
    durations do not depend on the step index, so an exponential per segment serves the whole
    run. Doing it inside the trajectory loop instead costs an exponential per segment per step
    per trajectory, which is the difference between seconds and hours.

    With `c_ops` the propagator is the no-jump one, exp(-i (H - i/2 sum c^dag c) t), and its
    norm loss is what tells a trajectory that a jump happened.
    """
    damp = 0
    if c_ops:
        damp = sum((c.dag() * c for c in c_ops), 0 * lat.n[0])
    props = []
    for H, t in segs:
        M = (-1j * (H.full() + (-0.5j * damp.full() if c_ops else 0.0)) * t)
        props.append(BlockProp(M, lat))
    return props


# ---------------------------------------------------------------- the three runs
def run_ideal(lat, psi0, H, times):
    """exp(-i H t) at each t: the answer the experiment is for, with no circuit and no loss.

    H is hermitian AND block diagonal in the cavity photon number, so it is diagonalised once
    per block and every time then costs two small matrix-vector products. Exponentiating it
    afresh at each time -- which is what BlockProp would do -- cost 5.6 s per point at
    D = 1680, i.e. a minute of the run spent re-deriving the same eigenvectors.
    """
    Hm = H.full()
    labels = gradings(lat)["cavity"]
    rows, cols = np.nonzero(np.abs(Hm) > 1e-12 * max(np.abs(Hm).max(), 1e-300))
    if np.any(labels[rows] != labels[cols]):            # not graded: fall back, still correct
        return [BlockProp(-1j * Hm * t, lat)(psi0.full().ravel()) for t in times]

    v0 = psi0.full().ravel()
    with one_blas_thread():
        blocks = []
        for idx in sector_index(labels).values():
            w, V = np.linalg.eigh(Hm[np.ix_(idx, idx)])
            blocks.append((idx, w, V, V.conj().T @ v0[idx]))

        out = []
        for t in times:
            v = np.zeros_like(v0)
            for idx, w, V, u in blocks:
                v[idx] = V @ (np.exp(-1j * w * t) * u)
            out.append(v)
    return out


def run_trotter(lat, psi0, segs, steps_list, props=None):
    """A Trotter step applied repeatedly from psi0, snapshotted at each count in steps_list.

    `segs` is one step as [(H, t), ...] -- the sideband circuit (step_operators) or the ideal
    splitting (ideal_trotter_operators). Pass `props` from build_propagators(lat, segs) to reuse
    them: they depend only on the step, not on the initial state, and building the sideband
    circuit's takes minutes while applying them takes seconds.
    """
    if props is None:
        props = build_propagators(lat, segs)
    cur, done, out = psi0.full().ravel(), 0, []
    with one_blas_thread():
        for n in steps_list:
            while done < n:
                for P in props:
                    cur = P(cur)
                done += 1
            out.append(cur.copy())
    return out


#: Set by run_lossy in the PARENT and inherited by forked workers. Trajectories are
#: independent, so they parallelise perfectly -- but the propagators are hundreds of megabytes
#: and pickling them to each worker costs far more than the work itself. Forking shares them
#: copy-on-write instead, which is why this is a module global rather than an argument.
_TRAJ = {}


def _one_trajectory(seed):
    """One quantum-jump trajectory. Returns the photon distribution at each snapshot, and each
    transmon's P(e) there (what the runtime's measure_before_counting records).

    POPULATIONS, not state vectors: a worker's result has to travel back through a pipe, and
    the distribution is the only part the observables need -- a few hundred floats per step
    instead of a full complex state.
    """
    props, C, v0 = _TRAJ["props"], _TRAJ["C"], _TRAJ["v0"]
    steps_list, lat = _TRAJ["steps_list"], _TRAJ["lat"]
    rng = np.random.default_rng(seed)
    cur, done, out = v0.copy(), 0, []
    target = rng.random()
    pe = []
    for n in steps_list:
        while done < n:
            for P in props:
                cur = P(cur)
                nrm2 = float(np.vdot(cur, cur).real)
                if nrm2 <= target:
                    # Which channel jumped, weighted by its rate on THIS state.
                    w = np.array([float(np.vdot(c @ cur, c @ cur).real) for c in C])
                    if w.sum() > 0:
                        c = C[rng.choice(len(C), p=w / w.sum())]
                        cur = c @ cur
                        cur = cur / np.linalg.norm(cur)
                    target = rng.random()
            done += 1
        out.append(populations(lat, cur / np.linalg.norm(cur)))
        pe.append(transmon_excited(lat, cur / np.linalg.norm(cur)))
    return out, pe


def run_lossy(lat, psi0, segs, steps_list, c_ops, ntraj=400, seed=0, substeps=1, workers=None,
              props=None, return_transmons=False):
    """The circuit with the device's collapse operators, by quantum trajectories.

    Returns the AVERAGE photon distribution at each snapshot -- one array per step, already
    averaged over trajectories.

    Trajectories rather than a master equation because the Liouvillian is a D^2 x D^2 object:
    at these cutoffs that is millions of elements square, which cannot be formed, while a
    trajectory is a single state vector.

    The unravelling is the standard one and its bookkeeping is the part that is easy to get
    wrong: between jumps the state is propagated by the NON-UNITARY no-jump propagator and its
    norm is allowed to decay monotonically, because that decaying norm IS the probability that
    no jump has happened yet. A jump is due when the norm squared falls below a uniform random
    number drawn once per jump. Renormalising the state between jumps -- the obvious-looking
    tidy-up -- throws that history away and makes every jump look as unlikely as the last
    segment alone, which silently under-counts loss. Checked against the lossless circuit with
    an empty c_ops: the two agree to 2e-16.

    `props` reuses no-jump propagators already built with
    build_propagators(lat, segs, c_ops=c_ops) -- they do not depend on the initial state, so a
    study of several preparations builds them once. (They must match `substeps`.)

    `substeps` splits each segment so a jump lands at a finer time than one whole pulse. It
    defaults to 1 because it buys nothing here and costs a lot: a segment is a ~30 ns sideband
    pulse and the worst cavity T1 is 80 us, so the chance of a jump inside one segment is about
    4e-4, and resolving WHERE in that 30 ns it happened changes no observable. Each substep is
    another matrix exponential per segment per block, so substeps=4 quadrupled the most
    expensive part of the run for that non-difference.
    """
    import os, sys, time, multiprocessing as mp
    t0 = time.time()
    if props is None:
        fine = [(H, t / substeps) for H, t in segs for _ in range(substeps)]
        props = build_propagators(lat, fine, c_ops=c_ops)
    _TRAJ.update(props=props, C=[c.full() for c in c_ops], lat=lat,
                 v0=psi0.full().ravel(), steps_list=list(steps_list))
    print(f"      {len(props)} propagators in {time.time()-t0:.0f}s "
          f"({props[-1].grading} grading, {len(props[-1].blocks)} blocks); "
          f"{ntraj} trajectories", file=sys.stderr, flush=True)

    seeds = [seed * 1_000_003 + i for i in range(ntraj)]
    if workers is None:
        workers = max(1, min(os.cpu_count() or 1, ntraj))
    t0 = time.time()
    if workers > 1:
        # One BLAS thread per worker: these are small block matrix-vector products, so the
        # threads fight over them and lose. The parallelism that pays is over trajectories.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        with mp.get_context("fork").Pool(workers) as pool:
            got = pool.map(_one_trajectory, seeds, chunksize=1)
    else:
        got = [_one_trajectory(sd) for sd in seeds]
    print(f"      {ntraj} trajectories on {workers} workers in {time.time()-t0:.0f}s",
          file=sys.stderr, flush=True)
    pops = [np.mean([g[0][k] for g in got], axis=0) for k in range(len(steps_list))]
    if not return_transmons:
        return pops
    # (steps, transmons): each transmon's P(e), averaged over trajectories
    return pops, np.mean([g[1] for g in got], axis=0)


# ---------------------------------------------------------------- observables
# Deliberately the SAME quantities, under the same names, that the runtime's
# process_current_data computes, so a simulated panel and a measured one can be read against
# each other without translating an axis. The difference is only in where the distribution
# comes from: the runtime histograms shots, this takes |<n|psi>|^2 exactly, so there is no
# sampling noise and no counting infidelity in these curves.


def populations(lat, vec):
    """P(n_0, n_1, n_2, n_3) from a state vector: the transmons traced out."""
    v = np.asarray(vec).ravel()
    p = (np.abs(v) ** 2).reshape(lat.dq, lat.nb).sum(axis=0)
    return p / p.sum()


def transmon_excited(lat, vec):
    """Each transmon's P(e) from a state vector. Basis |q> with transmon 0 the most significant
    bit and |1> = e (Sm = |0><1| lowers 1 -> 0)."""
    v = np.asarray(vec).ravel()
    pq = (np.abs(v) ** 2).reshape(lat.dq, lat.nb).sum(axis=1)
    pq = pq / pq.sum()
    return np.array([sum(pq[q] for q in range(lat.dq) if (q >> (NQ - 1 - l)) & 1)
                     for l in range(NQ)])


def observables(lat, p, postselect_N):
    """Every panel's number, from one photon-number distribution.

    Returns raw and post-selected versions of each, exactly as the runtime does: the ideal
    dynamics conserves the total, so a shot whose total is not `postselect_N` lost a photon,
    and the fraction of such weight is the erasure the experiment has to pay.
    """
    occ = np.array(lat.bs, float)                       # (nb, 4)
    tot = occ.sum(axis=1)
    keep = tot == postselect_N
    kept_weight = float(p[keep].sum())

    def moments(w):
        if w.sum() <= 0:
            nan4 = np.full(4, np.nan)
            return nan4, nan4, np.nan
        w = w / w.sum()
        mean = (w[:, None] * occ).sum(axis=0)
        var = (w[:, None] * occ ** 2).sum(axis=0) - mean ** 2
        return mean, var, float(np.mean(var))

    mean, var, F = moments(p.copy())
    ps_w = np.where(keep, p, 0.0)
    ps_mean, ps_var, ps_F = moments(ps_w)

    top = int(postselect_N)
    fcs = np.full((4, top + 1), np.nan)
    if kept_weight > 0:
        w = ps_w / ps_w.sum()
        for i in range(4):
            for kk in range(top + 1):
                fcs[i, kk] = float(w[occ[:, i] == kk].sum())

    nn = np.full(3, np.nan)
    ipr = np.nan
    if kept_weight > 0:
        w = ps_w / ps_w.sum()
        for b in range(3):
            nn[b] = (float((w * occ[:, b] * occ[:, b + 1]).sum())
                     - float((w * occ[:, b]).sum()) * float((w * occ[:, b + 1]).sum()))
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = occ / np.maximum(tot[:, None], 1)
            ipr = float((w * (frac ** 2).sum(axis=1)).sum())

    return {"mean_n": mean, "var_n": var, "F_bunch": F,
            "ps_mean_n": ps_mean, "ps_var_n": ps_var, "ps_F_bunch": ps_F,
            "erased_fraction": 1.0 - kept_weight, "fcs": fcs,
            "nn_corr": nn, "ipr_mean": ipr}


def series(lat, snaps, postselect_N, already_populations=False):
    """Run `observables` over a list of snapshots -> {name: array indexed by step}.

    `snaps` is either one state vector per step (ideal, trotter) or one ready-made photon
    distribution per step (lossy, which averaged over its trajectories in the workers).
    """
    rows = [observables(lat, snap if already_populations else populations(lat, snap),
                        postselect_N)
            for snap in snaps]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# ---------------------------------------------------------------- plots
# One figure per runtime page, same panels in the same order, so a simulated page and a
# measured page can be laid side by side. The one difference is the three curves: the runtime
# has a single measured curve where each panel here carries ideal / trotter / lossy.

#: (key in runs, legend label, colour, line style), drawn in this order.
CURVES = (("ideal", "exact unitary", "C0", "-"),
          ("trotter_ideal", "ideal Trotter steps", "C2", ":"),
          ("trotter", "sideband circuit", "C1", "--"),
          ("lossy", "circuit + decoherence", "C3", "-."))


def finish(fig, path):
    """Save the page, or show it when there is no path -- which is what a notebook wants."""
    import matplotlib.pyplot as plt
    fig.tight_layout()
    if path is None:
        plt.show()
    else:
        fig.savefig(path, dpi=140)
        plt.close(fig)


def plot_order_parameter(steps, runs, postselect_N, path=None):
    """Page 1: the bunching order parameter F = mean_i Var(n_i). THE measurement."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, label, col, ls in CURVES:
        if name not in runs:
            continue
        ax[0].plot(steps, runs[name]["F_bunch"], ls, color=col, marker="o", ms=4, label=label)
        ax[1].plot(steps, runs[name]["ps_F_bunch"], ls, color=col, marker="o", ms=4, label=label)
        ax[2].plot(steps, runs[name]["erased_fraction"], ls, color=col, marker="o", ms=4,
                   label=label)
    for k, title in enumerate(("bunching order parameter, every shot",
                               f"bunching order parameter, exactly {postselect_N} photons",
                               "erased by post-selection")):
        ax[k].set_xlabel("Trotter steps")
        ax[k].grid(True, alpha=0.3)
        ax[k].set_title(title, fontsize=9)
        ax[k].legend(fontsize=8)
    ax[0].set_ylabel(r"$\mathcal{F} = \langle \mathrm{Var}(n_i)\rangle_i$")
    ax[1].set_ylabel(r"$\mathcal{F}$")
    ax[2].set_ylabel("fraction discarded")
    finish(fig, path)


def plot_sites(steps, runs, path=None):
    """Page 2: per-cavity occupation and variance. Where the photons actually are."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    for i in range(4):
        for name, label, col, ls in CURVES:
            if name not in runs:
                continue
            ax[0, i].plot(steps, runs[name]["ps_mean_n"][:, i], ls, color=col, marker="o", ms=3,
                          label=label)
            ax[1, i].plot(steps, runs[name]["ps_var_n"][:, i], ls, color=col, marker="o", ms=3)
        ax[0, i].set_title(f"C{i+1}", fontsize=10)
        ax[1, i].set_xlabel("Trotter steps")
        for r in (0, 1):
            ax[r, i].grid(True, alpha=0.3)
    ax[0, 0].set_ylabel(r"$\langle n_i \rangle$")
    ax[1, 0].set_ylabel(r"$\mathrm{Var}(n_i)$")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("site occupations and variances, post-selected", fontsize=11)
    finish(fig, path)


def plot_fcs(steps, runs, postselect_N, path=None):
    """Page 3: full counting statistics -- P(n_i = k) per cavity, against Trotter step.

    This is what a variance summarises: a crossover into bunching is weight leaving a narrow
    distribution for a long-tailed one, and the tail is visible here well before F moves much.
    Even and odd k are drawn differently, as in the runtime, because the parity is what the
    transmon is locked to and confusing the two hides a real structure.
    """
    import matplotlib.pyplot as plt
    which = [n for n, _l, _c, _s in CURVES if n in runs]
    labels = {n: l for n, l, _c, _s in CURVES}
    fig, ax = plt.subplots(len(which), 4, figsize=(15, 3.4 * len(which)), squeeze=False,
                           sharex=True, sharey=True)
    top = int(postselect_N)
    for r, name in enumerate(which):
        for i in range(4):
            for k in range(top + 1):
                even = (k % 2 == 0)
                ax[r, i].plot(steps, runs[name]["fcs"][:, i, k],
                              "-" if even else "--",
                              color=plt.cm.winter(k / max(top, 1)) if even
                              else plt.cm.autumn(k / max(top, 1)),
                              marker="o", ms=2.5, lw=1.2, label=f"n={k}" if i == 0 else None)
            ax[r, i].grid(True, alpha=0.3)
            if r == 0:
                ax[r, i].set_title(f"C{i+1}", fontsize=10)
            if r == len(which) - 1:
                ax[r, i].set_xlabel("Trotter steps")
        ax[r, 0].set_ylabel(f"{labels[name]}\nP(n)", fontsize=9)
    ax[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"full counting statistics, post-selected on {postselect_N} photons",
                 fontsize=11)
    finish(fig, path)


def plot_correlations(steps, runs, path=None):
    """Page 4: nearest-neighbour number correlation and how concentrated the photons are."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.0))
    for b in range(3):
        for name, label, col, ls in CURVES:
            if name not in runs:
                continue
            ax[b].plot(steps, runs[name]["nn_corr"][:, b], ls, color=col, marker="o", ms=3,
                       label=label)
        ax[b].axhline(0.0, color="0.85", lw=0.8, ls=":")
        ax[b].set_title(f"bond C{b+1}-C{b+2}", fontsize=10)
        ax[b].set_xlabel("Trotter steps")
        ax[b].set_ylabel(r"$\langle n_i n_j\rangle - \langle n_i\rangle\langle n_j\rangle$")
        ax[b].grid(True, alpha=0.3)
    for name, label, col, ls in CURVES:
        if name not in runs:
            continue
        ax[3].plot(steps, runs[name]["ipr_mean"], ls, color=col, marker="o", ms=3, label=label)
    ax[3].axhline(0.25, color="0.85", lw=0.8, ls=":")
    ax[3].set_title("participation (1 = all on one site, 1/4 = even)", fontsize=9)
    ax[3].set_xlabel("Trotter steps"); ax[3].set_ylabel("IPR")
    ax[3].grid(True, alpha=0.3); ax[3].legend(fontsize=8)
    ax[0].legend(fontsize=8)
    finish(fig, path)


def plot_prep_comparison(steps, runs_by_prep, postselect_N, path=None):
    """Several preparations side by side: the order parameter and its erasure, every curve.

    One column per preparation, so the same curve can be read across them: a flat state and a
    pre-bunched one should differ in F, and that difference -- not either curve alone -- is the
    measurement.
    """
    import matplotlib.pyplot as plt
    preps = list(runs_by_prep)
    fig, ax = plt.subplots(2, len(preps), figsize=(4.6 * len(preps), 7), sharex=True,
                           sharey="row", squeeze=False)
    for c, prep in enumerate(preps):
        runs = runs_by_prep[prep]
        for name, label, col, ls in CURVES:
            if name not in runs:
                continue
            ax[0, c].plot(steps, runs[name]["ps_F_bunch"], ls, color=col, marker="o", ms=3,
                          label=label)
            ax[1, c].plot(steps, runs[name]["erased_fraction"], ls, color=col, marker="o", ms=3)
        ax[0, c].set_title(f"|{prep}>", fontsize=11)
        ax[1, c].set_xlabel("Trotter steps")
        for r in (0, 1):
            ax[r, c].grid(True, alpha=0.3)
    ax[0, 0].set_ylabel(rf"$\mathcal{{F}}$, exactly {postselect_N} photons")
    ax[1, 0].set_ylabel("erased by post-selection")
    ax[0, 0].legend(fontsize=8)
    finish(fig, path)


# ---------------------------------------------------------------- the study
#: The two initial states the crossover is read from. Both hold the same number of photons, so
#: the same post-selection applies to each and the difference between them is only HOW the
#: photons are arranged: flat, or already piled two-and-two on the ends.
STUDY = {"probe": (1, 1, 1, 1), "control": (2, 0, 0, 2)}


def run_study(occ, steps_list, g, J, U, dt, g_side, loc_time, ntraj, n_cut=None, seed=0,
              do_lossy=True):
    """Ideal, Trotter and lossy for one initial occupation. Returns (lattice, {name: series})."""
    N = int(sum(occ))
    # The cutoff has to sit ABOVE the photon number, not at it. Gauss' law puts every transmon
    # in an X eigenstate -- an equal superposition of g and e -- so the total EXCITATION is not
    # sharp, and a sideband's whole job is to turn a transmon excitation into a cavity photon.
    # Three transmons can therefore hand three extra photons to the cavities, and a basis cut
    # at N clips states the circuit really visits. Measured on |1,1,0,0> (6 steps, hopping
    # only): n_cut = N gave fidelity 0.146 / 0.664 / 0.906 at dt = 100/50/25 ns, N+1 gave
    # 0.942 / 0.996 / 0.9997, and N+2 moved only the fourth decimal (0.956 / 0.997 / 0.9998).
    # N+3 is the first cutoff that cannot clip anything at all, so that is the default.
    lat = Lattice(n_cut=N + 3 if n_cut is None else n_cut)
    psi0 = lat.gauss_state(occ)
    H = lat.H(g, J, U)
    segs, eps = step_operators(lat, g, J, U, dt, g_side, loc_time=loc_time)

    t_step = sum(t for _H, t in segs)
    print(f"  |{''.join(str(x) for x in occ)}>  N={N}  dim={lat.D}  eps={eps:.4f} rad  "
          f"physical step {t_step*1e9:.0f} ns  total {t_step*max(steps_list)*1e6:.1f} us")

    runs = {}
    # The ideal curve is the CONTINUOUS evolution at the same physical times the circuit
    # reaches, which is what makes the ideal/trotter gap a Trotterization error and nothing else.
    runs["ideal"] = series(lat, run_ideal(lat, psi0, H, [n * dt for n in steps_list]), N)
    runs["trotter"] = series(lat, run_trotter(lat, psi0, segs, steps_list), N)
    if do_lossy:
        runs["lossy"] = series(lat, run_lossy(lat, psi0, segs, steps_list, lat.collapse_ops(),
                                              ntraj=ntraj, seed=seed), N,
                               already_populations=True)
    return lat, runs, N


def main():
    import argparse, os
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=12, help="highest Trotter step count")
    ap.add_argument("--g", type=float, default=2 * np.pi * 0.20e6, help="electric term, rad/s")
    ap.add_argument("--J", type=float, default=2 * np.pi * 0.20e6, help="hopping, rad/s")
    ap.add_argument("--U", type=float, default=2 * np.pi * 0.05e6, help="on-site, rad/s")
    ap.add_argument("--dt", type=float, default=0.05e-6, help="Trotter step, s")
    ap.add_argument("--g-side", type=float, default=2 * np.pi * 1.0e6,
                    help="sideband coupling, rad/s -- sets how long a leg lasts")
    ap.add_argument("--loc-time", type=float, default=150e-9, help="LOC pulse length, s")
    ap.add_argument("--ntraj", type=int, default=200)
    ap.add_argument("--state", choices=sorted(STUDY) + ["both"], default="both")
    ap.add_argument("--no-lossy", action="store_true")
    ap.add_argument("--n-cut", type=int, default=None,
                    help="boson cutoff; default N+3, the first value that cannot clip "
                         "(three transmons can hand three photons back to the cavities)")
    ap.add_argument("--outdir", default="z2_decoherence_figs")
    args = ap.parse_args()

    steps_list = list(range(0, args.steps + 1))
    os.makedirs(args.outdir, exist_ok=True)
    names = sorted(STUDY) if args.state == "both" else [args.state]

    print(f"Z2-Higgs decoherence study   T1 cav {[f'{T1_CAV[i]*1e6:.0f}' for i in range(L)]} us,"
          f" transmon {T1_QUBIT[0]*1e6:.0f} us, T2 {T2_ALL*1e6:.0f} us")
    for name in names:
        occ = STUDY[name]
        lat, runs, N = run_study(occ, steps_list, args.g, args.J, args.U, args.dt,
                                 args.g_side, args.loc_time, args.ntraj,
                                 n_cut=args.n_cut, do_lossy=not args.no_lossy)
        tag = f"{name}_{''.join(str(x) for x in occ)}"
        plot_order_parameter(steps_list, runs, N, f"{args.outdir}/1_order_{tag}.png")
        plot_sites(steps_list, runs, f"{args.outdir}/2_sites_{tag}.png")
        plot_fcs(steps_list, runs, N, f"{args.outdir}/3_fcs_{tag}.png")
        plot_correlations(steps_list, runs, f"{args.outdir}/4_corr_{tag}.png")

        print(f"    {'step':>5s}" + "".join(f"{k:>14s}" for k in
                                            ("F ideal", "F trotter", "F lossy", "erased")))
        for i, n in enumerate(steps_list):
            row = [runs["ideal"]["ps_F_bunch"][i], runs["trotter"]["ps_F_bunch"][i],
                   runs["lossy"]["ps_F_bunch"][i] if "lossy" in runs else np.nan,
                   runs["lossy"]["erased_fraction"][i] if "lossy" in runs else np.nan]
            print(f"    {n:5d}" + "".join(f"{v:14.4f}" for v in row))
    print(f"\nfigures written to {args.outdir}/")


if __name__ == "__main__":
    main()
