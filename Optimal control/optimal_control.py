"""
Unified optimal control for unitary gate synthesis.

===============================================================================
WHAT THIS MODULE IS
===============================================================================
It replaces ``Grape.py`` and ``parameter_oc.py`` with one class hierarchy:

    OptimalControl          <- all the shared machinery lives here
      |- GrapeOC            <- piecewise-constant pulse ansatz (was GrapeLBFGS)
      |- ParameterOC        <- parameterized gate-sequence ansatz

The two old files solved the *same* mathematical problem -- "pick parameters so
that a product of unitaries matches a target" -- and differed only in how the
parameters produce the unitaries.  So everything except that one detail lives in
``OptimalControl``: fidelity, projector, gauge freedom, adiabatic frame,
penalties, gradients, the optimizer drivers (local + global), progress
reporting, cancellation, async running, and saving/loading.

A subclass only has to answer two questions:

    1. ``_forward(x_core, need_grad)``
         Given the parameters, what are the individual propagators U_0..U_{M-1}?
    2. ``_slice_dc(cache, j, L, out)``
         Given a co-state matrix L, what is Tr(L dU_j/dp) for each parameter p
         belonging to propagator j?

Everything else -- how those turn into a fidelity and a gradient -- is shared.

===============================================================================
PHYSICS CONVENTIONS  (hbar = 1 unless a value is passed)
===============================================================================
The quantity being optimized is built up in four stages:

    U_final = U_{M-1} ... U_1 U_0     time-ordered product; U_0 acts FIRST
    U_ad    = Ad^dag U_final Ad       optional basis change (e.g. bare <-> dressed)
    U_corr  = G(theta) U_ad           gauge freedom, G = exp(+i sum_j theta_j A_j)
    c       = Tr(Ut^dag U_corr P)     overlap with the target (P = I if unprojected)
    F       = |c|^2 / d_eff^2         fidelity in [0, 1]; d_eff = dim or Tr(P)
    cost    = 1 - F + sum(penalties)  what the optimizer minimizes

Why each stage exists:

  * ``P`` (projector) lets you demand the gate be correct only on a subspace --
    e.g. the lowest 8 cavity levels -- while still simulating higher levels so
    leakage out of that subspace is penalized honestly.
  * ``Ad`` (adiabatic_unitary) handles the case where the control Hamiltonian is
    written in one basis (bare) but the target is defined in another (dressed).
  * ``G(theta)`` (gauge_ops) absorbs physically irrelevant phases.  A gate that
    is right up to a known frame rotation -- e.g. accumulated single-mode phases
    exp(i theta n) -- is a *correct* gate, because that rotation can be undone in
    software.  The theta_j are optimized alongside the controls, so the optimizer
    is not wasting effort chasing a phase you do not care about.

===============================================================================
WHY THE GRADIENT LOOKS THE WAY IT DOES
===============================================================================
The naive gradient forms dU_final/dp for every parameter p, which costs several
d x d matrix products each.  Instead, note that a parameter p living in
propagator j only enters through that one factor:

    dU_final/dp = post_j (dU_j/dp) prefix_j
        prefix_j = U_{j-1} ... U_0        (everything before j)
        post_j   = U_{M-1} ... U_{j+1}    (everything after j)

Push that through the trace and use cyclicity:

    dc/dp = Tr(B dU_final/dp) = Tr(B post_j (dU_j/dp) prefix_j)
          = Tr( (prefix_j B post_j) dU_j/dp )
          = Tr( L_j dU_j/dp )                   with  L_j = prefix_j B post_j

L_j is the "co-state" at slice j and does not depend on which parameter of that
slice we differentiate.  Better still, the L_j satisfy a backward recursion, so
the whole set costs two matrix products per slice rather than a fresh product
per parameter.  See ``OptimalControl.cost_and_grad`` for the implementation.

===============================================================================
SPEED NOTES  (all measured; see README.md and test_optimal_control.py)
===============================================================================
Every one of these is EXACT -- no approximation was introduced anywhere.

  * ``cost`` and ``grad`` are computed together and SciPy is driven with
    ``jac=True``, so the forward propagation happens once per evaluation.
    The old code propagated twice per L-BFGS step.                      -> 2x
  * The backward co-state recursion above, plus every trace written as an
    O(d^2) contraction instead of ``np.trace(A @ B)`` (which builds the whole
    product just to read its diagonal).                                 -> 3-4x on traces
  * GRAPE slice derivatives use the spectral (Daleckii-Krein) formula rather
    than ``expm_frechet``.                                              -> 2.4-4x
  * ParameterOC gates use a closed-form derivative built on an
    eigendecomposition cached once at construction.                     -> ~30x
  * Gauge unitaries built from commuting operators (the usual case: number
    operators) are simultaneously diagonalized once at construction, so G and
    dG/dtheta are O(d^2) instead of one ``expm_frechet`` per parameter.

End to end at notebook sizes: ParameterOC 16.6 ms -> 0.53 ms per cost+gradient,
GRAPE 306 ms -> 62 ms.
"""

from __future__ import annotations

# `from __future__ import annotations` makes all type hints lazy strings, which
# lets us write forward references like -> "OptimalControl" without quoting.

import json          # run metadata is stored as human-readable JSON
import sys           # default stream for the progress display
import threading     # run_async worker thread + the cancellation flag
import time          # wall-clock timing and progress throttling
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import qutip as qt
from scipy.linalg import expm, expm_frechet
from scipy.optimize import (
    OptimizeResult,            # the result container; we also build our own
    basinhopping,              # global: hops between gradient-found local minima
    differential_evolution,    # global: population based, gradient-free
    dual_annealing,            # global: annealing + gradient local search
    minimize,                  # all the local gradient methods
    shgo,                      # global: simplicial homology
)

try:
    # scipy.stats.qmc provides Sobol / Latin-hypercube / Halton sequences, which
    # cover a high-dimensional box far more evenly than uniform random draws.
    # Only used to pick multistart starting points, so a missing qmc is not
    # fatal -- _sample_starts falls back to uniform random.
    from scipy.stats import qmc as _qmc
except Exception:  # pragma: no cover
    _qmc = None


# Public names.  Anything not listed here is an implementation detail and may
# change; the leading-underscore names are private by convention.
__all__ = [
    # --- the classes you actually build ---
    "OptimalControl",              # base class (subclass it for a new ansatz)
    "GrapeOC",                     # piecewise-constant pulses
    "ParameterOC",                 # sequence of named gates
    "FrameGate",                   # the gate primitive ParameterOC is built from
    "RunHandle",                   # returned by run_async()
    # --- introspection ---
    "available_methods",           # list the optimizer names
    # --- reading saved runs ---
    "load_run",                    # new folder format
    "load_qobjs",                  # QuTiP operators (new folder or legacy .npz)
    "load_pulses_theta",           # GRAPE parameters (new folder or legacy .npz)
    "load_rotation_phase_theta",   # gate parameters (new folder or legacy .npz)
    "load_parameters_dict",        # metadata (new folder or legacy .npz)
    # --- stand-alone propagation, for analysis without building a class ---
    "propagate_pulses",
    "propagate_gates",
    "gauge_unitary",
]

# Relative tolerance used when deciding whether an operator counts as Hermitian.
# Deliberately loose: operators assembled from QuTiP products accumulate a few
# ulps of asymmetry, and rejecting those would needlessly disable the fast
# spectral derivative path.
_TOL_HERM = 1e-10


# =============================================================================
# Small helpers
# =============================================================================

class _Cancelled(Exception):
    """Raised inside the objective to unwind out of a running SciPy optimizer.

    SciPy's minimizers offer no clean "stop now" API, but they also do not catch
    exceptions raised by the objective function.  So cancellation is implemented
    by raising this from ``_note_evaluation`` (which runs after every single
    evaluation) and catching it in ``_run_optimize``.  The best point seen so far
    is already stored on the optimizer, so nothing is lost by unwinding.

    It is private and never propagates to the caller: ``_run_optimize`` converts
    it into ``result["cancelled"] = True``.
    """


def _is_hermitian(A: np.ndarray, tol: float = _TOL_HERM) -> bool:
    """True if ``A`` equals its conjugate transpose to within a relative tolerance.

    Used to decide whether the fast spectral (eigendecomposition) derivative path
    is valid -- ``np.linalg.eigh`` assumes a Hermitian matrix and silently reads
    only the lower triangle, so feeding it a non-Hermitian matrix would give a
    wrong answer rather than an error.

    The tolerance is scaled by the magnitude of ``A`` (with a floor of 1) so the
    test means "asymmetric by a relative 1e-10", not "asymmetric by an absolute
    1e-10" -- the latter would spuriously reject operators with large entries,
    such as a Hamiltonian in rad/us with 2*pi*160 anharmonicity.
    """
    return bool(np.max(np.abs(A - A.conj().T)) <= tol * max(1.0, np.max(np.abs(A))))


def _trace_prod(A: np.ndarray, B: np.ndarray) -> complex:
    """Tr(A @ B) without forming the product: O(d^2) instead of O(d^3).

    ``np.trace(A @ B)`` computes all d^2 entries of the product and then throws
    away everything off the diagonal.  Since

        Tr(A B) = sum_{i,j} A_ij B_ji

    only the d^2 pairwise products that land on the diagonal are needed.  This
    matters because the gradient does one of these per parameter, per evaluation
    -- measured 3-4x faster on the traces alone at d = 24..100.

    ``optimize=False`` skips einsum's contraction-path planner, which is pure
    overhead for a two-operand expression this simple.
    """
    return np.einsum("ij,ji->", A, B, optimize=False)


# Which SciPy options each local minimizer actually accepts.  Passing an
# unknown key raises TypeError in some methods (trust-constr) and is silently
# ignored in others, so the same user-supplied `scipy_options` dict is filtered
# per method instead of being forwarded blindly.
_METHOD_OPTION_KEYS: Dict[str, set] = {
    "L-BFGS-B": {"maxiter", "maxfun", "ftol", "gtol", "maxls", "maxcor", "eps", "disp"},
    "TNC": {"maxfun", "maxiter", "ftol", "gtol", "xtol", "eps", "scale", "offset",
            "stepmx", "accuracy", "minfev", "rescale", "disp"},
    "SLSQP": {"maxiter", "ftol", "eps", "disp"},
    "trust-constr": {"maxiter", "gtol", "xtol", "barrier_tol", "initial_tr_radius",
                     "verbose", "disp"},
    "BFGS": {"maxiter", "gtol", "eps", "norm", "xrtol", "disp"},
    "CG": {"maxiter", "gtol", "eps", "norm", "disp"},
}


def _filter_options(method: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the option keys ``method`` understands.

    The user passes one ``scipy_options`` dict, but it may be forwarded to
    several different minimizers over the course of a run (a global method's
    inner local searches, then the final polish).  Rather than make the caller
    track which keys are legal where, filter here.

    Unknown method names pass through untouched, so a future SciPy method still
    receives whatever the user asked for.
    """
    allowed = _METHOD_OPTION_KEYS.get(method)
    if allowed is None:
        return dict(options)                    # unknown method: don't second-guess

    out = {k: v for k, v in options.items() if k in allowed}

    # TNC is the odd one out: it budgets *function evaluations* (maxfun) rather
    # than iterations, and emits an OptimizeWarning if handed 'maxiter'.
    # Translate so that one maxiter= from the caller means roughly the same
    # thing across every method.
    if method == "TNC" and "maxiter" in options:
        out.pop("maxiter", None)
        out.setdefault("maxfun", int(options["maxiter"]))
    return out


def _divided_difference(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build the Daleckii-Krein divided-difference matrix for exp().

    For a diagonalizable A = V diag(a) V^dag, the Frechet derivative of the
    matrix exponential in the direction E is

        d/dt exp(A + tE)|_0 = V [ (V^dag E V) o Phi ] V^dag        ('o' = elementwise)

    where Phi is the matrix of divided differences of exp over the spectrum:

        Phi_jk = (e^{a_j} - e^{a_k}) / (a_j - a_k)      for a_j != a_k
        Phi_jj = e^{a_j}                                (the limit j -> k)

    This is what makes the fast GRAPE derivative possible: one eigendecomposition
    yields the propagator *and* its exact derivative in any direction.

    Parameters
    ----------
    a:
        The eigenvalues of the exponent A (complex; for us a = -i*dt*w with w the
        real eigenvalues of a Hermitian Hamiltonian, so these sit on the
        imaginary axis).

    Returns
    -------
    (ea, phi):
        ``ea = exp(a)`` (reused by the caller to build the propagator) and the
        d x d divided-difference matrix.
    """
    ea = np.exp(a)

    # Pairwise differences a_j - a_k as a d x d matrix via broadcasting.
    da = a[:, None] - a[None, :]

    # The diagonal is exactly zero and degenerate pairs are near zero, so this
    # division deliberately produces inf/nan; they are all overwritten below.
    # errstate silences the warnings rather than letting them spam the console
    # once per time slice per iteration.
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (ea[:, None] - ea[None, :]) / da

    # Repair the diagonal and any (near-)degenerate eigenvalue pair.  Degeneracy
    # is common and not exotic -- e.g. an uncoupled Hamiltonian at zero drive --
    # so this branch is a correctness requirement, not an edge case.
    close = np.abs(da) < 1e-10
    if close.any():
        # lim_{y->x} (e^x - e^y)/(x - y) = e^x.  Evaluating at the midpoint,
        # exp((a_j + a_k)/2), is the symmetric choice and is slightly more
        # accurate than exp(a_j) for pairs that are near-degenerate but not
        # exactly equal.
        mid = np.exp(0.5 * (a[:, None] + a[None, :]))
        phi = np.where(close, mid, phi)
    return ea, phi


# =============================================================================
# Progress reporting
# =============================================================================

class _Progress:
    """Throttled progress display, cheap enough to call on every evaluation.

    The design constraint is that ``update()`` runs after *every* objective
    evaluation -- tens of thousands of times in a short run -- so it must cost
    essentially nothing.  It does two things: keep a running maximum (a float
    compare), and decide whether enough wall time has passed to print (one
    ``perf_counter()`` call).  Measured at **59 ns** per call, against an
    objective that costs 0.4-60 ms.  The actual printing is throttled by
    ``every`` seconds, so its cost does not scale with iteration count.

    This is why the fidelity can be shown live for free, whereas the old code
    recomputed the fidelity from scratch inside the callback -- a whole extra
    propagation per iteration.

    Modes
    -----
    ``"line"``:  one self-updating line via carriage return.  Best interactively.
    ``"log"``:   one line per update.  Use from a background thread, where "\\r"
                 does not render properly in most notebook front-ends.
    ``"none"``:  silent, but still tracks the best fidelity and the eval count.
    """

    def __init__(self, mode: str = "line", every: float = 0.25, stream=None, label: str = ""):
        if mode not in ("line", "log", "none"):
            raise ValueError("progress must be 'line', 'log' or 'none'.")
        self.mode = mode
        self.every = float(every)              # minimum seconds between prints
        # Resolved at construction rather than at print time so that redirecting
        # sys.stdout mid-run cannot change where an in-flight run writes.
        self.stream = stream if stream is not None else sys.stdout
        self.label = label                     # optional prefix, e.g. "start 3: "
        self.best = -np.inf                    # best fidelity seen this run
        self.n_eval = 0                        # objective evaluations this run
        self.start_ts = time.perf_counter()    # for the elapsed-time column
        self._last_ts = 0.0                    # when we last printed
        self._dirty = False                    # True once a "\r" line is pending
        self.extra = ""                        # optional suffix, appended to each line

    def update(self, fidelity: float, cost: float) -> None:
        """Record one evaluation and print if the throttle interval has elapsed.

        Everything before the ``mode == "none"`` check is bookkeeping that must
        happen regardless of whether anything is displayed, because
        ``result["n_evaluations"]`` and the progress line both read it.
        """
        self.n_eval += 1
        if fidelity > self.best:
            self.best = fidelity
        if self.mode == "none":
            return                                   # silent: skip the clock read
        now = time.perf_counter()
        if now - self._last_ts < self.every:
            return                                   # too soon; stay quiet
        self._last_ts = now
        self._emit(now, fidelity, cost)

    def _emit(self, now: float, fidelity: float, cost: float) -> None:
        """Write one progress line.  Called only from the throttled path."""
        el = now - self.start_ts
        # Both the current and the best fidelity are shown: the current value
        # tells you whether the optimizer is exploring (it fluctuates during a
        # line search), the best tells you what you would actually keep.
        msg = (f"{self.label}eval {self.n_eval:6d} | {el:7.1f}s | "
               f"F = {fidelity:.10f} | best F = {self.best:.10f} | "
               f"1-best = {1.0 - self.best:.3e}{self.extra}")
        if self.mode == "line":
            # "\r" returns to column 0 without a newline so the line overwrites
            # itself.  The trailing spaces erase leftovers from a longer message.
            self.stream.write("\r" + msg + "   ")
        else:
            self.stream.write(msg + "\n")
        try:
            self.stream.flush()                      # otherwise nothing appears until exit
        except Exception:
            pass                                     # a closed/odd stream must not kill the run
        self._dirty = True

    def finish(self, note: str = "") -> None:
        """Print the final summary line.

        ``note`` explains an early exit, e.g. "(cancelled)" or "(target reached)".
        """
        if self.mode == "none":
            return
        el = time.perf_counter() - self.start_ts
        msg = (f"{self.label}done  {self.n_eval:6d} evals | {el:7.1f}s | "
               f"best F = {self.best:.10f} | 1-best = {1.0 - self.best:.3e}  {note}")
        # Only overwrite-and-newline if a "\r" line is actually pending; if the
        # run was so short that nothing was ever printed, a bare "\r" would
        # clobber whatever the user's previous output was.
        if self.mode == "line" and self._dirty:
            self.stream.write("\r" + msg + "   \n")
        else:
            self.stream.write(msg + "\n")
        try:
            self.stream.flush()
        except Exception:
            pass


class RunHandle:
    """Handle for an optimization started with :meth:`OptimalControl.run_async`.

    The point is to keep a notebook responsive while a long optimization runs:
    the work happens on a background thread, and you can watch the fidelity,
    stop it when it is good enough, and collect the result.

    Typical notebook use::

        h = opt.run_async(pulses0=p0, maxiter=2000)
        h.best_fidelity           # readable at any time, from any cell
        h.stop()                  # graceful stop; keeps the best iterate
        out = h.wait()            # blocks until the thread is finished

    Threading notes
    ---------------
    The thread is a **daemon**, so a forgotten run cannot stop the interpreter
    from exiting.  Cancellation goes through ``threading.Event`` on the
    optimizer, which is the only piece of shared mutable state that both threads
    touch -- and Event is designed for exactly this.  Everything else the worker
    writes (``best_x``, ``best_fidelity``) is written by the worker and only read
    by the main thread, so a torn read is the worst case and that just means the
    displayed number is one evaluation stale.

    Do not run two optimizations on the *same* optimizer object concurrently:
    they would share ``best_x`` and the cancel flag and fight over both.
    """

    def __init__(self, optimizer: "OptimalControl", fn: Callable[[], Dict[str, Any]]):
        self._opt = optimizer
        self.result: Optional[Dict[str, Any]] = None        # filled in on success
        self.exception: Optional[BaseException] = None      # filled in on failure
        self._done = threading.Event()                      # set when the thread exits

        def _target():
            """Thread body: run the optimization, capturing whatever happens.

            An exception in a bare thread would otherwise print a traceback to
            stderr and vanish, leaving ``wait()`` to return ``None`` with no
            explanation.  Stashing it and re-raising from ``wait()`` puts the
            error where the user is actually looking.
            """
            try:
                self.result = fn()
            except BaseException as exc:  # noqa: BLE001 - deliberately broad; re-raised in wait()
                self.exception = exc
            finally:
                # In a finally block so the waiter is released even if the
                # optimization blew up in an unexpected way.
                self._done.set()

        self._thread = threading.Thread(target=_target, daemon=True)
        self._thread.start()

    # -- control ---------------------------------------------------------
    def stop(self) -> None:
        """Ask the optimizer to stop; the best iterate so far is kept.

        Returns immediately -- this only sets a flag.  The run stops at its next
        objective evaluation (sub-millisecond in practice).  Follow with
        :meth:`wait` to collect the result.
        """
        self._opt.cancel()

    def wait(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Block until the run finishes and return its result dict.

        Parameters
        ----------
        timeout:
            Seconds to wait, or ``None`` to wait forever.

        Returns
        -------
        The result dict, or ``None`` if ``timeout`` elapsed while still running.
        That ``None`` is the documented way to poll: ``if h.wait(60) is None:
        h.stop()``.

        Raises whatever the optimization raised, if anything.
        """
        self._done.wait(timeout)
        if self.exception is not None:
            raise self.exception
        return self.result

    # -- status ----------------------------------------------------------
    @property
    def running(self) -> bool:
        """True while the background thread is still working."""
        return self._thread.is_alive()

    def is_alive(self) -> bool:
        """Alias for :attr:`running` (matches ``threading.Thread`` naming)."""
        return self._thread.is_alive()

    @property
    def best_fidelity(self) -> float:
        """Best fidelity seen so far -- updates live while the run is going."""
        return self._opt.best_fidelity

    def __repr__(self) -> str:
        """So that a bare ``handle`` in a notebook cell shows something useful."""
        state = "running" if self.running else "finished"
        return f"<RunHandle {state} best_F={self._opt.best_fidelity:.8f}>"


# =============================================================================
# Gauge unitary helper (commuting operators get an O(d^2) fast path)
# =============================================================================

class _GaugeBlock:
    """Computes G(theta) = exp(+i sum_j theta_j A_j) and every dG/dtheta_j.

    The gauge unitary is rebuilt on *every* objective evaluation, so how it is
    computed matters.  The generic way -- one ``expm`` plus one ``expm_frechet``
    per gauge parameter -- is by far the most expensive thing in the evaluation
    for small problems.  But in practice the A_j are almost always mode number
    operators, which are diagonal, or number operators conjugated into a dressed
    basis, which are not diagonal but still commute.  Both cases collapse the
    matrix exponential into a scalar one.

    Three code paths, chosen automatically once at construction:

      ``diagonal``   every A_j is diagonal (e.g. ``[n_c, n_q]``).  G is then just
                     a diagonal of phases: O(d) work, O(d^2) to materialize.
      ``commuting``  the A_j commute pairwise, so they share one eigenbasis W
                     found once here.  Each call is then two matrix products --
                     no expm, no expm_frechet.
      ``general``    no structure: fall back to expm + expm_frechet per parameter.

    ``mode`` is exposed (and saved into the run metadata) so you can confirm
    which path a given problem actually took.
    """

    def __init__(self, gauge_ops: Sequence[qt.Qobj], dim: int):
        self.dim = dim
        self.ops = list(gauge_ops)
        self.n = len(self.ops)
        # Convert to dense NumPy once.  QuTiP Qobj arithmetic carries dims
        # bookkeeping that is pure overhead in the inner loop.
        self.dense = [op.full() for op in self.ops]
        # "none" means no gauge freedom at all -- G is the identity forever.
        self.mode = "none" if self.n == 0 else self._classify()

    def _classify(self) -> str:
        """Pick the cheapest valid code path, and cache whatever it needs.

        Runs once at construction, so it can afford to be thorough: every fast
        path is *verified* numerically rather than assumed, and anything that
        does not verify falls through to the general path.  A wrong answer here
        would silently corrupt every gradient, so the checks are not optional.
        """
        # --- fastest path: all operators already diagonal --------------
        # np.diag(np.diag(A)) zeroes everything off the diagonal, so this is the
        # largest off-diagonal magnitude across all the gauge operators.
        offdiag = max(np.max(np.abs(A - np.diag(np.diag(A)))) for A in self.dense)
        if offdiag <= 1e-12:
            # Store the real diagonals: A_j diagonal and Hermitian => real
            # eigenvalues sitting on the diagonal.  .copy() detaches from the
            # dense array so a later mutation cannot corrupt the cache.
            self.diag_vals = [np.real(np.diag(A)).copy() for A in self.dense]
            return "diagonal"

        # --- second path: Hermitian and mutually commuting -------------
        # Commuting Hermitian operators are simultaneously diagonalizable, which
        # is the property being exploited.  Hermiticity is required for eigh and
        # for the eigenvalues to be real.
        herm = all(_is_hermitian(A) for A in self.dense)
        commutes = herm and all(
            # [A_i, A_j] == 0, with the tolerance scaled by the operator
            # magnitudes so it is a relative rather than absolute test.
            np.max(np.abs(self.dense[i] @ self.dense[j] - self.dense[j] @ self.dense[i]))
            <= 1e-9 * max(1.0, np.max(np.abs(self.dense[i])) * np.max(np.abs(self.dense[j])))
            for i in range(self.n) for j in range(i + 1, self.n)
        )
        if commutes:
            # A *generic* real linear combination of commuting Hermitian
            # operators has the same eigenvectors as all of them, and generically
            # has no accidental degeneracies -- which matters because eigh only
            # returns a well-defined eigenbasis within a degenerate block.  Fixed
            # seed so the classification is deterministic across runs.
            rng = np.random.default_rng(12345)
            mix = sum(float(w) * A for w, A in zip(rng.normal(size=self.n), self.dense))
            _, W = np.linalg.eigh(mix)
            Wh = W.conj().T

            # Verify: W must diagonalize every A_j, not just the mixture.  This
            # catches the unlucky case where the random combination happened to
            # be degenerate and eigh returned an arbitrary basis inside a block.
            vals = []
            ok = True
            for A in self.dense:
                T = Wh @ A @ W                       # A in the candidate basis
                d = np.diag(T)
                if np.max(np.abs(T - np.diag(d))) > 1e-8 * max(1.0, np.max(np.abs(A))):
                    ok = False
                    break
                vals.append(np.real(d).copy())       # eigenvalues of A_j
            if ok:
                self.W, self.Wh, self.diag_vals = W, Wh, vals
                return "commuting"

        # --- fallback: no exploitable structure ------------------------
        return "general"

    def __call__(self, theta: Optional[np.ndarray], need_grad: bool = True
                 ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Evaluate G(theta), and dG/dtheta_j for every j if ``need_grad``.

        Parameters
        ----------
        theta:
            Gauge angles, shape ``(n,)``.  ``None`` means all zeros (G = I).
        need_grad:
            Skip the derivatives when only the value is wanted (``cost``,
            ``fidelity``, ``unitary``), which is the majority of calls during a
            derivative-free global search.

        Returns
        -------
        ``(G, [dG/dtheta_0, ...])``; the list is empty when ``need_grad`` is False.
        """
        # No gauge freedom: G is the identity and there is nothing to differentiate.
        if self.n == 0:
            return np.eye(self.dim, dtype=np.complex128), []

        theta = np.zeros(self.n) if theta is None else np.asarray(theta, dtype=float)
        if theta.shape != (self.n,):
            raise ValueError(f"theta must be shape {(self.n,)}, got {theta.shape}.")

        if self.mode == "diagonal":
            # G = exp(+i sum_j theta_j A_j) with every A_j diagonal, so the whole
            # exponential is elementwise on the diagonal:
            #     G_kk = exp(i * sum_j theta_j * (A_j)_kk)
            phase = np.exp(1j * sum(t * v for t, v in zip(theta, self.diag_vals)))
            G = np.diag(phase)
            # dG/dtheta_j = i A_j G, and both are diagonal, so this is elementwise.
            dG = ([np.diag(1j * v * phase) for v in self.diag_vals] if need_grad else [])
            return G, dG

        if self.mode == "commuting":
            # Same scalar exponential, just expressed in the shared eigenbasis W:
            #     G = W diag(exp(i sum_j theta_j lambda_j)) W^dag
            phase = np.exp(1j * sum(t * v for t, v in zip(theta, self.diag_vals)))
            # (W * phase) scales the COLUMNS of W by `phase`, i.e. W @ diag(phase)
            # without building the diagonal matrix.
            G = (self.W * phase) @ self.Wh
            # dG/dtheta_j = W diag(i lambda_j exp(...)) W^dag -- the A_j commute
            # with G, so no ordering subtlety arises.
            dG = ([(self.W * (1j * v * phase)) @ self.Wh for v in self.diag_vals]
                  if need_grad else [])
            return G, dG

        # --- general path: build the exponent and use expm/expm_frechet ---
        A_g = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for t, A in zip(theta, self.dense):
            A_g = A_g + 1j * t * A               # A_g = +i sum_j theta_j A_j
        G = expm(A_g)
        # d/dtheta_j exp(A_g) is the Frechet derivative in the direction
        # dA_g/dtheta_j = i A_j.  compute_expm=False because G is already known.
        dG = ([expm_frechet(A_g, 1j * A, compute_expm=False) for A in self.dense]
              if need_grad else [])
        return G, dG


# =============================================================================
# Base class
# =============================================================================

class OptimalControl:
    """Shared machinery for gate-synthesis optimal control.

    Subclasses provide the *ansatz*: how a flat parameter vector becomes a
    sequence of propagators, and how to contract a co-state against the
    derivative of one propagator.  See :class:`GrapeOC` and
    :class:`ParameterOC`.

    Parameters
    ----------
    U_target:
        Target unitary, full Hilbert space, same dimension as the generators.
    cost_type:
        ``"unitary"`` for full-space fidelity, ``"projected"`` for the fidelity
        restricted to the subspace of ``projector``.
    projector:
        Orthogonal projector P (required when ``cost_type="projected"``).
    gauge_ops:
        Operators A_j defining the free gauge unitary G = exp(+i sum theta_j A_j).
        Their angles are optimized alongside the control parameters.
    adiabatic_unitary:
        Optional basis change Ad; the compared unitary becomes G Ad^dag U Ad.
    """

    # Short tag identifying the ansatz; written into the saved metadata so a run
    # folder can be recognized without importing anything.  Subclasses override.
    ansatz_name: str = "base"

    def __init__(
        self,
        U_target: qt.Qobj,
        *,                                    # everything below is keyword-only,
        cost_type: str = "unitary",           # so call sites stay self-documenting
        projector: Optional[qt.Qobj] = None,
        gauge_ops: Optional[Sequence[qt.Qobj]] = None,
        adiabatic_unitary: Optional[qt.Qobj] = None,
    ):
        # ---- target unitary ------------------------------------------
        if not isinstance(U_target, qt.Qobj):
            raise TypeError("U_target must be a QuTiP Qobj.")
        self.Ut = U_target
        # Keep the tensor structure (e.g. [[12, 2], [12, 2]]) so results handed
        # back as Qobj carry the same subsystem layout the user started with.
        self.dims = U_target.dims
        self.dim = int(U_target.shape[0])       # total Hilbert space dimension d

        # ---- fidelity type -------------------------------------------
        if cost_type not in ("unitary", "projected"):
            raise ValueError("cost_type must be 'unitary' or 'projected'.")
        self.cost_type = cost_type

        # ---- projector (subspace fidelity) ---------------------------
        # P selects the subspace the gate must be correct on.  The simulation
        # still runs in the full space, so population that leaks out of P simply
        # fails to contribute to the overlap and is penalized automatically.
        self.P: Optional[qt.Qobj] = None
        if cost_type == "projected":
            if projector is None:
                raise ValueError("projector=... is required for cost_type='projected'.")
            if not isinstance(projector, qt.Qobj):
                raise TypeError("projector must be a QuTiP Qobj.")
            if projector.shape != (self.dim, self.dim):
                raise ValueError(f"projector must be shape {(self.dim, self.dim)}.")
            self.P = projector

        # ---- gauge operators -----------------------------------------
        # Each A_j contributes one free angle theta_j, optimized alongside the
        # controls, absorbing a phase the experiment does not care about.
        self.gauge_ops = list(gauge_ops) if gauge_ops is not None else []
        self.n_gauge = len(self.gauge_ops)
        for j, Aj in enumerate(self.gauge_ops):
            if not isinstance(Aj, qt.Qobj):
                raise TypeError(f"gauge_ops[{j}] must be a QuTiP Qobj.")
            if Aj.shape != (self.dim, self.dim):
                raise ValueError(f"gauge_ops[{j}] must be shape {(self.dim, self.dim)}.")

        # ---- adiabatic frame -----------------------------------------
        # Used when the controls act in one basis but the target is written in
        # another (typically bare vs dressed): compare Ad^dag U Ad to the target.
        if adiabatic_unitary is not None:
            if not isinstance(adiabatic_unitary, qt.Qobj):
                raise TypeError("adiabatic_unitary must be a QuTiP Qobj.")
            if adiabatic_unitary.shape != (self.dim, self.dim):
                raise ValueError("adiabatic_unitary must match the Hilbert space dimension.")
        self.adiabatic_unitary = adiabatic_unitary
        # Cache the dense form and its adjoint; both are used on every evaluation.
        self._Ad = None if adiabatic_unitary is None else adiabatic_unitary.full()
        self._Ad_dag = None if self._Ad is None else self._Ad.conj().T

        # ---- dense caches for the inner loop -------------------------
        # Everything below is precomputed once so the objective touches only
        # plain NumPy arrays.  QuTiP Qobj arithmetic re-validates dims on every
        # operation, which is real overhead at tens of thousands of evaluations.
        self._Ut_dense = self.Ut.full()
        self._P_dense = None if self.P is None else self.P.full()

        # The single most useful precomputation.  The overlap is
        #     c = Tr(Ut^dag U_corr P) = Tr(P Ut^dag U_corr)      [cyclicity]
        # so folding M := P Ut^dag (or just Ut^dag when unprojected) turns the
        # overlap into ONE O(d^2) contraction instead of two d x d products.
        Ut_dag = self._Ut_dense.conj().T
        self._M = Ut_dag if self._P_dense is None else self._P_dense @ Ut_dag

        # Effective dimension used to normalize the fidelity into [0, 1]:
        # the full dimension, or the rank of the projector (= Tr P, since P^2 = P).
        self._d_eff = (float(self.dim) if cost_type == "unitary"
                       else float(np.real(np.trace(self._P_dense))))
        if self._d_eff <= 0:
            # Catches an all-zero "projector", which would otherwise divide by 0
            # and produce nan fidelities that are hard to trace back.
            raise ValueError("Effective dimension Tr(P) must be positive.")
        # Store the reciprocal: the objective multiplies rather than divides.
        self._inv_d2 = 1.0 / (self._d_eff * self._d_eff)

        # Gauge block: knows how to build G(theta) and dG/dtheta_j as cheaply as
        # the structure of the gauge operators allows (see _GaugeBlock).
        self._gauge = _GaugeBlock(self.gauge_ops, self.dim)

        # ---- mutable run state ---------------------------------------
        self.penalties: List[Callable] = []      # added via add_*_penalty()
        self._cancel_event = threading.Event()   # set by cancel(), read per evaluation
        self._progress: Optional[_Progress] = None   # non-None only during a run
        self._in_run = False                     # arms the cancellation check
        # best_* track the best point across the whole run, including points the
        # optimizer visited and then walked away from.  This is what makes a
        # cancelled run still return something useful.
        self.best_fidelity: float = -np.inf
        self.best_x: Optional[np.ndarray] = None
        self._history: List[Dict[str, Any]] = []
        self._store_history_x = False
        # Remembered so save() can be called with no arguments after optimize().
        self.last_result: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------------------
    # Abstract ansatz interface
    #
    # These are the only methods a new ansatz has to implement.  Together they
    # answer: how many parameters are there, what propagators do they produce,
    # and how does a co-state contract against one propagator's derivative.
    # -----------------------------------------------------------------
    @property
    def n_core(self) -> int:
        """Number of ansatz parameters, excluding the gauge angles.

        The full optimization vector is ``n_core + n_gauge`` long.
        """
        raise NotImplementedError

    def _forward(self, x_core: np.ndarray, need_grad: bool) -> Dict[str, Any]:
        """Turn parameters into propagators; the forward half of an evaluation.

        Must return a dict with at least::

            "U_list"    list of the M propagators, index 0 acting first
            "U_final"   their time-ordered product U_{M-1} ... U_0
            "n_slices"  M
            "prefix"    [I, U_0, U_1 U_0, ...] of length M+1  (only if need_grad)

        plus whatever per-slice data ``_slice_dc`` needs.  When ``need_grad`` is
        False it should skip the prefix products and the derivative data, since
        the propagators can then be folded directly into ``U_final``.
        """
        raise NotImplementedError

    def _slice_dc(self, cache: Dict[str, Any], j: int, L: np.ndarray,
                  out: np.ndarray) -> None:
        """Accumulate ``dc/dp = Tr(L dU_j/dp)`` into ``out``, for slice j's parameters.

        ``L`` is the co-state matrix at slice j (see ``cost_and_grad``), ``out``
        is the complex vector of length ``n_core`` being filled in.  The method
        writes into the slots belonging to slice j and leaves the rest alone --
        it accumulates with ``+=`` rather than assigning, so a parameter shared
        between slices would compose correctly.
        """
        raise NotImplementedError

    def _core_to_natural(self, x_core: np.ndarray) -> Dict[str, Any]:
        """Re-express the flat core vector the way the user thinks about it.

        The returned dict is merged into the result of ``optimize()``, which is
        why the keys are things like ``"pulses_opt"`` or ``"rotation_opt"``.
        """
        raise NotImplementedError

    def _metadata(self) -> Dict[str, Any]:
        """Ansatz-specific fields to record in the saved ``metadata.json``.

        Default: nothing.  Subclasses add what is needed to interpret (and
        reconstruct) the run later, e.g. ``dt``/``n_steps`` or the gate sequence.
        """
        return {}

    def _qobjs_to_save(self) -> Dict[str, Any]:
        """Ansatz-specific QuTiP operators to write into ``qobjs/``.

        Default: nothing.  The base class already saves the target, projector,
        gauge operators and adiabatic unitary.
        """
        return {}

    # -----------------------------------------------------------------
    # Penalties
    #
    # A penalty adds a term to the cost and its gradient.  It is a plain
    # callable (x_core, theta) -> (cost, grad_core, grad_theta), stored in a
    # list and applied in cost_and_grad.  Penalties are how you express "and
    # also, make the pulse physically reasonable".
    # -----------------------------------------------------------------
    def add_l2_amplitude_penalty(self, lam: float) -> None:
        """Penalize total control power:  ``lam * sum(x_core^2)``.

        Discourages large amplitudes, which usually means less drive power and
        less leakage.  For GRAPE the sum is multiplied by ``dt`` so the penalty
        approximates the time integral of u(t)^2 and its size does not change
        when you re-discretize the same pulse with more steps.

        ``lam`` trades off against infidelity directly; start small (1e-5) and
        increase until the pulse looks reasonable without the fidelity suffering.
        """
        lam = float(lam)
        # getattr because only GrapeOC has a dt; ParameterOC gate angles are
        # dimensionless, so no scaling applies.  Captured now, by value, so a
        # later change to self.dt cannot silently redefine an existing penalty.
        scale = getattr(self, "dt", 1.0)

        def penalty(x_core: np.ndarray, theta: Optional[np.ndarray]):
            """Closure capturing lam and scale; called once per evaluation."""
            cost = lam * scale * float(np.sum(x_core ** 2))
            g = 2.0 * lam * scale * x_core          # d/dx (x^2) = 2x
            # No theta dependence, but the shape has to match when a gauge exists.
            return cost, g, (None if theta is None else np.zeros_like(theta))

        self.penalties.append(penalty)

    def add_smoothness_penalty(self, lam: float) -> None:
        """Penalize jaggedness:  ``lam * sum_n (x[n+1] - x[n])^2``.

        Successive differences are taken along the *sequence* index (time slice
        for GRAPE, gate index for ParameterOC), so this pushes toward pulses an
        instrument can actually play rather than sample-to-sample noise.
        """
        lam = float(lam)

        def penalty(x_core: np.ndarray, theta: Optional[np.ndarray]):
            """Closure capturing lam; called once per evaluation."""
            arr = self._core_as_sequence(x_core)          # (n_slices, n_per_slice)
            diffs = arr[1:] - arr[:-1]                    # forward differences
            cost = lam * float(np.sum(diffs ** 2))

            # d/dx[n] of sum_m (x[m+1] - x[m])^2.  Each difference d_n = x[n+1] -
            # x[n] contributes -2*lam*d_n to entry n and +2*lam*d_n to entry n+1,
            # so the two slice-assignments below accumulate exactly the right
            # thing without an explicit loop.  Endpoints appear in one term only,
            # which the [:-1] / [1:] slices handle automatically.
            g = np.zeros_like(arr)
            g[:-1] -= 2.0 * lam * diffs
            g[1:] += 2.0 * lam * diffs
            return (cost, self._sequence_as_core(g),
                    (None if theta is None else np.zeros_like(theta)))

        self.penalties.append(penalty)

    def add_custom_penalty(self, penalty_fn: Callable) -> None:
        """Register ``penalty_fn(x_core, theta) -> (cost, grad_core, grad_theta)``.

        ``x_core``:      flat ansatz parameter vector, ``n_core`` entries, in the
                         layout documented on each subclass.
        ``grad_core``:   same length as ``x_core``, or ``None`` for no dependence.
        ``grad_theta``:  shape ``(n_gauge,)``, or ``None``.

        The gradient you supply is trusted -- it is added directly to the
        analytic gradient.  If it is wrong the optimizer will quietly converge to
        the wrong place, so check a new penalty against finite differences (see
        ``_grad_case`` in the test suite for the pattern).
        """
        self.penalties.append(penalty_fn)

    def clear_penalties(self) -> None:
        """Remove all registered penalties.

        Useful in a notebook, where re-running an ``add_*_penalty`` cell would
        otherwise stack another copy of the same term onto the list.
        """
        self.penalties = []

    def _core_as_sequence(self, x_core: np.ndarray) -> np.ndarray:
        """View the core vector as ``(n_slices, n_per_slice)``.

        Needed by the smoothness penalty, which has to know which entries are
        neighbours in time/sequence order.  Subclasses implement it because the
        flat layout differs between ansatzes.
        """
        raise NotImplementedError

    def _sequence_as_core(self, arr: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`_core_as_sequence`, for mapping a gradient back.

        The default row-major flatten is correct for GRAPE; ParameterOC overrides
        it because its layout groups all rotations before all phases.
        """
        return np.asarray(arr, dtype=float).reshape(-1)

    # -----------------------------------------------------------------
    # Packing
    #
    # SciPy wants one flat real vector.  The layout is always
    #     x = [ ...n_core ansatz parameters..., ...n_gauge angles... ]
    # with the gauge angles last so that the ansatz block keeps the same indices
    # whether or not a gauge is in use.
    # -----------------------------------------------------------------
    def _pack(self, x_core: np.ndarray, theta: Optional[np.ndarray]) -> np.ndarray:
        """Concatenate ansatz parameters and gauge angles into the SciPy vector."""
        x_core = np.asarray(x_core, dtype=float).reshape(-1)
        if self.n_gauge == 0:
            return x_core                          # nothing to append
        # theta=None is legal and means "start the gauge at zero", which is what
        # a caller who has not thought about the gauge wants.
        theta = np.zeros(self.n_gauge) if theta is None else np.asarray(theta, dtype=float)
        return np.concatenate([x_core, theta.reshape(-1)])

    def _unpack(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Split the flat SciPy vector back into ``(x_core, theta)``.

        Returns views into ``x``, not copies -- this runs on every evaluation and
        nothing downstream mutates them.  ``theta`` is ``None`` when there is no
        gauge, which is the signal the rest of the code branches on.
        """
        x = np.asarray(x, dtype=float)
        n = self.n_core
        if self.n_gauge == 0:
            return x[:n], None
        return x[:n], x[n:n + self.n_gauge]

    # -----------------------------------------------------------------
    # Core evaluation
    # -----------------------------------------------------------------
    def _corrected_unitary(self, U_final: np.ndarray, G: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the adiabatic frame and then the gauge: ``U_corr = G Ad^dag U Ad``.

        Both optional stages are skipped entirely when unused, so a problem with
        neither a frame nor a gauge pays nothing for their existence.

        Returns
        -------
        ``(U_ad, U_corr)``.  Both are returned because the gauge gradient needs
        ``U_ad`` -- differentiating ``U_corr = G U_ad`` with respect to theta
        gives ``dG U_ad``, so the un-gauged unitary has to survive.
        """
        U_ad = U_final if self._Ad is None else self._Ad_dag @ U_final @ self._Ad
        U_corr = U_ad if self.n_gauge == 0 else G @ U_ad
        return U_ad, U_corr

    def cost_and_grad(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Cost and its exact gradient at the flat parameter vector ``x``.

        This is the primitive every optimizer is wired to (SciPy ``jac=True``):
        one forward propagation serves both the value and the derivative, which
        alone halves the work compared with separate ``cost``/``grad`` calls.

        The gradient uses a backward co-state recursion rather than building
        ``dU_final/dp`` for each parameter.  Starting from

            c  = Tr(M U_corr),          M = P Ut^dag  (precomputed once)
            dc = Tr(M G Ad^dag dU Ad)  = Tr(B dU),     B = Ad M G Ad^dag

        and substituting ``dU = post_j (dU_j/dp) prefix_j`` for a parameter
        living in slice j, cyclicity of the trace gives

            dc = Tr(L_j dU_j/dp),       L_j = prefix_j B post_j.

        The ``post_j`` products are never formed explicitly: with
        ``C_j = B post_j`` the recursion ``C_{j-1} = C_j U_j`` walks backwards for
        one matrix product per slice, and ``L_j = prefix_j C_j`` costs one more.
        Every remaining contraction is then O(d^2) instead of O(d^3).

        Returns
        -------
        (cost, gradient) with the gradient packed in the same flat layout as ``x``.
        """
        x_core, theta = self._unpack(x)

        # ---- forward pass: propagators + prefix products -------------
        # need_grad=True asks the ansatz to also keep the prefix products and
        # whatever per-slice data its derivative needs.
        cache = self._forward(x_core, need_grad=True)
        U_final = cache["U_final"]

        G, dG_list = self._gauge(theta, need_grad=True)
        U_ad, U_corr = self._corrected_unitary(U_final, G)

        # ---- fidelity ------------------------------------------------
        # c = Tr(Ut^dag U_corr P), folded into a single contraction against the
        # precomputed M = P Ut^dag.  Complex in general: its phase is the global
        # phase mismatch, which |c|^2 then discards (a global phase is physically
        # meaningless, so the fidelity must not depend on it).
        c = _trace_prod(self._M, U_corr)              # O(d^2), never trace(A @ B)
        F = float((abs(c) ** 2) * self._inv_d2)
        cost = 1.0 - F

        # Chain rule from the overlap to the cost:
        #     F = |c|^2 / d^2   =>   dF = 2 Re(conj(c) dc) / d^2
        #     cost = 1 - F      =>   d(cost)/dp = -2 Re(conj(c) dc/dp) / d^2
        # So every parameter's gradient is the same fixed transform of its dc/dp,
        # and the two constants can be hoisted out of the loop.
        cbar = np.conj(c)
        w = -2.0 * self._inv_d2

        # ---- co-state matrix B (see the docstring) -------------------
        # dc = Tr(M dU_corr) and U_corr = G Ad^dag U Ad, so
        #     dc = Tr(M G Ad^dag dU Ad) = Tr( (Ad M G Ad^dag) dU ) = Tr(B dU).
        # Built once per evaluation; both stages are skipped when unused.
        B = self._M if self.n_gauge == 0 else self._M @ G
        if self._Ad is not None:
            B = self._Ad @ B @ self._Ad_dag

        # ---- backward sweep over the propagator sequence -------------
        # Walk from the last propagator to the first, maintaining C_j = B post_j.
        # post_j (everything after slice j) is never materialized: at the last
        # slice post = I so C = B, and stepping down one slice right-multiplies
        # by U_j.  Two matrix products per slice total, independent of how many
        # parameters that slice owns.
        dc_core = np.zeros(self.n_core, dtype=np.complex128)
        prefix = cache["prefix"]
        U_list = cache["U_list"]
        n_slices = cache["n_slices"]
        C = B                                          # C_{last} = B, since post_last = I
        for j in range(n_slices - 1, -1, -1):
            L = prefix[j] @ C                          # L_j = prefix_j B post_j
            self._slice_dc(cache, j, L, dc_core)       # ansatz-specific contraction
            if j > 0:
                C = C @ U_list[j]                      # C_{j-1} = C_j U_j
        # dc_core now holds dc/dp for every ansatz parameter; convert to d(cost)/dp.
        grad_core = w * np.real(cbar * dc_core)

        # ---- gauge gradient ------------------------------------------
        # U_corr = G U_ad, so dU_corr/dtheta_j = (dG/dtheta_j) U_ad and
        #     dc_j = Tr(M dG_j U_ad) = Tr( (U_ad M) dG_j )       [cyclicity]
        # Forming Y = U_ad M once makes each parameter an O(d^2) contraction.
        grad_theta = None
        if self.n_gauge > 0:
            Y = U_ad @ self._M
            dc_t = np.array([_trace_prod(Y, dG) for dG in dG_list], dtype=np.complex128)
            grad_theta = w * np.real(cbar * dc_t)

        # ---- penalties (value and gradient) --------------------------
        # Added after the fidelity gradient so `F` above stays the pure fidelity
        # -- that is what gets reported and tracked as the best point, while
        # `cost` is what the optimizer actually descends.
        for pen in self.penalties:
            p_cost, p_gc, p_gt = pen(x_core, theta)
            cost += float(p_cost)
            if p_gc is not None:
                # Rebind rather than += : grad_core may be a fresh array each
                # call, but a penalty returning a read-only or broadcast array
                # would make in-place addition fail.
                grad_core = grad_core + np.asarray(p_gc, dtype=float).reshape(-1)
            if grad_theta is not None and p_gt is not None:
                grad_theta = grad_theta + np.asarray(p_gt, dtype=float).reshape(-1)

        # Records the best point, feeds the progress line, and raises _Cancelled
        # if a stop was requested.  Must come after `cost` is final.
        self._note_evaluation(x, F, cost)
        return float(cost), self._pack(grad_core, grad_theta)

    def cost(self, x: np.ndarray) -> float:
        """Cost only, without the gradient.

        Cheaper than :meth:`cost_and_grad` because ``need_grad=False`` lets the
        forward pass skip the prefix products and the per-slice derivative data,
        folding the propagators straight into ``U_final``.

        Used by the derivative-free global optimizers, which demand a scalar
        objective.  Prefer :meth:`cost_and_grad` everywhere else -- computing the
        gradient alongside the value is far cheaper than computing it separately.
        """
        x_core, theta = self._unpack(x)
        cache = self._forward(x_core, need_grad=False)
        G, _ = self._gauge(theta, need_grad=False)
        _, U_corr = self._corrected_unitary(cache["U_final"], G)
        c = _trace_prod(self._M, U_corr)
        F = float((abs(c) ** 2) * self._inv_d2)
        cost = 1.0 - F
        for pen in self.penalties:
            cost += float(pen(x_core, theta)[0])      # [0] = the penalty's value
        self._note_evaluation(x, F, cost)
        return float(cost)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """Gradient only (computes the cost too and throws it away)."""
        return self.cost_and_grad(x)[1]

    # -----------------------------------------------------------------
    # Fidelity / unitary conveniences
    # -----------------------------------------------------------------
    def unitary(self, *args, **kwargs) -> qt.Qobj:
        """Corrected unitary U_corr = G Ad^dag U_final Ad as a QuTiP Qobj.

        Accepts the same arguments as :meth:`fidelity`.
        """
        x_core, theta = self._unpack(self._resolve_x(*args, **kwargs))
        cache = self._forward(x_core, need_grad=False)
        G, _ = self._gauge(theta, need_grad=False)
        _, U_corr = self._corrected_unitary(cache["U_final"], G)
        return qt.Qobj(U_corr, dims=self.dims)

    def raw_unitary(self, *args, **kwargs) -> qt.Qobj:
        """U_final alone -- no gauge, no adiabatic frame.

        Accepts the same arguments as :meth:`fidelity`.
        """
        x_core, _ = self._unpack(self._resolve_x(*args, **kwargs))
        cache = self._forward(x_core, need_grad=False)
        return qt.Qobj(cache["U_final"], dims=self.dims)

    def fidelity(self, *args, **kwargs) -> float:
        """Configured fidelity F in [0, 1] (penalties are NOT included).

        Four equivalent ways to say which parameters to use::

            opt.fidelity(out)                          # a result dict from optimize()
            opt.fidelity(out["pulses_opt"], theta=out["theta_opt"])   # natural params
            opt.fidelity(x=some_flat_vector)           # the packed vector
            opt.fidelity()                             # the best point seen so far
        """
        x_core, theta = self._unpack(self._resolve_x(*args, **kwargs))
        cache = self._forward(x_core, need_grad=False)
        G, _ = self._gauge(theta, need_grad=False)
        _, U_corr = self._corrected_unitary(cache["U_final"], G)
        return float((abs(_trace_prod(self._M, U_corr)) ** 2) * self._inv_d2)

    def _resolve_x(self, *args, **kwargs) -> np.ndarray:
        """Turn whatever the caller passed into a flat parameter vector.

        Handles, in order: an explicit ``x=`` vector, a result dict from
        ``optimize()``, no arguments at all (use the running best), and finally
        the ansatz's natural parameters via :meth:`_natural_x`.
        """
        # --- form 3: an explicit packed vector, x=... ------------------
        # Checked first because it is unambiguous; everything else is inference.
        x = kwargs.pop("x", None)
        if x is not None:
            if args or kwargs:
                # Silently ignoring one of the two would be much worse than
                # refusing: the user would get a number computed from parameters
                # they did not mean.
                raise TypeError("Pass either x=... or the natural parameters, not both.")
            return np.asarray(x, dtype=float)

        # --- form 1: the result dict straight from optimize() ----------
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            res = args[0]
            if "x_opt" not in res:
                raise TypeError("dict argument must be a result from optimize().")
            return np.asarray(res["x_opt"], dtype=float)

        # --- form 4: nothing at all -> the best point seen so far ------
        if not args and not kwargs:
            if self.best_x is None:
                raise ValueError("No parameters given and no optimization has been run "
                                 "yet. Pass the parameters explicitly.")
            return self.best_x

        # --- form 2: the ansatz's natural parameters ------------------
        # e.g. (pulses, theta=...) or (rotation, phase, theta=...).  The subclass
        # validates shapes, so a packed vector passed positionally by mistake
        # raises a shape error naming the expected form rather than being
        # silently misread.
        return self._natural_x(*args, **kwargs)

    def _natural_x(self, *args, **kwargs) -> np.ndarray:
        """Pack the ansatz's natural parameters into a flat vector (subclass API)."""
        raise NotImplementedError

    def _check_theta0(self, theta0) -> Optional[np.ndarray]:
        """Validate gauge angles, defaulting to zeros; ``None`` when there is no gauge.

        Shared by both subclasses and by ``_natural_x``, so that "theta=None"
        means the same thing (start at zero / no gauge) everywhere.
        """
        if self.n_gauge == 0:
            return None                      # no gauge: theta is meaningless
        if theta0 is None:
            return np.zeros(self.n_gauge, dtype=float)
        theta0 = np.asarray(theta0, dtype=float)
        if theta0.shape != (self.n_gauge,):
            raise ValueError(f"theta0 must be shape {(self.n_gauge,)}, got {theta0.shape}.")
        return theta0

    # -----------------------------------------------------------------
    # Evaluation bookkeeping (progress + cancellation)
    # -----------------------------------------------------------------
    def _note_evaluation(self, x: np.ndarray, F: float, cost: float) -> None:
        """Bookkeeping run after every objective evaluation.

        Three jobs, all deliberately O(1) so they never show up in a profile:
        remember the best point seen, feed the progress display, and turn a
        cancellation request into an exception that unwinds out of SciPy.

        Tracking the best point *here* rather than trusting the optimizer's
        return value is what makes cancellation safe, and it also protects
        against a global method that ends its run somewhere worse than the best
        place it visited.

        The cancellation check is gated on ``_in_run`` so that the *final*
        fidelity/cost evaluations ``_run_optimize`` performs after a cancelled
        run do not themselves raise.
        """
        if F > self.best_fidelity:
            self.best_fidelity = F
            # copy=True because `x` is usually a SciPy-owned buffer that will be
            # overwritten on the next iteration -- storing a view would give us
            # whatever the optimizer happened to be holding at the end.
            self.best_x = np.array(x, dtype=float, copy=True)
        if self._progress is not None:
            self._progress.update(F, cost)
        if self._in_run and self._cancel_event.is_set():
            raise _Cancelled

    def cancel(self) -> None:
        """Signal a running optimize()/run_async() to stop; best iterate is kept.

        The flag is cleared at the start of every run, so calling this before a
        run starts has no effect -- it only stops a run that is already going.
        """
        self._cancel_event.set()

    # -----------------------------------------------------------------
    # Optimization
    # -----------------------------------------------------------------
    def _run_optimize(
        self,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[Optional[float], Optional[float]]]],
        *,
        method: str = "multistart",
        maxiter: int = 500,
        n_starts: int = 8,
        sampling: str = "sobol",
        seed: Optional[int] = None,
        scipy_options: Optional[Dict[str, Any]] = None,
        local_method: str = "L-BFGS-B",
        progress: str = "line",
        progress_every: float = 0.25,
        store_history: bool = True,
        store_history_x: bool = False,
        target_fidelity: Optional[float] = None,
        polish: bool = True,
        method_options: Optional[Dict[str, Any]] = None,
        reset_best: bool = True,
    ) -> Dict[str, Any]:
        """Shared optimizer driver -- see :func:`available_methods` for ``method``.

        Parameters
        ----------
        x0, bounds:
            Flat start vector and per-variable ``(lo, hi)`` bounds (or ``None``).
        method:
            Optimizer name.  Local gradient methods use the analytic gradient
            directly; derivative-free globals get a gradient-based polish
            afterwards when ``polish`` is True.
        maxiter, scipy_options:
            Passed to the underlying SciPy local minimizer.
        n_starts, sampling, seed:
            Multistart controls (``"sobol"``, ``"lhs"``, ``"halton"``, ``"random"``).
        progress, progress_every:
            ``"line"`` (single self-updating line), ``"log"`` or ``"none"``, and
            the minimum wall-clock gap between printed updates.
        target_fidelity:
            Stop as soon as this fidelity is reached.
        method_options:
            Extra knobs for the global methods (``niter``, ``popsize``, ``perturb``,
            ``n``, ``n_calls``, ``sigma0``, ``workers``, ...).
        reset_best:
            When False the running best fidelity/point survives from the previous
            call, so successive optimize() calls can be chained.

        Returns
        -------
        dict with ``x_opt``, ``fidelity``, ``cost``, ``result``, ``history``,
        ``cancelled``, ``method``, ``n_evaluations``, ``elapsed_s``, ``starts``,
        plus the ansatz's natural parameters (``pulses_opt`` or
        ``rotation_opt``/``phase_opt``) and ``theta_opt``.
        """
        method = str(method)
        # maxiter is promoted into the options dict so callers can pass it as a
        # plain argument; anything in scipy_options wins if it collides.
        options = {"maxiter": int(maxiter)}
        if scipy_options:
            options.update(dict(scipy_options))
        # Copied so that mutating our local view cannot surprise the caller, and
        # so `or {}` gives a usable dict when nothing was passed.
        method_options = dict(method_options or {})

        # ---- reset per-run state ------------------------------------
        self._cancel_event.clear()          # a stale cancel must not kill a new run
        if reset_best:
            self.best_fidelity = -np.inf
            self.best_x = None
        self._history = []
        self._store_history_x = bool(store_history_x)
        self._progress = _Progress(mode=progress, every=progress_every)
        self._in_run = True                 # arms the cancellation check
        # Seeded generator: with a fixed `seed` the whole run (start points,
        # perturbations) is reproducible.
        rng = np.random.default_rng(seed)

        x0 = np.asarray(x0, dtype=float)

        # ---- objective wrappers -------------------------------------
        # fun_grad is the primitive: one propagation gives cost and gradient, so
        # SciPy is always called with jac=True.  fun_only/jac_only exist only for
        # the global methods that insist on a scalar objective.
        def fun_grad(xv):
            """Cost and gradient together -- the form SciPy gets with jac=True."""
            f, g = self.cost_and_grad(xv)
            self._check_target(target_fidelity)
            return f, g

        def fun_only(xv):
            """Scalar cost, for the global methods that cannot take a gradient."""
            f = self.cost(xv)
            self._check_target(target_fidelity)
            return f

        def jac_only(xv):
            """Gradient alone, for inner local searches that want a separate jac."""
            # Wasteful in isolation (it computes the cost and discards it), but
            # only used by global methods whose inner local search demands a
            # separate jac callable.
            return self.cost_and_grad(xv)[1]

        def make_callback(tag: str):
            """Build a per-iteration history hook labelled with ``tag``.

            ``tag`` records which phase produced the entry ("start3", "polish",
            "anneal", ...), which is what makes the history readable afterwards
            for a method that runs many inner searches.

            The signature is deliberately loose: SciPy passes a different number
            of callback arguments per method (``(xk)``, ``(xk, f, accept)``,
            ``(xk, state)``), and the lambdas at each call site below normalize
            them down to just ``xk``.
            """
            def cb(xk, *args, **kwargs):
                """SciPy iteration callback; extra args differ per method."""
                if store_history:
                    # Record the running best rather than this iterate's value,
                    # so the trace is monotonic and reads as "progress".
                    rec = {"start": tag, "fidelity": self.best_fidelity}
                    if store_history_x and xk is not None:
                        rec["x"] = np.array(xk, dtype=float, copy=True)
                    self._history.append(rec)
            return cb

        # Argument bundle shared by every call into scipy.optimize.minimize.
        local_kwargs = dict(method=local_method, jac=True, bounds=bounds,
                            options=_filter_options(local_method, options))

        cancelled = False
        res: Optional[OptimizeResult] = None
        starts: List[Dict[str, Any]] = []       # per-start records, multistart only
        t0 = time.perf_counter()

        # The whole dispatch is inside one try so that _Cancelled raised deep
        # inside any method unwinds to the same place.
        try:
            # ---- plain local gradient minimizers ---------------------
            if method in ("L-BFGS-B", "TNC", "SLSQP", "trust-constr", "BFGS", "CG", "local"):
                # "local" means "whatever local_method says"; the others name
                # themselves.
                lm = local_method if method == "local" else method
                kw = dict(local_kwargs)
                kw["method"] = lm
                kw["options"] = _filter_options(lm, options)
                if lm in ("BFGS", "CG"):
                    # These two are unconstrained algorithms and SciPy errors if
                    # handed bounds.  NOTE: that means they may return a point
                    # outside your physical limits -- avoid them when the bounds
                    # are real constraints rather than a search box.
                    kw["bounds"] = None
                res = minimize(fun_grad, x0, callback=make_callback("local"), **kw)

            # ---- restart-based globals (both use the analytic gradient)
            # Same driver; `iterated` switches between independent quasi-random
            # restarts (multistart) and perturbing the running best (ils).
            elif method in ("multistart", "ils"):
                res, starts = self._multistart(
                    fun_grad, x0, bounds, local_kwargs, make_callback, rng,
                    n_starts=n_starts, sampling=sampling,
                    iterated=(method == "ils"),
                    perturb=method_options.get("perturb", 0.25),
                    target_fidelity=target_fidelity,
                )

            # ---- Monte-Carlo hopping between gradient-found local minima
            elif method == "basinhopping":
                # basinhopping forwards minimizer_kwargs straight to minimize(),
                # so jac=True from local_kwargs means our analytic gradient drives
                # every inner descent.  T is the Metropolis temperature (accept a
                # worse minimum with probability ~exp(-dF/T)) and stepsize is the
                # random displacement between hops; both are auto-tuned by SciPy
                # as it goes.
                mk = dict(local_kwargs)
                mk.setdefault("options", options)
                res = basinhopping(
                    fun_grad, x0, niter=int(method_options.get("niter", n_starts)),
                    T=float(method_options.get("T", 1.0)),
                    stepsize=float(method_options.get("stepsize", 0.5)),
                    minimizer_kwargs=mk,
                    seed=seed,
                    callback=lambda xk, f, acc: make_callback("basinhop")(xk),
                )

            # ---- annealing: global scalar search + gradient local search
            elif method == "dual_annealing":
                # The annealing walk itself needs a scalar objective (fun_only),
                # but the local refinement it runs from promising points does use
                # the analytic gradient, passed as jac_only.
                fb = self._finite_bounds(bounds, x0)
                res = dual_annealing(
                    fun_only, bounds=fb,
                    maxiter=int(method_options.get("niter", max(50, n_starts * 10))),
                    x0=x0, seed=seed,
                    minimizer_kwargs={"method": local_method, "jac": jac_only,
                                      "options": _filter_options(local_method, options)},
                    callback=lambda xk, f, ctx: make_callback("anneal")(xk),
                )

            # ---- population search; polish=False here because we run our own
            #      gradient polish below (SciPy's polish cannot use our jac)
            elif method == "differential_evolution":
                fb = self._finite_bounds(bounds, x0)
                res = differential_evolution(
                    fun_only, bounds=fb,
                    maxiter=int(method_options.get("niter", 200)),
                    popsize=int(method_options.get("popsize", 15)),
                    tol=float(method_options.get("tol", 1e-8)),
                    mutation=method_options.get("mutation", (0.5, 1.0)),
                    recombination=float(method_options.get("recombination", 0.7)),
                    init=method_options.get("init", "sobol"),   # even initial coverage
                    polish=False, seed=seed, x0=x0,
                    # workers != 1 forks the population; SciPy then requires
                    # "deferred" updating because immediate updating is inherently
                    # sequential.  Note that forked workers cannot report progress
                    # or observe cancel().
                    updating="deferred" if method_options.get("workers", 1) != 1 else "immediate",
                    workers=int(method_options.get("workers", 1)),
                    # DE's callback signature has changed across SciPy versions,
                    # so swallow whatever it passes and log the running best.
                    callback=lambda *a, **k: make_callback("de")(self.best_x),
                )

            # ---- simplicial homology global optimization
            elif method == "shgo":
                # SHGO builds a simplicial complex whose size grows exponentially
                # with the number of variables; above ~15 it stops responding
                # rather than finishing, and because that work happens outside the
                # objective, cancel() cannot interrupt it.  Refuse loudly instead.
                n_dim = x0.size
                max_dim = int(method_options.get("max_dim", 15))
                if n_dim > max_dim:
                    raise ValueError(
                        f"method='shgo' is impractical for {n_dim} variables (the "
                        f"simplicial complex blows up beyond ~{max_dim}). Use "
                        f"'multistart', 'ils' or 'basinhopping' instead, or pass "
                        f"method_options={{'max_dim': {n_dim}}} to force it.")
                fb = self._finite_bounds(bounds, x0)
                res = shgo(
                    fun_only, bounds=fb,
                    n=int(method_options.get("n", 128)),
                    iters=int(method_options.get("iters", 1)),
                    sampling_method=method_options.get("sampling_method", "sobol"),
                    minimizer_kwargs={"method": local_method, "jac": jac_only,
                                      "options": _filter_options(local_method, options)},
                )

            # ---- optional third-party backend ------------------------
            elif method == "cma":
                res = self._run_cma(fun_only, x0, bounds, method_options, seed, options)

            else:
                raise ValueError(
                    f"Unknown method {method!r}. Available: {', '.join(available_methods())}")

            # Derivative-free globals land near a minimum but rarely in it.  One
            # L-BFGS-B run with the analytic gradient is cheap and typically buys
            # several digits of infidelity, so it is on by default.
            if polish and method in ("differential_evolution", "cma", "shgo",
                                     "dual_annealing"):
                xp = self.best_x if self.best_x is not None else np.asarray(res.x, float)
                res_p = minimize(fun_grad, xp, callback=make_callback("polish"),
                                 **local_kwargs)
                if res is None or res_p.fun < float(getattr(res, "fun", np.inf)):
                    res = res_p

        except _Cancelled:
            # cancel() or target_fidelity fired; self.best_x holds the best point
            cancelled = True
        finally:
            # Disarm the cancellation check *before* the final evaluations below,
            # otherwise they would immediately re-raise _Cancelled.
            self._in_run = False

        elapsed = time.perf_counter() - t0
        n_eval = self._progress.n_eval
        best_x = np.asarray(self.best_x if self.best_x is not None else x0, dtype=float)

        note = ""
        if cancelled:
            note = ("(target reached)" if target_fidelity is not None
                    and self.best_fidelity >= target_fidelity else "(cancelled)")
        self._progress.finish(note)
        self._progress = None            # stop counting the final evaluations

        x_core, theta_opt = self._unpack(best_x)
        out: Dict[str, Any] = {
            "x_opt": best_x,
            "fidelity": float(self.fidelity(x=best_x)),
            "cost": float(self.cost(best_x)),
            "result": res,
            "history": self._history,
            "cancelled": cancelled,
            "method": method,
            "n_evaluations": int(n_eval),
            "elapsed_s": elapsed,
            "starts": starts,
        }
        out.update(self._core_to_natural(x_core))   # pulses_opt / rotation_opt+phase_opt
        out["theta_opt"] = theta_opt
        self.last_result = out
        return out

    def _check_target(self, target_fidelity: Optional[float]) -> None:
        """Trip the cancel flag once the requested fidelity has been reached.

        The flag (rather than an immediate raise) is used so the unwinding always
        happens at the same place, inside :meth:`_note_evaluation`.
        """
        if target_fidelity is not None and self.best_fidelity >= target_fidelity:
            self._cancel_event.set()

    # -- multistart / iterated local search -----------------------------
    def _multistart(self, fun_grad, x0, bounds, local_kwargs, make_callback, rng,
                    *, n_starts, sampling, iterated, perturb, target_fidelity):
        """Run many gradient descents and keep the best; the engine behind
        ``method="multistart"`` and ``method="ils"``.

        Both are the same loop with a different rule for choosing start points:

        ``multistart`` (iterated=False)
            Start points are drawn up front from a quasi-random sequence spread
            over the whole bounding box.  The runs are independent, so the search
            never gets stuck -- but it also never exploits anything it learns.

        ``ils`` -- iterated local search (iterated=True)
            Each new start is a random perturbation of the *best minimum found so
            far*.  This exploits the fact that in these landscapes good minima
            tend to cluster: once you find a decent basin, its neighbours are
            often better than a fresh random point.  ``perturb`` is the step size
            as a fraction of the box width -- too small and it re-finds the same
            minimum, too large and it degenerates into multistart.

        Start 0 is always the user's ``x0`` in both modes, so supplying a good
        physical guess is never wasted.

        Returns
        -------
        ``(best_result, starts)`` where ``starts`` is one record per start,
        exposed as ``result["starts"]`` so you can see the spread of outcomes and
        judge whether more restarts would help.
        """
        n_starts = max(1, int(n_starts))
        best_res = None
        starts: List[Dict[str, Any]] = []
        lo, hi = self._bounds_arrays(bounds, x0)

        # For multistart, draw every start point now: quasi-random sequences are
        # only well distributed as a complete block, not point by point.  ILS
        # cannot precompute, since each start depends on the previous results.
        if iterated:
            samples = None
        else:
            samples = self._sample_starts(n_starts - 1, lo, hi, sampling, rng)

        anchor = np.array(x0, dtype=float)      # ILS: the point being perturbed
        for s in range(n_starts):
            if s == 0:
                xs = np.array(x0, dtype=float)                   # the user's guess
            elif iterated:
                # Gaussian kick scaled per-coordinate by the box width, then
                # clipped back inside the box so the start is always feasible.
                step = perturb * (hi - lo)
                xs = np.clip(anchor + rng.normal(scale=step), lo, hi)
            else:
                xs = samples[s - 1]                              # s-1: start 0 was x0

            r = minimize(fun_grad, xs, callback=make_callback(f"start{s}"), **local_kwargs)

            # Record the fidelity (not just the cost) so the summary is readable
            # even when penalties make the cost hard to interpret.
            F_s = float(self.fidelity(x=r.x))
            starts.append({"start": s, "fidelity": F_s, "cost": float(r.fun),
                           "nit": int(getattr(r, "nit", -1))})

            # Compare on cost, which is what is being minimized (fidelity alone
            # would ignore the penalties).
            if best_res is None or r.fun < best_res.fun:
                best_res = r
                if iterated:
                    anchor = np.array(r.x, dtype=float)   # ILS moves to the new best

            # Stop early if the caller only wanted "good enough".  Checked here
            # as well as in the objective so we do not launch another whole
            # descent after the target has already been met.
            if target_fidelity is not None and self.best_fidelity >= target_fidelity:
                break
        return best_res, starts

    @staticmethod
    def _sample_starts(n: int, lo: np.ndarray, hi: np.ndarray, sampling: str,
                       rng) -> np.ndarray:
        """Draw ``n`` start points in the box [lo, hi].

        Quasi-random sequences (Sobol by default) cover a high-dimensional box far
        more evenly than uniform random draws for the same number of points, which
        is exactly what multistart wants.  Sobol's balance guarantees only hold for
        powers of two, so a full 2^m block is drawn and trimmed.
        """
        if n <= 0:
            return np.zeros((0, lo.size))       # e.g. n_starts=1: only x0 is used
        sampling = str(sampling).lower()
        d = lo.size

        # The n <= 2**16 guard keeps the up-front allocation sane: Sobol has to
        # generate a whole power-of-two block, so an absurd n_starts would
        # otherwise try to materialize millions of rows before the first descent.
        if sampling in ("sobol", "lhs", "halton") and _qmc is not None and n <= 2 ** 16:
            # Derive the engine's seed from rng so the whole run stays governed
            # by the single user-supplied seed.
            seed = int(rng.integers(0, 2 ** 31 - 1))
            if sampling == "sobol":
                # Sobol's equidistribution properties only hold for 2^m points,
                # and SciPy warns if asked for anything else -- so request the
                # next power of two and trim.
                m = int(np.ceil(np.log2(max(n, 1))))
                u = _qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(m)[:n]
            else:
                eng = {"lhs": _qmc.LatinHypercube, "halton": _qmc.Halton}[sampling]
                u = eng(d=d, seed=seed).random(n)
        else:
            u = rng.random((n, d))              # plain uniform fallback

        # u is in [0,1)^d; affinely map it onto the box.
        return lo + (hi - lo) * u

    def _bounds_arrays(self, bounds, x0) -> Tuple[np.ndarray, np.ndarray]:
        """Build a finite ``(lo, hi)`` box, inventing limits for unbounded axes.

        Sampling start points and the global methods all need a *finite* box, but
        ``bounds`` may be ``None`` or contain ``None`` entries (SciPy's way of
        writing "unbounded").  For those axes a box of width ``2*max(|x0_i|, 1)``
        centred on ``x0`` is used: proportional to the scale the user's own guess
        implies, with a floor of 1 so that an axis starting at exactly 0 still
        gets something to explore.

        This box is only ever used for *sampling*.  The true bounds (including
        the ``None``s) are what get passed to the minimizer, so nothing here can
        turn an unbounded variable into a bounded one.
        """
        n = np.asarray(x0).size
        if bounds is None:
            # Fully unbounded: one vectorized expression covers every axis.
            span = np.maximum(np.abs(np.asarray(x0, float)), 1.0)
            return np.asarray(x0, float) - span, np.asarray(x0, float) + span

        lo = np.empty(n)
        hi = np.empty(n)
        for i, (a, b) in enumerate(bounds):
            xi = float(np.asarray(x0).ravel()[i])
            span = max(abs(xi), 1.0)
            # Substitute the invented limit only on the side that is None, so a
            # half-bounded variable keeps its real constraint on the other side.
            lo[i] = xi - span if a is None else float(a)
            hi[i] = xi + span if b is None else float(b)
        return lo, hi

    def _finite_bounds(self, bounds, x0) -> List[Tuple[float, float]]:
        """``_bounds_arrays`` as a list of ``(lo, hi)`` tuples.

        The global SciPy methods want that shape rather than two arrays.
        """
        lo, hi = self._bounds_arrays(bounds, x0)
        return list(zip(lo.tolist(), hi.tolist()))

    def _run_cma(self, fun_only, x0, bounds, mopts, seed, options):
        """CMA-ES via the optional ``cma`` package, driven through its ask/tell API.

        CMA-ES is gradient-free: it adapts a multivariate Gaussian over the
        parameters, sampling a population each generation and re-fitting the
        covariance to the best of them.  That makes it strong on rugged
        landscapes but blind to the exact gradient we already have -- which is
        why ``_run_optimize`` follows it with a gradient polish.

        Kept optional (a plain ``ImportError`` with install instructions) so the
        module has no hard dependency beyond NumPy/SciPy/QuTiP.
        """
        try:
            import cma  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "method='cma' needs the 'cma' package (pip install cma)."
            ) from exc

        lo, hi = self._bounds_arrays(bounds, x0)
        # sigma0 is the initial search radius.  A quarter of the mean box width
        # is the usual rule of thumb: wide enough to escape the starting basin,
        # narrow enough not to spend the first generations at the boundary.
        sigma0 = float(mopts.get("sigma0", 0.25 * float(np.mean(hi - lo))))
        es = cma.CMAEvolutionStrategy(
            np.asarray(x0, float).tolist(), sigma0,
            {"bounds": [lo.tolist(), hi.tolist()],
             "maxiter": int(mopts.get("niter", options.get("maxiter", 500))),
             # cma rejects seed=None in some versions; +1 because it also treats
             # 0 as "pick randomly", which would break reproducibility.
             "seed": 0 if seed is None else int(seed) + 1,
             "verbose": -9},                    # silence cma's own printing
        )
        # ask/tell rather than es.optimize() so that our objective wrapper --
        # and therefore progress reporting and cancellation -- stays in the loop.
        while not es.stop():
            xs = es.ask()                                        # a new population
            es.tell(xs, [fun_only(np.asarray(v)) for v in xs])   # its costs
        r = es.result
        # Repackage into a SciPy-shaped result so downstream code is uniform.
        return OptimizeResult(x=np.asarray(r.xbest, float), fun=float(r.fbest),
                              success=True, message="cma finished", nit=int(es.countiter))

    # -----------------------------------------------------------------
    # Async
    # -----------------------------------------------------------------
    def run_async(self, **kwargs) -> RunHandle:
        """Run :meth:`optimize` in a background thread.

        Takes exactly the same arguments as :meth:`optimize`.  Returns a
        :class:`RunHandle`; call ``handle.stop()`` to cancel and ``handle.wait()``
        to collect the result.

        The two defaults differ from the blocking version: ``progress="log"``
        because a "\\r" self-updating line does not render correctly when written
        from a background thread in most notebook front-ends, and a longer
        ``progress_every`` so a long run does not fill the output with thousands
        of lines.  Both can be overridden.
        """
        kwargs.setdefault("progress", "log")
        kwargs.setdefault("progress_every", 2.0)
        # The lambda defers the call to the worker thread; RunHandle starts it.
        return RunHandle(self, lambda: self.optimize(**kwargs))

    def optimize(self, *args, **kwargs):
        """Run the optimization.  Implemented by each subclass.

        Subclasses expose their natural arguments (``pulses0``/``pulse_bounds``
        or ``rotation0``/``phase0``/``rotation_bounds``), build the flat ``x0``
        and ``bounds``, and hand off to :meth:`_run_optimize`, which does the
        actual work and is shared.
        """
        raise NotImplementedError

    # -----------------------------------------------------------------
    # Saving
    # -----------------------------------------------------------------
    def save(
        self,
        folder: Union[str, Path],
        results: Optional[Dict[str, Any]] = None,
        *,
        override: bool = False,
        extra: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> Path:
        """Save a run into ``folder``.  Never silently overwrites.

        If ``folder`` already exists and ``override`` is False, the next free
        name ``folder_1``, ``folder_2``, ... is used instead.  Optimization runs
        are expensive and easy to re-trigger by re-running a notebook cell, so
        the default is to preserve what is already on disk; the path actually
        written is printed and returned.

        Layout::

            <folder>/results.npz      arrays (x_opt, natural params, history)
            <folder>/metadata.json    human-readable run configuration
            <folder>/qobjs/*.qu       QuTiP operators (target, projector, ...)

        The split exists because the two halves want different formats: NumPy
        handles arrays compactly, JSON keeps the configuration readable without
        Python, and QuTiP's own format preserves the ``dims`` tensor structure
        that a bare array would lose.

        Parameters
        ----------
        folder:
            Destination directory (created, including parents).
        results:
            A result dict from ``optimize()``.  Defaults to the most recent run.
        override:
            Overwrite ``folder`` instead of picking a new name.
        extra:
            Anything else worth recording -- drive parameters, gate times, ...
            Stored under ``metadata["extra"]``; non-JSON values are repr'd rather
            than causing a failure.
        notes:
            Free-text description, stored under ``metadata["notes"]``.
        """
        # Default to the last run so `opt.save("folder")` just works.
        results = results if results is not None else self.last_result
        if results is None:
            raise ValueError("Nothing to save: run optimize() first or pass results=...")

        out_dir = self._resolve_save_dir(Path(folder), override=override)
        out_dir.mkdir(parents=True, exist_ok=True)
        qdir = out_dir / "qobjs"
        qdir.mkdir(parents=True, exist_ok=True)

        # ---- metadata: everything needed to interpret the run later ----
        meta: Dict[str, Any] = {
            "ansatz": self.ansatz_name,
            "class": type(self).__name__,
            "dim": int(self.dim),
            "dims": self.dims,                            # tensor structure
            "cost_type": self.cost_type,
            "n_gauge": int(self.n_gauge),
            "gauge_mode": self._gauge.mode,               # which fast path was used
            "has_adiabatic_unitary": self.adiabatic_unitary is not None,
            "n_penalties": len(self.penalties),           # penalties are closures,
                                                          # so only the count survives
            "fidelity": float(results.get("fidelity", np.nan)),
            "method": results.get("method"),
            "n_evaluations": int(results.get("n_evaluations", 0)),
            "elapsed_s": float(results.get("elapsed_s", np.nan)),
            "cancelled": bool(results.get("cancelled", False)),
            "starts": results.get("starts", []),          # per-start spread
            "notes": notes,
        }
        meta.update(self._metadata())                     # ansatz-specific fields
        if extra:
            meta["extra"] = _jsonify(extra)

        # ---- arrays ----------------------------------------------------
        payload: Dict[str, np.ndarray] = {
            "x_opt": np.asarray(results["x_opt"], dtype=float),
            # The metadata is duplicated inside the npz so the archive is
            # self-describing even if metadata.json is separated from it.
            "metadata_json": np.array(json.dumps(_jsonify(meta), indent=2)),
        }
        # Sweep up every other array in the result dict -- pulses_opt, or
        # rotation_opt/phase_opt -- without the base class needing to know which
        # ansatz produced them.  The skipped keys are either already stored or
        # not arrays (result is a SciPy object, history/starts are lists).
        for key, val in results.items():
            if key in ("x_opt", "result", "history", "starts"):
                continue
            if isinstance(val, np.ndarray):
                payload[key] = np.asarray(val, dtype=float)

        # npz cannot store None, so "no gauge" is encoded as an empty array;
        # the loaders turn a zero-size array back into None.
        theta = results.get("theta_opt", None)
        payload["theta_opt"] = (np.array([], dtype=float) if theta is None
                                else np.asarray(theta, dtype=float))

        # Just the fidelity trace, not the full history: the x-vectors would
        # dominate the file size and are rarely wanted after the fact.
        hist = [h.get("fidelity", np.nan) for h in results.get("history", [])]
        payload["history_fidelity"] = np.asarray(hist, dtype=float)

        np.savez_compressed(out_dir / "results.npz", **payload)

        # Also written standalone so it can be read without NumPy (or by eye).
        (out_dir / "metadata.json").write_text(json.dumps(_jsonify(meta), indent=2))

        # ---- QuTiP operators -------------------------------------------
        # Saved so a run can be fully reconstructed from the folder alone,
        # without re-running whatever notebook cell originally built them.
        qobjs: Dict[str, Any] = {"U_target": self.Ut, "projector": self.P,
                                 "gauge_op": self.gauge_ops,
                                 "adiabatic_unitary": self.adiabatic_unitary}
        qobjs.update(self._qobjs_to_save())          # H_drift/H_control, or a_c/a_q
        for name, obj in qobjs.items():
            _qsave_safe(obj, qdir, name)             # skips None, expands lists

        print(f"Saved run to {out_dir}")
        return out_dir

    @staticmethod
    def _resolve_save_dir(folder: Path, *, override: bool) -> Path:
        """Return ``folder``, or the first free ``folder_1`` / ``folder_2`` / ...

        Counts up rather than using a timestamp so the names stay short and sort
        in creation order.  The collision is announced, because a silently
        renamed output is exactly the kind of thing that gets noticed three days
        later while wondering why the results did not change.
        """
        if override or not folder.exists():
            return folder
        i = 1
        while True:
            # with_name replaces only the final component, so the parent
            # directory is preserved.
            cand = folder.with_name(f"{folder.name}_{i}")
            if not cand.exists():
                print(f"'{folder}' already exists -> writing to '{cand}' "
                      f"(pass override=True to overwrite instead).")
                return cand
            i += 1


# =============================================================================
# GRAPE ansatz: piecewise-constant pulses
# =============================================================================

class GrapeOC(OptimalControl):
    """GRAPE: gradient ascent with piecewise-constant control pulses.

    Time is chopped into ``n_steps`` slices of length ``dt``.  Within slice n the
    controls are held constant, so the Hamiltonian and propagator are

        H_n = H0 + sum_k u[n,k] Hk
        U_n = exp(-i dt H_n / hbar)

    and the free parameters are the amplitudes ``u[n,k]`` -- one per (time slice,
    control channel) pair.  This is the standard GRAPE ansatz: very expressive
    (any pulse shape the time grid can resolve) at the price of many parameters.

    Core parameter layout: ``u`` flattened row-major, i.e.
    ``[u[0,0], u[0,1], ..., u[1,0], ...]`` -- ``n_steps * n_ctrl`` entries.

    Parameters
    ----------
    H_drift:
        The always-on Hamiltonian H0.
    H_controls:
        The operators Hk whose amplitudes are being optimized.
    U_target, cost_type, projector, gauge_ops, adiabatic_unitary:
        See :class:`OptimalControl`.
    dt, n_steps:
        Slice duration and count; the total gate time is ``dt * n_steps``.
    derivative:
        How ``dU_n/du`` is computed:

        ``"spectral"`` (default)
            Exact, from the eigendecomposition of H_n via the Daleckii-Krein
            formula.  2.4-4x faster than ``"frechet"``, and the advantage grows
            with the number of control channels because the expensive part of
            the calculation is shared between them.  Requires Hermitian
            generators; falls back to ``"frechet"`` automatically (with a
            printed notice) if they are not.
        ``"frechet"``
            Exact, via ``scipy.linalg.expm_frechet``.  Works for any generator.
        ``"approx"``
            The small-dt approximation ``dU_n/du ~ (-i dt Hk / hbar) U_n``, which
            ignores ``[H_n, Hk]``.  Only valid when ``dt * ||Hk||`` is small.
            Kept for comparison; there is no speed reason to prefer it now that
            the spectral path exists.
    hbar:
        Set to 1 by default, i.e. Hamiltonians are given in angular frequency.
    """

    ansatz_name = "grape"

    def __init__(
        self,
        H_drift: qt.Qobj,
        H_controls: Sequence[qt.Qobj],
        U_target: qt.Qobj,
        dt: float,
        n_steps: int,
        *,
        cost_type: str = "unitary",
        projector: Optional[qt.Qobj] = None,
        gauge_ops: Optional[Sequence[qt.Qobj]] = None,
        adiabatic_unitary: Optional[qt.Qobj] = None,
        derivative: str = "spectral",
        hbar: float = 1.0,
    ):
        # The base class validates and caches everything that is not specific to
        # the pulse ansatz (target, projector, gauge, adiabatic frame).
        super().__init__(U_target, cost_type=cost_type, projector=projector,
                         gauge_ops=gauge_ops, adiabatic_unitary=adiabatic_unitary)

        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.hbar = float(hbar)
        self.H0 = H_drift
        self.Hc = list(H_controls)        # list() so a generator or tuple works
        self.n_ctrl = len(self.Hc)

        # ---- validation ----------------------------------------------
        # Shape mismatches here would otherwise surface as an opaque broadcasting
        # error thousands of evaluations later.
        for op, name in [(self.H0, "H_drift")]:
            if not isinstance(op, qt.Qobj):
                raise TypeError(f"{name} must be a QuTiP Qobj.")
            if op.shape != (self.dim, self.dim):
                raise ValueError(f"{name} must be shape {(self.dim, self.dim)}.")
        for k, Hk in enumerate(self.Hc):
            if not isinstance(Hk, qt.Qobj):
                raise TypeError(f"H_controls[{k}] must be a QuTiP Qobj.")
            if Hk.shape != (self.dim, self.dim):
                raise ValueError(f"H_controls[{k}] must be shape {(self.dim, self.dim)}.")
        if self.n_ctrl == 0:
            # Nothing to optimize; almost certainly a mistake at the call site.
            raise ValueError("At least one control Hamiltonian is required.")

        # ---- dense caches --------------------------------------------
        self._H0 = self.H0.full()
        self._Hc = [Hk.full() for Hk in self.Hc]
        # Stacked as one (n_ctrl, d, d) array so that building H_n is a single
        # tensordot rather than a Python loop over control channels.
        self._Hc_stack = np.stack(self._Hc)

        # ---- derivative method ---------------------------------------
        if derivative not in ("spectral", "frechet", "approx"):
            raise ValueError("derivative must be 'spectral', 'frechet' or 'approx'.")
        # eigh assumes Hermiticity and reads only one triangle, so a
        # non-Hermitian generator would give a silently wrong answer rather than
        # an error.  Check once and downgrade rather than refuse: a non-Hermitian
        # H is unusual but legitimate (e.g. an effective non-unitary model).
        hermitian = _is_hermitian(self._H0) and all(_is_hermitian(H) for H in self._Hc)
        if derivative == "spectral" and not hermitian:
            print("GrapeOC: generators are not Hermitian -> falling back to "
                  "derivative='frechet'.")
            derivative = "frechet"
        self.derivative = derivative
        self._hermitian = hermitian

        # The exponent is always A_n = s * H_n with this fixed scalar; hoisted so
        # the inner loop never recomputes -1j * dt / hbar.
        self._s = -1j * self.dt / self.hbar

    # -- ansatz interface ----------------------------------------------
    @property
    def n_core(self) -> int:
        """One parameter per (time slice, control channel)."""
        return self.n_steps * self.n_ctrl

    def _core_as_sequence(self, x_core: np.ndarray) -> np.ndarray:
        """View the flat vector as the ``(n_steps, n_ctrl)`` pulse array.

        Row-major reshape, so row n holds every control amplitude at time slice
        n -- which is also the ordering the smoothness penalty needs (successive
        rows are successive times).
        """
        return np.asarray(x_core, dtype=float).reshape(self.n_steps, self.n_ctrl)

    def _forward(self, x_core: np.ndarray, need_grad: bool) -> Dict[str, Any]:
        """Build every slice propagator U_n = exp(-i dt H_n / hbar).

        When ``need_grad`` is True the prefix products and the per-slice data the
        derivative needs (eigenvectors/eigenvalues, or the exponent A_n) are kept
        as well; otherwise the propagators are folded straight into ``U_final``
        and nothing else is stored.
        """
        pulses = self._core_as_sequence(x_core)          # (n_steps, n_ctrl)
        N, d = self.n_steps, self.dim
        U_list: List[np.ndarray] = [None] * N            # type: ignore[list-item]
        V_list: List[Optional[np.ndarray]] = [None] * N  # eigenvectors of H_n
        a_list: List[Optional[np.ndarray]] = [None] * N  # exponent eigenvalues s*w
        A_list: List[Optional[np.ndarray]] = [None] * N  # A_n, for the Frechet path

        use_spec = self.derivative == "spectral"
        for n in range(N):
            # H_n = H0 + sum_k u[n,k] Hk, contracted in one BLAS call
            Hn = self._H0 + np.tensordot(pulses[n], self._Hc_stack, axes=(0, 0))
            if use_spec:
                # H_n is Hermitian, so exp(-i dt H_n) is just a rescaling of its
                # eigenvalues -- and the same eigenbasis gives the exact
                # derivative below, for every control channel at once.
                w, V = np.linalg.eigh(Hn)
                a = self._s * w                          # s = -i dt / hbar
                ea = np.exp(a)
                U_list[n] = (V * ea) @ V.conj().T        # (V diag(ea)) V^dag
                if need_grad:
                    V_list[n] = V
                    a_list[n] = a
            else:
                A = self._s * Hn
                U_list[n] = expm(A)
                if need_grad:
                    A_list[n] = A

        # prefix[n] = U_{n-1} ... U_0, with prefix[0] = I; U_final = prefix[N]
        prefix: List[np.ndarray] = []
        if need_grad:
            I = np.eye(d, dtype=np.complex128)
            prefix = [I]
            for n in range(N):
                prefix.append(U_list[n] @ prefix[n])
            U_final = prefix[N]
        else:
            U_final = np.eye(d, dtype=np.complex128)
            for n in range(N):
                U_final = U_list[n] @ U_final

        return {"U_list": U_list, "prefix": prefix, "U_final": U_final,
                "n_slices": N, "V": V_list, "a": a_list, "A": A_list}

    def _slice_dc(self, cache, j: int, L: np.ndarray, out: np.ndarray) -> None:
        """Accumulate dc/du[j,k] = Tr(L dU_j/du[j,k]) for every control k.

        Spectral path (default).  With A = V diag(a) V^dag, the Frechet
        derivative of the matrix exponential in direction dA is

            dU = V [ (V^dag dA V) o Phi ] V^dag,
            Phi_pq = (e^{a_p} - e^{a_q}) / (a_p - a_q)      (Daleckii-Krein)

        so, using cyclicity and writing Ltil = V^dag L V,

            Tr(L dU) = sum_pq (Ltil^T o Phi)_pq (V^dag dA V)_pq.

        The factor ``(Ltil^T o Phi)`` does not depend on k, so it is built once
        per slice and reused for every control channel -- which is why adding
        control channels is much cheaper here than with ``expm_frechet``.
        """
        base = j * self.n_ctrl                   # first parameter index of this slice
        if self.derivative == "spectral":
            V = cache["V"][j]
            Vh = V.conj().T
            _, phi = _divided_difference(cache["a"][j])
            Mmat = (Vh @ L @ V).T * phi          # k-independent, computed once
            for k in range(self.n_ctrl):
                W = Vh @ self._Hc[k] @ V         # V^dag (dA/du) V, up to the factor s
                out[base + k] += self._s * np.einsum("ij,ij->", Mmat, W, optimize=False)
        elif self.derivative == "frechet":
            # Exact but slower: one 2d x 2d matrix exponential per control channel.
            A = cache["A"][j]
            for k in range(self.n_ctrl):
                dU = expm_frechet(A, self._s * self._Hc[k], compute_expm=False)
                out[base + k] += _trace_prod(L, dU)
        else:
            # Small-dt approximation dU_n/du ~ (-i dt Hk / hbar) U_n. Only valid
            # when dt * ||Hk|| is small and [H_n, Hk] is negligible.
            U = cache["U_list"][j]
            for k in range(self.n_ctrl):
                out[base + k] += self._s * _trace_prod(L, self._Hc[k] @ U)

    def _core_to_natural(self, x_core: np.ndarray) -> Dict[str, Any]:
        """Expose the optimized parameters as the 2-D pulse array users expect."""
        return {"pulses_opt": self._core_as_sequence(x_core)}

    def _natural_x(self, pulses=None, theta=None) -> np.ndarray:
        """Pack ``(pulses, theta)`` into the flat vector, checking the pulse shape.

        The shape check doubles as disambiguation: a packed 1-D vector passed
        positionally by mistake fails here with a message naming both the
        expected shape and the ``x=`` alternative, instead of being silently
        reinterpreted.
        """
        if pulses is None:
            raise TypeError("GrapeOC needs pulses (positional or pulses=...).")
        pulses = np.asarray(pulses, dtype=float)
        if pulses.shape != (self.n_steps, self.n_ctrl):
            raise ValueError(f"pulses must be shape {(self.n_steps, self.n_ctrl)}, "
                             f"got {pulses.shape}. For the packed vector use x=... ")
        return self._pack(pulses.reshape(-1), self._check_theta0(theta))

    def _metadata(self) -> Dict[str, Any]:
        """Everything needed to interpret a saved GRAPE run.

        ``T_total`` is redundant (dt * n_steps) but saved anyway -- it is the
        number you actually want when reading a run months later.
        """
        return {"n_steps": self.n_steps, "n_ctrl": self.n_ctrl, "dt": self.dt,
                "hbar": self.hbar, "derivative": self.derivative,
                "T_total": self.dt * self.n_steps}

    def _qobjs_to_save(self) -> Dict[str, Any]:
        """The generators, so a saved run can be re-propagated from disk alone."""
        return {"H_drift": self.H0, "H_control": self.Hc}

    # -- public API -----------------------------------------------------
    def optimize(
        self,
        pulses0: Optional[np.ndarray] = None,
        theta0: Optional[np.ndarray] = None,
        *,
        pulse_bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
        theta_bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
        method: str = "L-BFGS-B",
        **kwargs,
    ) -> Dict[str, Any]:
        """Optimize the control pulses (and the gauge angles, if any).

        Parameters
        ----------
        pulses0:
            Initial pulses, shape ``(n_steps, n_ctrl)``.  Defaults to all zeros,
            but a physically motivated guess is usually worth a lot here -- see
            the benchmark results in README.md.
        theta0:
            Initial gauge angles, shape ``(n_gauge,)``.  Defaults to zeros.
        pulse_bounds:
            One ``(lo, hi)`` per *control channel*, i.e. length ``n_ctrl``, and
            applied at every time slice.  ``None`` means unbounded.
        theta_bounds:
            One ``(lo, hi)`` per gauge angle.  Usually left unbounded: a gauge
            angle is a phase and wrapping it is harmless.
        method:
            Defaults to ``"L-BFGS-B"``.  For a good physical starting guess,
            ``"trust-constr"`` measured about 8x better; for a random start, use
            a global method.  See :func:`available_methods` and README.md.
        **kwargs:
            Forwarded to :meth:`_run_optimize` -- ``maxiter``, ``scipy_options``,
            ``n_starts``, ``progress``, ``target_fidelity``, ``seed``, ...

        Returns
        -------
        The result dict described in :meth:`_run_optimize`, including
        ``"pulses_opt"`` and ``"theta_opt"``.
        """
        if pulses0 is None:
            pulses0 = np.zeros((self.n_steps, self.n_ctrl), dtype=float)
        pulses0 = np.asarray(pulses0, dtype=float)
        if pulses0.shape != (self.n_steps, self.n_ctrl):
            raise ValueError(f"pulses0 must be shape {(self.n_steps, self.n_ctrl)}.")

        theta0 = self._check_theta0(theta0)

        # ---- expand the bounds to one entry per optimization variable ----
        # The user gives per-channel bounds; SciPy wants one per variable, in the
        # same order as the packed vector.
        bounds: List[Tuple[Optional[float], Optional[float]]] = []
        if pulse_bounds is None:
            bounds.extend([(None, None)] * self.n_core)
        else:
            if len(pulse_bounds) != self.n_ctrl:
                raise ValueError("pulse_bounds must have length n_ctrl.")
            for _ in range(self.n_steps):
                # Repeat the per-channel bounds once per slice, matching the
                # row-major layout of the flattened pulse array.
                bounds.extend(list(pulse_bounds))

        if self.n_gauge > 0:
            if theta_bounds is None:
                bounds.extend([(None, None)] * self.n_gauge)
            else:
                if len(theta_bounds) != self.n_gauge:
                    raise ValueError("theta_bounds must have length n_gauge.")
                bounds.extend(list(theta_bounds))

        # If nothing is actually bounded, hand SciPy None rather than a list of
        # (None, None): some methods reject a bounds argument outright, and
        # others take a slower constrained code path for no reason.
        if all(b == (None, None) for b in bounds):
            bounds = None  # type: ignore[assignment]

        return self._run_optimize(self._pack(pulses0.reshape(-1), theta0), bounds,
                                  method=method, **kwargs)



# =============================================================================
# Parameterized gate-sequence ansatz
# =============================================================================

class FrameGate:
    """A gate of the form  U(r, phi) = exp(-i r * D(phi) K0 D(phi)^dag).

    with ``D(phi) = exp(i * s * phi * diag(N))``.  Every gate in this module (and
    every gate whose phase enters as a rotation of the drive axis, which is all
    of them in practice) has this form.  Because ``K0`` is fixed, its
    eigendecomposition is computed once here and reused forever, which makes the
    gate and both of its exact derivatives cost two matrix products:

        U      = Vp diag(e^{-i r lam}) Vp^dag,      Vp = D(phi) V
        dU/dr  = Vp diag(-i lam e^{-i r lam}) Vp^dag
        dU/dphi = i s (N U - U N)                   (N diagonal -> O(d^2))

    Parameters
    ----------
    K0:
        Hermitian generator at phi = 0.
    phase_diag:
        Real diagonal of the operator N generating the phase frame, or ``None``
        for a gate without a phase parameter.
    phase_sign:
        The sign ``s`` above.
    """

    def __init__(self, name: str, K0: np.ndarray,
                 phase_diag: Optional[np.ndarray], phase_sign: float = 1.0):
        K0 = np.asarray(K0)
        # Hermiticity is required twice over: eigh assumes it, and without it
        # exp(-i r K0) would not be unitary in the first place.
        if not _is_hermitian(K0):
            raise ValueError(f"gate '{name}': K0 must be Hermitian.")
        self.name = name
        # THE key precomputation.  K0 does not depend on either parameter, so
        # this single eigendecomposition serves every evaluation of this gate
        # for the whole lifetime of the object -- that is where the ~30x comes
        # from, since the old code ran expm + two expm_frechet calls per gate
        # per evaluation.
        self.lam, self.V = np.linalg.eigh(K0)
        self.has_phase = phase_diag is not None
        self.N = None if phase_diag is None else np.asarray(phase_diag, dtype=float)
        self.s = float(phase_sign)
        self.dim = K0.shape[0]

    def evaluate(self, r: float, phi: float, need_grad: bool):
        """Return ``(U, dU/dr, dU/dphi)``; the derivatives are ``None`` if not needed.

        Cost: one O(d^2) row scaling to move into the phase frame, then one d^3
        product for U and one more for dU/dr.  dU/dphi is O(d^2) because N is
        diagonal.  No matrix exponential and no eigendecomposition at call time.
        """
        # D(phi) V, i.e. scale row p of V by e^{i s phi N_p}. phi == 0 is common
        # (phases disabled) and skipping the scaling there is measurably faster.
        if self.has_phase and phi != 0.0:
            Vp = np.exp(1j * self.s * phi * self.N)[:, None] * self.V
        else:
            Vp = self.V
        Vph = Vp.conj().T
        e = np.exp(-1j * r * self.lam)                 # eigenvalues of exp(-i r K)
        U = (Vp * e) @ Vph
        if not need_grad:
            return U, None, None

        # d/dr exp(-i r K) = -i K exp(-i r K); diagonal in the same basis.
        dU_r = (Vp * (-1j * self.lam * e)) @ Vph

        # U(phi) = D U0 D^dag with D = e^{i s phi N}, so dU/dphi = i s [N, U].
        dU_p = None
        if self.has_phase:
            dU_p = (1j * self.s) * (self.N[:, None] * U - U * self.N[None, :])
        return U, dU_r, dU_p

    def hamiltonian(self, r: float, phi: float) -> np.ndarray:
        """Reconstruct H(r, phi) = r * D(phi) K0 D(phi)^dag.

        Not used in the optimization at all -- the gate is built straight from
        the eigendecomposition.  Provided for cross-checks and for anyone who
        wants the actual Hamiltonian, e.g. to feed a time-domain ``mesolve``
        simulation of the same pulse.
        """
        if self.has_phase:
            D = np.exp(1j * self.s * phi * self.N)
            Vp = D[:, None] * self.V
        else:
            Vp = self.V
        # K0 = V diag(lam) V^dag, so D K0 D^dag = (D V) diag(lam) (D V)^dag.
        return r * ((Vp * self.lam) @ Vp.conj().T)


class ParameterOC(OptimalControl):
    """Optimize the parameters of a fixed sequence of named gates.

    Instead of a free pulse shape, the control is a *circuit*: a short list of
    physically available gates, applied in a fixed order, repeated ``num_apply``
    times.  Each gate instance gets its own rotation amplitude and (optionally)
    its own phase.  Far fewer parameters than GRAPE, and every one of them maps
    directly onto something an experiment can dial -- at the cost of a rugged
    landscape, which is why a global optimizer matters much more here.

    ``unitary_strings`` names one repetition, so::

        unitary_strings=["r", "bs"], num_apply=10

    means twenty gates: rotation, beamsplitter, rotation, beamsplitter, ...

    Core parameter layout::

        [rot_0 ... rot_{M-1},  phase_0 ... phase_{M-1}]      (phases optional)

    with ``M = len(unitary_strings) * num_apply`` in application order (index 0
    acts first).  Rotations and phases are grouped rather than interleaved so
    the two blocks can take different bounds.

    Built-in gates (Hilbert space is cavity ``a_c`` tensor qubit ``a_q``):

    ``"bs"``  beamsplitter / sideband exchange between cavity and qubit,
              ``H = g (a_c^dag a_q e^{i phi} + h.c.)``.  ``g = pi/2`` is a full
              swap; ``g = pi`` returns to the identity up to signs.
    ``"r"``   qubit rotation restricted to the 0-1 subspace,
              ``H = theta (sigma_+ e^{i phi} + h.c.)``.  ``theta = pi/2`` is an
              X/2 pulse; the phase sets the rotation axis in the equatorial plane.

    Add your own with :meth:`register_gate`.

    Parameters
    ----------
    unitary_strings:
        Gate names for ONE repetition of the sequence.
    target_unitary, cost_type, projector, gauge_ops, adiabatic_unitary:
        See :class:`OptimalControl`.
    n_c, n_q:
        Cavity and qubit dimensions; ``n_c * n_q`` must equal the target's.
    num_apply:
        How many times the sequence is repeated.
    optimize_phases:
        If False the phases are frozen at 0 and drop out of the parameter
        vector entirely, halving the dimensionality.
    """

    ansatz_name = "parameter"

    def __init__(
        self,
        unitary_strings: Sequence[str],
        target_unitary: qt.Qobj,
        n_c: int,
        n_q: int,
        *,
        num_apply: int = 1,
        optimize_phases: bool = True,
        cost_type: str = "unitary",
        projector: Optional[qt.Qobj] = None,
        gauge_ops: Optional[Sequence[qt.Qobj]] = None,
        adiabatic_unitary: Optional[qt.Qobj] = None,
    ):
        super().__init__(target_unitary, cost_type=cost_type, projector=projector,
                         gauge_ops=gauge_ops, adiabatic_unitary=adiabatic_unitary)
        self.unitary_strings = list(unitary_strings)
        self.n_base = len(self.unitary_strings)     # gates per repetition
        if self.n_base == 0:
            raise ValueError("unitary_strings must not be empty.")

        self.n_c = int(n_c)
        self.n_q = int(n_q)
        # Catches the common slip of changing a dimension in one place only;
        # without it the mismatch surfaces much later as a shape error.
        if self.n_c * self.n_q != self.dim:
            raise ValueError(f"n_c * n_q = {self.n_c * self.n_q} does not match the "
                             f"target dimension {self.dim}.")

        # Ladder operators in the full space, tensor order (cavity, qubit).
        # The gate generators are built from these.
        self.a_c = qt.tensor(qt.destroy(self.n_c), qt.qeye(self.n_q))
        self.a_q = qt.tensor(qt.qeye(self.n_c), qt.destroy(self.n_q))

        # Registry of available gates, keyed by the name used in unitary_strings.
        self.gates: Dict[str, FrameGate] = {}
        self._register_builtin_gates()

        self.num_apply = int(num_apply)
        self.optimize_phases = bool(optimize_phases)
        # Expands unitary_strings x num_apply into the concrete gate list; must
        # run after the registry is populated.
        self._rebuild_sequence()

    # -- gate registry ---------------------------------------------------
    def _register_builtin_gates(self) -> None:
        """Create the "bs" and "r" gates for this Hilbert space.

        Both are expressed in the FrameGate form ``exp(-i r D(phi) K0 D(phi)^dag)``
        with ``D(phi) = exp(i s phi n_q)``.  The phase frame is generated by the
        qubit number operator in both cases, but with OPPOSITE signs, because the
        two Hamiltonians put ``e^{+i phi}`` on operators that transform
        oppositely under ``n_q``:

            bs:  e^{i phi} multiplies a_q      (lowers n_q)  -> s = -1
            r:   e^{i phi} multiplies sigma_+  (raises n_q)  -> s = +1

        Getting that sign wrong still yields a unitary, just one whose phase runs
        backwards -- which is why both gates are checked against the literal
        ``expm`` of their Hamiltonian in the test suite.
        """
        a_c, a_q = self.a_c, self.a_q
        n_q_diag = np.real(np.diag((a_q.dag() * a_q).full())).copy()

        # bs: H = g (a_c^dag a_q e^{i phi} + a_c a_q^dag e^{-i phi})
        #        = g * D K0 D^dag  with D = exp(-i phi n_q)
        K_bs = (a_c.dag() * a_q + a_c * a_q.dag()).full()
        self.gates["bs"] = FrameGate("bs", K_bs, n_q_diag, phase_sign=-1.0)

        # r: H = theta (sigma_+ e^{i phi} + sigma_- e^{-i phi}) on the 0-1 subspace
        #      = theta * D K0 D^dag  with D = exp(+i phi n_q)
        sp = _sigma_plus_01(a_q)
        K_r = (sp + sp.dag()).full()
        self.gates["r"] = FrameGate("r", K_r, n_q_diag, phase_sign=+1.0)

    def register_gate(self, name: str, K0: Union[qt.Qobj, np.ndarray],
                      phase_diag: Optional[Union[qt.Qobj, np.ndarray]] = None,
                      phase_sign: float = 1.0) -> None:
        """Register a custom gate ``U(r, phi) = exp(-i r D(phi) K0 D(phi)^dag)``.

        Parameters
        ----------
        name:
            The string to use in ``unitary_strings``.  Re-registering an existing
            name replaces it.
        K0:
            Hermitian generator at ``phi = 0``, as a Qobj or a dense array.
        phase_diag:
            The operator N generating the phase frame, which must be diagonal --
            pass the operator or just its diagonal.  ``None`` gives a gate with
            no phase parameter (only its rotation amplitude is optimized).
        phase_sign:
            The sign ``s`` in ``D(phi) = exp(i s phi N)``.  Get this wrong and
            the gate is still unitary but its phase runs backwards, so check a
            new gate against ``expm`` of its Hamiltonian before trusting it.

        Note: if ``name`` is already in ``unitary_strings`` the sequence is
        rebuilt, so re-registering mid-session takes effect immediately.
        Otherwise, set ``unitary_strings`` and call :meth:`configure`.
        """
        K = K0.full() if isinstance(K0, qt.Qobj) else np.asarray(K0)
        if K.shape != (self.dim, self.dim):
            raise ValueError(f"K0 must be shape {(self.dim, self.dim)}.")

        if phase_diag is None:
            diag = None
        else:
            N = phase_diag.full() if isinstance(phase_diag, qt.Qobj) else np.asarray(phase_diag)
            if N.ndim == 2:
                # Accept a full operator, but only if it really is diagonal --
                # the O(d^2) dU/dphi formula depends on it.
                if np.max(np.abs(N - np.diag(np.diag(N)))) > 1e-12:
                    raise ValueError("phase_diag must be diagonal.")
                N = np.diag(N)
            diag = np.real(N).copy()

        self.gates[name] = FrameGate(name, K, diag, phase_sign)
        # Only rebuild if this gate is actually in use; otherwise registering a
        # gate would pointlessly invalidate the current sequence.
        if name in self.unitary_strings:
            self._rebuild_sequence()

    def _rebuild_sequence(self) -> None:
        """Expand ``unitary_strings x num_apply`` into the concrete gate list.

        Called whenever the structure changes.  The list holds references to the
        shared ``FrameGate`` objects, so repeating a gate costs nothing extra --
        all ten beamsplitters in a ten-gate sequence share one eigendecomposition.
        """
        seq: List[FrameGate] = []
        for _ in range(self.num_apply):
            for s in self.unitary_strings:
                if s not in self.gates:
                    raise ValueError(f"Unknown gate {s!r}. Known: {sorted(self.gates)}")
                seq.append(self.gates[s])
        self._seq = seq
        self.n_gates = len(seq)          # M: total gate count = n_base * num_apply
        # Which gates actually have a phase parameter; a mixed sequence is legal.
        self._phase_mask = np.array([g.has_phase for g in seq], dtype=bool)

    def configure(self, *, num_apply: Optional[int] = None,
                  optimize_phases: Optional[bool] = None) -> None:
        """Change the repetition count and/or whether phases are free.

        Both change ``n_core``, so any previously optimized parameter vector
        stops being valid.  Use this to sweep circuit depth without rebuilding
        the class (and re-doing every gate's eigendecomposition).
        """
        if num_apply is not None:
            self.num_apply = int(num_apply)
        if optimize_phases is not None:
            self.optimize_phases = bool(optimize_phases)
        self._rebuild_sequence()

    # -- ansatz interface -------------------------------------------------
    @property
    def n_core(self) -> int:
        """One rotation per gate, plus one phase per gate if phases are free."""
        return self.n_gates * (2 if self.optimize_phases else 1)

    def _split_core(self, x_core: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split the core vector into ``(rotations, phases)``.

        When phases are frozen they are not in the vector at all, so an array of
        zeros is synthesized -- that way everything downstream can assume both
        arrays exist and has no ``optimize_phases`` branch of its own.
        """
        x_core = np.asarray(x_core, dtype=float)
        rot = x_core[:self.n_gates]
        phase = (x_core[self.n_gates:2 * self.n_gates] if self.optimize_phases
                 else np.zeros(self.n_gates))
        return rot, phase

    def _core_as_sequence(self, x_core: np.ndarray) -> np.ndarray:
        """View the core vector as ``(n_gates, n_per_gate)`` for sequence penalties.

        Column 0 is the rotations, column 1 (if present) the phases, so a
        smoothness penalty sees consecutive gates as neighbouring rows.
        """
        rot, phase = self._split_core(x_core)
        return (np.stack([rot, phase], axis=1) if self.optimize_phases
                else rot.reshape(-1, 1))

    def _sequence_as_core(self, arr: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`_core_as_sequence`.

        Overridden because the core layout groups all rotations before all
        phases, whereas the sequence view interleaves them -- so mapping back is
        a transpose-then-flatten, not the base class's plain flatten.
        """
        arr = np.asarray(arr, dtype=float)
        return arr.T.reshape(-1) if self.optimize_phases else arr.reshape(-1)

    def _forward(self, x_core: np.ndarray, need_grad: bool) -> Dict[str, Any]:
        """Evaluate every gate in the sequence and the prefix products.

        Unlike the GRAPE ansatz, each gate's exact derivatives come straight out
        of :meth:`FrameGate.evaluate`, so they are simply cached here and the
        backward sweep only has to contract them against the co-state.
        """
        rot, phase = self._split_core(x_core)
        M, d = self.n_gates, self.dim
        U_list: List[np.ndarray] = [None] * M          # type: ignore[list-item]
        dU_r: List[Optional[np.ndarray]] = [None] * M  # dU_j / d(rotation_j)
        dU_p: List[Optional[np.ndarray]] = [None] * M  # dU_j / d(phase_j)

        for j, gate in enumerate(self._seq):
            U, ur, up = gate.evaluate(float(rot[j]), float(phase[j]), need_grad)
            U_list[j] = U
            dU_r[j] = ur
            dU_p[j] = up

        # prefix[j] = U_{j-1} ... U_0, prefix[0] = I; gate 0 acts first
        prefix: List[np.ndarray] = []
        if need_grad:
            I = np.eye(d, dtype=np.complex128)
            prefix = [I]
            for j in range(M):
                prefix.append(U_list[j] @ prefix[j])
            U_final = prefix[M]
        else:
            U_final = np.eye(d, dtype=np.complex128)
            for j in range(M):
                U_final = U_list[j] @ U_final

        return {"U_list": U_list, "prefix": prefix, "U_final": U_final,
                "n_slices": M, "dU_r": dU_r, "dU_p": dU_p}

    def _slice_dc(self, cache, j: int, L: np.ndarray, out: np.ndarray) -> None:
        """Contract the co-state with gate j's two derivatives (both O(d^2)).

        Core layout is ``[rot_0..rot_{M-1}, phase_0..phase_{M-1}]``, so the phase
        entry for gate j sits at ``n_gates + j``.  A gate registered without a
        phase parameter simply contributes nothing there.
        """
        out[j] += _trace_prod(L, cache["dU_r"][j])
        if self.optimize_phases and cache["dU_p"][j] is not None:
            out[self.n_gates + j] += _trace_prod(L, cache["dU_p"][j])

    def _core_to_natural(self, x_core: np.ndarray) -> Dict[str, Any]:
        """Expose the optimized parameters as separate rotation/phase arrays.

        ``phase_opt`` is present even when phases were frozen (all zeros), so
        downstream code and saved files have a uniform shape.
        """
        rot, phase = self._split_core(x_core)
        return {"rotation_opt": rot, "phase_opt": phase}

    def _natural_to_core(self, rotation=None, phase=None) -> np.ndarray:
        """Pack rotations (and phases, if free) into the core layout."""
        if rotation is None:
            raise TypeError("ParameterOC needs rotation (positional or rotation=...).")
        rot = np.asarray(rotation, dtype=float).reshape(-1)
        if rot.size != self.n_gates:
            raise ValueError(f"rotation must have {self.n_gates} entries, got {rot.size}.")
        if not self.optimize_phases:
            return rot                       # phases are not part of the vector
        # A phase array is accepted even when the caller omits it, so that
        # passing only rotations means "all phases zero".
        ph = (np.zeros(self.n_gates) if phase is None
              else np.asarray(phase, dtype=float).reshape(-1))
        if ph.size != self.n_gates:
            raise ValueError(f"phase must have {self.n_gates} entries, got {ph.size}.")
        return np.concatenate([rot, ph])

    def _natural_x(self, rotation=None, phase=None, theta=None) -> np.ndarray:
        """Pack ``(rotation, phase, theta)`` into the full flat vector."""
        return self._pack(self._natural_to_core(rotation, phase),
                          self._check_theta0(theta))

    def _metadata(self) -> Dict[str, Any]:
        """Enough to rebuild the circuit from a saved run.

        ``n_gates`` is redundant (n_base * num_apply) but saved so the parameter
        arrays can be interpreted without recomputing it.
        """
        return {"unitary_strings": list(self.unitary_strings), "n_base": self.n_base,
                "num_apply": self.num_apply, "n_gates": self.n_gates,
                "optimize_phases": self.optimize_phases,
                "n_c": self.n_c, "n_q": self.n_q}

    def _qobjs_to_save(self) -> Dict[str, Any]:
        """The ladder operators, so saved runs can be re-propagated from disk."""
        return {"a_c": self.a_c, "a_q": self.a_q}

    # -- public API ---------------------------------------------------
    def optimize(
        self,
        rotation0: Optional[np.ndarray] = None,
        phase0: Optional[np.ndarray] = None,
        theta0: Optional[np.ndarray] = None,
        *,
        num_apply: Optional[int] = None,
        optimize_phases: Optional[bool] = None,
        rotation_bounds: Optional[Tuple[float, float]] = (0.0, np.pi),
        phase_bounds: Optional[Tuple[float, float]] = (0.0, 2 * np.pi),
        gauge_bounds: Optional[Tuple[float, float]] = (0.0, 2 * np.pi),
        method: str = "multistart",
        n_starts: int = 8,
        **kwargs,
    ) -> Dict[str, Any]:
        """Optimize the gate rotation angles, phases and gauge angles.

        Parameters
        ----------
        rotation0, phase0:
            Initial values, one per gate (length ``n_gates``).  Default zeros --
            but zeros are a poor start here (the gradient nearly vanishes), so
            prefer random values or a global ``method``.
        theta0:
            Initial gauge angles, length ``n_gauge``.  Default zeros.
        num_apply, optimize_phases:
            Optional structure overrides; passing either reconfigures the
            sequence before optimizing, so circuit depth can be swept from a
            loop without rebuilding the class.
        rotation_bounds, phase_bounds, gauge_bounds:
            A single ``(lo, hi)`` per group, applied to every parameter in that
            group.  Unlike GRAPE these default to finite ranges, because the
            parameters are angles with a natural period and because the global
            methods need a finite box to sample.
        method:
            Defaults to ``"multistart"``.  This landscape is rugged, so a global
            method matters: ``"dual_annealing"`` measured best, then ``"ils"``,
            with a single local descent about 30x worse.  See README.md.
        n_starts:
            Number of restarts for ``multistart``/``ils``.
        **kwargs:
            Forwarded to :meth:`_run_optimize`.

        Returns
        -------
        The result dict described in :meth:`_run_optimize`, including
        ``"rotation_opt"``, ``"phase_opt"`` and ``"theta_opt"``.
        """
        # Structure first: this can change n_gates, which everything below uses.
        if num_apply is not None or optimize_phases is not None:
            self.configure(num_apply=num_apply, optimize_phases=optimize_phases)

        M = self.n_gates
        rot0 = np.zeros(M) if rotation0 is None else np.asarray(rotation0, dtype=float)
        if rot0.shape != (M,):
            raise ValueError(f"rotation0 must be shape {(M,)}.")
        ph0 = np.zeros(M) if phase0 is None else np.asarray(phase0, dtype=float)
        if ph0.shape != (M,):
            raise ValueError(f"phase0 must be shape {(M,)}.")

        theta0 = self._check_theta0(theta0)

        # One bound per variable, in the packed order: rotations, then phases
        # (if free), then gauge angles.
        bounds = [tuple(rotation_bounds)] * M
        if self.optimize_phases:
            bounds += [tuple(phase_bounds)] * M
        bounds += [tuple(gauge_bounds)] * self.n_gauge

        x0 = self._pack(self._natural_to_core(rotation=rot0, phase=ph0), theta0)
        return self._run_optimize(x0, bounds, method=method, n_starts=n_starts, **kwargs)

    # -- introspection --------------------------------------------------
    def gate_unitaries(self, rotation: np.ndarray, phase: Optional[np.ndarray] = None
                       ) -> List[qt.Qobj]:
        """The individual gate unitaries, in application order (index 0 acts first).

        Useful for inspecting what the circuit does step by step, or for feeding
        the gates one at a time into a lossy time-domain simulation.
        """
        rot = np.asarray(rotation, dtype=float)
        ph = np.zeros(self.n_gates) if phase is None else np.asarray(phase, dtype=float)
        # [0] picks U out of (U, dU/dr, dU/dphi); need_grad=False so the
        # derivatives are never computed.
        return [qt.Qobj(g.evaluate(float(rot[j]), float(ph[j]), False)[0], dims=self.dims)
                for j, g in enumerate(self._seq)]


def _sigma_plus_01(a_q: qt.Qobj) -> qt.Qobj:
    """Build sigma_+ = |1><0| on the qubit, embedded in the full tensor space.

    The qubit is taken to be the *last* tensor factor, matching the (cavity,
    qubit) ordering used throughout.  Restricting to the 0-1 subspace (rather
    than using the full a_q^dag) is what makes the "r" gate a genuine two-level
    rotation even when the transmon is simulated with more levels -- population
    driven to level 2 and above is then leakage, and is penalized.
    """
    subdims = a_q.dims[0]                    # e.g. [n_c, n_q]
    n_q = subdims[-1]
    if n_q < 2:
        raise ValueError("Need at least 2 qubit levels to define |1><0|.")
    local = qt.basis(n_q, 1) * qt.basis(n_q, 0).dag()      # |1><0| on the qubit alone
    # Identity on every other factor, |1><0| on the last.
    return qt.tensor(*[qt.qeye(d) for d in subdims[:-1]], local)


# =============================================================================
# Optimizer catalogue
# =============================================================================

_METHOD_DOCS: Dict[str, str] = {
    "L-BFGS-B": "Bounded quasi-Newton, analytic gradient. Fast local refinement.",
    "TNC": "Truncated Newton, bounded, analytic gradient.",
    "SLSQP": "Sequential least squares, bounded, analytic gradient.",
    "trust-constr": "Trust-region, bounded, analytic gradient. Robust but slower.",
    "BFGS": "Unbounded quasi-Newton, analytic gradient.",
    "CG": "Unbounded conjugate gradient.",
    "multistart": "n_starts independent L-BFGS-B runs from Sobol/LHS/random points. "
                  "Best general-purpose global strategy here.",
    "ils": "Iterated local search: L-BFGS-B from a perturbation of the running best. "
           "Good when good minima cluster.",
    "basinhopping": "Monte-Carlo hopping between L-BFGS-B local minima (uses gradient).",
    "dual_annealing": "Generalized simulated annealing with gradient-based local search.",
    "differential_evolution": "Population-based global search (gradient-free) + "
                              "gradient polish.",
    "shgo": "Simplicial homology global optimization with gradient-based local "
            "search. Only usable up to ~15 variables (the complex blows up).",
    "cma": "CMA-ES (needs the optional 'cma' package) + gradient polish.",
}


def available_methods(verbose: bool = False) -> List[str]:
    """List the optimizer names accepted by ``method=``.

    Pass ``verbose=True`` to also print a one-line description of each.  For the
    measured comparison on real problems -- and which one to actually pick --
    see README.md and ``benchmark_optimizers.py``.

    Note that ``"cma"`` is listed even when the optional ``cma`` package is not
    installed; selecting it then raises an ``ImportError`` explaining how to get
    it, rather than the name silently disappearing.
    """
    if verbose:
        for k, v in _METHOD_DOCS.items():
            print(f"  {k:24s} {v}")
    return list(_METHOD_DOCS)


# =============================================================================
# Saving / loading helpers
# =============================================================================

def _jsonify(obj: Any) -> Any:
    """Recursively coerce a value into something ``json.dumps`` accepts.

    Metadata comes from the user (``extra=...``) and from NumPy, so it routinely
    contains ``np.float64``, arrays and the occasional un-serializable object.
    Anything unrecognized is ``repr``'d rather than raising: losing the exact
    type of a note is much better than losing the whole saved run to a
    TypeError after an hour of optimization.
    """
    if isinstance(obj, dict):
        # str(k) because JSON keys must be strings, and NumPy scalars are common.
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Checked after the NumPy cases because np.bool_ is not a Python bool and
    # np.float64 IS a float subclass -- order matters here.
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)                          # last resort: keep something readable


def _qsave_safe(obj: Any, qdir: Path, name: str) -> None:
    """Write a Qobj -- or a list of them -- with QuTiP's own serializer.

    "Safe" in that a ``None`` (an unused projector, no adiabatic unitary) is
    silently skipped, so ``save()`` can hand over its whole operator dict without
    checking each entry.  Lists are numbered ``name_00``, ``name_01``, ... which
    sorts correctly on load.

    QuTiP's format is used rather than NumPy because it preserves ``dims``, the
    tensor-product structure that a bare matrix would lose.
    """
    if obj is None:
        return
    if isinstance(obj, qt.Qobj):
        qt.qsave(obj, str(qdir / name))
        return
    if isinstance(obj, (list, tuple)) and all(isinstance(x, qt.Qobj) for x in obj):
        for i, x in enumerate(obj):
            qt.qsave(x, str(qdir / f"{name}_{i:02d}"))


def load_run(folder: Union[str, Path]) -> Dict[str, Any]:
    """Load a run written by :meth:`OptimalControl.save`.

    Returns
    -------
    A dict with every saved array (``x_opt``, ``pulses_opt`` or
    ``rotation_opt``/``phase_opt``, ``theta_opt``, ``history_fidelity``), plus
    ``"metadata"`` (the parsed JSON) and ``"qobjs"`` (the QuTiP operators).

    Together those are enough to rebuild the unitary and re-check the fidelity
    without the notebook that produced them -- see the round-trip test.
    """
    d = Path(folder)
    if not d.is_dir():
        raise FileNotFoundError(f"{d} is not a directory. For the old single-file "
                                f"format use load_qobjs()/load_pulses_theta().")
    out: Dict[str, Any] = {}
    # allow_pickle=False on purpose: loading a run should never be able to
    # execute code.  Everything saved is a plain array or a JSON string.
    with np.load(d / "results.npz", allow_pickle=False) as data:
        for k in data.files:
            out[k] = data[k]

    # Prefer the copy embedded in the npz; fall back to the standalone file.
    if "metadata_json" in out:
        out["metadata"] = json.loads(str(out.pop("metadata_json")))
    elif (d / "metadata.json").exists():
        out["metadata"] = json.loads((d / "metadata.json").read_text())

    # An empty theta array was the on-disk encoding of "no gauge"; restore None.
    if out.get("theta_opt") is not None and np.size(out["theta_opt"]) == 0:
        out["theta_opt"] = None

    out["qobjs"] = load_qobjs(d / "qobjs")
    return out


def load_qobjs(qdir: Union[str, Path]) -> Dict[str, Any]:
    """Load the QuTiP operators saved with a run.

    Accepts any of the ways you might naturally refer to a run::

        load_qobjs("results/run2")            # a new run folder
        load_qobjs("results/run2/qobjs")      # its qobjs/ subfolder
        load_qobjs("results/run1.npz")        # an old-style file
        load_qobjs("results/run1")            # ... or its stem

    Every key is always present; entries that were not saved come back as
    ``None`` (or an empty list), so callers can just read what they need.
    """
    p = Path(qdir)

    # A new run folder: the operators live one level down, in qobjs/.  Checked
    # first because the run folder is the obvious thing for a caller to name.
    if p.is_dir() and (p / "qobjs").is_dir():
        p = p / "qobjs"
    # Legacy layout: either the caller handed us the .npz itself, or a stem whose
    # .npz sibling exists.  Map it to <stem>_qobjs.
    elif p.suffix == ".npz" or (not p.is_dir() and p.with_suffix(".npz").exists()):
        p = p.with_suffix("")
        p = p.parent / f"{p.name}_qobjs"

    if not p.is_dir():
        raise FileNotFoundError(
            f"No operators found for {qdir!r}: expected a run folder containing "
            f"qobjs/, a qobjs/ folder, or a legacy <name>.npz with a sibling "
            f"<name>_qobjs/ folder.")

    out: Dict[str, Any] = {}

    def _one(name):
        """Load a single optional operator, or None if it was never saved."""
        f = p / f"{name}.qu"
        # qload wants the stem, exists() wants the suffix -- hence both forms.
        return qt.qload(str(p / name)) if f.exists() else None

    def _many(prefix):
        """Load a numbered series (gauge_op_00, gauge_op_01, ...) in order."""
        files = sorted(p.glob(f"{prefix}_*.qu"))
        return [qt.qload(str(f.with_suffix(""))) for f in files]

    for name in ("U_target", "projector", "adiabatic_unitary", "H_drift", "a_c", "a_q"):
        out[name] = _one(name)
    out["gauge_ops"] = _many("gauge_op")
    out["H_controls"] = _many("H_control")
    return out


def load_parameters_dict(path: Union[str, Path]) -> Dict[str, Any]:
    """Just the metadata of a run, from either a new folder or an old ``.npz``."""
    p = Path(path)
    if p.is_dir():
        return load_run(p)["metadata"]
    # Legacy path.  allow_pickle=True is needed because the old saver stored the
    # JSON as a 0-d object array; "parameters_json" was its key.
    with np.load(p, allow_pickle=True) as data:
        for key in ("metadata_json", "parameters_json"):
            if key in data.files:
                # A 0-d object array needs .item(); a 0-d str array does not.
                return json.loads(str(data[key].item()) if data[key].dtype == object
                                  else str(data[key]))
    raise KeyError(f"No metadata found in {p}.")


def load_pulses_theta(path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """``(pulses, theta)`` from a new run folder or an old GRAPE ``.npz``."""
    p = Path(path)
    if p.is_dir():
        r = load_run(p)
        return r["pulses_opt"], r.get("theta_opt")
    with np.load(p, allow_pickle=True) as data:
        pulses = data["pulses_opt"]
        theta = data["theta_opt"]
    # Both formats encode "no gauge" as an empty array.
    return pulses, (None if np.size(theta) == 0 else theta)


def load_rotation_phase_theta(path: Union[str, Path]):
    """``(rotation, phase, theta)`` from a new run folder or an old ``.npz``."""
    p = Path(path)
    if p.is_dir():
        r = load_run(p)
        return r["rotation_opt"], r["phase_opt"], r.get("theta_opt")
    with np.load(p, allow_pickle=True) as data:
        rot = data["rotation_opt"]
        ph = data["phase_opt"]
        theta = data["theta_opt"]
    return rot, ph, (None if np.size(theta) == 0 else theta)


# =============================================================================
# Stand-alone propagation helpers (QuTiP in, QuTiP out)
#
# These reproduce the forward propagation without needing an optimizer object,
# for analysis and for re-checking a saved run from its files alone.  They are
# written for clarity over speed -- do not use them inside an optimization loop.
# =============================================================================

def propagate_pulses(H0: qt.Qobj, Hc: Sequence[qt.Qobj], dt: float,
                     pulses: np.ndarray, hbar: float = 1.0) -> qt.Qobj:
    """``U = prod_n exp(-i dt (H0 + sum_k u[n,k] Hk) / hbar)``, time-ordered.

    The GRAPE forward pass, independent of :class:`GrapeOC`.  Uses ``expm``
    directly rather than the spectral shortcut, which makes it a genuinely
    independent check of the optimizer's own propagation.
    """
    pulses = np.asarray(pulses, dtype=float)
    U = np.eye(H0.shape[0], dtype=np.complex128)
    H0d = H0.full()
    Hcd = [h.full() for h in Hc]
    for n in range(pulses.shape[0]):
        Hn = H0d.copy()                        # copy: H0d must not be mutated
        for k in range(len(Hcd)):
            Hn += pulses[n, k] * Hcd[k]
        # Left-multiply: slice n acts after everything already accumulated.
        U = expm((-1j * dt / hbar) * Hn) @ U
    return qt.Qobj(U, dims=H0.dims)


def propagate_gates(unitary_strings: Sequence[str], rotation: np.ndarray,
                    phase: Optional[np.ndarray], num_apply: int,
                    a_c: qt.Qobj, a_q: qt.Qobj,
                    optimize_phases: bool = True) -> qt.Qobj:
    """Time-ordered product of the named gate sequence, repeated ``num_apply`` times.

    Builds a throwaway :class:`ParameterOC` purely to reuse its gate registry and
    forward pass -- the target unitary is irrelevant here, so the identity is
    passed as a placeholder.
    """
    n_c, n_q = a_c.dims[0]
    dummy = qt.qeye(a_c.dims[0])               # placeholder target; never used
    tmp = ParameterOC(list(unitary_strings), dummy, n_c, n_q,
                      num_apply=num_apply, optimize_phases=optimize_phases)
    rot = np.asarray(rotation, dtype=float)
    ph = (np.asarray(phase, dtype=float) if (phase is not None and optimize_phases)
          else np.zeros(tmp.n_gates))
    core = tmp._natural_to_core(rotation=rot, phase=ph)
    return qt.Qobj(tmp._forward(core, need_grad=False)["U_final"], dims=a_c.dims)


def gauge_unitary(gauge_ops: Sequence[qt.Qobj], theta: np.ndarray) -> qt.Qobj:
    """``G(theta) = exp(+i sum_j theta_j A_j)`` as a QuTiP Qobj.

    Reuses :class:`_GaugeBlock`, so the same diagonal/commuting fast paths apply.
    Building the block costs one classification pass, which is negligible for a
    one-off call but means you should not call this in a loop.
    """
    if len(gauge_ops) == 0:
        raise ValueError("gauge_ops is empty.")
    block = _GaugeBlock(gauge_ops, gauge_ops[0].shape[0])
    G, _ = block(np.asarray(theta, dtype=float), need_grad=False)
    return qt.Qobj(G, dims=gauge_ops[0].dims)
