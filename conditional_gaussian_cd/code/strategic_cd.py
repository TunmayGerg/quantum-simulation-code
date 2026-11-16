"""Magnus-motivated CD recipes with a constant sideband plateau.

Units are microseconds and radians.  Every qubit envelope occupies 30 ns;
fractional rotations reduce its amplitude rather than its duration.  Search
scores use exact native sample matrices; accepted recipes are independently
replayed from SDK words after the full RFDC interpolation response.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import time
import itertools
from functools import lru_cache
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, brentq, linprog, least_squares
from project_paths import ROOT
import cd_trains as cd
import hardware_check as hw

OUT = ROOT / 'runs/strategic_cd'
OUT.mkdir(parents=True, exist_ok=True)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1., -1.])
SP = (X - 1j*Y)/2
MAX_G = np.pi/(2*.200)


def status(message):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    (OUT/'STATUS.md').write_text(f'# Strategic CD search\n\nUpdated {stamp}.\n\n{message}\n')
    print(message, flush=True)


def inverse_mirror(n_outer=1, n_inner=1, psi=np.pi/4, dwell_samples=4):
    """B,X,B,X; B=[X_a,psi_b,(psi+pi)_b,(-X)_a].

    The dwell in the first X frame balances the 5 ns combined outer-ramp
    area.  The sideband remains at one complex value on the entire plateau.
    """
    block = [(0., 1/n_outer), (psi, 1/n_inner),
             (psi+np.pi, 1/n_inner), (np.pi, 1/n_outer)]
    return block + [(0., 1.), ('idle', dwell_samples)] + block + [(0., 1.)]


def raw_waveform(recipe, g, pump_phase, centered=False):
    q = [0j]*4
    pi_shape = (np.pi/cd.PI_TIME*np.sin(np.pi*(np.arange(24)+.5)/24)**2
                if centered else cd.pi_samples())
    for axis, amount in recipe:
        q.extend([0j]*int(amount) if axis == 'idle' else pi_shape*amount*np.exp(1j*axis))
    q.extend([0j]*4)
    q = np.asarray(q, complex)
    s = np.full(len(q), g*np.exp(1j*pump_phase), complex)
    if centered:
        ramp=np.sin(np.pi*(np.arange(4)+.5)/8)**2
        s[:4]*=ramp;s[-4:]*=ramp[::-1]
    else:
        s[:4] *= np.sin(np.pi*np.arange(4)/8)**2
        s[-4:] *= np.cos(np.pi*np.arange(4)/8)**2
    assert len(q)%4 == 0
    return q, s


def first_magnus(recipe, centered=False):
    """Integrate Q(t)^dag sigma+ Q(t), including the outer pump ramps.

    Four-point Gauss integration within each constant DAC sample computes
    the coefficient multiplying a in the first averaged interaction.
    """
    q, s = raw_waveform(recipe, 1., 0., centered)
    nodes, weights = np.polynomial.legendre.leggauss(4)
    Q = np.eye(2, dtype=complex)
    M = np.zeros((2, 2), complex)
    for drive, pump in zip(q, s):
        h = drive*SP + drive.conjugate()*SP.conj().T
        for u, w in zip((nodes+1)/2, weights/2):
            T = expm(-1j*h*(u*cd.DT))@Q
            M += pump*cd.DT*w*(T.conj().T@SP@T)
        Q = expm(-1j*h*cd.DT)@Q
    coefficients = np.array([np.trace(A.conj().T@M)/2 for A in [np.eye(2), X, Y, Z]])
    global_phase = np.angle(np.trace(Q)/2)
    closure = np.linalg.norm(Q*np.exp(-1j*global_phase)-np.eye(2))
    return M, coefficients, float(closure)


@lru_cache(maxsize=None)
def native_qubit_moment(fraction=1., centered=False):
    """One pulse's exact zero-coupling frame and first sideband moment."""
    recipe = [(0., fraction)]
    M, _, _ = first_magnus(recipe, centered)
    # first_magnus includes both outer ramps; remove them in their frames.
    Q = expm(-1j*X*np.pi*fraction/2)
    up_area = .0025 if centered else cd.DT*sum(np.sin(np.pi*np.arange(4)/8)**2)
    down_area = .0025 if centered else cd.DT*sum(np.cos(np.pi*np.arange(4)/8)**2)
    return Q, M-up_area*SP-down_area*(Q.conj().T@SP@Q)


def fast_moment(recipe, centered=False):
    Q = np.eye(2, dtype=complex)
    M = (.0025 if centered else cd.DT*sum(np.sin(np.pi*np.arange(4)/8)**2))*SP
    for phi, amount in recipe:
        if phi == 'idle':
            M += amount*cd.DT*(Q.conj().T@SP@Q)
        else:
            Q0, M0 = native_qubit_moment(float(amount),centered)
            D = np.diag(np.exp(np.array([-1., 1.])*1j*phi/2))
            pulse_Q = D@Q0@D.conj().T
            pulse_M = np.exp(-1j*phi)*D@M0@D.conj().T
            M += Q.conj().T@pulse_M@Q
            Q = pulse_Q@Q
    M += (.0025 if centered else cd.DT*sum(np.cos(np.pi*np.arange(4)/8)**2))*(Q.conj().T@SP@Q)
    return np.array([np.trace(A.conj().T@M)/2 for A in [X, Y, Z]])


def symmetric_recipe(half_phases, idle_samples=None, fractions=None):
    phases = np.r_[half_phases, np.asarray(half_phases)[::-1]+np.pi]
    n = len(phases)
    fractions = np.ones(n) if fractions is None else np.r_[fractions, fractions[::-1]]
    idle_samples = np.zeros(n+1, int) if idle_samples is None else np.asarray(idle_samples, int)
    recipe = []
    for j in range(n+1):
        if idle_samples[j]:
            recipe.append(('idle', int(idle_samples[j])))
        if j<n:
            recipe.append((float(phases[j]), float(fractions[j])))
    return recipe


class NativeRecipe:
    """Exact DAC-hold propagators, reused for each qubit pulse phase."""
    def __init__(self, swap_ns, levels=28, centered=False):
        self.g = np.pi/(2*swap_ns*.001)
        self.levels = levels
        self.centered=centered
        self.space = cd.Space(levels, 9)
        self.target = cd.target_columns(self.space, 'cd', 1.)
        self.initial = np.eye(2*levels, dtype=complex)[:, self.space.inputs]
        H = self.space.hamiltonian(self.g, 0.)
        uparea=.0025 if centered else sum(np.sin(np.pi*np.arange(4)/8)**2)*cd.DT
        downarea=.0025 if centered else sum(np.cos(np.pi*np.arange(4)/8)**2)*cd.DT
        self.up = expm(-1j*H*uparea)
        self.down = expm(-1j*H*downarea)
        self.idle_sample = expm(-1j*H*cd.DT)
        self.cells = {}

    def cell(self, fraction):
        if fraction not in self.cells:
            U = np.eye(2*self.levels, dtype=complex)
            shape=(np.pi/cd.PI_TIME*np.sin(np.pi*(np.arange(24)+.5)/24)**2
                   if self.centered else cd.pi_samples())
            for omega in shape*fraction:
                eigenvalues, eigenvectors = np.linalg.eigh(self.space.hamiltonian(self.g, omega))
                U = (eigenvectors*np.exp(-1j*cd.DT*eigenvalues))@eigenvectors.T@U
            self.cells[fraction] = U
        return self.cells[fraction]

    def output(self, recipe, pump_phase):
        # A constant pump phase is one oscillator conjugation for the whole
        # recipe.  It does not change during a single gate.
        pump_D = np.exp(-1j*pump_phase*self.space.n)[:, None]
        out = self.up@(pump_D.conj()*self.initial)
        for axis, amount in recipe:
            if axis == 'idle':
                out = np.linalg.matrix_power(self.idle_sample, int(amount))@out
            else:
                d = np.exp(1j*axis*self.space.k)[:, None]
                out = d*(self.cell(amount)@(d.conj()*out))
        return pump_D*(self.down@out)

    def fidelity(self, recipe, pump_phase):
        out = self.output(recipe, pump_phase)
        return float(abs(np.vdot(self.target, out)/18)**2)


def validate():
    rows = []
    for n in [1, 2, 3, 4, 6, 8]:
        for psi in np.linspace(0., np.pi, 9):
            _, c, closure = first_magnus(inverse_mirror(n, 1, psi))
            rows.append(dict(n_outer=n, psi=float(psi), coefficient_real=c.real.tolist(),
                             coefficient_imag=c.imag.tolist(), closure=closure,
                             unwanted=float(np.linalg.norm(c[[0, 2, 3]]))))
    assert len(rows) == 54
    assert max(r['unwanted'] for r in rows) < 2e-14
    assert max(r['closure'] for r in rows) < 2e-13
    (OUT/'first_magnus_validation.json').write_text(json.dumps(rows, indent=2)+'\n')
    status(f'Completed first-Magnus check: {len(rows)} closed finite-pulse recipes; '
           f"largest unwanted Pauli coefficient {max(r['unwanted'] for r in rows):.3g} us; "
           f"largest qubit closure error {max(r['closure'] for r in rows):.3g}. "
           'Searching target-beta1 recipes next; no fidelity claim yet.')


def search_inverse(swap_ns=200.):
    """Search one continuous block angle and its global pump orientation.

    The integer rotation denominators are explicit finite enumeration.  The
    target remains beta1 in every objective; no fitted target displacement.
    """
    model = NativeRecipe(swap_ns)
    rows = []
    evaluations = 0
    # Uniform additional idle time is balanced between I and X frames.
    # At this stage the minimal315ns recipe is screened atstrong coupling.
    for n_outer in [1, 2, 3, 4, 6, 8]:
        for n_inner in [1, 2, 3]:
            for seed_psi in [np.pi/4, 3*np.pi/4]:
                def objective(p):
                    nonlocal evaluations
                    evaluations += 1
                    recipe = inverse_mirror(n_outer, n_inner, p[0])
                    return 1-model.fidelity(recipe, p[1])
                recipe = inverse_mirror(n_outer, n_inner, seed_psi)
                _, coefficients, _ = first_magnus(recipe)
                pump_phase = -np.pi/2-np.angle(coefficients[1])
                result = minimize(objective, [seed_psi, pump_phase], method='Nelder-Mead',
                                  options=dict(maxiter=100, xatol=1e-7, fatol=1e-10))
                row = dict(n_outer=n_outer, n_inner=n_inner, psi=float(result.x[0]),
                           pump_phase=float(result.x[1]), native_fidelity=float(1-result.fun),
                           swap_ns=swap_ns, command_us=.315, evaluations=result.nfev)
                rows.append(row)
                print('INVERSE MIRROR', row, flush=True)
                (OUT/f'inverse_swap{swap_ns:g}.json').write_text(json.dumps(rows, indent=2)+'\n')
    assert len(rows) == 36
    best = max(rows, key=lambda r:r['native_fidelity'])
    status(f'Completed {len(rows)} inverse-mirror fits / {evaluations} exact native gate evaluations '
           f'({18*evaluations} input columns). Best beta1 native fidelity {best["native_fidelity"]:.10f}, '
           f'315ns, swap{swap_ns:g}ns, denominators {best["n_outer"]}/{best["n_inner"]}. '
           'This is a screening score; SDK/RFDC and cutoff checks pending.')
    return best


def balanced_word_search(swap_ns=200., cells=8):
    """Enumerate true inverse-mirror pi words and symmetric idle allocations.

    q(T-t)=-q(t) makes the qubit interaction frame retrace itself, so the
    toggling-frame Hamiltonian is time symmetric and its second Magnus term
    vanishes (up to the start-of-sample discretization).  Balanced signed
    half-turns also remove the transverse first-order term.  Equal I/X idle
    weights preserve the desired force; the extra X weight balances ramps.
    """
    model = NativeRecipe(swap_ns, 32)
    rows = []
    # The first-order force area is g/2*(30N+5+idle_ns).
    required_ns = 2/model.g*1000-30*cells-5
    # Raw sequences must contain a multiple of4samples. Atleast4samples
    # are needed in the X frame to balance the outer pump ramps.
    base_samples = max(4, int(np.ceil(required_ns/(4*cd.DT*1000)))*4)
    if base_samples % 8 != 4:
        base_samples += 4
    # Ramps give an I-frame area4samples; choose totalI=(d-4)/2,
    # totalX=(d+4)/2. Symmetric paired dwells require these totals even.
    half_count = cells//2
    for half in itertools.product([1, -1], repeat=half_count):
        signs = list(half)+[-int(v) for v in half[::-1]]
        # Integral sin(theta) for each signedpi interval.
        if sum((-1)**j*sign for j, sign in enumerate(signs)) != 0:
            continue
        for total_samples in [base_samples, base_samples+8, base_samples+16]:
            i_each = (total_samples-4)//4
            x_each = (total_samples+4)//4
            i_slots = list(range(0, cells//2+1, 2))
            x_slots = list(range(1, cells//2+1, 2))
            for i_slot in i_slots:
                for x_slot in x_slots:
                    idle = np.zeros(cells+1, int)
                    for slot, amount in [(i_slot, i_each), (x_slot, x_each)]:
                        idle[slot] += amount
                        idle[cells-slot] += amount
                    recipe = []
                    for j in range(cells+1):
                        if idle[j]:
                            recipe.append(('idle', int(idle[j])))
                        if j < cells:
                            recipe.append((0. if signs[j]>0 else np.pi, 1.))
                    # A singleconstantg correction is allowed downward from
                    # the requestedmaximumstrength, never a fittedtarget.
                    score = model.fidelity(recipe, -np.pi/2)
                    rows.append(dict(signs=signs, idle_samples=idle.tolist(),
                                     native_fidelity=score, cells=cells,
                                     nominal_swap_ns=swap_ns,
                                     command_us=(30*cells+10+sum(idle)*1.25)*.001))
    assert len(rows)>0
    rows.sort(key=lambda r:r['native_fidelity'], reverse=True)
    (OUT/f'balanced_n{cells}_swap{swap_ns:g}.json').write_text(json.dumps(rows, indent=2)+'\n')
    status(f'Completed {len(rows)} properinverse-mirror CD sign/timing candidates '
           f'({18*len(rows)} inputcolumns), beta1 fixed. BestnativeF {rows[0]["native_fidelity"]:.10f}, '
           f'{rows[0]["command_us"]*1000:.1f}ns, swap{swap_ns:g}ns. '
           'No SDK/RFDC certification yet.')
    return rows[0]


def symmetric_axes_search(swap_ns=200., cells=8):
    """Four-axis inverse mirror: exactclosure; time symmetry cancels Omega2.

    First-moment matching is an explicit optimization constraint rather
    than a fitted target.  The optimizer has only the halfword axes and one
    constant pump quadrature.  Idle positions are a finite discrete scan.
    """
    model = NativeRecipe(swap_ns, 28)
    rng = np.random.default_rng(947)
    rows = []
    native_evaluations = 0
    constraint_evaluations = 0
    base = np.resize([0., 0., np.pi, np.pi], cells)[:cells//2]
    total_idle_start = max(12, int(np.ceil((2/model.g/.001-30*cells-5)/5))*4)
    for total_idle in range(total_idle_start, total_idle_start+49, 8):
        # Symmetry: equal dwellatpairedslots, soeachvariableis2samples.
        idle = np.zeros(cells+1, int)
        idle[1] = idle[-2] = total_idle//2
        for seed in range(3):
            p0 = np.r_[base+(0. if seed==0 else rng.normal(0,.4,cells//2)), -np.pi/2]
            best = [np.inf, p0.copy()]
            def objective(p):
                nonlocal native_evaluations
                native_evaluations += 1
                recipe = symmetric_recipe(p[:-1], idle)
                cost = 1-model.fidelity(recipe, p[-1])
                # Penalty is a strict proper-first-average design condition;
                # its residual is stored independently of the gate fidelity.
                c = np.exp(1j*p[-1])*fast_moment(recipe)*model.g
                residual = c-np.array([-1j, 0., 0.])
                value = cost+5*float(np.vdot(residual, residual).real)
                if value<best[0]:
                    best[:] = [value, p.copy()]
                return value
            result = minimize(objective, p0, method='BFGS',
                              options=dict(maxiter=90, gtol=2e-6))
            p = best[1]
            recipe = symmetric_recipe(p[:-1], idle)
            c = np.exp(1j*p[-1])*fast_moment(recipe)*model.g
            row = dict(half_phases=p[:-1].tolist(), pump_phase=float(p[-1]),
                       idle_samples=idle.tolist(), native_fidelity=model.fidelity(recipe,p[-1]),
                       first_moment_error=float(np.linalg.norm(c-np.array([-1j,0.,0.]))),
                       coefficients_real=c.real.tolist(), coefficients_imag=c.imag.tolist(),
                       swap_ns=swap_ns, command_us=(30*cells+10+total_idle*1.25)*.001,
                       evaluations=result.nfev, seed=seed)
            rows.append(row)
            print('SYMMETRIC AXES',row,flush=True)
            (OUT/f'symmetric_axes_n{cells}_swap{swap_ns:g}.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert len(rows)==21
    best = max(rows,key=lambda r:r['native_fidelity'])
    status(f'Completed21symmetricaxis/timingfits / {native_evaluations}nativeevaluations '
           f'({18*native_evaluations}inputcolumns). Bestbeta1 F{best["native_fidelity"]:.10f} '
           f'T{best["command_us"]*1000:.1f}ns, firstmomenterror{best["first_moment_error"]:.5g}. '
           'Nonzero first-moment residuals are explicitlyreported; no certification.')


def all_sign_search(cells=10, gap_samples=0):
    """Finite enumeration of every pure-X pi word with proper first force.

    Unlike an assumedXX-- repetition, this tests all balanced signed angular
    paths.  The five-ns X-frame idle cancels the transverse outer-ramp area.
    One constantg is chosen from the common beta1 first-moment condition.
    """
    total_ns = 30*cells+5+5+gap_samples*1.25
    swap_ns = np.pi*total_ns/4
    assert swap_ns>=200
    model = NativeRecipe(swap_ns,32)
    rows=[]
    x_slots = list(range(1,cells,2))
    # The totaladditionaldwellisbalancedI/X; useone symmetriclocationpair.
    for signs in itertools.product([1,-1],repeat=cells):
        if signs[0]!=1 or sum((-1)**j*s for j,s in enumerate(signs))!=0:
            continue
        for x_slot in x_slots:
            idle=np.zeros(cells+1,int)
            idle[x_slot]+=4
            if gap_samples:
                idle[x_slot]+=gap_samples//2
                idle[0]+=gap_samples//2
            recipe=[]
            for j in range(cells+1):
                if idle[j]:recipe.append(('idle',int(idle[j])))
                if j<cells:recipe.append((0. if signs[j]>0 else np.pi,1.))
            rows.append(dict(signs=list(signs),idle_samples=idle.tolist(),
                             native_fidelity=model.fidelity(recipe,-np.pi/2),
                             cells=cells,nominal_swap_ns=swap_ns,pump_phase=-np.pi/2,
                             command_us=(30*cells+10+sum(idle)*1.25)*.001))
    rows.sort(key=lambda r:r['native_fidelity'],reverse=True)
    assert len(rows)>0
    tag=f'allsigns_n{cells}_gap{gap_samples}'
    (OUT/(tag+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
    status(f'Completed{len(rows)}balancedpure-Xsignedpaths/timinglocations '
           f'({18*len(rows)}inputcolumns); fixedbeta1, gfromproperfirstmoment. '
           f'BestnativeF{rows[0]["native_fidelity"]:.10f}, T{rows[0]["command_us"]*1000:.1f}ns, '
           f'swap{swap_ns:.4f}ns. SDK/RFDCpending.')
    return rows[0]


def centroid_search(cells=10):
    """Geometric first-force cancellation with discrete pi/N rotations.

    All controls rotate about +/-X, so their desired CD force is constant.
    The unwanted force traces exp(i theta(t)); positive dwell weights at
    angular vertices are found by a two-constraint linear program.  A
    reversed negative second half makes the whole trajectory retrace and
    removes the even Magnus terms before optimizing any gate fidelity.
    """
    half=cells//2
    options=[1.,-1.,.5,-.5]
    rows=[];feasible=0;model_cache={};solved=0
    # A pi/N pulse always occupies30ns. No shorter fractionalenvelopes.
    for signed in itertools.product(options,repeat=half):
        if signed[0]<0:continue
        axes=np.array([0. if v>0 else np.pi for v in signed])
        fractions=np.abs(signed)
        recipe=symmetric_recipe(axes,fractions=fractions)
        c=fast_moment(recipe,True)
        # The first-moment unwanted quadratures are proportional to the
        # cosine/sine centroid of the qubit angularpath.
        residual=np.array([-2*c[1].imag,2*c[2].imag])
        theta=np.r_[0.,np.cumsum(signed)*np.pi]
        A=np.array([np.cos(theta),np.sin(theta)])
        lp=linprog(np.ones(half+1),A_eq=A,b_eq=-residual,
                   bounds=[(0.,None)]*(half+1),method='highs')
        solved+=1
        if not lp.success:continue
        # Each paired idle consists of two equal DACsample counts. Round
        # tothatgrid andstoretheactualremainingfirst-moment residual.
        totals=np.rint(lp.x/(2*cd.DT)).astype(int)*2
        idle=np.zeros(cells+1,int)
        for j,total in enumerate(totals):
            if j==half:idle[j]+=int(total)
            else:
                idle[j]+=int(total//2);idle[cells-j]+=int(total//2)
        # SDKwaveformlength is a multipleof4; addbalancedI/X dwellpair if
        # needed (oppositecirclevertices preservecentroidwhenavailable).
        if sum(idle)%4:
            opposite=next(((j,k) for j in range(half+1) for k in range(j+1,half+1)
                           if np.linalg.norm(A[:,j]+A[:,k])<1e-12),None)
            if opposite is None:continue
            for j in opposite:
                if j==half:idle[j]+=1
                else:idle[j]+=1;idle[cells-j]+=1
            if sum(idle)%4:continue
        duration_ns=30*cells+10+sum(idle)*1.25
        if duration_ns>500:continue
        effective_ns=duration_ns-5
        swap_ns=np.pi*effective_ns/4
        if swap_ns<200:continue
        recipe=symmetric_recipe(axes,idle,fractions)
        coefficient=fast_moment(recipe,True)
        direction_error=float(np.linalg.norm(coefficient[1:])/abs(coefficient[0]))
        if direction_error>.015:continue
        feasible+=1
        if effective_ns not in model_cache:
            model_cache[effective_ns]=NativeRecipe(swap_ns,28,True)
        model=model_cache[effective_ns]
        score=model.fidelity(recipe,-np.pi/2)
        rows.append(dict(signed_half=list(signed),half_phases=axes.tolist(),
                         half_fractions=fractions.tolist(),idle_samples=idle.tolist(),
                         native_fidelity=score,first_direction_error=direction_error,
                         nominal_swap_ns=float(swap_ns),command_us=duration_ns*.001,
                         cells=cells,pump_phase=-np.pi/2,centered_envelopes=True))
    rows.sort(key=lambda r:r['native_fidelity'],reverse=True)
    assert solved==len(options)**half//2
    assert feasible==len(rows) and len(rows)>0
    (OUT/f'centroid_n{cells}.json').write_text(json.dumps(rows,indent=2)+'\n')
    status(f'Geometricpi/Nsearch: {solved}angularpathschecked; {feasible}feasiblequantizedcentroidrecipes '
           f'({18*feasible}inputcolumns). BestnativeF{rows[0]["native_fidelity"]:.10f}, '
           f'T{rows[0]["command_us"]*1000:.1f}ns, swap{rows[0]["nominal_swap_ns"]:.3f}ns; '
           f'firstdirectionresidual{rows[0]["first_direction_error"]:.3g}. SDK/RFDCpending.')
    return rows[0]


def export_recipe(recipe,g,pump_phase,name,centered=False,extra=None):
    q,s=raw_waveform(recipe,g,pump_phase,centered)
    pulses=[];start=4
    for axis,amount in recipe:
        if axis=='idle':start+=int(amount)
        else:
            pulses.append(dict(start_sample=start,samples=24,angle_fraction=amount,
                               phase_rad=axis,reversed_envelope=False))
            start+=24
    np.savez(OUT/(name+'.npz'),qubit=q,sideband=s,dt_us=cd.DT)
    metadata=dict(gate='cd',strength=1.,name=name,pulses=pulses,
                  qubit_frame_Rz=0.,g=float(g),nominal_swap_ns=float(1000*np.pi/(2*g)),
                  pump_phase_rad=float(pump_phase),centered_envelopes=centered,
                  command_us=len(q)*cd.DT,source='code/strategic_cd.py',
                  all_input_columns=18,certified=False)
    if extra:metadata.update(extra)
    (OUT/(name+'.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def export_best10():
    row=json.loads((OUT/'allsigns_n10_gap0.json').read_text())[0]
    recipe=[]
    for j in range(11):
        if row['idle_samples'][j]:recipe.append(('idle',row['idle_samples'][j]))
        if j<10:recipe.append((0. if row['signs'][j]>0 else np.pi,1.))
    swap_ns=239.4387033953878
    export_recipe(recipe,np.pi/(2*swap_ns*.001),-np.pi/2,'best_proper10',False,
                  dict(native_fidelity=.9823297321411291,
                       design='Enumerated proper first-force words;oneconstantgaincalibration',
                       source_word=row['signs'],first_moment_coefficients=str(fast_moment(recipe))))


def fractional11_search():
    """Eleven envelopes: nine pi and two pi/2; qubit closes at g=0.

    This tests a different finite-angle discretization of the rotary echo.
    Every candidate must have an almostclosed forcecentroid before its
    full quantum evolution is scored.  No pump modulation is introduced.
    """
    rows=[];checked=0;scored=0;model_cache={}
    n=11
    for i,j in itertools.combinations(range(n),2):
        for signs_other in itertools.product([1.,-1.],repeat=9):
            signed=np.ones(n);signed[i]=signed[j]=.5
            signed[[k for k in range(n) if k not in [i,j]]]=signs_other
            # sum(signed) is always even: two positivehalfturns plusninepi.
            theta=np.r_[0.,np.cumsum(signed)*np.pi]
            recipe=[(0. if v>0 else np.pi,abs(v)) for v in signed]
            c=fast_moment(recipe,True);checked+=1
            residual=np.array([-2*c[1].imag,2*c[2].imag])
            A=np.array([np.cos(theta),np.sin(theta)])
            lp=linprog(np.ones(n+1),A_eq=A,b_eq=-residual,
                       bounds=[(0.,None)]*(n+1),method='highs')
            if not lp.success:continue
            idle=np.rint(lp.x/cd.DT).astype(int)
            if sum(idle)%4:continue
            duration_ns=30*n+10+sum(idle)*1.25
            if duration_ns>=370:continue
            recipe=[]
            for k in range(n+1):
                if idle[k]:recipe.append(('idle',int(idle[k])))
                if k<n:recipe.append((0. if signed[k]>0 else np.pi,abs(signed[k])))
            c=fast_moment(recipe,True)
            residual=float(np.linalg.norm(c[1:])/abs(c[0]))
            if residual>.008:continue
            swap_ns=np.pi*(duration_ns-5)/4
            if duration_ns not in model_cache:
                model_cache[duration_ns]=NativeRecipe(swap_ns,28,True)
            model=model_cache[duration_ns]
            score=model.fidelity(recipe,-np.pi/2);scored+=1
            if score>.97:
                row=dict(signed_angles=signed.tolist(),idle_samples=idle.tolist(),
                         native_fidelity=score,first_direction_error=residual,
                         nominal_swap_ns=swap_ns,command_us=duration_ns*.001,
                         centered_envelopes=True,cells=n)
                rows.append(row)
        if rows:
            print('FRACTIONAL11',i,j,'checked',checked,'scored',scored,
                  'bestF',max(r['native_fidelity'] for r in rows),flush=True)
    assert checked==55*512
    rows.sort(key=lambda r:r['native_fidelity'],reverse=True)
    (OUT/'fractional11.json').write_text(json.dumps(dict(checked=checked,scored=scored,
                                                       retained=rows),indent=2)+'\n')
    if not rows:
        status(f'Fractional11searchchecked{checked}closedangularpaths andscored{scored}centroidmatchedgates '
               f'({18*scored}inputcolumns). Noneofthesegatesreached97%nativefidelitybelow370ns. '
               'Noacceptedcandidate; thisfinitefamilydoesnotimprovethe12pulsebaseline.')
        return None
    status(f'Fractional11searchchecked{checked}closedangularpaths, scored{scored}centroidmatchedgates '
           f'({18*scored}inputcolumns). BestnativeF{rows[0]["native_fidelity"]:.10f}, '
           f'T{rows[0]["command_us"]*1000:.1f}ns. StillrequiresSDK/RFDCcertification.')
    best=rows[0];recipe=[]
    for j in range(n+1):
        if best['idle_samples'][j]:recipe.append(('idle',best['idle_samples'][j]))
        if j<n:
            v=best['signed_angles'][j];recipe.append((0. if v>0 else np.pi,abs(v)))
    export_recipe(recipe,np.pi/(2*best['nominal_swap_ns']*.001),-np.pi/2,
                  'best_fractional11',True,best)
    return rows[0]


def mixed_axes_constrained(cells=10):
    """Time-symmetric mixed-axis paths with exact first-force constraints.

    The firstmomentis constrained explicitly. Oneconstantpumporientation
    aligns its complexsigma_xcoefficient; oneconstantg matchesbeta1.
    Mixedaxes aretested becauseX-onlypaths have asign-definite cubicerror.
    """
    half=cells//2;rng=np.random.default_rng(7302);rows=[]
    target_min=1/MAX_G
    attempts=0;scored=0;force_maxima=[]
    for idle_ns in [0,10,20,30]:
        idle=np.zeros(cells+1,int)
        idle[0]=idle[-1]=int(idle_ns/2/1.25)
        # Allpulseanglespi; onlythehalfwordaxes arecontinuousparameters.
        def constraints(p):
            c=fast_moment(symmetric_recipe(p,idle),True)
            return np.r_[c[1:].real,c[1:].imag]/.15
        for seed in range(18):
            p0=rng.uniform(-np.pi,np.pi,half)
            fit=least_squares(constraints,p0,max_nfev=100,ftol=1e-11,xtol=1e-11,gtol=1e-11)
            def force_cost(p):
                c=fast_moment(symmetric_recipe(p,idle),True)
                return -abs(c[0])/.15
            # Optimize the remaining constraint-manifold freedom for force
            # efficiency before any full-Hilbert-space fidelity fit.
            maximum=minimize(force_cost,fit.x,method='SLSQP',
                             constraints=[dict(type='eq',fun=constraints)],
                             options=dict(maxiter=100,ftol=1e-10))
            attempts+=1
            p=maximum.x;c=fast_moment(symmetric_recipe(p,idle),True)
            residual=float(np.linalg.norm(c[1:])/max(abs(c[0]),1e-12))
            force_maxima.append(dict(idle_ns=idle_ns,seed=seed,
                                     force_coefficient=float(abs(c[0])),
                                     direction_error=residual,success=bool(maximum.success)))
            if residual>1e-7 or abs(c[0])<target_min:continue
            g=1/abs(c[0]);pump=-np.pi/2-np.angle(c[0]);swap_ns=1000*np.pi/(2*g)
            model=NativeRecipe(swap_ns,28,True)
            score=model.fidelity(symmetric_recipe(p,idle),pump);scored+=1
            row=dict(half_phases=p.tolist(),idle_samples=idle.tolist(),
                     nominal_swap_ns=swap_ns,pump_phase=float(pump),native_fidelity=score,
                     first_direction_error=residual,command_us=(30*cells+10+idle_ns)*.001,
                     cells=cells,centered_envelopes=True,seed=seed)
            rows.append(row)
            print('MIXED EXACTFIRST',row,flush=True)
        (OUT/f'mixed_exactfirst_n{cells}.json').write_text(json.dumps(dict(attempts=attempts,scored=scored,
                                                  force_maxima=force_maxima,rows=rows),indent=2)+'\n')
    assert attempts==72
    if rows:
        best=max(rows,key=lambda r:r['native_fidelity'])
        status(f'Mixed-axisfirstconstraintsearch: {attempts}closedpathsfit; {scored}physicallyfeasibleexactfirstforcegates '
               f'({18*scored}inputcolumns). BestnativeF{best["native_fidelity"]:.10f}, '
               f'T{best["command_us"]*1000:.1f}ns. Theseareproperaveragecandidates, notcertifiedgates.')
        return best
    status(f'Mixed-axisfirstconstraintsearch: {attempts}closedpathsfit; noexactfirstforcesolution withg<=cap. '
           'Thisisafiniteinitial-seedfailure, notano-gotheorem.')


def fourth_order_cost():
    """Audit canonical signed-JC Suzuki synthesis before expensive fitting.

    A=JC, B=XAX, -A=ZAZ, -B=YAY, all realizable in Pauli frames with the
    physicalpumpfixed.  The mergedseven-dwellfourth-orderwordhas a large
    absolute-timecost even beforefinite30nspulses areincluded.
    """
    p=1/(2-2**(1/3));q=1-2*p
    # Each tuple names an available toggled Hamiltonian and signed duration.
    terms=[('A',p/2),('B',p),('A',(p+q)/2),('B',q),
           ('A',(q+p)/2),('B',p),('A',p/2)]
    physical_time=sum(abs(v) for _,v in terms)/MAX_G
    space=cd.Space(12,9)
    A=space.hamiltonian(1.,0.).astype(complex)
    qx=np.kron(np.eye(12),X);B=qx@A@qx
    errors=[]
    for scale in [.1,.05,.025]:
        U=np.eye(24,dtype=complex)
        for label,amount in terms:
            U=expm(-1j*scale*amount*(A if label=='A' else B))@U
        exact=expm(-1j*scale*(A+B))
        errors.append(dict(scale=scale,error=float(np.linalg.norm(U-exact))))
    ratios=[errors[j]['error']/errors[j+1]['error'] for j in range(2)]
    # SixphysicalbridgesvisitI,X,Z,Y,Z,X,I. Actualnativebridgesarepart
    # oftheHamiltonianandmaynotbe replacedbyzero-widthPaulis.
    bridges=[(0.,1.),(np.pi/2,1.),(0.,1.),(np.pi,1.),(3*np.pi/2,1.),(np.pi,1.)]
    coefficients=fast_moment(bridges,True)
    report=dict(p=p,q=q,merged_terms=terms,minimum_swap_ns=200.,
                ideal_dwell_lower_bound_us=physical_time,
                six_bridge_floor_us=6*.030,
                total_with_bridge_floor_us=physical_time+6*.030+.010,
                ideal_order_errors=errors,halving_error_ratios=ratios,
                native_bridge_first_real=coefficients.real.tolist(),
                native_bridge_first_imag=coefficients.imag.tolist(),
                conclusion='CanonicalmergedSuzukiS4dwelltimealoneexceeds370ns; '
                           'finitebridgesalsoaddfirst-orderforceandneedcancellation. '
                           'Nofast-gateclaimandnoglobaloptimalityclaim.')
    assert len(errors)==3 and all(29<r<33 for r in ratios)
    assert physical_time>.370
    (OUT/'signed_jc_fourth_order_cost.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SIGNED-JC S4:',report,flush=True)
    return report


def refine_mixed_geometry():
    """Refine geometric pi/N seeds along the proper-first-force manifold.

    There are five halfword axes plus one constant coupling.  Four real
    first-momentconstraints andexactinverseclosurepreventtheoptimizer
    from changing the targetaxis;thefulltargetremainsbeta1.  Thecoupling
    maycorrectfinite-ordergain,butneverexceedthe200nsswapcap.
    """
    source=json.loads((OUT/'centroid_n10.json').read_text())
    rows=[];total_evaluations=0
    for index,seed in enumerate(source[:12]):
        fractions=np.asarray(seed['half_fractions']);idle=np.asarray(seed['idle_samples'])
        p0=np.r_[seed['half_phases'],np.log(np.pi/(2*seed['nominal_swap_ns']*.001))]
        def constraints(p):
            c=fast_moment(symmetric_recipe(p[:5],idle,fractions),True)
            return np.r_[c[1:].real,c[1:].imag]/.15
        # Feasible-seed solve is cheap two-levelgeometry, not a quantumgate
        # search. ItrepairsDACdwellroundingbeforethefidelityoptimization.
        feasible=least_squares(lambda a:constraints(np.r_[a,p0[-1]]),p0[:5],
                               max_nfev=80,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        p0[:5]=feasible.x
        evaluations=0;cache={};best=[1.,p0.copy()]
        def objective(p):
            nonlocal evaluations,total_evaluations
            evaluations+=1;total_evaluations+=1
            c=fast_moment(symmetric_recipe(p[:5],idle,fractions),True)
            pump=-np.pi/2-np.angle(c[0]);g=np.exp(p[-1])
            key=float(g)
            if key not in cache:
                cache.clear();cache[key]=NativeRecipe(1000*np.pi/(2*g),28,True)
            score=cache[key].fidelity(symmetric_recipe(p[:5],idle,fractions),pump)
            cost=1-score
            if np.linalg.norm(constraints(p))<1e-6 and cost<best[0]:best[:]=[cost,p.copy()]
            return cost
        result=minimize(objective,p0,method='SLSQP',
                        bounds=[(None,None)]*5+[(np.log(2.),np.log(MAX_G*.998))],
                        constraints=[dict(type='eq',fun=constraints)],
                        options=dict(maxiter=80,ftol=1e-9))
        p=best[1] if best[0]<1 else result.x
        c=fast_moment(symmetric_recipe(p[:5],idle,fractions),True)
        pump=-np.pi/2-np.angle(c[0]);g=np.exp(p[-1])
        model=NativeRecipe(1000*np.pi/(2*g),32,True)
        score=model.fidelity(symmetric_recipe(p[:5],idle,fractions),pump)
        row=dict(seed_index=index,half_phases=p[:5].tolist(),half_fractions=fractions.tolist(),
                 idle_samples=idle.tolist(),g=float(g),pump_phase=float(pump),
                 nominal_swap_ns=float(1000*np.pi/(2*g)),native_fidelity=score,
                 first_direction_error=float(np.linalg.norm(c[1:])/abs(c[0])),
                 first_beta=float(g*abs(c[0])),command_us=seed['command_us'],
                 evaluations=evaluations,optimizer_message=str(result.message),
                 centered_envelopes=True,cells=10)
        rows.append(row)
        (OUT/'mixed_geometric_refinement.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('MIXED GEOMETRIC',row,flush=True)
    assert len(rows)==12
    best=max(rows,key=lambda r:r['native_fidelity'])
    recipe=symmetric_recipe(best['half_phases'],best['idle_samples'],np.array(best['half_fractions']))
    export_recipe(recipe,best['g'],best['pump_phase'],'best_mixed_geometric',True,best)
    status(f'Completed12geometricallyseededmixed-axisconstrainedfits, {total_evaluations}nativegateevaluations '
           f'({18*total_evaluations}inputcolumns). BestnativeF{best["native_fidelity"]:.10f}, '
           f'T{best["command_us"]*1000:.1f}ns, firstaxiserror{best["first_direction_error"]:.3g}. '
           'SDK/RFDCandcutoffcertificationpending; nofinalfidelityclaim.')
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['validate', 'inverse', 'balanced', 'symmetric', 'allsigns', 'centroid',
                                         'export10','fractional11','mixed','fourth','refine'])
    parser.add_argument('--swap', type=float, default=200.)
    parser.add_argument('--cells', type=int, default=8)
    parser.add_argument('--gap-samples', type=int, default=0)
    args = parser.parse_args()
    if args.stage == 'validate':
        validate()
    elif args.stage == 'inverse':
        search_inverse(args.swap)
    elif args.stage == 'balanced':
        balanced_word_search(args.swap, args.cells)
    elif args.stage=='symmetric':
        symmetric_axes_search(args.swap,args.cells)
    elif args.stage=='allsigns':
        all_sign_search(args.cells,args.gap_samples)
    elif args.stage=='centroid':
        centroid_search(args.cells)
    elif args.stage=='export10':
        export_best10()
    elif args.stage=='fractional11':
        fractional11_search()
    elif args.stage=='mixed':
        mixed_axes_constrained(args.cells)
    elif args.stage=='fourth':
        fourth_order_cost()
    else:
        refine_mixed_geometry()
