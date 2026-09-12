"""Error ANGLE of the group-commutator synthesis at a FIXED target angle Theta.

N is chosen FIRST and eps = sqrt(Theta/(2N)) follows, so the achieved angle is always Theta
and integer rounding cannot pollute the scaling.
Metric: the operator-norm distance to the converged synthesis -- an ANGLE, the same kind of
quantity a Trotter error bound quotes. (Infidelity is its square, which is why a naive
fidelity fit reads 4 instead of 2.)
"""
import numpy as np, qutip as qt
exec(open("scaling.py").read().split("def ideal(")[0])

THETA = np.pi / 4
eps_ref = 0.015
n_ref = int(round(THETA / (2 * eps_ref ** 2)))
u_ref = synthesized(np.sqrt(THETA / (2 * n_ref)), n_ref)

print(f"{'N':>6} {'eps':>8} {'error angle':>13} {'angle/eps^2':>12} "
      f"{'T_phys (g=1)':>13}")
rows = []
for n in (2, 4, 8, 16, 32, 64, 128, 256):
    eps = np.sqrt(THETA / (2 * n))
    u = synthesized(eps, n)
    ang = float((u - u_ref).norm())          # operator-norm distance
    t = 8 * n * eps                          # total pulses x dt, dt = eps/g with g = 1
    rows.append((eps, ang, t))
    print(f"{n:6d} {eps:8.4f} {ang:13.3e} {ang/eps**2:12.4f} {t:13.2f}")
e = np.array([r[0] for r in rows]); a = np.array([r[1] for r in rows])
t = np.array([r[2] for r in rows])
ok = a > 1e-10
print(f"\nerror angle ~ eps^{np.polyfit(np.log(e[ok]), np.log(a[ok]), 1)[0]:.2f}")
print(f"T_phys      ~ eps^{np.polyfit(np.log(e), np.log(t), 1)[0]:.2f}   (fixed g)")
print(f"=> T_phys   ~ (1/accuracy)^{-np.polyfit(np.log(a[ok]), np.log(t[ok]), 1)[0]:.2f}"
      f"  i.e. improving accuracy by k costs k^{-np.polyfit(np.log(a[ok]), np.log(t[ok]), 1)[0]:.2f} in time")
