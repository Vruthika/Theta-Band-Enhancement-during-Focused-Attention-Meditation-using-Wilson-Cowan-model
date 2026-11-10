"""
Step 4 - Bifurcation diagrams.

For each value of a parameter this script records
  * every fixed point and whether it is stable (black) or unstable (red),
  * the minimum and maximum of E on the limit cycle, found by simulation
    (blue), which shows where oscillations exist and how large they are.
It then locates the bifurcation points accurately:
  * saddle-node: where the NUMBER of fixed points changes (bisection),
  * Hopf: where the real part of the complex eigenvalues crosses zero (brentq),
  * the point where the limit cycle disappears (bisection).

Three one-parameter diagrams are produced (SN = saddle-node):
   A) input P   -> oscillation switches on through a saddle-node
   B) weight wEE -> a Hopf bifurcation (oscillation grows from zero amplitude)
   C) input Q   -> a second Hopf bifurcation
Saves fig4_bifurcation.png and prints the bifurcation values.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from wc_model import Params, simulate, find_fixed_points, jacobian, classify

base = Params()          # classic set; P = 1.25 unless the sweep changes it


def fixed_points_info(p):
    out = []
    for (E, I) in find_fixed_points(p, n_grid=4001):
        label, ev, tr, det = classify(jacobian(E, I, p))
        out.append((E, label.startswith("stable")))
    return out


def cycle_range(p, t_end=500.0):
    """(Emin, Emax) of the long-time behaviour, or None if activity settles.
    Starts near the central fixed point, where a limit cycle (if any) lives."""
    t, E, I = simulate(p, y0=(0.2, 0.1), t_end=t_end, n=5001)
    tail = E[int(0.6 * len(E)):]
    if tail.max() - tail.min() > 1e-3:
        return tail.min(), tail.max()
    return None


def sweep(name, values):
    fps, cyc = [], []
    for k, v in enumerate(values):
        p = base.with_(**{name: float(v)})
        fps.append(fixed_points_info(p))
        cyc.append(cycle_range(p))
    return fps, cyc


def count_fp(name, v):
    return len(find_fixed_points(base.with_(**{name: v}), n_grid=8001))


def bisect_change(f, a, b, tol=1e-4):
    """Find where the integer/boolean function f changes between a and b."""
    fa = f(a)
    while b - a > tol:
        m = 0.5 * (a + b)
        if f(m) == fa:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def saddle_nodes(name, values):
    """Parameter values where the number of fixed points changes."""
    counts = [count_fp(name, v) for v in values]
    found = []
    for k in range(len(values) - 1):
        if counts[k] != counts[k + 1]:
            found.append(bisect_change(lambda v: count_fp(name, v), values[k], values[k + 1]))
    return found


def hopf_point(name, lo, hi):
    """Where the largest real eigenvalue part is zero (single fixed point)."""
    def g(v):
        p = base.with_(**{name: v})
        E, I = find_fixed_points(p, n_grid=8001)[0]
        return np.linalg.eigvals(jacobian(E, I, p)).real.max()
    return brentq(g, lo, hi, xtol=1e-6)


def draw(ax, name, values, fps, cyc, marks):
    for v, info in zip(values, fps):
        for E, stable in info:
            ax.plot(v, E, ".", ms=2.5, color="k" if stable else "r")
    for v, c in zip(values, cyc):
        if c is not None:
            ax.plot([v, v], c, ".", ms=2.5, color="tab:blue")
    for v, txt in marks:
        ax.axvline(v, color="0.5", ls="--", lw=0.8)
        ax.text(v, 1.02, txt, rotation=90, va="bottom", ha="center", fontsize=8,
                transform=ax.get_xaxis_transform())
    ax.plot([], [], "k.", label="stable fixed point")
    ax.plot([], [], "r.", label="unstable fixed point")
    ax.plot([], [], ".", color="tab:blue", label="limit cycle (min, max of E)")
    ax.set(xlabel=name, ylabel="E", ylim=(0, 1.02), xlim=(values[0], values[-1]))


# ------------------------------------------------------------------ A) input P
Ps = np.linspace(0.0, 4.5, 181)
print("Sweeping P ...")
fpsP, cycP = sweep("P", Ps)
snP = saddle_nodes("P", Ps)
# where the limit cycle disappears at large P
has = [c is not None for c in cycP]
k_end = max(i for i, h in enumerate(has) if h)
P_end = bisect_change(lambda v: cycle_range(base.with_(P=v)) is not None,
                      Ps[k_end], Ps[min(k_end + 1, len(Ps) - 1)], tol=1e-3)
k_on = min(i for i, h in enumerate(has) if h)
P_on = bisect_change(lambda v: cycle_range(base.with_(P=v)) is not None,
                     Ps[max(k_on - 1, 0)], Ps[k_on], tol=1e-3)
print("Saddle-node values of P:", [round(x, 4) for x in snP])
print(f"Limit cycle exists from P = {P_on:.3f} to P = {P_end:.3f}")

# --------------------------------------------------------------- B) weight wEE
wEEs = np.linspace(10.0, 18.0, 81)
print("Sweeping wEE ...")
fpsW, cycW = sweep("wEE", wEEs)
hopf_wEE = hopf_point("wEE", 13.0, 13.25)
print(f"Hopf bifurcation in wEE at {hopf_wEE:.4f}  (P = 1.25)")

# ------------------------------------------------------------------ C) input Q
Qs = np.linspace(0.0, 1.2, 61)
print("Sweeping Q ...")
fpsQ, cycQ = sweep("Q", Qs)
hopf_Q = hopf_point("Q", 0.6, 0.7)
print(f"Hopf bifurcation in Q at {hopf_Q:.4f}  (P = 1.25)")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
marksP = []
for x in snP:
    ends_here = abs(x - P_end) < 0.02          # merge labels that coincide
    marksP.append((x, f"SN {x:.2f}" + (" (cycle ends)" if ends_here else "")))
if not any(abs(x - P_end) < 0.02 for x in snP):
    marksP.append((P_end, f"end {P_end:.2f}"))
draw(ax[0], "P", Ps, fpsP, cycP, marksP)
ax[0].legend(loc="center right", fontsize=8)
ax[0].set_title("A) input P (other parameters classic)", pad=70)
draw(ax[1], "wEE", wEEs, fpsW, cycW, [(hopf_wEE, f"Hopf {hopf_wEE:.2f}")])
ax[1].set_title("B) weight wEE at P = 1.25", pad=70)
draw(ax[2], "Q", Qs, fpsQ, cycQ, [(hopf_Q, f"Hopf {hopf_Q:.3f}")])
ax[2].set_title("C) input Q at P = 1.25", pad=70)
plt.tight_layout(); plt.savefig("fig4_bifurcation.png", dpi=200)
plt.show()
