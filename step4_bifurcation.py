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
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import brentq
from wc_model import Params, simulate, find_fixed_points, jacobian, classify, oscillation_stats

base = Params()


def fixed_points_info(p):
    return [(E, classify(jacobian(E, I, p))[0].startswith("stable"))
            for (E, I) in find_fixed_points(p, n_grid=8001)]


def cycle_range(p, t_end=500.0):
    t, E, I = simulate(p, y0=(0.2, 0.1), t_end=t_end, n=5001)
    tail = E[int(0.6 * len(E)):]
    if tail.max() - tail.min() > 1e-3:
        return tail.min(), tail.max()
    return None


def sweep(name, values):
    fps, cyc = [], []
    for v in values:
        p = base.with_(**{name: float(v)})
        fps.append(fixed_points_info(p))
        cyc.append(cycle_range(p))
    return fps, cyc


def count_fp(name, v):
    return len(find_fixed_points(base.with_(**{name: v}), n_grid=8001))


def bisect_change(f, a, b, tol=1e-4):
    if a == b:
        return a
    fa = f(a)
    while b - a > tol:
        m = 0.5 * (a + b)
        if f(m) == fa:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def saddle_nodes(name, values):
    counts = [count_fp(name, v) for v in values]
    out = []
    for k in range(len(values) - 1):
        if counts[k] != counts[k + 1]:
            out.append(bisect_change(lambda v: count_fp(name, v), values[k], values[k + 1]))
    return out


def hopf_point(name, lo, hi):
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


# A) input P
Ps = np.linspace(0.0, 4.5, 181)
print("Sweeping P ...")
fpsP, cycP = sweep("P", Ps)
snP = saddle_nodes("P", Ps)
has = [c is not None for c in cycP]
k_end = max(i for i, h in enumerate(has) if h)
P_end = bisect_change(lambda v: cycle_range(base.with_(P=v)) is not None,
                      Ps[k_end], Ps[min(k_end + 1, len(Ps) - 1)], tol=1e-3)
k_on = min(i for i, h in enumerate(has) if h)
P_on = bisect_change(lambda v: cycle_range(base.with_(P=v)) is not None,
                     Ps[max(k_on - 1, 0)], Ps[k_on], tol=1e-3)
print("Saddle-node values of P:", [round(x, 4) for x in snP])
print(f"Limit cycle exists from P = {P_on:.3f} to P = {P_end:.3f}")

# B) weight wEE
wEEs = np.linspace(10.0, 18.0, 81)
print("Sweeping wEE ...")
fpsW, cycW = sweep("wEE", wEEs)
hopf_wEE = hopf_point("wEE", 13.0, 13.25)
print(f"Hopf bifurcation in wEE at {hopf_wEE:.4f}  (P = 1.25)")

# C) input Q
Qs = np.linspace(0.0, 1.2, 61)
print("Sweeping Q ...")
fpsQ, cycQ = sweep("Q", Qs)
hopf_Q = hopf_point("Q", 0.6, 0.7)
print(f"Hopf bifurcation in Q at {hopf_Q:.4f}  (P = 1.25)")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
marksP = []
for x in snP:
    ends_here = abs(x - P_end) < 0.02
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

# ---------------------------------------------------------------- SNIC test
print("\nSNIC check: period scaling near onset")
dPs = np.logspace(-4, -1.3, 14)
Ts = []
for dP in dPs:
    p = base.with_(P=float(P_on + dP))
    t, E, I = simulate(p, y0=(0.2, 0.1), t_end=6000, n=240001)
    T = oscillation_stats(t, E, discard=0.5)[2]
    Ts.append(T if not np.isnan(T) else np.nan)
Ts = np.array(Ts)
mask = ~np.isnan(Ts)
slope, icpt = np.polyfit(np.log(dPs[mask]), np.log(Ts[mask]), 1)
print(f"  log-log slope = {slope:.3f}  (SNIC theory: -0.500)")

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(dPs[mask], Ts[mask], "o-", label="simulation")
ax.loglog(dPs[mask], np.exp(icpt) * dPs[mask]**slope, "r--",
          label=f"fit: T ~ (P - P_on)^{{{slope:.2f}}}")
ax.set(xlabel="P - P_on", ylabel="period T",
       title=f"SNIC signature near P_on = {P_on:.3f}")
ax.legend()
plt.tight_layout(); plt.savefig("fig4c_snic.png", dpi=200)

# ------------------------------------------------------- two-parameter diagram
print("\nTwo-parameter regime map (P, wEE) ...")
Ps_2d = np.linspace(0.5, 3.0, 35)
wEE_2d = np.linspace(10, 18, 35)
regime = np.zeros((len(wEE_2d), len(Ps_2d)))
for i, w in enumerate(wEE_2d):
    for j, P in enumerate(Ps_2d):
        p = base.with_(P=float(P), wEE=float(w))
        has_cycle = cycle_range(p) is not None
        any_stable = any(s for _, s in fixed_points_info(p))
        regime[i, j] = 2 if (has_cycle and any_stable) else (1 if has_cycle else 0)

cmap = ListedColormap(["#e8e8e8", "#3b82f6", "#f59e0b"])
fig, ax = plt.subplots(figsize=(7, 5))
ax.pcolormesh(Ps_2d, wEE_2d, regime, cmap=cmap, shading="auto", vmin=0, vmax=2)
ax.contour(Ps_2d, wEE_2d, regime, levels=[0.5, 1.5], colors="k", linewidths=1)
ax.plot(1.25, 16.0, "r*", ms=14, label="classic set")
ax.set(xlabel="input P", ylabel="wEE",
       title="Regime map: grey=steady, blue=oscillation, orange=coexist")
ax.legend()
plt.tight_layout(); plt.savefig("fig4b_twoparam.png", dpi=200)

plt.show()