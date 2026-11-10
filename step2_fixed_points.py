"""
Step 2 - Fixed points and their stability.

What it does
  1. Finds every fixed point for several values of the input P.
  2. Computes the Jacobian, its eigenvalues, trace and determinant, and
     classifies each point (stable/unstable node or focus, saddle).
  3. Sweeps P and plots the fixed-point branches (black = stable,
     red = unstable). This is the equilibrium part of the bifurcation diagram
     you will finish in Phase 4 by adding the limit-cycle amplitude.
"""
import numpy as np
import matplotlib.pyplot as plt
from wc_model import Params, find_fixed_points, jacobian, classify

base = Params()

print("Fixed points for selected P (tau_E = tau_I = 1)")
print("-" * 78)
for P in (0.0, 0.5, 0.9, 1.25, 2.0, 3.5, 4.0):
    p = base.with_(P=P)
    print(f"\nP = {P}")
    for (E, I) in find_fixed_points(p):
        label, ev, tr, det = classify(jacobian(E, I, p))
        evs = ", ".join(f"{z:.3f}" for z in ev)
        print(f"  (E*, I*) = ({E:.4f}, {I:.4f})  {label:15s} trace={tr:7.3f} det={det:7.3f}  eig = [{evs}]")

# sweep P --------------------------------------------------------------------
Ps = np.linspace(0.0, 4.5, 451)
pts = {"stable": [], "unstable": []}
for P in Ps:
    p = base.with_(P=P)
    for (E, I) in find_fixed_points(p, n_grid=8001):
        label, ev, tr, det = classify(jacobian(E, I, p))
        pts["stable" if label.startswith("stable") else "unstable"].append((P, E))

fig, ax = plt.subplots(figsize=(7, 4.5))
for key, c in (("stable", "k"), ("unstable", "r")):
    a = np.array(pts[key])
    if len(a):
        ax.plot(a[:, 0], a[:, 1], ".", ms=3, color=c, label=f"{key} fixed point")
ax.set(xlabel="external input P", ylabel="E* (fixed point)",
       title="Fixed points vs P ")
ax.legend(); plt.tight_layout(); plt.savefig("fig2_fixed_points_vs_P.png", dpi=200)
plt.show()
