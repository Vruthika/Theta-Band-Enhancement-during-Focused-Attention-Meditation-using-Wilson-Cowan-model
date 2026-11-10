"""
Step 3 - Phase-plane analysis.

For three values of P it draws, in the (E, I) plane:
  * the vector field (arrows show the direction of motion),
  * the nullclines (blue: dE/dt = 0, orange: dI/dt = 0),
  * the fixed points (filled = stable, open = unstable, x = saddle),
  * several trajectories, and the limit cycle (thick red) if one exists.
"""
import numpy as np
import matplotlib.pyplot as plt
from wc_model import (Params, rhs, simulate, find_fixed_points, jacobian, classify,
                      I_on_E_nullcline, E_on_I_nullcline, oscillation_stats)

base = Params()
XMAX, YMAX = 0.55, 0.50


def draw(ax, P, starts):
    p = base.with_(P=P)
    # vector field
    Eg, Ig = np.meshgrid(np.linspace(0.01, XMAX, 22), np.linspace(0.01, YMAX, 20))
    dE, dI = rhs(0, [Eg, Ig], p)
    mag = np.hypot(dE, dI) + 1e-12
    ax.quiver(Eg, Ig, dE / mag, dI / mag, color="0.75", angles="xy", width=0.003)
    # nullclines (explicit formulas, no root-finding needed)
    Ee = np.linspace(1e-4, 1 - 1e-4, 4000)
    ax.plot(Ee, I_on_E_nullcline(Ee, p), color="tab:blue", lw=2, label="dE/dt = 0")
    Ii = np.linspace(1e-4, 1 - 1e-4, 4000)
    ax.plot(E_on_I_nullcline(Ii, p), Ii, color="tab:orange", lw=2, label="dI/dt = 0")
    # trajectories
    for y0 in starts:
        t, E, I = simulate(p, y0=y0, t_end=60, n=6001)
        ax.plot(E, I, color="0.35", lw=0.8)
        ax.plot(*y0, "o", color="0.35", ms=3)
    # limit cycle: integrate long, keep the last part
    t, E, I = simulate(p, y0=(0.2, 0.1), t_end=400, n=40001)
    lo, hi, T = oscillation_stats(t, E)
    if not np.isnan(T):
        k = len(t) // 2
        ax.plot(E[k:], I[k:], color="crimson", lw=2.5, label="limit cycle")
    # fixed points
    for (E0, I0) in find_fixed_points(p):
        label, ev, tr, det = classify(jacobian(E0, I0, p))
        if label == "saddle":
            ax.plot(E0, I0, "kx", ms=9, mew=2)
        elif label.startswith("stable"):
            ax.plot(E0, I0, "ko", ms=8)
        else:
            ax.plot(E0, I0, "ko", ms=8, mfc="white", mew=2)
    ax.set(xlim=(0, XMAX), ylim=(0, YMAX), xlabel="E", ylabel="I", title=f"P = {P}")


starts = [(0.02, 0.30), (0.50, 0.05), (0.45, 0.45), (0.10, 0.02), (0.30, 0.20)]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, P in zip(axes, (0.5, 0.9, 1.25)):
    draw(ax, P, starts)
axes[0].legend(loc="upper right", fontsize=8)
plt.suptitle("Phase planes: P = 0.5 (one stable state), 0.9 (stable node + saddle + unstable focus, no cycle), 1.25 (limit cycle)")
plt.tight_layout(); plt.savefig("fig3_phase_planes.png", dpi=200)
plt.show()