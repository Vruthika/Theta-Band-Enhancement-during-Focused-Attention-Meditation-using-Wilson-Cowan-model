"""
Step 8 - Robustness and critical evaluation

Question: how fragile is the theta-capable oscillation to the choices made?
Each parameter is changed one at a time by -20, -10, -5, +5, +10, +20 percent
(Q is zero in the classic set, so it is shifted by an absolute -0.2 ... +0.2).
For each change the script checks whether a sustained oscillation survives and
how the period changes relative to the classic set. A second test replaces the
two sigmoids by a single shared sigmoid (as in the abstract's S(x)).
"""
import numpy as np
import matplotlib.pyplot as plt
from wc_model import Params, simulate, oscillation_stats

base = Params()


def period(p, t_end=800.0):
    t, E, I = simulate(p, y0=(0.2, 0.1), t_end=t_end, n=int(t_end * 40) + 1)
    return oscillation_stats(t, E, discard=0.5)[2]


T0 = period(base)
print(f"Classic set: period = {T0:.3f}\n")

names = ["wEE", "wEI", "wIE", "wII", "P", "aE", "thE", "aI", "thI", "Q"]
changes = np.array([-0.20, -0.10, -0.05, 0.05, 0.10, 0.20])
ratio = np.full((len(names), len(changes)), np.nan)
for i, nm in enumerate(names):
    for j, c in enumerate(changes):
        v0 = getattr(base, nm)
        v = c if nm == "Q" else v0 * (1.0 + c)            # Q: absolute shift from 0
        T = period(base.with_(**{nm: v}))
        ratio[i, j] = T / T0 if not np.isnan(T) else np.nan
    row = "  ".join("  --  " if np.isnan(r) else f"{r:6.2f}" for r in ratio[i])
    print(f"{nm:4s} {row}")
print("\n('--' = no sustained oscillation; numbers = period / classic period)")

# ---- shared sigmoid test ------------------------------------------------------
print("\nShared sigmoid (S_I = S_E):")
shared = [("aI = aE = 1.3, thI = thE = 4.0", base.with_(aI=1.3, thI=4.0)),
          ("aI = aE = 2.0, thI = thE = 3.7", base.with_(aE=2.0, thE=3.7))]
for label, p in shared:
    T = period(p)
    print(f"  {label}: " + ("no sustained oscillation" if np.isnan(T) else f"period = {T:.2f}"))

# ---- figure -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
cmap = plt.get_cmap("coolwarm").copy(); cmap.set_bad("0.85")
im = ax.imshow(np.ma.masked_invalid(ratio), cmap=cmap, vmin=0.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(len(changes)))
ax.set_xticklabels([f"{c * 100:+.0f}%  (Q: {c:+.2f})" for c in changes], fontsize=8)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
for i in range(len(names)):
    for j in range(len(changes)):
        ax.text(j, i, "none" if np.isnan(ratio[i, j]) else f"{ratio[i, j]:.2f}",
                ha="center", va="center", fontsize=8)
ax.set(xlabel="change of one parameter (Q: absolute shift)", title="Period relative to the classic set (grey = no oscillation)")
plt.colorbar(im, label="period / classic period")
plt.tight_layout(); plt.savefig("fig8_robustness.png", dpi=200)
plt.show()