"""
Step 1 - Set up the model and CONFIRM it oscillates before changing anything.

What it does
  1. Simulates the classic parameter set from three different starting points.
  2. Checks they all end on the same closed loop (a limit cycle).
  3. Shows that with a low input (P = 0) the model just settles (no oscillation).
  4. Checks the nondimensionalisation: with tau_E = tau_I = tau, the period
     is (dimensionless period) x tau, so tau sets the real-world frequency.
"""
import numpy as np
import matplotlib.pyplot as plt
from wc_model import Params, simulate, oscillation_stats

p = Params()                       # classic set, P = 1.25
print("Parameters:", p, "\n")

# 1) three different starting points -----------------------------------------
starts = [(0.10, 0.05), (0.90, 0.90), (0.30, 0.10)]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for y0 in starts:
    t, E, I = simulate(p, y0=y0, t_end=300, n=30001)
    lo, hi, T = oscillation_stats(t, E)
    print(f"start {y0}:  E in [{lo:.4f}, {hi:.4f}],  period = {T:.3f} time units")
    ax[0].plot(t, E, lw=1, label=f"start {y0}")
    ax[1].plot(E, I, lw=0.8)
ax[0].set(xlim=(0, 60), xlabel="time (units of tau)", ylabel="E(t)",
          title="E(t) from three starts (P = 1.25)")
ax[0].legend(fontsize=8)
ax[1].set(xlabel="E", ylabel="I", title="Phase plane: all starts reach one loop")

# 2) no oscillation at low input ---------------------------------------------
t, E, I = simulate(p.with_(P=0.0), y0=(0.5, 0.5), t_end=100, n=10001)
ax[2].plot(t, E, label="E"); ax[2].plot(t, I, label="I")
ax[2].set(xlabel="time (units of tau)", title="P = 0: activity just settles")
ax[2].legend()
lo, hi, T = oscillation_stats(t, E)
print(f"\nP = 0: E range after transient = [{lo:.4f}, {hi:.4f}]  (period = {T})")
plt.tight_layout(); plt.savefig("fig1_simulation.png", dpi=200)

# 3) nondimensional check: how tau sets the frequency ------------------------
t, E, I = simulate(p, y0=(0.1, 0.05), t_end=300, n=30001)
_, _, T1 = oscillation_stats(t, E)
print(f"\nDimensionless period (tau = 1): {T1:.3f}")
for tau_ms in (20.0, 25.0, 30.0, 35.0):
    pp = p.with_(tauE=tau_ms, tauI=tau_ms)           # time now in ms
    t, E, I = simulate(pp, y0=(0.1, 0.05), t_end=300 * tau_ms, n=30001)
    _, _, T = oscillation_stats(t, E)
    print(f"tau = {tau_ms:4.0f} ms -> period = {T:7.2f} ms -> frequency = {1000/T:5.2f} Hz")
plt.show()
