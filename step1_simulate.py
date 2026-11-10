"""Step 1: simulate the model, confirm the limit cycle and nondimensional scaling."""
import numpy as np
import matplotlib.pyplot as plt
from wc_model import Params, simulate, oscillation_stats, rhs, rhs_nd

p = Params()
print("Parameters:", p, "\n")

# three starting points should reach the same loop
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

# P = 0 control: no oscillation
t, E, I = simulate(p.with_(P=0.0), y0=(0.5, 0.5), t_end=100, n=10001)
ax[2].plot(t, E, label="E"); ax[2].plot(t, I, label="I")
ax[2].set(xlabel="time (units of tau)", title="P = 0: activity just settles")
ax[2].legend()
lo, hi, T = oscillation_stats(t, E)
print(f"\nP = 0: E range after transient = [{lo:.4f}, {hi:.4f}]  (period = {T})")
plt.tight_layout(); plt.savefig("fig1_simulation.png", dpi=200)

# nondimensional vs dimensional form must agree when tau_E = tau_I = 1
print("\nNondimensional sanity check (tau_E = tau_I = 1):")
for pt in [(0.3, 0.2), (0.1, 0.5), (0.8, 0.1)]:
    d_dim = np.array(rhs(0, pt, p))
    d_nd  = np.array(rhs_nd(0, pt, p))
    err = np.abs(d_dim - d_nd).max()
    print(f"  at {pt}: max|rhs - rhs_nd| = {err:.2e}")

# tau scaling check
t, E, I = simulate(p, y0=(0.1, 0.05), t_end=300, n=30001)
_, _, T1 = oscillation_stats(t, E)
print(f"\nDimensionless period (tau = 1): {T1:.3f}")
for tau_ms in (20.0, 25.0, 30.0, 35.0):
    pp = p.with_(tauE=tau_ms, tauI=tau_ms)
    t, E, I = simulate(pp, y0=(0.1, 0.05), t_end=300 * tau_ms, n=30001)
    _, _, T = oscillation_stats(t, E)
    print(f"tau = {tau_ms:4.0f} ms -> period = {T:7.2f} ms -> frequency = {1000/T:5.2f} Hz")
plt.show()  