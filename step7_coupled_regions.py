"""
Step 7 - Two coupled regions and synchronization.

Region 2 receives k * E1 and region 1 receives k * E2 (excitatory coupling).
Two experiments, both swept over the coupling strength k:

  Experiment 1: IDENTICAL regions (P1 = P2 = 1.25), started in anti-phase.
     Question: does coupling pull them together, and to what phase lag?
  Experiment 2: DETUNED regions (P2 = P1 + dP), started in phase.
     Question: how large must k be to make the faster region and the slower
     region run at the same frequency (frequency locking)?

Measures (see wc_model.sync_measures): phase difference, frequency difference,
phase-locking value (PLV) and correlation. Time is in units of tau; the
frequency difference is converted to Hz with tau = 25 ms.
Saves fig7_synchronization.png.
"""
import numpy as np
import matplotlib.pyplot as plt
from wc_model import Params, simulate, oscillation_stats, simulate_coupled, sync_measures

base = Params()
TAU_MS = 25.0
DP = 0.05                                         # detuning for experiment 2
ks = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0, 2.5])

# two states on the limit cycle half a period apart (for the anti-phase start)
t, E, I = simulate(base, y0=(0.2, 0.1), t_end=300, n=300001)
T0 = oscillation_stats(t, E)[2]
dt = t[1] - t[0]
i0, sh = 200000, int(round(T0 / 2 / dt))
ya, yb = (E[i0], I[i0]), (E[i0 + sh], I[i0 + sh])

res1, res2 = [], []
for k in ks:
    t, y = simulate_coupled(base, base, k, (*ya, *yb), t_end=2000.0)
    res1.append(sync_measures(t, y))
    t, y = simulate_coupled(base, base.with_(P=base.P + DP), k, (*ya, *ya), t_end=2000.0)
    res2.append(sync_measures(t, y))
    print(f"k = {k:4.2f}  identical: {res1[-1] if res1[-1] is None else {a: round(v, 3) for a, v in res1[-1].items()}}")


def col(res, key):
    return np.array([np.nan if r is None else r[key] for r in res])


# locking threshold for experiment 2
fd = np.abs(col(res2, "dfreq")) * 1000.0 / TAU_MS          # Hz
locked = np.where(fd < 0.02)[0]
k_lock = ks[locked[0]] if len(locked) else np.nan
print(f"\nDetuned (dP = {DP}): frequency difference at k = 0 is {fd[0]:.3f} Hz;"
      f" locked from about k = {k_lock}")

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
a = ax[0, 0]
a.plot(ks, np.abs(col(res1, "dphi")), "o-", label="|phase lag| (rad)")
a.axhline(np.pi, color="0.6", ls=":"); a.text(0.9, np.pi + 0.05, "anti-phase (pi)", ha="center", fontsize=8)
a.set(xlabel="coupling k", ylabel="|phase difference| (rad)",
      title="1) identical regions, started in anti-phase")
b = a.twinx(); b.plot(ks, col(res1, "corr"), "s--", color="tab:orange", label="correlation")
b.set_ylabel("correlation of E1, E2"); b.set_ylim(-1, 1)
a.legend(loc="upper left", fontsize=8); b.legend(loc="upper right", fontsize=8)

a = ax[0, 1]
a.plot(ks, fd, "o-"); a.axhline(0, color="0.7")
a.set(xlabel="coupling k", ylabel="frequency difference (Hz)",
      title=f"2) detuned regions (dP = {DP}): frequency locking")
if not np.isnan(k_lock):
    a.axvline(k_lock, color="r", ls="--"); a.text(k_lock, fd.max() * 0.6, f" locked from k ~ {k_lock}", color="r")

a = ax[1, 0]
a.plot(ks, col(res1, "plv"), "o-", label="identical regions")
a.plot(ks, col(res2, "plv"), "s-", label=f"detuned (dP = {DP})")
a.set(xlabel="coupling k", ylabel="phase-locking value (PLV)", ylim=(0, 1.05),
      title="Phase locking vs coupling"); a.legend()

# example time series: identical regions, weak coupling
k_ex = 0.3
t, y = simulate_coupled(base, base, k_ex, (*ya, *yb), t_end=2000.0)
a = ax[1, 1]
m = (t > 1900) & (t < 1960)
a.plot(t[m], y[0][m], label="E1"); a.plot(t[m], y[2][m], label="E2")
a.set(xlabel="time (units of tau)", ylabel="E", title=f"Example: identical regions, k = {k_ex}")
a.legend()
plt.tight_layout(); plt.savefig("fig7_synchronization.png", dpi=200)
plt.show()
