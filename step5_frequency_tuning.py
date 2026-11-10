"""
Step 5 - Place the oscillation frequency in the theta band

Because of the nondimensionalisation, the real period is
        T = T_hat(eps, parameters) * tau_E ,   eps = tau_E / tau_I
where T_hat is the dimensionless period found with tau_E = 1. So we only need
to simulate in dimensionless time and then convert to Hz.

Panels
  A) frequency vs tau (tau_E = tau_I = tau) at P = 1.25, theta band shaded,
     with direct millisecond simulations as a check.
  B) frequency map over (tau_E, tau_I). Contours mark 4 and 8 Hz. Grey means
     no sustained oscillation for that ratio.
  C) frequency vs input P at tau = 25 ms: the rhythm speeds up as P grows.
"""
import numpy as np
import matplotlib.pyplot as plt
from wc_model import Params, simulate, oscillation_stats

base = Params()
THETA = (4.0, 8.0)


def T_hat(p, t_end=800.0):
    """Dimensionless period (tau_E = 1 scale) and E range. nan if no oscillation."""
    t, E, I = simulate(p, y0=(0.2, 0.1), t_end=t_end, n=int(t_end * 40) + 1)
    lo, hi, T = oscillation_stats(t, E, discard=0.5)
    return T


# ---- A) frequency vs tau ----------------------------------------------------
T1 = T_hat(base)                                   # tau_E = tau_I = 1
print(f"Dimensionless period at P = 1.25: {T1:.3f}")
taus = np.linspace(10, 60, 200)
f_tau = 1000.0 / (T1 * taus)
tau_hi, tau_lo = 1000.0 / (T1 * THETA[0]), 1000.0 / (T1 * THETA[1])
print(f"Theta band (4-8 Hz) needs tau between {tau_lo:.1f} ms and {tau_hi:.1f} ms")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].plot(taus, f_tau, lw=2, label="1000 / (T_hat x tau)")
ax[0].axhspan(*THETA, color="gold", alpha=0.3, label="theta band 4-8 Hz")
for tau in (20, 25, 30, 35, 40):                   # direct check in milliseconds
    p = base.with_(tauE=float(tau), tauI=float(tau))
    t, E, I = simulate(p, y0=(0.2, 0.1), t_end=300.0 * tau, n=30001)
    T = oscillation_stats(t, E)[2]
    ax[0].plot(tau, 1000.0 / T, "ko", ms=6)
ax[0].plot([], [], "ko", label="direct simulation (ms)")
ax[0].set(xlabel="tau_E = tau_I  (ms)", ylabel="frequency (Hz)", title="A) frequency vs time constant (P = 1.25)")
ax[0].legend()

# ---- B) frequency map over (tau_E, tau_I) ------------------------------------
eps_grid = np.exp(np.linspace(np.log(0.25), np.log(2.5), 60))
Th = np.array([T_hat(base.with_(tauE=1.0, tauI=1.0 / e)) for e in eps_grid])
valid = ~np.isnan(Th)
print(f"Sustained oscillation found for eps = tau_E/tau_I up to about {eps_grid[valid].max():.2f}")
tauE_v = np.linspace(5, 60, 120)
tauI_v = np.linspace(5, 60, 120)
TE, TI = np.meshgrid(tauE_v, tauI_v)
EPS = TE / TI
Thi = np.interp(np.log(EPS), np.log(eps_grid[valid]), Th[valid])
Thi[(EPS > eps_grid[valid].max()) | (EPS < eps_grid[valid].min())] = np.nan
F = 1000.0 / (Thi * TE)
im = ax[1].pcolormesh(TE, TI, F, shading="auto", cmap="viridis", vmin=0, vmax=30)
ax[1].set_facecolor("0.85")
cs = ax[1].contour(TE, TI, F, levels=list(THETA), colors=["white", "white"], linewidths=2)
ax[1].clabel(cs, fmt="%g Hz")
ax[1].plot([5, 60], [5, 60], "w--", lw=0.8)
ax[1].plot(25, 25, "r*", ms=14, label="chosen: 25 ms, 25 ms")
ax[1].set(xlabel="tau_E (ms)", ylabel="tau_I (ms)", title="B) frequency map (grey = no oscillation or not computed)")
ax[1].legend(loc="upper left", fontsize=8)
plt.colorbar(im, ax=ax[1], label="frequency (Hz)")

# ---- C) frequency vs input P at tau = 25 ms -----------------------------------
tau = 25.0
Ps = np.linspace(1.0, 3.0, 41)
fP = []
for P in Ps:
    T = T_hat(base.with_(P=float(P)), t_end=1200.0)
    fP.append(1000.0 / (T * tau) if not np.isnan(T) else np.nan)
fP = np.array(fP)
ax[2].plot(Ps, fP, "o-", ms=4)
ax[2].axhspan(*THETA, color="gold", alpha=0.3, label="theta band 4-8 Hz")
ax[2].set(xlabel="input P", ylabel="frequency (Hz)", title=f"C) frequency vs input P (tau = {tau:.0f} ms)")
ax[2].legend()
ok = (fP >= THETA[0]) & (fP <= THETA[1])
runs, start = [], None
for k, flag in enumerate(ok):                      # contiguous runs of P in the band
    if flag and start is None:
        start = k
    if (not flag or k == len(ok) - 1) and start is not None:
        end = k if flag else k - 1
        runs.append((Ps[start], Ps[end])); start = None
print("At tau = 25 ms the rhythm is in the theta band for P in:",
      ", ".join(f"[{a:.2f}, {b:.2f}]" for a, b in runs))
plt.tight_layout(); plt.savefig("fig5_frequency_tuning.png", dpi=200)
plt.show()