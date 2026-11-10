"""
Step 6 - Resting vs meditative regime: time series and power spectra 

EEG is mimicked by the excitatory activity E(t) (an assumption to state in
the report). Time is in ms (tau_E = tau_I = 25 ms, which puts the classic
rhythm near 6 Hz). The spectrum is estimated with Welch's method and the
theta-band (4-8 Hz) power is the integral of the spectrum over 4-8 Hz.

Two ways of defining "rest" vs "meditation" are compared, so you can discuss
with your guide which one to use:
  Scenario A - input P:  rest P = 0.70 (below onset, steady state)
                         meditation P = 1.25 (sustained limit cycle)
  Scenario B - weight wEE (Hopf route, P = 1.25):
                         rest wEE = 13.30 (just above Hopf, weak oscillation)
                         meditation wEE = 16.0 (strong oscillation)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.integrate import trapezoid
from wc_model import Params, simulate

TAU = 25.0
FS = 1000.0
T_TOTAL, T_DISCARD = 22000.0, 2000.0
THETA = (4.0, 8.0)
base = Params(tauE=TAU, tauI=TAU)

scenarios = {
    "A) input P": [("rest (P = 0.70)", base.with_(P=0.70)),
                   ("meditation (P = 1.25)", base.with_(P=1.25))],
    "B) weight wEE": [("rest (wEE = 13.3)", base.with_(wEE=13.3)),
                      ("meditation (wEE = 16)", base.with_(wEE=16.0))],
}


def analyse(p):
    t, E, I = simulate(p, y0=(0.2, 0.1), t_end=T_TOTAL, n=int(T_TOTAL) + 1)
    keep = t >= T_DISCARD
    t, x = t[keep], E[keep]
    f, Pxx = welch(x, fs=FS, window="hann", nperseg=4096, noverlap=2048,
                   detrend="constant", scaling="density")
    band = (f >= THETA[0]) & (f <= THETA[1])
    theta = trapezoid(Pxx[band], f[band])
    total = trapezoid(Pxx[(f >= 1) & (f <= 40)], f[(f >= 1) & (f <= 40)])
    peak = f[np.argmax(Pxx)] if Pxx.max() > 1e-18 else np.nan
    return dict(t=t, x=x, f=f, Pxx=Pxx, theta=theta, total=total, peak=peak,
                p2p=x.max() - x.min())


fig, axes = plt.subplots(2, 2, figsize=(13, 8))
print(f"{'scenario / state':32s} {'peak-to-peak':>12s} {'peak freq (Hz)':>15s} "
      f"{'theta power':>12s} {'theta / total':>14s}")
for row, (name, pair) in enumerate(scenarios.items()):
    res = []
    for label, p in pair:
        r = analyse(p); res.append(r)
        rel = r["theta"] / r["total"] if r["total"] > 1e-18 else np.nan
        print(f"{name + ' ' + label:32s} {r['p2p']:12.4f} {r['peak']:15.2f} "
              f"{r['theta']:12.3e} {rel:14.3f}")
        axes[row, 0].plot((r["t"] - T_DISCARD) / 1000.0, r["x"], label=label)
        axes[row, 1].semilogy(r["f"], np.maximum(r["Pxx"], 1e-20), label=label)
    if res[0]["theta"] < 1e-15:
        print(f"{'':32s} rest has no oscillation (theta power is numerically zero)\n")
    else:
        ratio = res[1]["theta"] / res[0]["theta"]
        print(f"{'':32s} theta-power ratio (meditation / rest) = {ratio:.1f}\n")
    axes[row, 0].set(xlim=(0, 2), xlabel="time (s)", ylabel="E(t)",
                     title=f"{name}: time series")
    axes[row, 0].legend(fontsize=8)
    axes[row, 1].axvspan(*THETA, color="gold", alpha=0.3, label="theta 4-8 Hz")
    axes[row, 1].set(xlim=(0, 30), ylim=(1e-12, None), xlabel="frequency (Hz)",
                     ylabel="power spectral density",
                     title=f"{name}: Welch spectrum")
    axes[row, 1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig6_rest_vs_meditation.png", dpi=200)
plt.show()