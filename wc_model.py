"""
wc_model.py
Core code for the Wilson-Cowan E-I project (simplified form, no refractory term).

    tau_E dE/dt = -E + S_E( wEE*E - wEI*I + P )
    tau_I dI/dt = -I + S_I( wIE*E - wII*I + Q )
    S(x)        = 1 / (1 + exp(-a (x - theta)))

Default parameters: the classic limit-cycle set from Wilson & Cowan (1972),
c1=16, c2=12, c3=15, c4=3, a_e=1.3, theta_e=4, a_i=2, theta_i=3.7, P=1.25, Q=0,
used here WITHOUT the refractory factor (1 - r*E). Time is in units of tau
(tau_E = tau_I = 1) until you tune the time constants.

Subscript rule: in wEI the first letter is the receiver and the second the
sender, so wEI is the strength of inhibition (from I) onto E.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import expit


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    wEE: float = 16.0   # excitation onto E from E
    wEI: float = 12.0   # inhibition onto E from I
    wIE: float = 15.0   # excitation onto I from E
    wII: float = 3.0    # inhibition onto I from I
    P: float = 1.25     # external input to E
    Q: float = 0.0      # external input to I
    aE: float = 1.3     # sigmoid steepness (E)
    thE: float = 4.0    # sigmoid threshold (E)
    aI: float = 2.0     # sigmoid steepness (I)
    thI: float = 3.7    # sigmoid threshold (I)
    tauE: float = 1.0   # time constant of E (ms once you choose real units)
    tauI: float = 1.0   # time constant of I

    def with_(self, **kw):
        """Return a copy with some parameters changed, e.g. p.with_(P=2.0)."""
        return replace(self, **kw)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def S(x, a, th):
    """Sigmoid response function (numerically safe)."""
    return expit(a * (x - th))


def dS(x, a, th):
    """Derivative of the sigmoid: S'(x) = a S (1 - S)."""
    s = S(x, a, th)
    return a * s * (1.0 - s)


def rhs(t, y, p: Params):
    """Right-hand side of the dimensional model (time in units of tau)."""
    E, I = y
    dE = (-E + S(p.wEE * E - p.wEI * I + p.P, p.aE, p.thE)) / p.tauE
    dI = (-I + S(p.wIE * E - p.wII * I + p.Q, p.aI, p.thI)) / p.tauI
    return [dE, dI]


def rhs_nd(s, y, p: Params):
    """Nondimensional form. With s = t / tauE and eps = tauE / tauI:
         dE/ds = -E + S_E(...)
         dI/ds = eps * ( -I + S_I(...) )
    E and I are already dimensionless (0..1), so only time needs scaling and
    the two time constants collapse into the single ratio eps."""
    eps = p.tauE / p.tauI
    E, I = y
    dE = -E + S(p.wEE * E - p.wEI * I + p.P, p.aE, p.thE)
    dI = eps * (-I + S(p.wIE * E - p.wII * I + p.Q, p.aI, p.thI))
    return [dE, dI]


def simulate(p: Params, y0=(0.1, 0.05), t_end=300.0, n=30001):
    """Integrate the model. Returns t, E, I."""
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(rhs, (0.0, t_end), y0, args=(p,), t_eval=t_eval,
                    method="LSODA", rtol=1e-9, atol=1e-12)
    return sol.t, sol.y[0], sol.y[1]


# ----------------------------------------------------------------------------
# Fixed points and stability
# ----------------------------------------------------------------------------
def jacobian(E, I, p: Params):
    """Analytic Jacobian of the dimensional model at the point (E, I)."""
    uE = p.wEE * E - p.wEI * I + p.P
    uI = p.wIE * E - p.wII * I + p.Q
    sE, sI = dS(uE, p.aE, p.thE), dS(uI, p.aI, p.thI)
    return np.array([
        [(-1.0 + sE * p.wEE) / p.tauE, (-sE * p.wEI) / p.tauE],
        [(sI * p.wIE) / p.tauI,        (-1.0 - sI * p.wII) / p.tauI],
    ])


def _logit(x):
    return np.log(x / (1.0 - x))


def I_on_E_nullcline(E, p: Params):
    """I as an explicit function of E on the E-nullcline (dE/dt = 0)."""
    return (p.wEE * E + p.P - p.thE - _logit(E) / p.aE) / p.wEI


def E_on_I_nullcline(I, p: Params):
    """E as an explicit function of I on the I-nullcline (dI/dt = 0)."""
    return (p.thI + _logit(I) / p.aI + p.wII * I - p.Q) / p.wIE


def find_fixed_points(p: Params, n_grid=20001):
    """All fixed points in 0 < E, I < 1.

    Method: put I = I_on_E_nullcline(E) into the I equation, which leaves one
    equation g(E) = 0 in one unknown. Scan E for sign changes of g and refine
    each with Brent's method. This finds every root on the grid, with no
    dependence on initial guesses."""
    E = np.linspace(1e-9, 1.0 - 1e-9, n_grid)
    I = I_on_E_nullcline(E, p)

    def g(e):
        i = I_on_E_nullcline(e, p)
        return -i + S(p.wIE * e - p.wII * i + p.Q, p.aI, p.thI)

    gv = g(E)
    roots = []
    for k in np.where(np.sign(gv[:-1]) * np.sign(gv[1:]) < 0)[0]:
        e = brentq(g, E[k], E[k + 1], xtol=1e-14)
        roots.append((e, float(I_on_E_nullcline(e, p))))
    return roots


def classify(J):
    """Return (label, eigenvalues, trace, determinant) for a 2x2 Jacobian."""
    ev = np.linalg.eigvals(J)
    tr, det = np.trace(J), np.linalg.det(J)
    if det < 0:
        label, stable = "saddle", False
    else:
        kind = "focus" if np.iscomplex(ev).any() else "node"
        stable = tr < 0
        label = f"{'stable' if stable else 'unstable'} {kind}"
    return label, ev, tr, det


# ----------------------------------------------------------------------------
# Oscillation measures
# ----------------------------------------------------------------------------
def oscillation_stats(t, E, discard=0.5):
    """Amplitude range and mean period of E(t), discarding an initial transient.
    Returns (E_min, E_max, period). period is nan if no sustained oscillation."""
    k0 = int(len(t) * discard)
    tt, e = t[k0:], E[k0:]
    lo, hi = e.min(), e.max()
    if hi - lo < 1e-4:
        return lo, hi, np.nan
    pk = np.where((e[1:-1] > e[:-2]) & (e[1:-1] > e[2:]))[0] + 1
    if len(pk) < 3:
        return lo, hi, np.nan
    return lo, hi, float(np.mean(np.diff(tt[pk])))


# ----------------------------------------------------------------------------
# Two coupled regions (Phase 6)
# ----------------------------------------------------------------------------
def rhs_coupled(t, y, p1: Params, p2: Params, k: float):
    """Two E-I regions. The E population of each region also receives
    k * (E of the other region) inside its sigmoid. k > 0 is excitatory."""
    E1, I1, E2, I2 = y
    return [
        (-E1 + S(p1.wEE * E1 - p1.wEI * I1 + p1.P + k * E2, p1.aE, p1.thE)) / p1.tauE,
        (-I1 + S(p1.wIE * E1 - p1.wII * I1 + p1.Q, p1.aI, p1.thI)) / p1.tauI,
        (-E2 + S(p2.wEE * E2 - p2.wEI * I2 + p2.P + k * E1, p2.aE, p2.thE)) / p2.tauE,
        (-I2 + S(p2.wIE * E2 - p2.wII * I2 + p2.Q, p2.aI, p2.thI)) / p2.tauI,
    ]


def simulate_coupled(p1: Params, p2: Params, k, y0, t_end=2000.0, n=100001):
    """Integrate two coupled regions. Returns t and the 4 x n array (E1, I1, E2, I2)."""
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(rhs_coupled, (0.0, t_end), y0, args=(p1, p2, k), t_eval=t_eval,
                    method="LSODA", rtol=1e-9, atol=1e-12)
    return sol.t, sol.y


def phase_from_peaks(t, x):
    """Phase (radians, increasing by 2*pi per cycle) from the peaks of x(t).
    Works well for spiky, non-sinusoidal waveforms. Returns (t_valid, phase)."""
    pk = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    if len(pk) < 4:
        return None, None
    tp = t[pk]
    m = (t >= tp[0]) & (t <= tp[-1])
    return t[m], np.interp(t[m], tp, 2.0 * np.pi * np.arange(len(tp)))


def sync_measures(t, y, discard=0.5):
    """Synchronization measures for two regions after discarding a transient.
    Returns a dict, or None if one region does not oscillate (e.g. saturates).
      dphi : final phase difference, wrapped to (-pi, pi]
      dfreq: mean frequency difference in cycles per time unit (0 if locked)
      plv  : phase-locking value |mean(exp(i*dphi))| (1 = perfectly locked)
      corr : Pearson correlation of E1 and E2"""
    m = t > discard * t[-1]
    E1, E2 = y[0][m], y[2][m]
    if (E1.max() - E1.min() < 1e-3) or (E2.max() - E2.min() < 1e-3):
        return None
    t1, ph1 = phase_from_peaks(t[m], E1)
    t2, ph2 = phase_from_peaks(t[m], E2)
    if t1 is None or t2 is None:
        return None
    g = np.linspace(max(t1[0], t2[0]), min(t1[-1], t2[-1]), 20000)
    d = np.interp(g, t1, ph1) - np.interp(g, t2, ph2)
    return dict(dphi=float((d[-1] + np.pi) % (2 * np.pi) - np.pi),
                dfreq=float(np.polyfit(g, d, 1)[0] / (2 * np.pi)),
                plv=float(abs(np.mean(np.exp(1j * d)))),
                corr=float(np.corrcoef(E1, E2)[0, 1]))
