"""Core Wilson-Cowan E-I model (simplified, no refractory term)."""
from dataclasses import dataclass, replace
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import expit

@dataclass(frozen=True)
class Params:
    wEE: float = 16.0
    wEI: float = 12.0
    wIE: float = 15.0
    wII: float = 3.0
    P: float = 1.25
    Q: float = 0.0
    aE: float = 1.3
    thE: float = 4.0
    aI: float = 2.0
    thI: float = 3.7
    tauE: float = 1.0
    tauI: float = 1.0

    def with_(self, **kw):
        return replace(self, **kw)


def S(x, a, th):
    return expit(a * (x - th))


def dS(x, a, th):
    s = S(x, a, th)
    return a * s * (1.0 - s)


def rhs(t, y, p: Params):
    E, I = y
    dE = (-E + S(p.wEE * E - p.wEI * I + p.P, p.aE, p.thE)) / p.tauE
    dI = (-I + S(p.wIE * E - p.wII * I + p.Q, p.aI, p.thI)) / p.tauI
    return [dE, dI]


def rhs_nd(s, y, p: Params):
    """Nondimensional form with s = t/tauE, eps = tauE/tauI."""
    eps = p.tauE / p.tauI
    E, I = y
    dE = -E + S(p.wEE * E - p.wEI * I + p.P, p.aE, p.thE)
    dI = eps * (-I + S(p.wIE * E - p.wII * I + p.Q, p.aI, p.thI))
    return [dE, dI]


def simulate(p: Params, y0=(0.1, 0.05), t_end=300.0, n=30001):
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(rhs, (0.0, t_end), y0, args=(p,), t_eval=t_eval,
                    method="LSODA", rtol=1e-9, atol=1e-12)
    return sol.t, sol.y[0], sol.y[1]


def jacobian(E, I, p: Params):
    uE = p.wEE * E - p.wEI * I + p.P
    uI = p.wIE * E - p.wII * I + p.Q
    sE, sI = dS(uE, p.aE, p.thE), dS(uI, p.aI, p.thI)
    return np.array([
        [(-1.0 + sE * p.wEE) / p.tauE, (-sE * p.wEI) / p.tauE],
        [(sI * p.wIE) / p.tauI,        (-1.0 - sI * p.wII) / p.tauI],
    ])


def jacobian_numeric(E, I, p: Params, h=1e-6):
    """Central-difference Jacobian, for checking the analytic one."""
    f = lambda y: np.array(rhs(0.0, y, p))
    J = np.zeros((2, 2))
    for j, d in enumerate(([h, 0.0], [0.0, h])):
        J[:, j] = (f(np.array([E, I]) + d) - f(np.array([E, I]) - d)) / (2 * h)
    return J


def _logit(x):
    return np.log(x / (1.0 - x))


def I_on_E_nullcline(E, p: Params):
    return (p.wEE * E + p.P - p.thE - _logit(E) / p.aE) / p.wEI


def E_on_I_nullcline(I, p: Params):
    return (p.thI + _logit(I) / p.aI + p.wII * I - p.Q) / p.wIE


def find_fixed_points(p: Params, n_grid=20001):
    """All fixed points in (0, 1)^2, found by scanning g(E) = 0 with brentq."""
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
    ev = np.linalg.eigvals(J)
    tr, det = np.trace(J), np.linalg.det(J)
    if det < 0:
        label = "saddle"
    else:
        kind = "focus" if np.iscomplex(ev).any() else "node"
        label = f"{'stable' if tr < 0 else 'unstable'} {kind}"
    return label, ev, tr, det


def oscillation_stats(t, E, discard=0.5):
    """(E_min, E_max, period) after discarding transient; period is nan if no oscillation."""
    k0 = int(len(t) * discard)
    tt, e = t[k0:], E[k0:]
    lo, hi = e.min(), e.max()
    if hi - lo < 1e-4:
        return lo, hi, np.nan
    pk = np.where((e[1:-1] > e[:-2]) & (e[1:-1] > e[2:]))[0] + 1
    if len(pk) < 3:
        return lo, hi, np.nan
    return lo, hi, float(np.mean(np.diff(tt[pk])))


def rhs_coupled(t, y, p1: Params, p2: Params, k: float):
    E1, I1, E2, I2 = y
    return [
        (-E1 + S(p1.wEE * E1 - p1.wEI * I1 + p1.P + k * E2, p1.aE, p1.thE)) / p1.tauE,
        (-I1 + S(p1.wIE * E1 - p1.wII * I1 + p1.Q, p1.aI, p1.thI)) / p1.tauI,
        (-E2 + S(p2.wEE * E2 - p2.wEI * I2 + p2.P + k * E1, p2.aE, p2.thE)) / p2.tauE,
        (-I2 + S(p2.wIE * E2 - p2.wII * I2 + p2.Q, p2.aI, p2.thI)) / p2.tauI,
    ]


def simulate_coupled(p1: Params, p2: Params, k, y0, t_end=2000.0, n=100001):
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(rhs_coupled, (0.0, t_end), y0, args=(p1, p2, k), t_eval=t_eval,
                    method="LSODA", rtol=1e-9, atol=1e-12)
    return sol.t, sol.y


def phase_from_peaks(t, x):
    """Phase in radians from peaks of x(t); returns (t_valid, phase)."""
    pk = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    if len(pk) < 4:
        return None, None
    tp = t[pk]
    m = (t >= tp[0]) & (t <= tp[-1])
    return t[m], np.interp(t[m], tp, 2.0 * np.pi * np.arange(len(tp)))


def sync_measures(t, y, discard=0.5):
    """Phase difference, frequency difference, PLV and correlation for two coupled regions."""
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