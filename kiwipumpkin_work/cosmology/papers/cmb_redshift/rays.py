"""Off-centre past null geodesics in LTB (Alnes & Amarzguioui 2006 formulation, planar polar angle, t as clock)."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import ltb

def start(p, t0, robs, psi):
    """psi: angle between the viewing direction and the direction to the centre (CoE)."""
    R, Rr, Rrr, Rt, Rtr, E, E1, M, M1 = ltb.geom(robs, t0, p)
    kr = -np.cos(psi) * np.sqrt(1 + 2 * E) / Rr
    return np.array([robs, 0.0, -1.0, kr]), R * np.sin(psi)

def trace(p, t0, robs, psi, t_end, rtol=1e-12, dense=True):
    y0, J = start(p, t0, robs, psi)
    xe = np.log(t0 / t_end)
    sol = solve_ivp(lambda x, y: ltb.rhs(x, y, p, J, t0), (0.0, xe), y0, method="DOP853",
                    rtol=rtol, atol=1e-13, dense_output=dense)
    sol.J = J
    return sol

def null_err(p, t0, sol):
    """max |g(k,k)| / kt^2 along the stored steps."""
    e = 0.0
    for x, y in zip(sol.t, sol.y.T):
        t = t0 * np.exp(-x); R, Rr, Rrr, Rt, Rtr, E, E1, M, M1 = ltb.geom(y[0], t, p)
        g = -y[2] ** 2 + Rr ** 2 / (1 + 2 * E) * y[3] ** 2 + (sol.J / R ** 2) ** 2 * R ** 2
        e = max(e, abs(g) / y[2] ** 2)
    return e

def z_at_t(sol, t0, t):
    return -sol.sol(np.log(t0 / t))[2] - 1.0

def z_at_rho(sol, p, t0, rho_rec):
    """first crossing (going back) of local density rho_rec; returns (z, t)."""
    f = lambda x: np.log(ltb.density(sol.sol(x)[0], t0 * np.exp(-x), p)) - np.log(rho_rec)
    xs = np.linspace(sol.t[0], sol.t[-1], 400); fv = np.array([f(x) for x in xs])
    i = np.where(np.diff(np.sign(fv)) != 0)[0]
    if len(i) == 0: return np.nan, np.nan
    x = brentq(f, xs[i[0]], xs[i[0] + 1], xtol=1e-14, rtol=1e-14)
    return -sol.sol(x)[2] - 1.0, t0 * np.exp(-x)

def legendre_coeffs(mu, wts, f, lmax=6):
    """f(mu) = sum c_l P_l(mu) from Gauss-Legendre nodes."""
    from numpy.polynomial.legendre import legval
    c = []
    for l in range(lmax + 1):
        e = np.zeros(l + 1); e[l] = 1
        c.append((2 * l + 1) / 2 * np.sum(wts * f * legval(mu, e)))
    return np.array(c)

def area_distance(p, t0, robs, psi, tq, dpsi=2e-5):
    """Angular-diameter distance at clock times tq (array) for direction psi, from neighbouring rays at fixed t.
    Uses g(xi,k)=0, so |xi|^2 = g(xi,xi) is the screen separation; the azimuthal side is R|sin theta|."""
    sols = [trace(p, t0, robs, psi + s * dpsi, tq.min() * 0.999) for s in (-1, 0, 1)]
    xq = np.log(t0 / tq)
    Ym, Y0, Yp = [s.sol(xq) for s in sols]
    out = []; zz = []
    for j, t in enumerate(tq):
        R, Rr, Rrr, Rt, Rtr, E, E1, M, M1 = ltb.geom(Y0[0, j], t, p)
        dr = (Yp[0, j] - Ym[0, j]) / (2 * dpsi); dth = (Yp[1, j] - Ym[1, j]) / (2 * dpsi)
        xi = np.sqrt(Rr ** 2 / (1 + 2 * E) * dr ** 2 + R ** 2 * dth ** 2)
        out.append(np.sqrt(xi * R * abs(np.sin(Y0[1, j])) / np.sin(psi)))
        zz.append(-Y0[2, j] - 1)
    return np.array(zz), np.array(out)
