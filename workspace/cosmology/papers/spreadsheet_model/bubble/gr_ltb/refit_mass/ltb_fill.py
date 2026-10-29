"""Copy of ../ltb_thin.py plus p[12] = uniform interior fill density (0 < r < a). LTB (dust, Lambda=0) for the thin-shell-plus-core configuration. Reuses papers/cmb_redshift/ltb.py
(same conventions: G=c=1, Gpc, Rdot^2 = 2M/R + 2E, R = M^(1/3) s^(2/3) g(w)); only the mass function differs.
Parameter vector p (float64, length 12):
 p[1] Mc (core geometric mass, Gpc), p[2] Lc (core e^{-r/Lc} scale, label units)
 p[3] modeE: 0 -> E = p[4] r^2 ;  1 -> E fixed by R(r,t0) = r (label = areal radius today) at age t0 - t_B
 p[5] t0 ; p[6], p[7]: bang time t_B = p[6] exp(-(r/p[7])^2) (0 = uniform bang)
 p[8] rho0 (shell density on the label, geometric, Gpc^-2), p[9] a = inner edge, p[10] b = outer edge, p[11] w = C2 smoothstep half-width
 Shell density on the label: rho0 [s((r-a)/w) - s((r-b)/w)],  s(u) = 1/2 + (15u - 10u^3 + 3u^5)/16 on [-1,1]."""
import sys, os, numpy as np, numba as nb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../../cmb_redshift"))
import ltb as L0
from ltb import gfun, ginv, Fexp, GYR, H0_70
XG, WG = np.polynomial.legendre.leggauss(8)

@nb.njit(cache=True)
def sstep(u):
    if u <= -1: return 0.0, 0.0, 0.0
    if u >= 1: return 1.0, 0.0, 0.0
    return 0.5 + (15 * u - 10 * u**3 + 3 * u**5) / 16, 15 * (1 - u * u)**2 / 16, -60 * u * (1 - u * u) / 16

@nb.njit(cache=True)
def J(r, c, w):
    """int_0^r x^2 s((x-c)/w) dx"""
    lo = c - w
    if r <= lo: return 0.0
    hi = min(r, c + w); h = 0.5 * (hi - lo); m = 0.5 * (hi + lo); tot = 0.0
    for k in range(8):
        x = m + h * XG[k]; tot += WG[k] * x * x * sstep((x - c) / w)[0]
    tot *= h
    if r > c + w: tot += (r**3 - (c + w)**3) / 3
    return tot

@nb.njit(cache=True)
def massfun(r, p):
    Mc, Lc, rho0, a, b, w = p[1], p[2], p[8], p[9], p[10], p[11]
    F, F1, F2 = Fexp(r / Lc)
    sa, sa1, sa2 = sstep((r - a) / w); sb, sb1, sb2 = sstep((r - b) / w)
    rho = rho0 * (sa - sb); rho1 = rho0 * (sa1 - sb1) / w
    Ms = 4 * np.pi * rho0 * (J(r, a, w) - J(r, b, w))
    ri = p[12]                                   # uniform interior fill rho_in (1 - s((r-a)/w))
    rho += ri * (1 - sa); rho1 -= ri * sa1 / w
    Ms += 4 * np.pi * ri * (r**3 / 3 - J(r, a, w))
    return Mc * F + Ms, Mc * F1 / Lc + 4 * np.pi * r * r * rho, Mc * F2 / (Lc * Lc) + 4 * np.pi * (2 * r * rho + r * r * rho1)

@nb.njit(cache=True)
def bangfun(r, p):
    A = p[6]
    if A == 0.0: return 0.0, 0.0, 0.0
    b = p[7]; x = r / b; e = A * np.exp(-x * x)
    return e, -2 * x / b * e, (4 * x * x - 2) / (b * b) * e

@nb.njit(cache=True)
def energy(r, p, M, M1, M2, tb, tb1, tb2):
    if p[3] == 0:
        return p[4] * r * r, 2 * p[4] * r, 2 * p[4]
    s0 = p[5] - tb; s01 = -tb1; s02 = -tb2
    G = r * M ** (-1 / 3) * s0 ** (-2 / 3)
    lG1 = 1 / r - M1 / (3 * M) - (2 / 3) * s01 / s0
    lG2 = -1 / (r * r) - (M2 / M - (M1 / M) ** 2) / 3 - (2 / 3) * (s02 / s0 - (s01 / s0) ** 2)
    G1 = G * lG1; G2 = G * (lG2 + lG1 * lG1)
    w = ginv(G); g, g1, g2 = gfun(w)
    w1 = G1 / g1; w2 = (G2 - g2 * w1 * w1) / g1
    B = M ** (2 / 3) * s0 ** (-2 / 3)
    lB1 = (2 / 3) * M1 / M - (2 / 3) * s01 / s0
    lB2 = (2 / 3) * (M2 / M - (M1 / M) ** 2) - (2 / 3) * (s02 / s0 - (s01 / s0) ** 2)
    B1 = B * lB1; B2 = B * (lB2 + lB1 * lB1)
    return w * B, w1 * B + w * B1, w2 * B + 2 * w1 * B1 + w * B2

@nb.njit(cache=True)
def geom(r, t, p):
    M, M1, M2 = massfun(r, p)
    tb, tb1, tb2 = bangfun(r, p)
    E, E1, E2 = energy(r, p, M, M1, M2, tb, tb1, tb2)
    s = t - tb; s1 = -tb1; s2 = -tb2
    a = M ** (1 / 3) * s ** (2 / 3)
    la1 = M1 / (3 * M) + (2 / 3) * s1 / s
    la2 = (M2 / M - (M1 / M) ** 2) / 3 + (2 / 3) * (s2 / s - (s1 / s) ** 2)
    a1 = a * la1; a2 = a * (la2 + la1 * la1)
    at = (2 / 3) * a / s; at1 = (2 / 3) * (a1 / s - a * s1 / (s * s))
    b = M ** (-2 / 3) * s ** (2 / 3)
    lb1 = -(2 / 3) * M1 / M + (2 / 3) * s1 / s
    lb2 = -(2 / 3) * (M2 / M - (M1 / M) ** 2) + (2 / 3) * (s2 / s - (s1 / s) ** 2)
    b1 = b * lb1; b2 = b * (lb2 + lb1 * lb1)
    w = E * b; w1 = E1 * b + E * b1; w2 = E2 * b + 2 * E1 * b1 + E * b2
    wt = (2 / 3) * w / s; wt1 = (2 / 3) * (w1 / s - w * s1 / (s * s))
    g, g1, g2 = gfun(w)
    R = a * g
    Rr = a1 * g + a * g1 * w1
    Rrr = a2 * g + 2 * a1 * g1 * w1 + a * g2 * w1 * w1 + a * g1 * w2
    Rt = at * g + a * g1 * wt
    Rtr = at1 * g + at * g1 * w1 + a1 * g1 * wt + a * g2 * wt * w1 + a * g1 * wt1
    return R, Rr, Rrr, Rt, Rtr, E, E1, M, M1

@nb.njit(cache=True)
def rhs(x, y, p, J_, t0):
    t = t0 * np.exp(-x); r = y[0]; kt = y[2]; kr = y[3]
    R, Rr, Rrr, Rt, Rtr, E, E1, M, M1 = geom(r, t, p)
    kth = J_ / (R * R); ee = 1 + 2 * E
    dkt = -(Rr * Rtr / ee) * kr * kr - R * Rt * kth * kth
    dkr = -2 * (Rtr / Rr) * kt * kr - (Rrr / Rr - E1 / ee) * kr * kr + (R * ee / Rr) * kth * kth
    f = -t / kt
    out = np.empty(4)
    out[0] = f * kr; out[1] = f * kth; out[2] = f * dkt; out[3] = f * dkr
    return out

@nb.njit(cache=True)
def density(r, t, p):
    R, Rr, Rrr, Rt, Rtr, E, E1, M, M1 = geom(r, t, p)
    return M1 / (4 * np.pi * R * R * Rr)

def local_rates(r, t, p):
    R, Rr, Rrr, Rt, Rtr, E, E1, M, M1 = geom(r, t, p)
    return Rt / R, Rtr / Rr
