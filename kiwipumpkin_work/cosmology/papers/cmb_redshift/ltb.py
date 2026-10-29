"""Lemaitre-Tolman-Bondi dust model, G = c = 1, lengths in Gpc, times in Gpc/c.
Metric: ds^2 = -dt^2 + R'^2/(1+2E) dr^2 + R^2 dOmega^2,  Rdot^2 = 2M/R + 2E.
Exact solution in universal form R = M^(1/3) s^(2/3) g(w), w = E s^(2/3) M^(-2/3), s = t - t_B(r).
Model parameter vector p (float64):
 p[0] modeM: 0 -> M = p[1] r^3 (FRW);  1 -> M = p[1] F(r/p[2]),  F(X) = 1 - e^-X (1+X+X^2/2)  (e^{-r/L} density)
 p[3] modeE: 0 -> E = p[4] r^2;  1 -> E fixed by R(r, t0) = r (label = areal radius today) with age t0 - t_B
 p[5] t0 ;  p[6], p[7]: bang time t_B(r) = p[6] exp(-(r/p[7])^2);  p[8]: background M = p[8] r^3 added if modeM = 2"""
import numpy as np, numba as nb, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
GC = np.array(json.load(open(os.path.join(HERE, "gseries.json"))))
WC = -(2 * np.pi) ** (2 / 3) / 2          # crunch
GPC_S = 3.0857e25 / 2.99792458e8           # 1 Gpc/c in s
GYR = GPC_S / 3.15576e16                   # 1 Gpc/c in Gyr (3.2616)
H0_70 = 70 / 299792.458 * 1000.0           # H0 = 70 in Gpc^-1

@nb.njit(cache=True)
def eta_hyp(T):
    e = np.cbrt(6 * T) if T < 1 else np.log(2 * T + 2.0)
    for _ in range(100):
        f = np.sinh(e) - e - T; d = np.cosh(e) - 1
        de = f / d; e -= de
        if abs(de) < 1e-15 * e: break
    return e

@nb.njit(cache=True)
def eta_ell(T):
    lo, hi = 0.0, 2 * np.pi
    e = min(np.cbrt(6 * T), 6.0)
    for _ in range(200):
        f = e - np.sin(e) - T
        if f > 0: hi = e
        else: lo = e
        d = 1 - np.cos(e)
        en = e - f / d if d > 0 else 0.5 * (lo + hi)
        if not (lo < en < hi): en = 0.5 * (lo + hi)
        if abs(en - e) < 1e-15 * e: e = en; break
        e = en
    return e

@nb.njit(cache=True)
def gfun(w):
    """g, dg/dw, d2g/dw2."""
    if abs(w) < 0.15:
        g = 0.0; g1 = 0.0; g2 = 0.0; n = GC.shape[0]
        for j in range(n - 1, -1, -1):
            g = g * w + GC[j]
        for j in range(n - 1, 0, -1):
            g1 = g1 * w + j * GC[j]
        for j in range(n - 1, 1, -1):
            g2 = g2 * w + j * (j - 1) * GC[j]
        return g, g1, g2
    T = (2 * abs(w)) ** 1.5
    if w > 0:
        e = eta_hyp(T); g = (np.cosh(e) - 1) / (2 * w)
        h = 1.5 * np.sqrt(2 * w) * np.sinh(e) / (np.cosh(e) - 1)
    else:
        e = eta_ell(T); g = (1 - np.cos(e)) / (2 * abs(w))
        h = 1.5 * np.sqrt(2 * abs(w)) * np.sin(e) / (1 - np.cos(e))
    g1 = (h - g) / w
    h1 = (h - 4.5 / (g * g)) / (2 * w)
    g2 = (h1 - 2 * g1) / w
    return g, g1, g2

@nb.njit(cache=True)
def ginv(G):
    lo, hi = WC + 1e-12, max(1.0, G * G)
    w = 0.0 if G < 1.7 else 0.5 * G * G - 1.0
    for _ in range(200):
        g, g1, g2 = gfun(w)
        f = g - G
        if f > 0: hi = w
        else: lo = w
        wn = w - f / g1
        if not (lo < wn < hi): wn = 0.5 * (lo + hi)
        if abs(wn - w) < 1e-15 * max(1.0, abs(w)): w = wn; break
        w = wn
    return w

@nb.njit(cache=True)
def Fexp(X):
    """F, F', F'' for F = 1 - e^-X (1+X+X^2/2)."""
    ex = np.exp(-X)
    if X < 2.0:
        s = 0.0; term = X ** 3 / 6.0; n = 3
        while True:
            s += term; n += 1; term *= X / n
            if term < 1e-18 * s: break
        F = ex * s
    else:
        F = 1 - ex * (1 + X + 0.5 * X * X)
    return F, 0.5 * X * X * ex, ex * (X - 0.5 * X * X)

@nb.njit(cache=True)
def massfun(r, p):
    if p[0] == 0:
        return p[1] * r ** 3, 3 * p[1] * r * r, 6 * p[1] * r
    L = p[2]; F, F1, F2 = Fexp(r / L)
    if p[0] == 2:     # homogeneous background p[8] r^3 plus the e^{-r/L} lump
        return p[1] * F + p[8] * r ** 3, p[1] * F1 / L + 3 * p[8] * r * r, p[1] * F2 / (L * L) + 6 * p[8] * r
    return p[1] * F, p[1] * F1 / L, p[1] * F2 / (L * L)

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
    """Returns R, R', R'', Rdot, Rdot', E, E', M, M'."""
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
def rhs(x, y, p, J, t0):
    """y = (r, theta, kt, kr); independent variable x = ln(t0/t) along the past light cone."""
    t = t0 * np.exp(-x); r = y[0]; kt = y[2]; kr = y[3]
    R, Rr, Rrr, Rt, Rtr, E, E1, M, M1 = geom(r, t, p)
    kth = J / (R * R); ee = 1 + 2 * E
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
    return Rt / R, Rtr / Rr        # H_perp, H_r
