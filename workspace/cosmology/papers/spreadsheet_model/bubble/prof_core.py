"""Alternative CoE density profiles, same integrated model and background as bubble_core.
The CoE profile enters only through a tabulated enclosed-mass fraction m(r) = M(<r)/M(<D) (Gauss's law), truncated at Rb."""
import numpy as np, numba as nb
from bubble_core import (geometry, AFfour, ML, MA, r2, A_from, best_sine, RKM, GPC, GSI, TH1, IMIG, TH0, HT12, rb_of, shifted)
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid

RG = np.concatenate([[0.0], np.geomspace(1e-4, 12.0, 1500)])
from sedov import sedov_G
ST = {k: sedov_G(g) for k, g in (("st53", 5/3), ("st43", 4/3))}

def shape(kind, r, a):
    L = a.get("LD", 1.0); x = r / L
    if kind == "exp":    return np.exp(-x)
    if kind == "eta":    return np.exp(-(x ** a["eta"]))   # Einasto-like / gen. exp; eta=1 -> exp
    if kind == "pl":     return (1 + x) ** (-a["n"])
    if kind == "gauss":  return np.exp(-0.5 * x * x)
    if kind == "cored":  return 1 / (1 + x * x)
    if kind == "soft":   return (1 + np.abs(x) ** a["alpha"]) ** (-a["beta"])  # soft core+tail
    if kind == "shell":  return np.exp(-0.5 * ((r - a["rs"]) / a["w"]) ** 2)
    if kind in ST:       xi, G = ST[kind]; return np.interp(r / a["rst"], xi, G, left=0.0, right=0.0)
    raise ValueError(kind)

def tables(kind, a, Rb):
    s = shape(kind, RG, a) * (RG <= Rb)
    m = cumulative_trapezoid(4 * np.pi * RG**2 * s, RG, initial=0.0)
    m1 = np.interp(1.0, RG, m)
    return s / m1, m / m1          # density per unit M(<D) (times 1), enclosed mass fraction

@nb.njit(cache=True)
def acc(x, y, mu, RGt, MT, mub, Rb):
    r = max(np.sqrt(x * x + y * y), 1e-12)
    m = np.interp(r, RGt, MT)
    f = -mu * m / (r * r * r)
    f -= mub if r < Rb else mub * Rb**3 / (r * r * r)
    return f * x, f * y

@nb.njit(cache=True, parallel=True)
def integrate(q, mu, RGt, MT, mub, Rb, taui, nsteps):
    N = q.shape[0]
    X = np.empty((N, 2)); V = np.empty((N, 2)); VR = np.empty(N)
    s0 = np.log(taui); h = -s0 / nsteps
    for i in nb.prange(N):
        x = q[i, 0] * taui; y = q[i, 1] * taui; vx = q[i, 0]; vy = q[i, 1]; s = s0; vrmin = 1e30
        for n in range(nsteps):
            t1 = np.exp(s); t2 = np.exp(s + 0.5 * h); t4 = np.exp(s + h)
            ax, ay = acc(x, y, mu, RGt, MT, mub, Rb)
            k1x = t1 * vx; k1y = t1 * vy; l1x = t1 * ax; l1y = t1 * ay
            ax, ay = acc(x + 0.5*h*k1x, y + 0.5*h*k1y, mu, RGt, MT, mub, Rb)
            k2x = t2 * (vx + 0.5*h*l1x); k2y = t2 * (vy + 0.5*h*l1y); l2x = t2 * ax; l2y = t2 * ay
            ax, ay = acc(x + 0.5*h*k2x, y + 0.5*h*k2y, mu, RGt, MT, mub, Rb)
            k3x = t2 * (vx + 0.5*h*l2x); k3y = t2 * (vy + 0.5*h*l2y); l3x = t2 * ax; l3y = t2 * ay
            ax, ay = acc(x + h*k3x, y + h*k3y, mu, RGt, MT, mub, Rb)
            k4x = t4 * (vx + h*l3x); k4y = t4 * (vy + h*l3y); l4x = t4 * ax; l4y = t4 * ay
            x += h/6*(k1x + 2*k2x + 2*k3x + k4x); y += h/6*(k1y + 2*k2y + 2*k3y + k4y)
            vx += h/6*(l1x + 2*l2x + 2*l3x + l4x); vy += h/6*(l1y + 2*l2y + 2*l3y + l4y)
            s += h
            vr = (x*vx + y*vy) / np.sqrt(x*x + y*y)
            if vr < vrmin: vrmin = vr
        X[i, 0] = x; X[i, 1] = y; V[i, 0] = vx; V[i, 1] = vy; VR[i] = vrmin
    return X, V, VR

def phi_coe(r, mu, MT):
    """exact potential of the (truncated) CoE profile: -mu[m(r)/r + int_r^inf m'(r')/r' dr']"""
    dm = np.gradient(MT, RG); integ = cumulative_trapezoid((dm / np.maximum(RG, 1e-12))[::-1], -RG[::-1], initial=0.0)[::-1]
    return -mu * (np.interp(r, RG, MT) / r + np.interp(r, RG, integ))

def run(kind, a, mc, Ob, fac=1.5, taui=0.01, nsteps=200, iters=3):
    DR = a["DR"]; q, rr = geometry(TH1, DR); Rb = rb_of(DR, fac); Ht = 0.95
    S, MT = tables(kind, a, Rb)
    for _ in range(iters):
        mu = mc * Ht**2 / 2; mub = Ob * Ht**2 / 2
        X, V, VR = integrate(q, mu, RG, MT, mub, Rb, taui, nsteps)
        Ht = np.hypot(*V[0]) / np.hypot(*X[0])
    vrel = V[1:] - V[0]; af = np.hypot(vrel[:, 0], vrel[:, 1]) / rr
    rE = np.hypot(*X[0]); vE = np.hypot(*V[0])
    ph = phi_coe(rE, mu, MT) - mub * (3 * Rb**2 - rE**2) / 2
    rhob = 3 * mub / (4 * np.pi)
    rhot = lambda r: mu * np.interp(r, RG, S) + rhob
    rs = 1.0 / DR; r34 = 3.4 / (DR * RKM / GPC); rD = rhot(1.0)
    return dict(af=af, vrmin=float(VR.min()), E=float(0.5 * vE**2 + ph), vesc2=float(vE**2 / (-2 * ph)), rE=rE, vE=vE, Ht=float(Ht), age=float(Ht * TH0),
                mu=mu, mub=mub, Rb=Rb, MT=MT, S=S,
                contrast=float((rhot(1 - rs) - rhot(1 + rs)) / rD), c34=float((rhot(1 - r34) - rhot(1 + r34)) / rD),
                m_out=float(np.interp(1 - rs, RG, MT)))       # fraction of M(<D) enclosed inside the bubble's near edge

def t_turn(rr):
    if rr["E"] >= 0: return np.inf
    mu, mub, Rb, MT = rr["mu"], rr["mub"], rr["Rb"], rr["MT"]
    def f(t, y):
        r = y[0]; g = mu * np.interp(r, RG, MT) / r**2 + (mub * r if r < Rb else mub * Rb**3 / r**2); return [y[1], -g]
    ev = lambda t, y: y[1]; ev.terminal = True; ev.direction = -1
    s = solve_ivp(f, (1.0, 1e5), [rr["rE"], rr["vE"]], events=ev, rtol=1e-10, atol=1e-12)
    return float(s.t_events[0][0]) if len(s.t_events[0]) else np.inf

def score(kind, p, target, norm="mean", cmax=None):
    rr = run(kind, p, p["mc"], p["Ob"])
    af = rr["af"]
    if (not np.all(np.isfinite(af))) or rr["vrmin"] <= 0: return -np.inf, None, None, rr
    if cmax is not None and abs(rr["contrast"]) > cmax: return -np.inf, None, None, rr
    if kind == "shell" and p["rs"] > 1 - 1 / p["DR"]: return -np.inf, None, None, rr   # shell centre must lie inside the bubble's near edge
    if norm == "mean": af = af / af.mean()
    if norm == "mean2": af = af**2 / np.mean(af**2)
    l0 = p.get("l0", 0.0); af = shifted(af, l0) if l0 else af; th = TH1 - l0
    if target == "mig": afx, thx, y = af[IMIG], th[IMIG], MA
    else: afx, thx, y = af, th, AFfour(TH1)
    a, ph = best_sine(afx, thx, y)
    return r2(y, A_from(afx, thx, a, ph)), a, ph, rr

LB = dict(DR=(1.05, 10.0), LD=(0.01, 3.0), mc=(1e-3, 30.0), n=(1.5, 8.0), eta=(0.05, 8.0), alpha=(0.25, 8.0), beta=(0.25, 8.0), rs=(0.02, 0.95), rst=(0.05, 4.0), w=(0.01, 0.5), l0=(0, 360))
LOGV = dict(DR=False, LD=True, mc=True, n=False, eta=False, alpha=False, beta=False, rs=False, rst=True, w=True, l0=False)
rng = np.random.default_rng(21)
def unpack(names, u):
    out = {}
    for n, ui in zip(names, u):
        lo, hi = LB[n]
        if n == "l0": out[n] = (ui % 1) * 360; continue
        ui = min(max(ui, 0), 1)
        out[n] = float(np.exp(np.log(lo) + ui*(np.log(hi)-np.log(lo)))) if LOGV[n] else lo + ui*(hi-lo)
    return out
def pack(names, p):
    u = []
    for n in names:
        lo, hi = LB[n]; v = p[n]
        if n == "l0": u.append(v/360); continue
        u.append((np.log(v)-np.log(lo))/(np.log(hi)-np.log(lo)) if LOGV[n] else (v-lo)/(hi-lo))
    return np.clip(np.array(u), 0, 1)
def fit(kind, target, free, fixed, starts=5, nscreen=200, seeds=(), **kw):
    def full(u): p = dict(fixed); p.update(unpack(free, u)); return p
    def obj(u):
        pen = sum(10*(max(0, -x) + max(0, x-1))**2 for n, x in zip(free, u) if n != "l0")
        v = score(kind, full(u), target, **kw)[0]
        return (1 - v if np.isfinite(v) else 50.0) + pen
    U = [rng.random(len(free)) for _ in range(nscreen)] + [pack(free, dict(fixed, **s)) for s in seeds if all(n in dict(fixed, **s) for n in free)]
    vals = np.array([obj(u) for u in U]); order = np.argsort(vals)
    best = (np.inf, None)
    for i in order[:starts]:
        if vals[i] >= 50: continue
        r = minimize(obj, U[i], method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-7, maxiter=600))
        if r.fun < best[0]: best = (r.fun, r.x)
    if best[1] is None: return None
    p = full(best[1]); v, a, ph, rr = score(kind, p, target, **kw); p.update(amp=a, phi=ph, R2=v); return p

def describe(kind, p, norm="mean"):
    rr = run(kind, p, p["mc"], p["Ob"])
    af = rr["af"] / (rr["af"].mean() if norm == "mean" else 1)
    if norm == "mean2": af = rr["af"]**2 / np.mean(rr["af"]**2)
    l0 = p.get("l0", 0.0); afs = shifted(af, l0) if l0 else af; A = A_from(afs, TH1 - l0, p["amp"], p["phi"])
    Dm = p["DR"] * RKM * 1e3; M = rr["mu"] * Dm**3 / (GSI * (rr["age"] * 3.156e16)**2)
    out = dict(p); out.update(kind=kind, R2_mig=float(r2(MA, A[IMIG])), R2_four=float(r2(AFfour(TH1), A)), D_Gpc=p["DR"] * RKM / GPC,
                              M_kg=M, M_sun=M / 1.989e30, contrast=rr["contrast"], c34=rr["c34"], vesc2=rr["vesc2"], age=rr["age"], vrmin=rr["vrmin"],
                              t_turn=t_turn(rr), m_out=rr["m_out"], slopeD=float(np.interp(1.0, RG, np.gradient(np.log(np.maximum(rr["MT"], 1e-300)), np.log(np.maximum(RG, 1e-12))))),
                              m_far=float(np.interp(1 + 1 / p["DR"], RG, rr["MT"])), mc_today=p["mc"])
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in out.items()}
