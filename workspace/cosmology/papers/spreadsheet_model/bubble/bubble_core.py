"""Integrated fixed-profile model + uniform background sphere (centred on CoE, fixed in space, radius Rb in units of D).
Units: length D, time t0 (duration of the integration). mu = G M t0^2/D^3 (CoE mass inside D);
background: density rho_b, mub = G (4/3 pi rho_b D^3) t0^2/D^3 = Omega_b (Ht)_E^2 / 2, with (Ht)_E = Earth's v t0/r today."""
import sys, numpy as np, numba as nb
sys.path.insert(0, "../integrated")
from integ_core import Fx, geometry, AFfour, ML, MA, MLO, MHI, r2, A_from, best_sine, MCRIT_FAC, K, H0, GSI, RKM, GPC
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
TH1 = np.arange(0, 360, 1.0); IMIG = ML.astype(int)
TH0 = 1 / H0 / 3.156e16                     # 1/H0 in Gyr (H0 = 70)
HT12 = 12.0 / TH0

@nb.njit(cache=True)
def acc(x, y, mu, LD, FD, mub, Rb):
    r = max(np.sqrt(x * x + y * y), 1e-12)
    f = -mu * Fx(r / LD) / FD / (r * r * r)
    f -= mub if r < Rb else mub * Rb**3 / (r * r * r)
    return f * x, f * y

@nb.njit(cache=True, parallel=True)
def integrate(q, mu, LD, mub, Rb, taui, nsteps):
    N = q.shape[0]; FD = Fx(1.0 / LD)
    X = np.empty((N, 2)); V = np.empty((N, 2)); VR = np.empty(N)
    s0 = np.log(taui); h = -s0 / nsteps
    for i in nb.prange(N):
        x = q[i, 0] * taui; y = q[i, 1] * taui; vx = q[i, 0]; vy = q[i, 1]; s = s0; vrmin = 1e30
        for n in range(nsteps):
            t1 = np.exp(s); t2 = np.exp(s + 0.5 * h); t4 = np.exp(s + h)
            ax, ay = acc(x, y, mu, LD, FD, mub, Rb)
            k1x = t1 * vx; k1y = t1 * vy; l1x = t1 * ax; l1y = t1 * ay
            ax, ay = acc(x + 0.5*h*k1x, y + 0.5*h*k1y, mu, LD, FD, mub, Rb)
            k2x = t2 * (vx + 0.5*h*l1x); k2y = t2 * (vy + 0.5*h*l1y); l2x = t2 * ax; l2y = t2 * ay
            ax, ay = acc(x + 0.5*h*k2x, y + 0.5*h*k2y, mu, LD, FD, mub, Rb)
            k3x = t2 * (vx + 0.5*h*l2x); k3y = t2 * (vy + 0.5*h*l2y); l3x = t2 * ax; l3y = t2 * ay
            ax, ay = acc(x + h*k3x, y + h*k3y, mu, LD, FD, mub, Rb)
            k4x = t4 * (vx + h*l3x); k4y = t4 * (vy + h*l3y); l4x = t4 * ax; l4y = t4 * ay
            x += h/6*(k1x + 2*k2x + 2*k3x + k4x); y += h/6*(k1y + 2*k2y + 2*k3y + k4y)
            vx += h/6*(l1x + 2*l2x + 2*l3x + l4x); vy += h/6*(l1y + 2*l2y + 2*l3y + l4y)
            s += h
            vr = (x*vx + y*vy) / np.sqrt(x*x + y*y)
            if vr < vrmin: vrmin = vr
        X[i, 0] = x; X[i, 1] = y; V[i, 0] = vx; V[i, 1] = vy; VR[i] = vrmin
    return X, V, VR

def phi_pot(r, mu, LD, mub, Rb):
    u = r / LD; Mt = mu / Fx(1.0 / LD)
    p = -Mt * (Fx(u) / r + (1 + u) * np.exp(-u) / (2 * LD))
    p += -mub * (3 * Rb**2 - r * r) / 2 if r < Rb else -mub * Rb**3 / r
    return p

def rho(r, mu, LD, mub):                 # G*rho*t0^2 (units), CoE profile + background
    return mu * np.exp(-r / LD) / (8 * np.pi * LD**3 * Fx(1.0 / LD)) + 3 * mub / (4 * np.pi)

def rb_of(DR, fac): return fac * (1 + 1.0 / DR)

def run(DR, LD, mc, Ob, fac=1.5, taui=0.01, nsteps=200, iters=3):
    """Ob = background fraction of critical density TODAY (H0 at Earth). mc = M/Mcrit today = 2 mu/(Ht)^2."""
    q, rr = geometry(TH1, DR); Rb = rb_of(DR, fac); Ht = 0.95
    for _ in range(iters):
        mu = mc * Ht**2 / 2; mub = Ob * Ht**2 / 2
        X, V, VR = integrate(q, mu, LD, mub, Rb, taui, nsteps)
        Ht = np.hypot(*V[0]) / np.hypot(*X[0])
    vrel = V[1:] - V[0]; af = np.hypot(vrel[:, 0], vrel[:, 1]) / rr
    rE = np.hypot(*X[0]); vE = np.hypot(*V[0])
    ph = phi_pot(rE, mu, LD, mub, Rb)
    rs = 1.0 / DR; r34 = 3.4 / (DR * RKM / GPC)
    rD = rho(1.0, mu, LD, mub)
    return dict(af=af, vrmin=float(VR.min()), E=float(0.5 * vE**2 + ph), vesc2=float(vE**2 / (-2 * ph)), rE=float(rE), vE=float(vE),
                Ht=float(Ht), mu=mu, mub=mub, Rb=Rb, age=float(Ht * TH0),
                contrast=float((rho(1 - rs, mu, LD, mub) - rho(1 + rs, mu, LD, mub)) / rD),
                c34=float((rho(1 - r34, mu, LD, mub) - rho(1 + r34, mu, LD, mub)) / rD),
                fcoe=float(rho(1.0, mu, LD, 0.0) / rD))

def t_turn(rr, LD):
    if rr["E"] >= 0: return np.inf
    mu, mub, Rb = rr["mu"], rr["mub"], rr["Rb"]; FD = Fx(1 / LD)
    def f(t, y):
        r = y[0]; g = mu * Fx(r / LD) / FD / r**2 + (mub * r if r < Rb else mub * Rb**3 / r**2); return [y[1], -g]
    ev = lambda t, y: y[1]; ev.terminal = True; ev.direction = -1
    s = solve_ivp(f, (1.0, 1e5), [rr["rE"], rr["vE"]], events=ev, rtol=1e-10, atol=1e-12)
    return float(s.t_events[0][0]) if len(s.t_events[0]) else np.inf

def shifted(af, l0): return np.interp((TH1 - l0) % 360, np.append(TH1, 360), np.append(af, af[0]))

def score(p, target, norm="mean", cmax=None, need_bound=False, need_age=False, fac=1.5):
    rr = run(p["DR"], p["LD"], p["mc"], p["Ob"], fac)
    af = rr["af"]
    bad = (not np.all(np.isfinite(af))) or rr["vrmin"] <= 0
    if cmax is not None and abs(rr["contrast"]) > cmax: bad = True
    if need_bound and rr["E"] > 0: bad = True
    if need_age and rr["Ht"] < HT12: bad = True
    if bad: return -np.inf, None, None, rr
    if norm == "mean": af = af / af.mean()
    l0 = p.get("l0", 0.0); af = shifted(af, l0) if l0 else af; th = TH1 - l0
    if target == "mig": afx, thx, y = af[IMIG], th[IMIG], MA
    else: afx, thx, y = af, th, AFfour(TH1)
    a, ph = best_sine(afx, thx, y)
    return r2(y, A_from(afx, thx, a, ph)), a, ph, rr

LB = dict(DR=(1.05, 10.0), LD=(0.01, 3.0), mc=(1e-3, 30.0), Ob=(0.1, 1.0), l0=(0, 360))
LOGV = dict(DR=False, LD=True, mc=True, Ob=False, l0=False)
rng = np.random.default_rng(11)
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
def fit(target, free, fixed, starts=6, nscreen=250, seeds=(), **kw):
    def full(u): p = dict(fixed); p.update(unpack(free, u)); return p
    def obj(u):
        pen = sum(10*(max(0, -x) + max(0, x-1))**2 for n, x in zip(free, u) if n != "l0")
        v = score(full(u), target, **kw)[0]
        return (1 - v if np.isfinite(v) else 50.0) + pen
    U = [rng.random(len(free)) for _ in range(nscreen)] + [pack(free, dict(fixed, **s)) for s in seeds]
    vals = np.array([obj(u) for u in U]); order = np.argsort(vals)
    best = (np.inf, None)
    for i in order[:starts]:
        if vals[i] >= 50: continue
        r = minimize(obj, U[i], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=1500))
        if r.fun < best[0]: best = (r.fun, r.x)
    if best[1] is None: return None
    p = full(best[1]); v, a, ph, rr = score(p, target, **kw)
    p.update(amp=a, phi=ph, R2=v); return p, rr

def describe(p, norm="mean", fac=1.5):
    rr = run(p["DR"], p["LD"], p["mc"], p["Ob"], fac)
    af = rr["af"] / (rr["af"].mean() if norm == "mean" else 1)
    l0 = p.get("l0", 0.0); afs = shifted(af, l0) if l0 else af; A = A_from(afs, TH1 - l0, p["amp"], p["phi"])
    Dm = p["DR"] * RKM * 1e3; t0s = rr["age"] * 3.156e16
    M = rr["mu"] * Dm**3 / (GSI * t0s**2)
    out = dict(p); out.update(R2_mig=float(r2(MA, A[IMIG])), R2_four=float(r2(AFfour(TH1), A)), D_Gpc=p["DR"] * RKM / GPC, L_Gpc=p["LD"] * p["DR"] * RKM / GPC,
                              M_kg=M, M_sun=M / 1.989e30, contrast=rr["contrast"], c34=rr["c34"], vesc2=rr["vesc2"], E=rr["E"], age=rr["age"], Ht=rr["Ht"],
                              vrmin=rr["vrmin"], t_turn=t_turn(rr, p["LD"]), fcoe=rr["fcoe"], young=rr["age"] < 12,
                              vesc2_Rb1=run(p["DR"], p["LD"], p["mc"], p["Ob"], 1.0)["vesc2"], vesc2_Rb3=run(p["DR"], p["LD"], p["mc"], p["Ob"], 3.0)["vesc2"])
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.bool_)) else v) for k, v in out.items()}
