"""Point (compact core) + constant-density Sedov-Taylor shell, built on prof_core (tabulated enclosed mass, same integrator).
Parameters: mcc = M_c/M_crit (today), Os = rho_s/rho_crit (today), rin, rout (units D), Lc = core scale (exp profile), DR, l0.
No separate background (Ob = 0): the shell is the uniform component."""
import numpy as np
import prof_core as P
from prof_core import RG
from scipy.integrate import cumulative_trapezoid

LC_DEFAULT = 0.02
def build(a):
    Lc = a.get("Lc", LC_DEFAULT)
    core = np.exp(-RG / Lc); mcore = cumulative_trapezoid(4 * np.pi * RG**2 * core, RG, initial=0.0)
    core = core / np.interp(1.0, RG, mcore) * a["mcc"]            # density in "M/Mcrit per unit (4pi/3 D^3)" units x (4pi/3)... consistent below
    shell = np.where((RG > a["rin"]) & (RG < a["rout"]), a["Os"] * 3 / (4 * np.pi), 0.0)
    s = core + shell
    m = cumulative_trapezoid(4 * np.pi * RG**2 * s, RG, initial=0.0)
    m1 = np.interp(1.0, RG, m)                                       # = M(<D)/Mcrit total
    return s / m1, m / m1, m1

_orig_tables = P.tables
def tables(kind, a, Rb):
    if kind != "ptshell": return _orig_tables(kind, a, Rb)
    S, MT, m1 = build(a); return S, MT
P.tables = tables
_orig_run = P.run
def run(kind, a, mc, Ob, fac=1.5, **kw):
    if kind == "ptshell":
        _, _, m1 = build(a); a = dict(a); fac = max(fac, 1.02 * a["rout"] / (1 + 1 / a["DR"]))
        return _orig_run(kind, a, m1, 0.0, fac=fac, **kw)
    return _orig_run(kind, a, mc, Ob, fac, **kw)
P.run = run
P.LB.update(mcc=(1e-4, 3.0), Os=(0.1, 1.0), rin=(0.01, 0.95), rout=(0.3, 4.0), Lc=(0.005, 0.1))
P.LOGV.update(mcc=True, Os=False, rin=False, rout=True, Lc=True)

def score(p, target="mig", norm="mean", inside=True, cmax=None):
    q = dict(p, mc=0.0, Ob=0.0)
    if inside and not (p["rin"] < 1 - 1 / p["DR"] and p["rout"] > 1 + 1 / p["DR"]): return -np.inf, None, None, None
    if p["rin"] >= p["rout"]: return -np.inf, None, None, None
    return P.score("ptshell", q, target, norm, cmax)
P_score_generic = P.score
def fit(free, fixed, inside=True, target="mig", norm="mean", starts=4, nscreen=200, seeds=(), cmax=None):
    def full(u): p = dict(fixed); p.update(P.unpack(free, u)); return p
    def obj(u):
        pen = sum(10*(max(0, -x) + max(0, x-1))**2 for n, x in zip(free, u) if n != "l0")
        v = score(full(u), target, norm, inside, cmax)[0]
        return (1 - v if np.isfinite(v) else 50.0) + pen
    U = [P.rng.random(len(free)) for _ in range(nscreen)] + [P.pack(free, dict(fixed, **s)) for s in seeds]
    vals = np.array([obj(u) for u in U]); order = np.argsort(vals); best = (np.inf, None)
    from scipy.optimize import minimize
    for i in order[:starts]:
        if vals[i] >= 50: continue
        r = minimize(obj, U[i], method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-7, maxiter=800))
        if r.fun < best[0]: best = (r.fun, r.x)
    if best[1] is None: return None
    p = full(best[1]); v, a, ph, rr = score(p, target, norm, inside, cmax); p.update(amp=a, phi=ph, R2=v); return p

def describe(p, norm="mean"):
    _, _, m1 = build(p); q = dict(p, mc=m1, Ob=0.0)
    d = P.describe("ptshell", q, norm)
    rs = 1 / p["DR"]
    d.update(mc_total=m1, Mhole=p["Os"] * p["rin"]**3, Meff=p["mcc"] - p["Os"] * p["rin"]**3,
             inside=bool(p["rin"] < 1 - rs and p["rout"] > 1 + rs), Rin_Gpc=p["rin"] * d["D_Gpc"], Rout_Gpc=p["rout"] * d["D_Gpc"])
    rr = P.run("ptshell", q, m1, 0.0)
    d["Mc_kg"] = d["M_kg"] * p["mcc"] / m1; d["Mc_sun"] = d["Mc_kg"] / 1.989e30
    return {k: (float(v) if isinstance(v, (int, float, np.floating, bool, np.bool_)) else v) for k, v in d.items()}
