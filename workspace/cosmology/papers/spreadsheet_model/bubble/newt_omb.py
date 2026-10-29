"""Newtonian CoE model (ptshell, as behind universe_thin.json) refit with Omega_s FORCED to 0.049 (H0 = 70).
Free: D, Mc (mcc = Mc/Mcrit(D)), shell edges R_in = D - kin*14.3, R_out = D + kout*14.3 Gpc (kin, kout >= 1), l0, (Amp, phi).
Model A = af = |dV|/separation, mean-normalised over the 360 directions, compared linearly with Migkas A/A_all (as prof_core.score).
Sampling sphere radius Rs: 14.3 Gpc (same as the 0.909 fit, headline); variants 0.316 Gpc (z=0.075) and 1.19 Gpc (z=0.3).
The Newtonian model is scale-free except through the 14.3 Gpc shell-thickness rule, so D enters via kin, kout and Rs/D.
Does not modify universe_thin*.py or the page."""
import json, sys, numpy as np
from scipy.optimize import minimize
import ptshell as S, prof_core as P
from bubble_core import MA, IMIG, TH1, r2, A_from, best_sine, shifted, H0, GSI, GPC
OS = 0.049; RV = 14.30048287260589; GM = GPC * 1e3
RHOC = 3 * H0**2 / (8 * np.pi * GSI)

def geo(D, mcc, kin, kout, Rs):
    return dict(DR=D / Rs, Os=OS, mcc=mcc, rin=1 - kin * RV / D, rout=1 + kout * RV / D, Lc=0.005, mc=0.0, Ob=0.0)

def af_of(a):
    if not (0.01 < a["rin"] < 1 and a["rout"] > 1): return None
    rr = P.run("ptshell", a, 0.0, 0.0)
    af = rr["af"]
    if (not np.all(np.isfinite(af))) or rr["vrmin"] <= 0: return None
    return af / af.mean(), rr

def fit_l0(af, sine):
    best = (-np.inf,)
    for l0 in np.arange(0, 360, 1.0):
        A = shifted(af, l0); x = A[IMIG]; th = (TH1 - l0)[IMIG]
        if sine:
            am, ph = best_sine(x, th, MA); v = r2(MA, A_from(x, th, am, ph))
        else: am, ph, v = 0.0, 0.0, r2(MA, x)
        if v > best[0]: best = (v, l0, am, ph)
    return best

def unpack(u):   # u = (log10 D, log10 mcc, log(kin-1), log(kout-1))
    D = 10**u[0]; mcc = 10**u[1]; kin = 1 + np.exp(u[2]); kout = 1 + np.exp(u[3]); return D, mcc, kin, kout

def objective(u, Rs, sine):
    D, mcc, kin, kout = unpack(u)
    if kin * RV >= 0.99 * D: return 5.0
    r = af_of(geo(D, mcc, kin, kout, Rs))
    return 5.0 if r is None else 1 - fit_l0(r[0], sine)[0]

def describe(D, mcc, kin, kout, Rs, sine):
    a = geo(D, mcc, kin, kout, Rs); af, rr = af_of(a); v, l0, am, ph = fit_l0(af, sine)
    A = A_from(shifted(af, l0), TH1 - l0, am, ph)
    Dm = D * GM; Mcrit = 4 * np.pi / 3 * RHOC * Dm**3; Mc = mcc * Mcrit; rhos = OS * RHOC
    Rin, Rout = D - kin * RV, D + kout * RV; rin, rout = Rin * GM, Rout * GM
    Msh = lambda r: 4 * np.pi / 3 * rhos * (np.clip(r, rin, rout)**3 - rin**3)
    def Phi(r):   # exact potential of core (point) + shell, as universe_thin.phys
        if r < rin: ps = -GSI * 2 * np.pi * rhos * (rout**2 - rin**2)
        elif r < rout: ps = -GSI * (4 * np.pi / 3 * rhos * (r**3 - rin**3) / r + 2 * np.pi * rhos * (rout**2 - r**2))
        else: ps = -GSI * Msh(rout) / r
        return -GSI * Mc / r + ps
    vE = rr["vE"] / rr["rE"] * Dm / (rr["age"] * 3.156e16) * rr["rE"] if False else None
    # Earth speed: V = H_E * D with H_E t0 = Ht, t0 = Ht/H0  ->  V_O = H0 * D * (rE/1) (units: Earth at rE ~ 1 D)
    VO = rr["vE"] / rr["Ht"] * H0 * Dm * 1.0
    v2_full = VO**2 / (2 * abs(Phi(Dm))); v2_in = VO**2 / (2 * GSI * (Mc + Msh(Dm)) / Dm)
    E = VO**2 / 2 + Phi(Dm)
    from scipy.optimize import brentq
    rt = brentq(lambda r: Phi(r) - E, Dm, 1e7 * Dm) / GM if E < 0 else np.inf
    c = 2.99792458e8
    out = dict(Rs=Rs, D=D, R_in=Rin, R_out=Rout, kin=kin, kout=kout, Mc_kg=Mc, Mc_sun=Mc / 1.989e30, mcc=mcc, Os=OS, l0=l0, amp=am, phi=ph,
               R2=v, sine=sine, shell_total_kg=Msh(rout), shell_inside_D_kg=Msh(Dm), M_inside_D_kg=Mc + Msh(Dm),
               mass_within_14p3_of_Earth_kg=4 * np.pi / 3 * rhos * (RV * GM)**3, deficit_kg=4 * np.pi / 3 * rhos * rin**3,
               VO_over_c=VO / c, vesc2_full_potential=v2_full, vesc2_enclosed_only=v2_in, bound=bool(E < 0), turnaround_Gpc=rt,
               rs_inside_D_Gpc=2 * GSI * (Mc + Msh(Dm)) / c**2 / GM, age_Gyr=rr["age"], A=A.tolist())
    return out

def fit(Rs, sine, seed=0, nscreen=400, starts=6):
    rng = np.random.default_rng(seed)
    U = np.c_[rng.uniform(1.2, 4.0, nscreen), rng.uniform(-6, 1.5, nscreen), rng.uniform(-4, 2.5, nscreen), rng.uniform(-4, 2.5, nscreen)]
    vals = np.array([objective(u, Rs, sine) for u in U]); order = np.argsort(vals)[:starts]; best = (np.inf, None)
    for i in order:
        r = minimize(objective, U[i], args=(Rs, sine), method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-5, maxiter=400))
        if r.fun < best[0]: best = (r.fun, r.x)
    return describe(*unpack(best[1]), Rs, sine), best[1]

if __name__ == "__main__":
    out = {}
    for tag, Rs in (("Rs14.3", RV), ("Rs0.316_z0.075", 0.316), ("Rs1.19_z0.3", 1.19)):
        for sine in (True, False):
            d, u = fit(Rs, sine); out["%s|%s" % (tag, "sine" if sine else "geom")] = d
            print(tag, "sine" if sine else "geom", {k: (float("%.4g" % v) if isinstance(v, float) else v) for k, v in d.items() if k != "A"}, "u", np.round(u, 3), flush=True)
    json.dump(out, open("universe_newt_omb.json", "w"), indent=1, default=float)
