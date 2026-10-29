"""Relativistic (exact LTB, GR-A, t_B = 0, H_loc = 70) refit with EVERYTHING free: D, Mc, shell edges R_in = D - kin*14.3,
R_out = D + kout*14.3 (kin, kout >= 1, independent), l0 (+ Amp, phi). Omega_s = 0.049 (ordinary matter). Model A = linear
directional expansion rate at fixed z (D_L^-1), A_all = mean over 360 directions; sampling z = 0.3 (1.19 Gpc), as the page column."""
import json, numpy as np, multiprocessing as mp
from scipy.optimize import minimize
import rm_core as C, ltb_fill as LT
OS = 0.049; RV = 14.30048287260589; Z = 0.3; C.ZL = (Z,)
def ev(Mc, D, kin, kout, full=False):
    out = dict(Mc_kg=Mc, D=D, kin=kin, kout=kout, R_in=D - kin * RV, R_out=D + kout * RV)
    if out["R_in"] <= 5 * 0.005 * D: out["fail"] = "geometry"; return out
    out["mcc"] = Mc / (4 * np.pi / 3 * 9.2042e-27 * (D * 3.0857e25)**3)
    if not 1e-5 <= out["mcc"] <= 1e3: out["fail"] = "mcc range (1e-5..1e3; tiny core = near-homogeneous, R2~0.28, ray solver very slow)"; return out
    p = C.params(Mc, D, RV, OS); p[9] = out["R_in"]; p[10] = out["R_out"]
    t0 = C.solve_t0(p, D)
    if t0 is None: out["fail"] = "no t0"; return out
    g = LT.geom(D, t0, p); Hp, Hr = g[3] / g[0], g[4] / g[1]
    out.update(t0=t0, age_Gyr=t0 * C.GYR, H_perp=Hp * C.K, H_r=Hr * C.K, twoE=2 * g[5], M_D_kg=g[7] * C.KG, rs_D=2 * g[7], along=bool(Hr > Hp), Rdot=g[3], twoM_over_R=2 * g[7] / g[0])
    out.update(C.validity(p, t0))
    if out["twoE_min"] <= -1: out["fail"] = "1+2E<=0"; return out
    try: nb_ = C.nearby_fast(p, t0, D, (Z,), 24)
    except Exception as e: out["fail"] = "rays"; return out
    DL = nb_["DL"][:, 0]
    if not np.all(np.isfinite(DL)) or (DL <= 0).any(): out["fail"] = "rays"; return out
    b, Ad = C.lfits(DL, nb_["mu"])
    out.update(R2_s=float(b["s"][0]), l0_s=float(b["s"][1]), amp=float(b["s"][2]), phi=float(b["s"][3]), R2_ns=float(b["ns"][0]), l0_ns=float(b["ns"][1]))
    if full: out["Ad"] = Ad.tolist()
    return out
def unpack(u): return 10**u[1] * 4 * np.pi / 3 * 9.2042e-27 * (10**u[0] * 3.0857e25)**3, 10**u[0], 1 + np.exp(u[2]), 1 + np.exp(u[3])
def obj(u, key):
    o = ev(*unpack(u)); return -o[key] if key in o else 5.0
def screen(u):
    import time; t = time.time(); o = ev(*unpack(u)); o["u"] = list(u); o["sec"] = time.time() - t; return o
def nm(a):
    u0, key = a
    r = minimize(obj, np.array(u0), args=(key,), method="Nelder-Mead", options=dict(xatol=0.01, fatol=3e-4, maxiter=150))
    o = ev(*unpack(r.x)); o["u"] = list(r.x); o["key"] = key; return o
if __name__ == "__main__":
    rng = np.random.default_rng(5); n = 1200
    U = np.c_[rng.uniform(np.log10(15), 3.0, n), rng.uniform(-5, 3, n), rng.uniform(-4, 3, n), rng.uniform(-4, 3, n)]
    with mp.Pool(7) as P:
        S = []
        for i, o in enumerate(P.imap_unordered(screen, U, chunksize=2)):
            S.append(o)
            if i % 100 == 0: print("screen", i, flush=True)
        starts = []
        for key in ("R2_s", "R2_ns"):
            V = sorted([o for o in S if key in o], key=lambda o: -o[key])[:6]; starts += [(o["u"], key) for o in V]
        F = P.map(nm, starts, chunksize=1)
    json.dump(dict(screen=S, refined=F), open("relfree.json", "w"), default=float)
    ok = [o for o in S if "R2_s" in o]; print("screen valid", len(ok), "of", n)
    for o in sorted(ok, key=lambda o: -o["R2_s"])[:5]: print(" scr D %.1f Mc %.2e Rin %.1f Rout %.1f R2s %.3f R2ns %.3f" % (o["D"], o["Mc_kg"], o["R_in"], o["R_out"], o["R2_s"], o["R2_ns"]))
    for o in F:
        if "R2_s" in o: print(" NM[%s] D %.2f Mc %.3e Rin %.2f Rout %.2f R2s %.4f R2ns %.4f l0 %.1f along %s 2E %.2f age %.2f Hp %.1f Hr %.1f" % (o["key"], o["D"], o["Mc_kg"], o["R_in"], o["R_out"], o["R2_s"], o["R2_ns"], o["l0_s"], o["along"], o["twoE"], o["age_Gyr"], o["H_perp"], o["H_r"]))
