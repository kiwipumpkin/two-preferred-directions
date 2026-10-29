"""Residual of Migkas 2-D map after the ring-blend model (a): do the patches sit at P2? multipole content + axes."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import importlib.util, json, sys
from pathlib import Path
import numpy as np, healpy as hp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ts", PKG + "/coe_review/09_values_blend/compare/two_scale.py")
# reuse pieces without running two_scale's main body: exec only up to the "out = {}" line
src = open(PKG + "/coe_review/09_values_blend/compare/two_scale.py").read().split("out = {}")[0]
G = {"__file__": PKG + "/coe_review/09_values_blend/compare/two_scale.py", "__name__": "ts"}; exec(src, G)
Amap, w, n, blend, linfit, lb = G["Amap"], G["w"], G["n"], G["blend"], G["linfit"], G["lb"]
NS = 16; npix = hp.nside2npix(NS)

# model (a): sigma_L 50 + sigma_S 20 band-pass, weighted LS
L = blend(50, n); S = blend(20, n)-L; beta, fitA = linfit([L, S]); res = Amap-fitA
beta0, fitL = linfit([L]); resL = Amap-fitL

# geometry: CoE (180,-15), roll 60.18, theta0 from AoE
def vec(l, b): l, b = np.radians(l), np.radians(b); return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])
def gal(v): return (round(float(np.degrees(np.arctan2(v[1], v[0])) % 360), 1), round(float(np.degrees(np.arcsin(np.clip(v[2], -1, 1)))), 1))
def rot(ax, a):
    ax = ax/np.linalg.norm(ax); a = np.radians(a); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]); return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
def frame(tilt, roll): R1 = rot(np.array([0, 1., 0]), tilt); return rot(R1@vec(180, 0), roll)@R1
geo = json.load(open(PKG + "/coe_review/06_consistent/geometry_from_aoe.json"))["Land & Magueijo 2005 (260, +60)"]
th0 = geo["theta0"]; Rm = frame(-15, 60.18)
P1, P2 = Rm@vec(180-th0, 0), Rm@vec(180+th0, 0)
COE = Rm@vec(180, 0)
axes = {"P1 (AoE)": P1, "P2": P2, "CoE axis": COE}
# model-plane normal (the plane containing CoE axis, P1, P2)
N = np.cross(P1, P2); N /= np.linalg.norm(N); axes["P1-P2 plane normal"] = N
out = {"geometry": {k: dict(lb=gal(v), antipode=gal(-v)) for k, v in axes.items()}}
sep = lambda a, b: float(np.degrees(np.arccos(np.clip(abs(a@b), -1, 1))))  # headless
sepd = lambda a, b: float(np.degrees(np.arccos(np.clip(a@b, -1, 1))))

# ---- patches (from two_scale.json, sigma 50) vs P1/P2
ts = json.load(open(PKG + "/coe_review/09_values_blend/compare/two_scale.json"))["blend_plus_two_patches"]
pt = {}
for s in ["30", "40", "50", "60"]:
    for k in ["patch1", "patch2"]:
        v = vec(ts[s][k]["l"], ts[s][k]["b"])
        pt[f"{k}@sig{s}"] = {nm: dict(to_axis=round(sepd(v, a), 1), to_antipode=round(sepd(v, -a), 1)) for nm, a in axes.items() if nm != "P1-P2 plane normal"}
out["patch_separations_deg"] = pt
# residual extrema
th_, ph_ = hp.pix2ang(NS, np.arange(npix)); pv = np.array(hp.pix2vec(NS, np.arange(npix))).T
sm = res
ext = {"max": gal(pv[np.argmax(sm)]), "min": gal(pv[np.argmin(sm)])}
out["residual_extrema"] = {k: dict(lb=v, to_P2=round(sepd(vec(*v), P2), 1), to_minusP2=round(sepd(vec(*v), -P2), 1), to_P1=round(sepd(vec(*v), P1), 1), to_minusP1=round(sepd(vec(*v), -P1), 1)) for k, v in ext.items()}
# values of the residual at the axes
out["residual_at_points_pct"] = {nm: dict(axis=round(100*float(hp.get_interp_val(res, *hp.vec2ang(a))), 2), antipode=round(100*float(hp.get_interp_val(res, *hp.vec2ang(-a))), 2))
                                 for nm, a in axes.items()}
out["residual_rms_pct"] = round(100*float(res.std()), 2)

# ---- multipoles
LMAX = 6
def cl(m): return hp.anafast(m-m.mean(), lmax=LMAX)
cl_res = cl(res)
z = np.load(PKG + "/coe_review/hmap/maps_cov.npz"); V, lam = z["MIG_t75__V"], z["MIG_t75__lam"]
rng = np.random.default_rng(3); sims = []
for _ in range(1000):
    dy = V@(np.sqrt(lam)*rng.normal(size=lam.size)); dA = 2*np.log(10)*Amap*dy
    _, fitn = linfit([L, S]) if False else (None, None)
    sims.append(cl(dA))
sims = np.array(sims)
out["Cl"] = {str(l): dict(resid=float(cl_res[l]), noise_mean=float(sims[:, l].mean()), ratio=round(float(cl_res[l]/sims[:, l].mean()), 2),
                          p_noise=float((sims[:, l] >= cl_res[l]).mean())) for l in range(1, LMAX+1)}
out["Cl_note"] = "noise sims from the Migkas map covariance (V diag(lam) V^T on log10 H), linearised to A; model-fit dof not subtracted"

# ---- preferred axis per multipole (de Oliveira-Costa: max sum m^2 |a_lm|^2) and planarity
alm = hp.map2alm(res-res.mean(), lmax=LMAX)
cand = np.array(hp.pix2vec(32, np.arange(hp.nside2npix(32)))).T; cand = cand[cand[:, 2] >= 0]
def axis_stats(l):
    best = (-1, None, None)
    for v in cand:
        t, p = hp.vec2ang(v); r = hp.Rotator(rot=[np.degrees(p[0]), -np.degrees(t[0]), 0], deg=True, eulertype="ZYX", inv=False)
        a = r.rotate_alm(alm.copy(), lmax=LMAX)
        am = np.array([a[hp.Alm.getidx(LMAX, l, m)] for m in range(l+1)]); pw = np.r_[abs(am[0])**2, 2*abs(am[1:])**2]
        ms = np.arange(l+1)
        S_ = float((ms**2*pw).sum()/(l**2*pw.sum())); planar = float(pw[l]/pw.sum())
        if S_ > best[0]: best = (S_, v, planar)
    return best
# sanity: does the rotation put axis v onto z? check with a pure Y_20 about a known direction
out["axes"] = {}
for l in (2, 3):
    S_, v, pl = axis_stats(l)
    out["axes"][str(l)] = dict(axis=gal(v), m2_concentration=round(S_, 3), sectoral_fraction=round(pl, 3),
                              sep_deg={nm: round(sep(v, a), 1) for nm, a in axes.items()},
                              p_random_axis={nm: round(1-np.cos(np.radians(sep(v, a))), 3) for nm, a in axes.items()})
json.dump(out, open(HERE/"resid_p2.json", "w"), indent=1); print(json.dumps(out, indent=1))
np.savez(HERE/"resid_maps.npz", res=res, resL=resL, alm=alm)
