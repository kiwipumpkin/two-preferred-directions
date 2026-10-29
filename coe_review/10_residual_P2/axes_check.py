import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import json, numpy as np, healpy as hp
LMAX = 6; NS = 16
cand = np.array(hp.pix2vec(32, np.arange(hp.nside2npix(32)))).T; cand = cand[cand[:, 2] >= 0]
def best_axis(alm, l):
    best = (-1, None, None)
    for v in cand:
        t, p = hp.vec2ang(v); r = hp.Rotator(rot=[np.degrees(p[0]), -np.degrees(t[0]), 0], deg=True, eulertype="ZYX")
        a = r.rotate_alm(alm.copy(), lmax=LMAX)
        am = np.array([a[hp.Alm.getidx(LMAX, l, m)] for m in range(l+1)]); pw = np.r_[abs(am[0])**2, 2*abs(am[1:])**2]
        S_ = float((np.arange(l+1)**2*pw).sum()/(l**2*pw.sum()))
        if S_ > best[0]: best = (S_, v, float(pw[l]/pw.sum()))
    return best
def gal(v): return (round(float(np.degrees(np.arctan2(v[1], v[0])) % 360), 1), round(float(np.degrees(np.arcsin(np.clip(v[2], -1, 1)))), 1))
# sanity: Y_ll sectoral about axis (120, 40) for l=2,3,4 -> recovered axis should be (120,40)
ax0 = np.array(hp.ang2vec(120, 40, lonlat=True))
pv = np.array(hp.pix2vec(NS, np.arange(hp.nside2npix(NS)))).T
e1 = np.cross(ax0, [0, 0, 1.]); e1 /= np.linalg.norm(e1); e2 = np.cross(ax0, e1)
phi = np.arctan2(pv@e2, pv@e1); st = np.sqrt(1-(pv@ax0)**2)
for l in (2, 3, 4):
    m = st**l*np.cos(l*phi); S_, v, pl = best_axis(hp.map2alm(m, lmax=LMAX), l)
    print("sanity l=%d recovered" % l, gal(v), "expected (120,40) or antipode; m2conc %.3f" % S_)
d = np.load("resid_maps.npz"); alm = d["alm"]
S4, v4, p4 = best_axis(alm, 4); print("l=4 axis", gal(v4), "m2conc %.3f sectoral %.3f" % (S4, p4))
# null for l=2 sectoral fraction & l=4 concentration from noise sims
z = np.load(PKG + "/coe_review/hmap/maps_cov.npz"); V, lam = z["MIG_t75__V"], z["MIG_t75__lam"]
y0 = z["MIG_t75__y0"]; A = 10**(2*y0); A /= A.mean()
rng = np.random.default_rng(5); s2, s4 = [], []
for _ in range(150):
    dA = 2*np.log(10)*A*(V@(np.sqrt(lam)*rng.normal(size=lam.size))); a = hp.map2alm(dA-dA.mean(), lmax=LMAX)
    s2.append(best_axis(a, 2)[2]); s4.append(best_axis(a, 4)[0])
print("null l=2 sectoral fraction: median %.3f, P(>=0.989)=%.3f" % (np.median(s2), np.mean(np.array(s2) >= 0.989)))
print("null l=4 m2conc: median %.3f, P(>=obs)=%.3f" % (np.median(s4), np.mean(np.array(s4) >= S4)))
json.dump(dict(l4_axis=gal(v4), l4_m2conc=S4, l4_sectoral=p4, null_l2_sectoral_median=float(np.median(s2)), p_l2_sectoral=float(np.mean(np.array(s2) >= 0.989)),
               null_l4_m2conc_median=float(np.median(s4)), p_l4_m2conc=float(np.mean(np.array(s4) >= S4))), open("axes_check.json", "w"), indent=1)
