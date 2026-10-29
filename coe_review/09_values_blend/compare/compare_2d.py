"""Blend model (1-D Migkas curve wrapped on sky) vs Migkas 2-D A/A_all map (theta=75 cones)."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root
import sys, json, importlib.util
from pathlib import Path
import numpy as np, healpy as hp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("blend", HERE.parent/"rot60/make_fig05_sky_orientation_values_nobands_blend.py")
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)

z = np.load(PKG + "/coe_review/hmap/maps_cov.npz")
y0, sig = z["MIG_t75__y0"], z["MIG_t75__sig"]
Amap = 10**(2*y0); Amap /= Amap.mean()           # A ∝ H^2
w = 1/(2*np.log(10)*sig)**2                       # weight from log10 H errors
NS = 16; th, ph = hp.pix2ang(NS, np.arange(hp.nside2npix(NS)))
b, l = 90-np.degrees(th), np.degrees(ph)
lb = lambda L, Bb: np.array([np.cos(np.radians(Bb))*np.cos(np.radians(L)), np.cos(np.radians(Bb))*np.sin(np.radians(L)), np.sin(np.radians(Bb))])
n = lb(l, b).T

def wcorr(a, c):
    a = a-np.average(a, weights=w); c = c-np.average(c, weights=w)
    return np.average(a*c, weights=w)/np.sqrt(np.average(a*a, weights=w)*np.average(c*c, weights=w))

mods = [B.load(s, i) for i, s in enumerate(B.base.RING_SCRIPTS)]
D_rows = [lb(0, 0), lb(0, 15), lb(0, 15), lb(0, 15)]
models = [B.A_on_dirs(m, D, n) for m, D in zip(mods, D_rows)]
res = {"rows": {}}
for i, M in enumerate(models):
    res["rows"][B.base.LAB[i]] = round(float(wcorr(M, Amap)), 3)

# roll scan for row 4 (patch module roll)
m4 = mods[3]; rolls = np.arange(0, 360, 5); rc = []
for r in rolls:
    m4.ROLL_BC_DEG = float(r); rc.append(float(wcorr(B.A_on_dirs(m4, D_rows[3], n), Amap)))
m4.ROLL_BC_DEG = 60.18
res["roll_scan"] = dict(zip([int(r) for r in rolls], [round(c, 3) for c in rc]))

# null: same model, random rigid rotations of the sky
rng = np.random.default_rng(1); M4 = models[3]; null = []
for _ in range(2000):
    R = hp.Rotator(rot=[rng.uniform(0, 360), np.degrees(np.arcsin(rng.uniform(-1, 1))), rng.uniform(0, 360)])
    tt, pp = R(th, ph); null.append(wcorr(M4[hp.ang2pix(NS, tt, pp)], Amap))
null = np.array(null); r0 = res["rows"][B.base.LAB[3]]
res["null_p_row4"] = float((null >= r0).mean()); res["null_sd"] = float(null.std())
json.dump(res, open(HERE/"compare_2d.json", "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "roll_scan"}, indent=1))
print("roll best", rolls[int(np.argmax(rc))], max(rc), "at 60:", rc[12])

# figure
vmin, vmax = 0.85, 1.15; cm = plt.get_cmap("coolwarm")
fig = plt.figure(figsize=(15, 8.6))
def moll(m, ttl, sub, mn=vmin, mx=vmax, cmap=cm):
    hp.mollview(m, fig=fig.number, sub=sub, title=ttl, min=mn, max=mx, cmap=cmap, flip="astro", cbar=True, notext=True)
    hp.graticule(dpar=30, dmer=90, alpha=.3)
    for (L, Bb, mk, c) in [(0, 15, "o", "k"), (180, -15, "*", "k"), (260, 60, "^", "g"), (80, -60, "v", "g")]:
        hp.projplot(L, Bb, mk, lonlat=True, color=c, ms=8)
moll(Amap, "Migkas 2-D map (θ=75° cones), A/A_all", 221)
moll(models[3], f"Blend model, row 4 (roll 60.18°)   r = {r0:+.2f}", 222)
moll(Amap - models[3] + 1, "Data − model (+1)", 223)
ax = fig.add_subplot(224)
ax.plot(rolls, rc, "k-"); ax.axvline(60.18, color="g", ls="--", label="AoE roll 60.18°")
ax.axvline(45, color="grey", ls=":", label="workspace 45°")
ax.axhspan(np.percentile(null, 2.5), np.percentile(null, 97.5), color="0.85", label="random-orientation 95%")
ax.set_xlabel("roll (deg)"); ax.set_ylabel("weighted correlation with 2-D map"); ax.legend(fontsize=8)
ax.set_title("Row-4 model vs data as roll varies")
fig.text(0.5, -0.03, "Markers: ● anti-CoE (0,+15)  ★ CoE (180,−15)  ▲▼ Axis of Evil (260,+60)/(80,−60).  Galactic, ℓ increases left.", ha="center", fontsize=9)
fig.savefig(HERE/"compare_2d.png", dpi=130, bbox_inches="tight"); print("wrote", HERE/"compare_2d.png")
