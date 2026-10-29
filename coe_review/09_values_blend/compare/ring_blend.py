"""Ring-to-ring blending: Gaussian-smooth A(theta) across theta (neighbouring rings), keep the
top-to-bottom psi blend. Scan smoothing width sigma x roll against the Migkas 2-D map."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root
import importlib.util, json
from pathlib import Path
import numpy as np, healpy as hp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("blend", HERE.parent/"rot60/make_fig05_sky_orientation_values_nobands_blend.py")
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
m4 = B.load(B.base.RING_SCRIPTS[3], 3)
A_raw = B.A_of

z = np.load(PKG + "/coe_review/hmap/maps_cov.npz")
y0, sig = z["MIG_t75__y0"], z["MIG_t75__sig"]
Amap = 10**(2*y0); Amap /= Amap.mean(); w = 1/(2*np.log(10)*sig)**2
NS = 16; th, ph = hp.pix2ang(NS, np.arange(hp.nside2npix(NS)))
lb = lambda L, b: np.stack([np.cos(np.radians(b))*np.cos(np.radians(L)), np.cos(np.radians(b))*np.sin(np.radians(L)), np.sin(np.radians(b))], -1)
n = lb(np.degrees(ph), 90-np.degrees(th)); D = lb(0, 15)

T = np.arange(0, 360, 0.5); A0 = A_raw(T)
def smoothed(sigma):
    if sigma == 0: return A_raw
    k = np.exp(-0.5*(((T+180) % 360 - 180)/sigma)**2); k /= k.sum()
    As = np.real(np.fft.ifft(np.fft.fft(A0)*np.fft.fft(k)))
    return lambda t: np.interp(np.mod(t, 360), T, As, period=360)

def wstats(M):
    a = M-np.average(M, weights=w); c = Amap-np.average(Amap, weights=w)
    r = np.average(a*c, weights=w)/np.sqrt(np.average(a*a, weights=w)*np.average(c*c, weights=w))
    s = np.average(a*c, weights=w)/np.average(a*a, weights=w)        # best amplitude scale
    chi2 = np.sum(w*(c-s*a)**2); chi2_iso = np.sum(w*c**2)
    return r, s, chi2, chi2_iso

def model(sigma, roll):
    B.A_of = smoothed(sigma); m4.ROLL_BC_DEG = float(roll)
    M = B.A_on_dirs(m4, D, n); B.A_of = A_raw; return M

sigmas = [0, 5, 10, 15, 20, 30, 40, 50, 60]; rolls = np.arange(0, 360, 10)
R = np.array([[wstats(model(s, r))[0] for r in rolls] for s in sigmas])
R60 = [wstats(model(s, 60.18)) for s in sigmas]
i, j = np.unravel_index(R.argmax(), R.shape)
best_s = sigmas[int(np.argmax([x[0] for x in R60]))]
out = {"sigma_deg": sigmas, "at_roll_60.18": {str(s): dict(r=round(x[0], 3), amp_scale=round(x[1], 3), dchi2_vs_iso=round(x[3]-x[2], 1)) for s, x in zip(sigmas, R60)},
       "global_best": dict(sigma=sigmas[i], roll=int(rolls[j]), r=round(float(R[i, j]), 3)),
       "best_roll_per_sigma": {str(s): int(rolls[R[k].argmax()]) for k, s in enumerate(sigmas)}}
json.dump(out, open(HERE/"ring_blend.json", "w"), indent=1); print(json.dumps(out, indent=1))

# figure: naive vs ring-blended vs Migkas, same style as side_by_side
lon = np.linspace(-np.pi, np.pi, 361); lat = np.linspace(-np.pi/2, np.pi/2, 181)
LO, LA = np.meshgrid(lon, lat); ell = np.mod(-np.rad2deg(LO), 360); bb = np.rad2deg(LA); ng = lb(ell, bb)
m4.ROLL_BC_DEG = 60.18
B.A_of = smoothed(30); naive = B.A_on_dirs(m4, D, ng); B.A_of = smoothed(60); rb = B.A_on_dirs(m4, D, ng); B.A_of = A_raw
sc60 = dict(zip(sigmas, R60))[60][1]; rb = 1 + sc60*(rb - rb.mean())
data = hp.get_interp_val(Amap, ell, bb, lonlat=True)
disp = lambda l: np.deg2rad(-((np.asarray(l)+180) % 360-180))
ra = np.linspace(0, 360, 721); th2, ph2 = hp.Rotator(coord=["C", "G"])(np.full_like(ra, np.pi/2), np.deg2rad(ra))
cl, cb_ = np.rad2deg(ph2) % 360, 90-np.rad2deg(th2)
pts = [((0, 15), "o", "#8b1a1a", "anti-CoE (0°,+15°)"), ((180, -15), "*", "#5b2a86", "CoE (180°,−15°)"),
       ((17, -9), "^", "#1f9bb0", "MAX (17°,−9°)"), ((185, 0), "^", "#d6197a", "MAX (185°,0°)"),
       ((115, 0), "s", "#6aa121", "MIN (115°,0°)"), ((274, -22), "s", "#2a62c9", "MIN (274°,−22°)"),
       ((260, 60), "P", "k", "AoE (260°,+60°)"), ((80, -60), "X", "k", "AoE (80°,−60°)")]
r0, rb_r = dict(zip(sigmas, R60))[30][0], dict(zip(sigmas, R60))[60][0]
fig = plt.figure(figsize=(16, 9.5))
panels = [(naive, f"Ours + ring-to-ring blend σ = 30° (no rescale needed)   r = {r0:.2f}"),
          (rb, f"Ours + ring-to-ring blend σ = 60° (amplitude ×{sc60:.1f})   r = {rb_r:.2f}"),
          (data, "Migkas 2-D map, θ = 75° cones")]
for k, (M, ttl) in enumerate(panels):
    ax = fig.add_subplot(2, 2, k+1, projection="mollweide")
    pc = ax.pcolormesh(LO, LA, M, cmap=B.CMAP, norm=B.NORM, shading="gouraud", rasterized=True)
    x = disp(cl); o = np.argsort(x); ax.plot(x[o], np.deg2rad(cb_[o]), "--", color="#1b7f3b", lw=1.4)
    for (l, b), mk, c, lab in pts: ax.plot(disp(l), np.deg2rad(b), mk, color=c, mec="k", ms=9, label=lab, zorder=5)
    ax.set_xticks(np.deg2rad([-150, -90, 0, 90, 150])); ax.set_xticklabels(["150°", "90°", "0°", "270°", "210°"])
    ax.grid(alpha=.3); ax.set_title(ttl, fontsize=11, pad=10)
ax = fig.add_subplot(2, 2, 4)
ax.plot(T, A0, color="0.6", lw=1, label="Migkas A(θ), as digitized")
ax.plot(T, smoothed(30)(T), "k", lw=2, label="ring-to-ring blended, σ = 30°"); ax.plot(T, smoothed(60)(T), "k--", lw=1.5, label="σ = 60°")
ax.set_xlabel("θ from anti-CoE (deg; 0–180 top half, 180–360 bottom half)"); ax.set_ylabel("A/A_all"); ax.set_xlim(0, 360)
ax2 = ax.inset_axes([0.62, 0.62, 0.36, 0.34])
ax2.plot(sigmas, [x[0] for x in R60], "o-k"); ax2.set_xlabel("σ (deg)", fontsize=7); ax2.set_ylabel("r @ roll 60°", fontsize=7); ax2.tick_params(labelsize=7)
ax.legend(fontsize=8, loc="lower left"); ax.set_title("Curve before/after blending across rings")
h, l = fig.axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=8, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.035))
cax = fig.add_axes([0.3, 0.035, 0.4, 0.015]); c = fig.colorbar(pc, cax=cax, orientation="horizontal", extend="both"); c.set_label("A/A_all (same scale all maps)")
fig.savefig(HERE/"ring_blend.png", dpi=140, bbox_inches="tight"); print("best sigma at 60:", best_s)
