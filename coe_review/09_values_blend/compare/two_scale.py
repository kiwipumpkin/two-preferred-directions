"""(1) sigma = 30/40/50/60 ring-to-ring blends next to Migkas.
(2) Second, smaller-scale component:
   (a) ring band-pass: same rings, finer scale  S = blend(sig_s) - blend(sig_L); fit M = c + a*L + b*S
   (b) two free Gaussian patches on top of the large-scale blend (centres, widths, amplitudes fitted)."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root
import importlib.util, json
from pathlib import Path
import numpy as np, healpy as hp, matplotlib
from scipy.optimize import least_squares
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("blend", HERE.parent/"rot60/make_fig05_sky_orientation_values_nobands_blend.py")
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
m4 = B.load(B.base.RING_SCRIPTS[3], 3); m4.ROLL_BC_DEG = 60.18
A_raw = B.A_of

z = np.load(PKG + "/coe_review/hmap/maps_cov.npz")
y0, sig = z["MIG_t75__y0"], z["MIG_t75__sig"]
Amap = 10**(2*y0); Amap /= Amap.mean(); w = 1/(2*np.log(10)*sig)**2; sw = np.sqrt(w)
NS = 16; th, ph = hp.pix2ang(NS, np.arange(hp.nside2npix(NS)))
lb = lambda L, b: np.stack([np.cos(np.radians(b))*np.cos(np.radians(L)), np.cos(np.radians(b))*np.sin(np.radians(L)), np.sin(np.radians(b))], -1)
n = lb(np.degrees(ph), 90-np.degrees(th)); D = lb(0, 15)
lon = np.linspace(-np.pi, np.pi, 361); lat = np.linspace(-np.pi/2, np.pi/2, 181)
LO, LA = np.meshgrid(lon, lat); ell = np.mod(-np.rad2deg(LO), 360); bb = np.rad2deg(LA); ng = lb(ell, bb)
data_g = hp.get_interp_val(Amap, ell, bb, lonlat=True)

T = np.arange(0, 360, 0.5); A0 = A_raw(T)
def smoothed(s):
    if s == 0: return A_raw
    k = np.exp(-0.5*(((T+180) % 360-180)/s)**2); k /= k.sum()
    As = np.real(np.fft.ifft(np.fft.fft(A0)*np.fft.fft(k))); return lambda t: np.interp(np.mod(t, 360), T, As, period=360)
def blend(s, dirs):
    B.A_of = smoothed(s); M = B.A_on_dirs(m4, D, dirs); B.A_of = A_raw; return M
def wr(M):
    a = M-np.average(M, weights=w); c = Amap-np.average(Amap, weights=w)
    return float(np.average(a*c, weights=w)/np.sqrt(np.average(a*a, weights=w)*np.average(c*c, weights=w)))
def linfit(cols):  # weighted LS of Amap on [1, cols...]
    X = np.column_stack([np.ones_like(Amap)]+cols); beta = np.linalg.lstsq(X*sw[:, None], Amap*sw, rcond=None)[0]
    return beta, X@beta

out = {}
# ---- (1) single-scale maps, each with its own fitted amplitude
single = {}
for s in [30, 40, 50, 60]:
    beta, fit = linfit([blend(s, n)]); single[s] = dict(r=round(wr(fit), 3), amp=round(float(beta[1]), 2))
out["single_scale"] = single

# ---- (2a) two-scale ring band-pass
best = None; grid = []
for sL in [30, 40, 50, 60]:
    L = blend(sL, n)
    for ss in [0, 5, 10, 15, 20]:
        S = blend(ss, n)-L; beta, fit = linfit([L, S]); r = wr(fit)
        grid.append(dict(sigL=sL, sigS=ss, r=round(r, 3), aL=round(float(beta[1]), 2), aS=round(float(beta[2]), 2)))
        if best is None or r > best["r"]: best = grid[-1]
out["two_scale_ring_grid"] = grid; out["two_scale_ring_best"] = best

# ---- (2b) large-scale blend + two free Gaussian patches
def patch(l0, b0, wd, dirs):
    c = dirs@lb(l0, b0); return np.exp((c-1)/np.radians(wd)**2)
Lb = {s: blend(s, n) for s in [30, 40, 50, 60]}
def resid(p, s):
    a0, aL, A1, l1, b1, w1, A2, l2, b2, w2 = p
    M = a0+aL*(Lb[s]-1)+A1*patch(l1, b1, w1, n)+A2*patch(l2, b2, w2, n); return (M-Amap)*sw
blob = {}
for s in [30, 40, 50, 60]:
    bestp = None
    for st in [(180, 15, 20, 15, -10, 15), (170, 20, 25, 20, -5, 20), (200, 10, 15, 10, -15, 25)]:
        p0 = [1, 1, .08, st[0], st[1], st[2], .08, st[3], st[4], st[5]]
        f = least_squares(resid, p0, args=(s,), bounds=([0.5, 0, -0.3, -360, -90, 5, -0.3, -360, -90, 5], [1.5, 5, 0.3, 720, 90, 60, 0.3, 720, 90, 60]))
        if bestp is None or f.cost < bestp.cost: bestp = f
    p = bestp.x; M = (resid(p, s)/sw)+Amap
    blob[s] = dict(r=round(wr(M), 3), aL=round(p[1], 2),
                   patch1=dict(l=round(p[3] % 360, 1), b=round(p[4], 1), width=round(p[5], 1), amp=round(p[2], 3)),
                   patch2=dict(l=round(p[7] % 360, 1), b=round(p[8], 1), width=round(p[9], 1), amp=round(p[6], 3)), params=list(map(float, p)))
out["blend_plus_two_patches"] = {k: {kk: vv for kk, vv in v.items() if kk != "params"} for k, v in blob.items()}
# where do the patches sit relative to the geometry?
ref = {"anti-CoE": (0, 15), "CoE": (180, -15), "AoE+": (260, 60), "AoE-": (80, -60), "MAX1": (17, -9), "MAX2": (185, 0)}
ang = lambda a, b: round(float(np.degrees(np.arccos(np.clip(lb(*a)@lb(*b), -1, 1)))), 1)
for s in blob:
    for k in ["patch1", "patch2"]:
        P = (blob[s][k]["l"], blob[s][k]["b"]); out["blend_plus_two_patches"][s][k]["sep_deg"] = {nm: ang(P, v) for nm, v in ref.items()}
json.dump(out, open(HERE/"two_scale.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "two_scale_ring_grid"}, indent=1))

# ---------- figures
disp = lambda l: np.deg2rad(-((np.asarray(l)+180) % 360-180))
ra = np.linspace(0, 360, 721); th2, ph2 = hp.Rotator(coord=["C", "G"])(np.full_like(ra, np.pi/2), np.deg2rad(ra))
cl, cb_ = np.rad2deg(ph2) % 360, 90-np.rad2deg(th2)
pts = [((0, 15), "o", "#8b1a1a", "anti-CoE (0°,+15°)"), ((180, -15), "*", "#5b2a86", "CoE (180°,−15°)"),
       ((17, -9), "^", "#1f9bb0", "MAX (17°,−9°)"), ((185, 0), "^", "#d6197a", "MAX (185°,0°)"),
       ((115, 0), "s", "#6aa121", "MIN (115°,0°)"), ((274, -22), "s", "#2a62c9", "MIN (274°,−22°)"),
       ((260, 60), "P", "k", "AoE (260°,+60°)"), ((80, -60), "X", "k", "AoE (80°,−60°)")]
def mpanel(fig, pos, M, ttl, cmap=B.CMAP, norm=B.NORM, extra=()):
    ax = fig.add_subplot(*pos, projection="mollweide")
    pc = ax.pcolormesh(LO, LA, M, cmap=cmap, norm=norm, shading="gouraud", rasterized=True)
    x = disp(cl); o = np.argsort(x); ax.plot(x[o], np.deg2rad(cb_[o]), "--", color="#1b7f3b", lw=1.3)
    for (l, b), mk, c, lab in pts: ax.plot(disp(l), np.deg2rad(b), mk, color=c, mec="k", ms=8, label=lab, zorder=5)
    for (l, b) in extra: ax.plot(disp(l), np.deg2rad(b), "o", mfc="none", mec="k", mew=2, ms=16, zorder=6)
    ax.set_xticks(np.deg2rad([-150, -90, 0, 90, 150])); ax.set_xticklabels(["150°", "90°", "0°", "270°", "210°"], fontsize=8)
    ax.tick_params(axis="y", labelsize=7); ax.grid(alpha=.3); ax.set_title(ttl, fontsize=10, pad=8); return ax, pc
def finish(fig, ax0, pc, fn, lab="A/A_all (same scale all maps)"):
    h, l = ax0.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    cax = fig.add_axes([0.3, 0.03, 0.4, 0.013]); c = fig.colorbar(pc, cax=cax, orientation="horizontal", extend="both"); c.set_label(lab)
    fig.savefig(HERE/fn, dpi=130, bbox_inches="tight"); plt.close(fig)

def scaled(s, a): M = blend(s, ng); return 1+a*(M-M.mean())
# Fig 1: sigma 30/40/50/60 + Migkas
fig = plt.figure(figsize=(16, 13))
for k, s in enumerate([30, 40, 50, 60]):
    ax, pc = mpanel(fig, (3, 2, k+1), scaled(s, single[s]["amp"]), f"Ring-to-ring blend σ = {s}°  (amplitude ×{single[s]['amp']})   r = {single[s]['r']:.2f}")
    if k == 0: ax0 = ax
mpanel(fig, (3, 2, 5), data_g, "Migkas 2-D map, θ = 75° cones")
ax = fig.add_subplot(3, 2, 6)
ax.plot(T, A0, color="0.6", lw=1, label="Migkas A(θ), as digitized")
for s, ls in zip([30, 40, 50, 60], ["-", "--", "-.", ":"]): ax.plot(T, smoothed(s)(T), "k", ls=ls, lw=1.6, label=f"σ = {s}°")
ax.set_xlim(0, 360); ax.set_xlabel("θ from anti-CoE (0–180 top half, 180–360 bottom half)"); ax.set_ylabel("A/A_all"); ax.legend(fontsize=8)
ax.set_title("Curve after blending across rings (before amplitude fit)", fontsize=10)
finish(fig, ax0, pc, "ring_blend_30_40_50_60.png")

# Fig 2: two components
sb = best["sigL"]; bp = blob[sb]["params"]
Lg = blend(sb, ng); Sg = blend(best["sigS"], ng)-Lg
X = linfit([blend(sb, n), blend(best["sigS"], n)-blend(sb, n)])[0]
ringfit = X[0]+X[1]*Lg+X[2]*Sg
blobfit = bp[0]+bp[1]*(Lg-1)+bp[2]*patch(bp[3], bp[4], bp[5], ng)+bp[6]*patch(bp[7], bp[8], bp[9], ng)
largefit = bp[0]+bp[1]*(Lg-1)
P = [(bp[3] % 360, bp[4]), (bp[7] % 360, bp[8])]
fig = plt.figure(figsize=(16, 13))
ax0, pc = mpanel(fig, (3, 2, 1), largefit, f"Large scale only: blend σ = {sb}°  r = {single[sb]['r']:.2f}")
mpanel(fig, (3, 2, 2), data_g, "Migkas 2-D map, θ = 75° cones", extra=P)
mpanel(fig, (3, 2, 3), ringfit, f"(a) + finer rings: σ_L = {best['sigL']}°, σ_S = {best['sigS']}°   r = {best['r']:.2f}")
mpanel(fig, (3, 2, 4), blobfit, f"(b) + two patches (circled)   r = {blob[sb]['r']:.2f}", extra=P)
mpanel(fig, (3, 2, 5), data_g-ringfit+1, "Data − (a)  (+1)")
mpanel(fig, (3, 2, 6), data_g-blobfit+1, "Data − (b)  (+1)", extra=P)
finish(fig, ax0, pc, "two_scale.png")
print("done")
