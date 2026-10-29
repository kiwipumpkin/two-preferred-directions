"""Close-up of the observer's sphere in the CoE model frame (observer at origin, CoE on -x, plane of evil = x-y).
Adds: CMB low-multipole plane (perpendicular to the Axis of Evil), the l = 4 residual of the Migkas map painted on the
sphere with its belt lobes marked, and the anomaly / dipole features of the joint catalog."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import json, numpy as np, healpy as hp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
g = json.load(open(PKG + "/coe_review/06_consistent/geometry_from_aoe.json"))["Land & Magueijo 2005 (260, +60)"]
DR, th0 = g["DR"], g["theta0"]; t0 = np.radians(th0); ROLL = 60.18

def vec(l, b): l, b = np.radians(l), np.radians(b); return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])
def rot(ax, a):
    ax = ax/np.linalg.norm(ax); a = np.radians(a); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]); return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
def frame(tilt, roll): R1 = rot(np.array([0, 1., 0]), tilt); return rot(R1@vec(180, 0), roll)@R1
Rm = frame(-15, ROLL)                       # model -> Galactic
to_model = lambda l, b: Rm.T@vec(l, b)
P1 = np.array([-np.cos(t0), np.sin(t0), 0]); P2 = np.array([-np.cos(t0), -np.sin(t0), 0])
assert np.allclose(Rm@P1, vec(80, -60), atol=2e-3), Rm@P1

# l = 4 part of the Migkas residual, evaluated on the sphere mesh
d = np.load(PKG + "/coe_review/10_residual_P2/resid_maps.npz"); alm = d["alm"]; LMAX = 6
a4 = np.zeros_like(alm)
for m in range(5): a4[hp.Alm.getidx(LMAX, 4, m)] = alm[hp.Alm.getidx(LMAX, 4, m)]
M4 = 100*hp.alm2map(a4, 64, lmax=LMAX)
A, B = np.meshgrid(np.linspace(0, 2*np.pi, 181), np.linspace(0, np.pi, 91))
X, Y, Z = np.cos(A)*np.sin(B), np.sin(A)*np.sin(B), np.cos(B)
G = np.einsum("ij,jkl->ikl", Rm, np.array([X, Y, Z]))
th, ph = np.arccos(np.clip(G[2], -1, 1)), np.arctan2(G[1], G[0]) % (2*np.pi)
V4 = hp.get_interp_val(M4, th, ph)
norm = matplotlib.colors.Normalize(-4, 4); cm = plt.get_cmap("coolwarm")

ELEV, AZIM = 28, -58
vv = np.array([np.cos(np.radians(ELEV))*np.cos(np.radians(AZIM)), np.cos(np.radians(ELEV))*np.sin(np.radians(AZIM)), np.sin(np.radians(ELEV))])
front = lambda p: p@vv > -0.05

fig = plt.figure(figsize=(15, 15)); ax = fig.add_subplot(projection="3d")
fc = cm(norm(V4)); fc[..., 3] = 0.38
ax.plot_surface(X, Y, Z, facecolors=fc, linewidth=0, shade=False, rstride=1, cstride=1)
# plane of evil (x-y) and CMB multipole plane (normal = AoE axis P1)
S = 1.55
ax.add_collection3d(Poly3DCollection([[[-4.0, -4.0, 0], [4.0, -4.0, 0], [4.0, 4.0, 0], [-4.0, 4.0, 0]]], color="#f2b233", alpha=.15))
n = (P1-P2)/np.linalg.norm(P1-P2); e1 = np.array([0, 0, 1.]); e2 = np.cross(n, e1)   # multipole plane normal = P1-P2 axis (contains the CoE)
nA = P1; eA2 = np.cross(nA, e1)   # plane normal to the published headless axis (reference)
sq = [z*e1+x*e2 for z, x in ((-S, -DR-0.45), (S, -DR-0.45), (S, S), (-S, S))]   # extended to reach the CoE, which lies in this plane
ax.add_collection3d(Poly3DCollection([sq], color="#2e9e8f", alpha=.15))
u = np.linspace(0, 2*np.pi, 400)
gc = lambda a, b: np.outer(a, np.cos(u))+np.outer(b, np.sin(u))
for circ, col, lab in ((gc([1, 0, 0], [0, 1, 0]), "#c98a0b", "plane of evil (great circle)"), (gc(e1, e2), "#1d7f73", "CMB multipole plane (⊥ P1–P2 axis; contains the CoE)")):
    f = front(circ.T); c2 = circ.copy(); c2[:, ~f] = np.nan; c3 = circ.copy(); c3[:, f] = np.nan
    ax.plot(*c2, color=col, lw=2.6, label=lab); ax.plot(*c3, color=col, lw=1, ls=":")
circA = gc(e1, eA2); fA = front(circA.T); cA = circA.copy(); cA[:, ~fA] = np.nan
ax.plot(*cA, color="0.45", lw=1.1, ls=(0, (4, 3)), label="plane ⊥ published headless axis (8.1° from the CoE)")
# celestial equator (Dec = 0) mapped Galactic -> model frame
_ra = np.linspace(0, 2*np.pi, 721); _th, _ph = hp.Rotator(coord=["C", "G"])(np.full_like(_ra, np.pi/2), _ra)
ceq = np.array([to_model(np.degrees(p), 90-np.degrees(t)) for t, p in zip(_th, _ph)]).T
_f = front(ceq.T); _c2 = ceq.copy(); _c2[:, ~_f] = np.nan; _c3 = ceq.copy(); _c3[:, _f] = np.nan
ax.plot(*_c2, color="#7a1fa2", lw=2.2, ls=(0, (6, 3)), label="celestial equator (Dec = 0)"); ax.plot(*_c3, color="#7a1fa2", lw=.9, ls=":")
# equal-distance circle and the arc of the CoE circle
ring = np.array([-np.cos(t0)+0*u, np.sin(t0)*np.cos(u), np.sin(t0)*np.sin(u)])
ax.plot(*ring, "--", color="#c0392b", lw=1.2, alpha=.8, label=f"equal-distance circle (θ0 = {th0:.2f}° from CoE)")
C = np.array([-DR, 0, 0]); w = np.linspace(-0.9, 0.9, 200)
w = np.linspace(-np.pi, np.pi, 800); arc = np.array([C[0]+DR*np.cos(w), DR*np.sin(w), 0*w]); arc[:, (np.abs(arc[0]-(0.0)) > 4.0) | (np.abs(arc[1]) > 4.0)] = np.nan
ax.plot(*arc, color="#c0392b", lw=3, label=f"circle about the CoE (radius D = {DR:.3f} R)")
# axes through the observer
L = 1.5
ax.plot([0, -DR], [0, 0], [0, 0], color="#5b2a86", lw=2.5, label="CoE axis"); ax.plot([0, L], [0, 0], [0, 0], color="#5b2a86", lw=1.2, ls=":")
ax.plot(*np.c_[-L*P1, L*P1], color="k", lw=1.4, ls="--", label="published Axis of Evil (P1 ↔ antipode)")
ax.plot(*np.c_[-L*n, L*n], color="#1d7f73", lw=2.6, label="P1–P2 axis (90° from the CoE)")
ax.plot(*np.c_[-L*P2, L*P2], color="#1b7f3b", lw=2.2, label="P2 axis")
ax.scatter(0, 0, 0, s=60, color="k", depthshade=False)
ax.scatter(*C, s=420, marker="*", color="#5b2a86", edgecolor="w", depthshade=False, zorder=20)
ax.text(C[0], C[1], C[2]+.32, "Center of expansion", fontsize=12, color="#5b2a86", weight="bold", ha="center")
for P in (P1, P2): ax.plot(*np.c_[C, P], color="#c0392b", lw=1, ls=":")
ax.text(-DR/2, 0, .35, f"D = {DR:.2f} R", fontsize=11, color="#5b2a86", ha="center")
for P, nm, col in ((P1, "P1", "k"), (P2, "P2", "#1b7f3b"), (-P1, "−P1", "k"), (-P2, "−P2", "#1b7f3b")):
    ax.scatter(*P, s=110, color=col, edgecolor="w", depthshade=False, alpha=1 if front(P) else .35)
    ax.text(*(P*1.08+[0, 0, .05]), nm, fontsize=13, color=col, weight="bold", alpha=1 if front(P) else .45)

# l = 4 belt lobes (from the residual-map analysis)
lobes = [((294, 47), +4.0), ((114, -47), +4.0), ((183, 15), +3.3), ((3, -15), +3.3),
         ((4, 36), -4.6), ((184, -36), -4.6), ((195, 60), -2.6), ((15, -60), -2.6)]
for (l, b), a in lobes:
    p = to_model(l, b)*1.01; f = front(p)
    ax.scatter(*p, s=240*abs(a)/4, facecolor="none", edgecolor="#b2182b" if a > 0 else "#2166ac", linewidth=2.2, depthshade=False, alpha=1 if f else .3)

# feature catalog (joint fit) + Migkas extrema + CMB dipole
feat = [("CMB dipole", 264, 48, "hot"), ("CMB dipole (cold)", 84, -48, "cold"), ("CMB Cold Spot", 209, -57, "cold"),
        ("Bulk flow toward", 287, 8, "hot"), ("Bulk flow away", 107, -8, "cold"),
        ("NVSS hot", 196, 46, "hot"), ("NVSS cold", 16, -46, "cold"), ("CatWISE hot", 238, 31, "hot"), ("CatWISE cold", 58, -31, "cold"),
        ("Hemisph. asym. hot", 230, -20, "hot"), ("Hemisph. asym. cold", 50, 20, "cold"), ("Loop I / NPS", 329, 17.5, "hot"),
        ("Migkas MAX", 17, -9, "hot"), ("Migkas MAX", 185, 0, "hot"), ("Migkas MIN", 115, 0, "cold"), ("Migkas MIN", 274, -22, "cold")]
mk = {"hot": ("^", "#d7301f"), "cold": ("v", "#2b6cb0")}
ax.set_xlim(-4.0, 4.0); ax.set_ylim(-4.0, 4.0); ax.set_zlim(-2.4, 2.4)
ax.set_box_aspect((1, 1, 0.6), zoom=1.45); ax.view_init(elev=ELEV, azim=AZIM); fig.canvas.draw()
from mpl_toolkits.mplot3d import proj3d
P2D = lambda q: np.array(proj3d.proj_transform(*q, ax.get_proj())[:2])
anch, lab = [], []
for nm, l, b, k in feat:
    p = to_model(l, b); f = front(p); m_, c = mk[k]
    ax.scatter(*(p*1.02), s=95, marker=m_, color=c, edgecolor="k", depthshade=False, alpha=1 if f else .3, zorder=10)
    a2 = P2D(p*1.02); r = a2/np.linalg.norm(a2) if np.linalg.norm(a2) > 1e-6 else np.array([1, 0])
    anch.append(a2); lab.append(dict(t=nm, c=c, f=f, pos=a2+r*0.06))
# labels on a ring outside the sphere, ordered by angle, with a minimum angular gap
A2 = np.array(anch); c0 = P2D(np.zeros(3)); Rs = np.linalg.norm(P2D(np.array([0, 0, 1.0]))-c0)
ang = np.arctan2(A2[:, 1]-c0[1], A2[:, 0]-c0[0]); o = np.argsort(ang); a_ = ang[o].copy(); gap = 2*np.pi/len(a_)*0.95
for _ in range(500):
    for k in range(len(a_)):
        nxt = (k+1) % len(a_); d = (a_[nxt]-a_[k]) % (2*np.pi)
        if d < gap: a_[k] -= (gap-d)/2; a_[nxt] += (gap-d)/2
ang2 = np.empty_like(ang); ang2[o] = a_
pos = np.c_[c0[0]+1.75*Rs*np.cos(ang2), c0[1]+1.6*Rs*np.sin(ang2)]
for d, a2, q in zip(lab, A2, pos):
    ax.annotate(d["t"], xy=a2, xytext=q, xycoords="data", textcoords="data", fontsize=7.5, color=d["c"], alpha=1 if d["f"] else .5,
                ha="center", va="center", arrowprops=dict(arrowstyle="-", color="0.4", lw=.6, alpha=1 if d["f"] else .35))

ax.set_xlim(-4.0, 4.0); ax.set_ylim(-4.0, 4.0); ax.set_zlim(-2.4, 2.4)
ax.set_axis_off()
from matplotlib.lines import Line2D
h, l = ax.get_legend_handles_labels()
h += [Line2D([], [], marker="^", ls="", color="#d7301f", mec="k", ms=9), Line2D([], [], marker="v", ls="", color="#2b6cb0", mec="k", ms=9),
      Line2D([], [], marker="o", ls="", mfc="none", mec="#b2182b", mew=2, ms=12), Line2D([], [], marker="o", ls="", mfc="none", mec="#2166ac", mew=2, ms=12)]
l += ["hot / high feature", "cold / low feature", "ℓ = 4 hot lobe (Migkas residual)", "ℓ = 4 cold lobe"]
fig.legend(h, l, loc="lower center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.06))
cax = fig.add_axes([0.35, 0.05, 0.3, 0.008]); cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm, cm), cax=cax, orientation="horizontal")
cb.set_label("sphere color: ℓ = 4 part of the Migkas residual (%)", fontsize=10)
fig.suptitle("Center of expansion and the observer's sphere (true scale): plane of evil, CMB multipole plane (⊥ P1–P2, contains the CoE), ℓ = 4 belt and features\n"
             f"P1 = Axis of Evil (80°,−60°)↔(260°,+60°) · P2 (234°,+52.5°)↔(54°,−52.5°) · CoE (180°,−15°), roll {ROLL}° · faded = far side",
             fontsize=13, y=0.97)
fig.savefig("plane_of_evil_coe_obs_ceq_p1p2.png", dpi=140, bbox_inches="tight"); print("ok")
