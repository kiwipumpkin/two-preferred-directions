"""Shared code for the 2-D H-map comparison (CoE directional Hubble rate vs Migkas+2020 clusters).
Read-only use of the repo; all outputs go to /home/user/coe_review/hmap/.
Single-threaded: set BLAS threads to 1 before numpy import."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
import healpy as hp

REPO = PKG + "/kiwipumpkin_work"
OUT = PKG + "/coe_review/hmap"
NSIDE = 16
NPIX = hp.nside2npix(NSIDE)
LOG10E = np.log10(np.e)


def lb2vec(l, b):
    l = np.radians(l); b = np.radians(b)
    return np.stack([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)], -1)


def vec2lb(v):
    v = np.asarray(v, float); v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360, np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))


def load_catalogue():
    L = [q for q in open(REPO + "/cosmology/migkas/migkas2020_tablec1_vizier.tsv") if not q.startswith("#") and q.strip()]
    h = L[0].rstrip("\n").split("\t"); R = [r.rstrip("\n").split("\t") for r in L[3:]]
    g = lambda k: np.array([float(r[h.index(k)]) for r in R])
    d = dict(z=g("z"), l=g("GLON"), b=g("GLAT"), T=g("T"), ET=g("E_T"), eT=g("e_T"), LX=g("LX"), eLX=g("e_LX"))
    E = np.sqrt(.3 * (1 + d["z"]) ** 3 + .7)
    d["y"] = np.log10(d["LX"] / 100 / E)          # log10 L/(1e44 erg/s E(z)); LX in 1e35 W = 1e42 erg/s
    d["x"] = np.log10(d["T"] / 4)                  # pivot 4 keV
    d["sy"] = LOG10E * d["eLX"] / 100              # e_LX is a percentage
    d["sx"] = LOG10E * 0.5 * (d["ET"] + d["eT"]) / d["T"]
    d["n"] = lb2vec(d["l"], d["b"])
    return d


def pix_centres():
    l, b = hp.pix2ang(NSIDE, np.arange(NPIX), lonlat=True)
    return l % 360, b, lb2vec(l, b)


def cone_angles(cat, pv):
    """angular separation (deg) between every pixel centre (rows) and cluster (cols)."""
    return np.degrees(np.arccos(np.clip(pv @ cat["n"].T, -1, 1)))


# ---------------------------------------------------------------- estimators
# Both are linear in y given x and the weights, so the forward model of a sky law is exact:
#   cone log10(A/A_all) for data  y -> y + delta  shifts by  K_cone.delta - K_all.delta.

def ols_maps(cat, Mem, W):
    """Repo-style OLS (figures/l50/jk.py): log L/E vs log T/4keV, slope free per cone, uniform weights.
    Mem: (Npix, Ncl) 0/1 membership; W: (Ncl, Nb) bootstrap integer weights.
    Returns y_map = 0.5*log10(A_cone/A_all) = log10(H/H_ref), shape (Npix, Nb)."""
    x = cat["x"][:, None]; y = cat["y"][:, None]
    cols = [W, W * x, W * y, W * x * x, W * x * y]
    S = [Mem @ c for c in cols]; Sa = [c.sum(0) for c in cols]
    def icpt(S0, Sx, Sy, Sxx, Sxy):
        B = (S0 * Sxy - Sx * Sy) / (S0 * Sxx - Sx ** 2); return (Sy - B * Sx) / S0
    return 0.5 * (icpt(*S) - icpt(*Sa)[None, :])


def ols_kernel(cat, Mem):
    """K (Npix, Ncl): intercept weights c_i = 1/n - xbar (x_i - xbar)/Sxx over cone members, minus full-sample row."""
    x = cat["x"]
    def row(m):
        n = m.sum(); xb = (m * x).sum() / n; Sxx = (m * (x - xb) ** 2).sum()
        return m * (1 / n - xb * (x - xb) / Sxx)
    Ka = row(np.ones_like(x))
    return np.array([row(m) for m in Mem]) - Ka[None, :]


def mig_weights(cat, ang, theta):
    """Migkas-like: B fixed at full-sample value, sigma_i^2 = sy^2 + B^2 sx^2 + sig_int^2 (global sig_int from chi2_red=1),
    cone weight cos^2(90deg * ang/theta) (they divide the errors by cos(...))."""
    x, y, sx, sy = cat["x"], cat["y"], cat["sx"], cat["sy"]
    B, si = 2.0, 0.2
    for _ in range(50):
        w = 1 / (sy ** 2 + B ** 2 * sx ** 2 + si ** 2)
        xb = (w * x).sum() / w.sum(); yb = (w * y).sum() / w.sum()
        B = (w * (x - xb) * (y - yb)).sum() / (w * (x - xb) ** 2).sum(); a = yb - B * xb
        r = y - a - B * x
        f = lambda s: np.sum(r ** 2 / (sy ** 2 + B ** 2 * sx ** 2 + s ** 2)) / (len(x) - 2) - 1
        from scipy.optimize import brentq
        si = brentq(f, 1e-4, 2.0)
    w = 1 / (sy ** 2 + B ** 2 * sx ** 2 + si ** 2)
    c2 = np.where(ang < theta, np.cos(np.radians(90 * ang / theta)) ** 2, 0.0)
    s2 = (sy ** 2 + B ** 2 * sx ** 2)[None, :]
    # 1/(sig_stat^2/cos^2 + sig_int^2): only the measurement errors are enlarged (Migkas Sect. 4.3);
    # sig_int = global full-sample value [approximation: Migkas re-fit sig_int per cone]
    Wc = np.where(c2 > 0, c2 / (s2 + si ** 2 * c2), 0.0)
    return dict(B=B, sig_int=si, a_all=a, w=w, Wc=Wc)


def mig_maps(cat, mw, Wb):
    """fixed-slope weighted-mean intercept; B re-fitted on each bootstrap sample (weighted OLS, full sample)."""
    x = cat["x"][:, None]; y = cat["y"][:, None]; w = mw["w"][:, None] * Wb
    xb = (w * x).sum(0) / w.sum(0); yb = (w * y).sum(0) / w.sum(0)
    B = (w * (x - xb) * (y - yb)).sum(0) / (w * (x - xb) ** 2).sum(0)
    yp = y - B[None, :] * x
    a_all = (w * yp).sum(0) / w.sum(0)
    Wc = mw["Wc"]
    a_c = (Wc @ (Wb * yp)) / (Wc @ Wb)
    return 0.5 * (a_c - a_all[None, :])


def mig_kernel(mw):
    Wc = mw["Wc"]; return Wc / Wc.sum(1, keepdims=True) - (mw["w"] / mw["w"].sum())[None, :]


# ---------------------------------------------------------------- CoE laws  h(psi) = H_obs/Hbar, psi from ANTI-CoE
def law_curve(name):
    """Returns psi grid (deg, 0..180) and h(psi). For the CatWISE-locked curves the generating code
    (catwise-density-levelset-migkas.py: sky_locked -> prof_core.run -> integ_core.geometry) puts test particles on a
    circle of radius d=R_vision=14.3 Gpc around the observer at angle theta, theta=0 pointing AWAY from the CoE;
    a_f(theta)=|v_rel|/d; curve(l) = a_f(l-l0)/<a_f>. Spherical potential => axisymmetric: h(psi)=a_f(psi)."""
    N = REPO + "/cosmology-course/notes/"
    psi = np.arange(0, 180.01, 0.5)
    if name == "AF_smooth":
        d = np.genfromtxt(REPO + "/cosmology/map_compare/amp_mean_param_noise/A_vs_angle_ideal_plus_chris_rotation.csv", delimiter=",", names=True)
        lg, a, l0 = d["ell_deg"], d["AF_smooth"], 0.0
    else:
        f = {"eta_d14.3": N + "catwise-eta-migkas-fit-d14.3.csv"}.get(name, N + "catwise-density-levelset-migkas-%s.csv" % name)
        d = np.genfromtxt(f, delimiter=",", names=True); lg, a = d["ell_deg"], d["A_over_Aall"]
        l0 = 19.9 if name == "eta_d14.3" else 15.2
    o = np.argsort(lg); lg, a = lg[o], a[o]
    lg2 = np.r_[lg - 360, lg, lg + 360]; a2 = np.tile(a, 3)
    h = 0.5 * (np.interp(l0 + psi, lg2, a2) + np.interp(l0 - psi, lg2, a2))
    return psi, h


LAWS = ["soft_fixed15.2", "eta_fixed15.2", "exp_fixed15.2", "pl_fixed15.2", "cored_fixed15.2", "gauss_fixed15.2", "eta_d14.3", "AF_smooth"]


def law_templates(cat, axes_v, psi, h, K):
    """u[a, pix] = K . log10 h(psi_a(cluster)) for each axis a (anti-CoE direction). (Naxes, Npix)"""
    cosp = np.clip(axes_v @ cat["n"].T, -1, 1); ps = np.degrees(np.arccos(cosp))
    lh = np.log10(np.interp(ps, psi, h))
    return lh @ K.T


def cluster_basis(cat):
    """real dipole (3) + quadrupole (5) basis evaluated at the clusters, (Ncl, 8)."""
    x, y, z = cat["n"].T
    return np.stack([x, y, z, x * y, x * z, y * z, x * x - y * y, 3 * z * z - 1], 1)
