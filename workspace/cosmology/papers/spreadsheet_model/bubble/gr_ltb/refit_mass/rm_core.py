"""Core-mass refit (headline): Chris's layout unchanged (CoE core Lc = 0.005 D, empty interior, uniform shell Omega_s = 1.012
from D-R to D+R, vacuum outside), only Mc raised; D, R, l0 refit (Amp, phi for the sine column). Exact LTB, GR-A construction
(today's areal mass distribution as specified, single bang t_B = 0, E(r) from R(r,t0) = r), t0 from local mean H = 70.
Nearby A(l) as in ../run_gr.py: analyze.nearby (D_L ray bundles, N = 24), A propto D_L^-2 at fixed z, b = 0, CoE at l0 + 180."""
import os, sys, json, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../.."))                      # bubble_core
sys.path.insert(0, os.path.join(HERE, "../../../integrated"))        # integ_core
import ltb_fill as LT
sys.path.insert(0, os.path.join(HERE, "../../../../cmb_redshift"))
import rays, analyze
rays.ltb = LT; analyze.ltb = LT
from bubble_core import MA, IMIG, TH1, r2, A_from, best_sine
from numpy.polynomial.legendre import legfit, legval
from scipy.optimize import brentq
H0 = LT.H0_70; GYR = LT.GYR; K = 70 / H0
KG = 3.0857e25 * 2.99792458e8**2 / 6.674e-11          # kg per geometric Gpc
RHOC = 3 * H0**2 / (8 * np.pi); OS = 1.012438625747782
ZL = (0.05, 0.1, 0.2, 0.3); IZ = 1                   # headline z = 0.1
POW = -1.0   # Chris: A is linear in the expansion rate (af = abs(dV)/separation), NOT H^2; H(psi) at fixed z propto 1/D_L

def params(Mc_kg, D, R, Os=OS, Oin=0.0):
    p = np.zeros(13); p[1] = Mc_kg / KG; p[2] = 0.005 * D; p[3] = 1
    p[8] = Os * RHOC; p[9] = D - R; p[10] = D + R; p[11] = min(0.5, 0.05 * R); p[12] = Oin * RHOC
    return p

def Hloc(p, r, t):
    Hp, Hr = LT.local_rates(r, t, p); return (Hr + 2 * Hp) / 3

def solve_t0(p, D):
    def f(lt):
        p[5] = np.exp(lt)
        try: v = Hloc(p, D, p[5]) / H0 - 1
        except Exception: return np.nan
        return v if np.isfinite(v) else np.nan
    grid = np.linspace(np.log(0.02), np.log(60.0), 60); fv = np.array([f(x) for x in grid])
    for i in range(len(grid) - 1):
        if np.isfinite(fv[i]) and np.isfinite(fv[i + 1]) and fv[i] > 0 >= fv[i + 1]:
            lt = brentq(f, grid[i], grid[i + 1], xtol=1e-12); p[5] = np.exp(lt); return p[5]
    return None

def validity(p, t0):
    a, b, w = p[9], p[10], p[11]
    rr = np.linspace(a - w, b + w, 81); e2 = []; rd = []
    for r in rr:
        g = LT.geom(r, t0, p); e2.append(2 * g[5]); rd.append(g[3])
    e2 = np.array(e2); rd = np.array(rd)
    return dict(twoE_min=float(e2.min()), n_1p2E_le0=int((1 + e2 <= 0).sum()), n_collapsing=int((rd <= 0).sum()),
                ok=bool((1 + e2 > 0).all() and (rd > 0).all()))

DGRID = np.arange(0, 360, 0.5)
def lfits(DLmu, mu):
    """Best l0 (0.5 deg grid) for no-sine and sine; A = D_L^-2 normalised to the mean over the 1-deg sky grid."""
    cf = legfit(mu, DLmu, len(mu) - 1)
    Ad = legval(np.cos(np.radians(DGRID)), cf) ** POW                  # A = model-A equivalent: H(psi) at fixed z propto D_L^-1 (POW = -1)
    best = dict(ns=(-np.inf, None), s=(-np.inf, None, None, None))
    for l0 in DGRID:
        idx = (((TH1 - l0 - 180.0) % 360) / 0.5).astype(int)
        A = Ad[idx]; A = A / A.mean(); x = A[IMIG]; th = (TH1 - l0)[IMIG]
        v = r2(MA, x)
        if v > best["ns"][0]: best["ns"] = (v, l0)
        am, ph = best_sine(x, th, MA); vs = r2(MA, A_from(x, th, am, ph))
        if vs > best["s"][0]: best["s"] = (vs, l0, am, ph)
    return best, Ad

def curve(Ad, l0, amp=0.0, phi=0.0):
    idx = (((TH1 - l0 - 180.0) % 360) / 0.5).astype(int); A = Ad[idx]; A = A / A.mean()
    return A_from(A, TH1 - l0, amp, phi) if amp else A

def evaluate(Mc_kg, D, R, Os=OS, Oin=0.0, full=False):
    out = dict(Mc_kg=Mc_kg, D=D, R=R, Os=Os, Oin=Oin)
    if not (R > 0 and D > R * 1.02): out["fail"] = "geometry"; return out
    p = params(Mc_kg, D, R, Os, Oin)
    t0 = solve_t0(p, D)
    if t0 is None: out["fail"] = "no t0 with H=70"; return out
    g = LT.geom(D, t0, p); Hp, Hr = g[3] / g[0], g[4] / g[1]
    out.update(t0=t0, age_Gyr=t0 * GYR, H_perp=Hp * K, H_r=Hr * K, twoE=2 * g[5], M_D_kg=g[7] * KG, rs_D=2 * g[7], along=bool(Hr > Hp),
               deficit_kg=4 * np.pi / 3 * (Os - Oin) * RHOC * (D - R)**3 * KG)
    out.update(validity(p, t0))
    if g[5] < 0: out["R_max"] = g[7] / -g[5]; out["t_turn_Gyr"] = np.pi * g[7] / (-2 * g[5])**1.5 * GYR
    try:
        nb_ = nearby_fast(p, t0, D, ZL, 24) if globals().get("FAST") else analyze.nearby(p, t0, D, zlist=ZL, N=24)
    except Exception as e:
        out["fail"] = "rays: %s" % e; return out
    DL = nb_["DL"]
    if not np.all(np.isfinite(DL)) or (DL <= 0).any(): out["fail"] = "rays nonfinite"; return out
    fz = {}
    for k, z in enumerate(ZL):
        b, Ad = lfits(DL[:, k], nb_["mu"])
        A = curve(Ad, b["ns"][1])
        fz[str(z)] = dict(R2_ns=float(b["ns"][0]), l0_ns=float(b["ns"][1]), R2_s=float(b["s"][0]), l0_s=float(b["s"][1]), amp=float(b["s"][2]), phi=float(b["s"][3]),
                          m2=float(analyze.fourier(TH1, A)[1]), corr=float(np.corrcoef(A[IMIG], MA)[0, 1]))
        if full: fz[str(z)]["Ad"] = Ad.tolist()
    out["fits"] = fz; out["R2_ns"] = fz[str(ZL[0])]["R2_ns"]; out["R2_s"] = fz[str(ZL[0])]["R2_s"]   # headline = first z in ZL
    return out

# ---- faster nearby for the wide scan: same method as analyze.nearby, rays traced only as far back as the target z needs
TMIN = {0.075: 0.85, 0.3: 0.6}
def nearby_fast(p, t0, ro, zlist, N=24):
    from numpy.polynomial.legendre import leggauss
    mu, wts = leggauss(N); psi = np.arccos(mu)
    tq = t0 * np.linspace(0.999, TMIN.get(max(zlist), 0.6), 80)
    DL = np.zeros((N, len(zlist)))
    for i, ps in enumerate(psi):
        z, DA = rays.area_distance(p, t0, ro, ps, tq)
        DL[i] = np.interp(zlist, z, (1 + z)**2 * DA)
    return dict(mu=mu, DL=DL)
analyze_nearby_orig = analyze.nearby
def _nb(p, t0, ro, zlist=ZL, N=24): return nearby_fast(p, t0, ro, zlist, N)
FAST = True
