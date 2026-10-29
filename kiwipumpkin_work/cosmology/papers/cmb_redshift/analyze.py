"""CMB redshift anisotropy and nearby Hubble anisotropy for an off-centre LTB observer."""
import numpy as np, ltb, rays
from numpy.polynomial.legendre import leggauss, legval, legfit
from scipy.optimize import brentq
ZREC = 1089.0
T0K = 2.7255
L_COE = 193.17            # Galactic longitude of the CoE direction (anti-CoE at l0 = 13.17 deg)

def cl_equiv(c, l):
    """C_l of an axisymmetric pattern c P_l, and D_l in muK^2."""
    C = 4 * np.pi * c**2 / (2 * l + 1)**2
    return C, l * (l + 1) * C / (2 * np.pi) * (T0K * 1e6)**2

def boost_multipoles(mu, wts, T, lmax=6):
    """Null the dipole with an axial boost beta (observer velocity along psi = 0).
    T'(mu') = T(mu) / (gamma (1 - beta mu')),  mu = (mu' - beta)/(1 - beta mu')."""
    from scipy.interpolate import PchipInterpolator
    mu2, wts2 = leggauss(400)
    cf = legfit(mu, T, len(mu) - 1)
    if np.max(np.abs(cf[len(mu) // 2:])) < 1e-8 * np.max(np.abs(cf[1:4])):   # smooth pattern: Legendre interpolation
        ip = lambda x: legval(x, cf)
    else:                                   # sharp features (cap edge): monotone cubic, constant at the poles
        ip = PchipInterpolator(np.r_[-1, mu, 1], np.r_[T[0], T, T[-1]])
    def mult(beta):
        g = 1 / np.sqrt(1 - beta**2); m0 = (mu2 - beta) / (1 - beta * mu2)
        Tb = ip(m0) / (g * (1 - beta * mu2))
        return rays.legendre_coeffs(mu2, wts2, Tb / (0.5 * np.sum(wts2 * Tb)) - 1, lmax)
    try:
        beta = brentq(lambda b: mult(b)[1], -0.999, 0.999, xtol=1e-15)
    except ValueError:
        return np.nan, np.full(lmax + 1, np.nan)
    return beta, mult(beta)

def cmb(p, t0, ro, N=48, t_end_frac=1e-6):
    mu, wts = leggauss(N); psi = np.arccos(mu)
    sols = [rays.trace(p, t0, ro, ps, t0 * t_end_frac) for ps in psi]
    nerr = max(rays.null_err(p, t0, s) for s in sols[::8])
    out = dict(N=N, null_err=nerr, mu=mu, psi=psi)
    # (a) fixed proper time since the bang: t_rec such that the monopole of z is ZREC
    zbar = lambda tr: 0.5 * np.sum(wts * np.array([rays.z_at_t(s, t0, tr) for s in sols]))
    lo, hi = t0 * t_end_frac * 1.001, t0 * 0.5
    tr = np.exp(brentq(lambda lt: np.log1p(zbar(np.exp(lt))) - np.log1p(ZREC), np.log(lo), np.log(hi), xtol=1e-14))
    za = np.array([rays.z_at_t(s, t0, tr) for s in sols])
    Ta = 1 / (1 + za); Ta = Ta / (0.5 * np.sum(wts * Ta))
    out.update(t_rec_Myr=tr * ltb.GYR * 1e3, z_a=za, dT_a=Ta - 1, c_a=rays.legendre_coeffs(mu, wts, Ta - 1, 6))
    out["beta_a"], out["cb_a"] = boost_multipoles(mu, wts, Ta)
    # (b) fixed local density: rho_rec such that the monopole of z (over directions where it exists) is ZREC
    def zb(lr):
        z = np.array([rays.z_at_rho(s, p, t0, np.exp(lr))[0] for s in sols])
        ok = np.isfinite(z); return z, ok
    rho_eds = lambda t: 1 / (6 * np.pi * t**2)
    g = lambda lr: (lambda z, ok: np.log1p(0.5 * np.sum(wts[ok] * z[ok]) / (0.5 * np.sum(wts[ok]))) - np.log1p(ZREC))(*zb(lr))
    try:
        lr = brentq(g, np.log(rho_eds(tr * 20)), np.log(rho_eds(t0 * t_end_frac * 1.5)), xtol=1e-13)
        zbv, ok = zb(lr)
    except ValueError:                      # no direction reaches the recombination density with z = ZREC
        lr = np.log(rho_eds(tr)); zbv, ok = zb(lr)
    out.update(rho_rec_over_eds_at_ta=np.exp(lr) / rho_eds(tr), z_b=zbv, sky_frac_b=0.5 * np.sum(wts[ok]))
    if ok.all():
        Tb = 1 / (1 + zbv); Tb = Tb / (0.5 * np.sum(wts * Tb))
        out.update(dT_b=Tb - 1, c_b=rays.legendre_coeffs(mu, wts, Tb - 1, 6))
        out["beta_b"], out["cb_b"] = boost_multipoles(mu, wts, Tb)
    else:
        out.update(dT_b=np.full(N, np.nan), c_b=np.full(7, np.nan), beta_b=np.nan, cb_b=np.full(7, np.nan))
        out["psi_max_b_deg"] = float(np.degrees(psi[ok].max())) if ok.any() else 0.0
    return out

def nearby(p, t0, ro, zlist=(0.05, 0.1, 0.2, 0.3), N=24):
    """D_L(z, psi) from ray bundles; returns fractional H0 variation and A = (D_L/<D_L>)^2 on the Galactic plane."""
    mu, wts = leggauss(N); psi = np.arccos(mu)
    tq = t0 * np.linspace(0.999, 0.6, 160)
    DL = np.zeros((N, len(zlist)))
    for i, ps in enumerate(psi):
        z, DA = rays.area_distance(p, t0, ro, ps, tq)
        dl = (1 + z)**2 * DA
        DL[i] = np.interp(zlist, z, dl)
    dH = -(DL / (0.5 * np.sum(wts[:, None] * DL, axis=0)) - 1)       # delta H0 / H0 on the sphere
    # Galactic plane, b = 0: psi = angle between (l, 0) and the CoE direction
    ell = np.arange(0, 360, 1.0); cpsi = np.cos(np.radians(ell - L_COE))
    DLp = np.array([legval(cpsi, legfit(mu, DL[:, k], N - 1)) for k in range(len(zlist))]).T
    A = (DLp / DLp.mean(axis=0))**2
    Hp, Hr = ltb.local_rates(ro, t0, p)
    return dict(mu=mu, psi=psi, DL=DL, dH=dH, ell=ell, A=A, zlist=np.array(zlist), Hperp=Hp, Hr=Hr,
                dH_leg=[rays.legendre_coeffs(mu, wts, dH[:, k], 4) for k in range(len(zlist))])

def fourier(ell, y):
    t = np.radians(ell); out = []
    for m in (1, 2, 3):
        c = 2 * np.mean(y * np.cos(m * t)); s = 2 * np.mean(y * np.sin(m * t)); out.append(np.hypot(c, s))
    return np.array(out)
