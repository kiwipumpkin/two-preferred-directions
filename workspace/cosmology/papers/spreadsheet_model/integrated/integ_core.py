"""Integrated version of Chris's model (his geometry, e^{-r/L} profile, sine factor).
Units: length D (today's Earth-CoE distance), time t0 = 1/k. Unperturbed flow x = x0*tau, v = x0.
Gravity: xdd = -mu * [m(<r,tau)/M] x / r^3,  mu = G_SI M t0^2 / D^3,  M = mass inside D.
M/M_crit = 2 mu (k/H0)^2  (M_crit = 4/3 pi rho_c D^3, H0 = 70)."""
import numpy as np, numba as nb, os, csv
HERE = os.path.dirname(os.path.abspath(__file__))
K = 2.18e-18; H0 = 70 / 3.0857e19; GSI = 6.674e-11; RKM = 4.4127e23; GPC = 3.0857e22
MCRIT_FAC = 2 * (K / H0) ** 2           # M/Mcrit = MCRIT_FAC * mu

@nb.njit(cache=True, fastmath=False)
def Fx(X):
    if X < 1e-2:
        return X**3 / 6.0 * (1 - 0.75 * X + 0.3 * X * X)
    return 1.0 - np.exp(-X) * (1.0 + X + 0.5 * X * X)

@nb.njit(cache=True)
def acc(x, y, tau, mu, LD, como, FD):
    r = max(np.sqrt(x * x + y * y), 1e-12)
    ell = LD * tau if como else LD
    m = Fx(r / ell) / FD
    f = -mu * m / (r * r * r)
    return f * x, f * y

@nb.njit(cache=True, parallel=True)
def integrate(q, mu, LD, como, taui, nsteps):
    """q: (N,2) comoving start positions (x = q*taui, v = q at taui). RK4 in s = ln tau.
    Returns x(1), v(1)."""
    N = q.shape[0]; FD = Fx(1.0 / LD)
    X = np.empty((N, 2)); V = np.empty((N, 2)); VR = np.empty(N)
    s0 = np.log(taui); h = -s0 / nsteps
    for i in nb.prange(N):
        x = q[i, 0] * taui; y = q[i, 1] * taui; vx = q[i, 0]; vy = q[i, 1]
        s = s0; vrmin = 1e30
        for n in range(nsteps):
            t1 = np.exp(s); t2 = np.exp(s + 0.5 * h); t4 = np.exp(s + h)
            ax, ay = acc(x, y, t1, mu, LD, como, FD)
            k1x = t1 * vx; k1y = t1 * vy; l1x = t1 * ax; l1y = t1 * ay
            ax, ay = acc(x + 0.5 * h * k1x, y + 0.5 * h * k1y, t2, mu, LD, como, FD)
            k2x = t2 * (vx + 0.5 * h * l1x); k2y = t2 * (vy + 0.5 * h * l1y); l2x = t2 * ax; l2y = t2 * ay
            ax, ay = acc(x + 0.5 * h * k2x, y + 0.5 * h * k2y, t2, mu, LD, como, FD)
            k3x = t2 * (vx + 0.5 * h * l2x); k3y = t2 * (vy + 0.5 * h * l2y); l3x = t2 * ax; l3y = t2 * ay
            ax, ay = acc(x + h * k3x, y + h * k3y, t4, mu, LD, como, FD)
            k4x = t4 * (vx + h * l3x); k4y = t4 * (vy + h * l3y); l4x = t4 * ax; l4y = t4 * ay
            x += h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x); y += h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
            vx += h / 6 * (l1x + 2 * l2x + 2 * l3x + l4x); vy += h / 6 * (l1y + 2 * l2y + 2 * l3y + l4y)
            s += h
            vr = x * vx + y * vy
            if vr < vrmin: vrmin = vr
        X[i, 0] = x; X[i, 1] = y; V[i, 0] = vx; V[i, 1] = vy; VR[i] = vrmin
    return X, V, VR

def geometry(theta_deg, DR):
    rho = 1.0 / DR; th = np.radians(theta_deg)
    q = np.empty((len(th) + 1, 2)); q[0] = (1.0, 0.0)
    q[1:, 0] = 1 + rho * np.cos(th); q[1:, 1] = rho * np.sin(th)
    return q, rho

def AF(theta_deg, DR, LD, mu, como=False, taui=0.01, nsteps=800, shoot=0):
    q0, rho = geometry(theta_deg, DR)
    q = q0.copy()
    X, V, VR = integrate(q, mu, LD, como, taui, nsteps)
    for _ in range(shoot):                        # adjust starts so x(t0) lands on his geometry
        q = q + (q0 - X); X, V, VR = integrate(q, mu, LD, como, taui, nsteps)
    vrel = V[1:] - V[0]
    af = np.hypot(vrel[:, 0], vrel[:, 1]) / rho
    if VR.min() <= 0 or not np.all(np.isfinite(af)):  # a particle turned around (fell back) before today
        af = af * np.nan
    return af, X

def AFfour(l):
    l = np.radians(l)
    return (0.9965 + 0.0296*np.cos(l) + 0.0434*np.sin(l) + 0.0636*np.cos(2*l) + 0.0366*np.sin(2*l)
            - 0.0299*np.cos(3*l) - 0.0114*np.sin(3*l))
mig = list(csv.DictReader(open(os.path.join(HERE, "../../../migkas/fig6_digitized.csv"))))
ML = np.array([float(r["l_deg"]) for r in mig]); MA = np.array([float(r["A_over_Aall"]) for r in mig])
MLO = np.array([float(r["A_err_low"]) for r in mig]); MHI = np.array([float(r["A_err_high"]) for r in mig])

def r2(y, m): return 1 - np.sum((y - m) ** 2) / np.sum((y - y.mean()) ** 2)
def A_from(af, th, amp, phi): return (1 - amp * np.sin(np.radians(th) + phi)) * af

def best_sine(af, th, y, ampmax=0.15):
    """A = af - c*af*sin(th) - s*af*cos(th), c = amp cos(phi), s = amp sin(phi): linear LS."""
    t = np.radians(th); Xm = np.vstack([af * np.sin(t), af * np.cos(t)]).T
    c, s = np.linalg.lstsq(Xm, af - y, rcond=None)[0]
    amp = np.hypot(c, s); phi = np.arctan2(s, c)
    if amp > ampmax:                              # clip to bound, re-optimise phase on a grid
        ph = np.linspace(-np.pi, np.pi, 721); best = None
        for sgn in (1, -1):
            for p in ph:
                v = np.sum((y - A_from(af, th, sgn * ampmax, p)) ** 2)
                if best is None or v < best[0]: best = (v, sgn * ampmax, p)
        amp, phi = best[1], best[2]
    if amp > 0: amp, phi = -amp, phi + np.pi          # quote with Amp < 0 (his convention)
    return amp, phi % (2 * np.pi)
