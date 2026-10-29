"""Tier B (Sec. 8): probability that a random headless axis falls in the patch the Migkas fit had fixed before the CMB comparison
(D/R 3-4.5 -> theta band; roll 45-80 deg, either roll branch), with the CoE direction fixed. 2e6 random axes."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import numpy as np
exec(open(PKG + "/coe_review/06_consistent/geometry_from_aoe.py").read().split("out={}")[0])   # frame(), vec()
R0 = frame(-15, 0); C = R0 @ vec(180, 0); Y0 = R0 @ np.array([0, 1., 0]); Z0 = np.cross(C, Y0)
def psi(A):
    if A @ C < 0: A = -A
    return np.degrees(np.arctan2(A @ Z0, A @ Y0)) % 360, np.degrees(np.arccos(A @ C))
p, t = psi(vec(80, -60)); print("Axis of Evil: roll %.2f, theta %.2f" % (p, t))
sgn = 1 if abs(((p - 60.18 + 180) % 360) - 180) < 1 else -1
rng = np.random.default_rng(4); N = 2_000_000
A = rng.normal(size=(N, 3)); A /= np.linalg.norm(A, axis=1)[:, None]; A *= np.sign(A @ C)[:, None]
ps = (sgn*np.degrees(np.arctan2(A @ Z0, A @ Y0))) % 360; th = np.degrees(np.arccos(np.clip(A @ C, -1, 1)))
inwin = ((ps >= 45) & (ps <= 80)) | ((ps >= 225) & (ps <= 260))
for lo, hi in [(3.0, 4.5), (2.0, 6.0), (2.0, 8.0)]:
    tlo, thi = np.degrees(np.arccos(1/(2*lo))), np.degrees(np.arccos(1/(2*hi))); band = (th >= tlo) & (th <= thi)
    print(f"D/R {lo}-{hi}: theta {tlo:.1f}-{thi:.1f}; P(band) = {band.mean():.4f}; P(band & roll window) = {(band & inwin).mean():.4f}")
