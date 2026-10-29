"""Generic Fourier series fitted to Migkas A (71 bins, published errors averaged): 2 harmonics (5 params) and 3 harmonics (7 params).
Paper Sec. 8 ('Why compare with a dipole') and Fig. 1 caption (R^2 = 0.81)."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import numpy as np, pandas as pd
MIGKAS_CSV = PKG + "/workspace/cosmology/migkas/fig6_digitized.csv"
m = pd.read_csv(MIGKAS_CSV); l = np.radians(m.l_deg.values)
y, e = m.A_over_Aall.values, (m.A_err_low.values + m.A_err_high.values) / 2
w = 1 / e**2; c0 = np.sum(w * (y - np.sum(w * y) / np.sum(w))**2)
for nh in (1, 2, 3):
    X = np.column_stack([np.ones_like(l)] + [f(k * l) for k in range(1, nh + 1) for f in (np.cos, np.sin)])
    b = np.linalg.lstsq(X / e[:, None], y / e, rcond=None)[0]; c = np.sum(((y - X @ b) / e)**2)
    r = y - X @ b; R2u = 1 - np.sum(r**2) / np.sum((y - y.mean())**2)
    print(f"{nh} harmonic(s), {X.shape[1]} params: chi2 = {c:.2f}, weighted R^2 = {1 - c / c0:.3f}, unweighted R^2 = {R2u:.3f}  (isotropy chi2 {c0:.2f})")
