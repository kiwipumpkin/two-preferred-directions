"""Appendix C (Table 6): significance under different treatments of the bins.
(1) 71 independent bins: from chi2_test.json (A) and the dipole comparison (Delta chi2 = 22.0, 4 extra params).
(2) Full map covariance, CoE term (2 params): Delta chi2 = chi2_iso - chi2_min from dir_fit (07_coe_direction/dir_fit.npz).
(3) Delta chi2 scaled to 36 effective bins."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import json, numpy as np
from scipy.stats import chi2, norm
s = lambda d, k: norm.isf(chi2.sf(max(d, 0), k)/2)
A = json.load(open(PKG + "/coe_review/17_paperfigs/chi2_test.json"))["A"]
z = np.load(PKG + "/coe_review/07_coe_direction/dir_fit.npz"); d2 = float(z["iso"] - z["chi"].min())
dA, ddip = A["dchi2"], 22.0
print("71 independent bins:      A vs isotropy %.1f sigma (dchi2 %.2f, 6 params);  A vs dipole %.1f sigma (dchi2 %.1f, 4 params)" % (s(dA, 6), dA, s(ddip, 4), ddip))
print("full map covariance:      dchi2 = %.2f (iso %.2f - min %.2f) -> %.1f sigma (2 params)" % (d2, float(z["iso"]), float(z["chi"].min()), s(d2, 2)))
f = 36/71
print("36 effective bins:        A vs isotropy %.1f sigma;  A vs dipole %.1f sigma" % (s(dA*f, 6), s(ddip*f, 4)))
