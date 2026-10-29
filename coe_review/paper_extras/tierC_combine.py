"""Tier C (Sec. 8, Table 3): T3 with both roll branches, and Fisher combination of T1+T2 (from combined.json) with T3."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import json, math
from scipy.stats import chi2, norm
J = json.load(open(PKG + "/coe_review/13_combined/combined.json"))
pT3 = 4*10.18/360          # roll within 10.18 deg of the Axis-of-Evil roll on either branch (two windows of +-10.18 deg)
print("T3 (either branch) p = %.4f" % pT3)
for k in ("null_A", "null_B"):
    pj = J[k]["p_joint_T1T2"]; F = -2*(math.log(pj) + math.log(pT3)); p = chi2.sf(F, 4)
    print("%s: T1+T2 p = %.4f (%.1f sigma); all three p = %.4f (%.2f sigma)" % (k, pj, J[k]["sigma_T1T2"], p, norm.isf(p/2)))
