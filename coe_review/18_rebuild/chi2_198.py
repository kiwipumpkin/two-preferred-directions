"""71 Migkas bins as independent points with their published errors. Isotropy (constant) vs the CoE model.
A: Newtonian CoE model, D fixed by the Axis of Evil, free mcc, kin, kout, l0, Amp, phi (6) -- fit from refit_newt_fixedD.
B: same geometry, free offset, scale, sine amp, sine phase (4).  Likelihood-ratio (Delta chi^2) test vs a constant (1 param)."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import sys, os, json, numpy as np, pandas as pd
from scipy.stats import chi2, norm
B = PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble'; sys.path.insert(0, B); os.chdir(B)
import newt_omb as N
from bubble_core import shifted, A_from, TH1
d = json.load(open(PKG + '/coe_review/06_consistent/fig4/prof_DR2.879.json'))
af, _ = N.af_of(N.geo(d['D'], d['mcc'], d['kin'], d['kout'], N.RV)); af = af/af.mean()
Amod = np.asarray(A_from(shifted(af, d['l0']), TH1-d['l0'], d['amp'], d['phi']), float)
m = pd.read_csv(PKG + '/workspace/cosmology/migkas/fig6_digitized.csv'); l = m.l_deg.values
yA, eA = m.A_over_Aall.values, (m.A_err_low.values+m.A_err_high.values)/2
yB, eB = m.B_over_Ball.values, (m.B_err_low.values+m.B_err_high.values)/2
def const_chi2(y, e): w = 1/e**2; c = np.sum(w*y)/np.sum(w); return np.sum(((y-c)/e)**2)
fA = np.interp(l, TH1, Amod, period=360)
# allow the model a free overall normalisation as the constant model has (fair comparison)
s = np.sum(fA*yA/eA**2)/np.sum(fA**2/eA**2); chiA_mod = np.sum(((yA-s*fA)/eA)**2)
shape = np.interp(l, TH1, shifted(af, d['l0']), period=360); L = np.radians(l)
X = np.column_stack([np.ones_like(l), shape, np.sin(L), np.cos(L)]); bB = np.linalg.lstsq(X/eB[:, None], yB/eB, rcond=None)[0]
chiB_mod = np.sum(((yB-X@bB)/eB)**2)
out = {}
for name, y, e, cm, k in (('A', yA, eA, chiA_mod, 6), ('B', yB, eB, chiB_mod, 3)):
    c0 = const_chi2(y, e); dchi = c0-cm; p = chi2.sf(dchi, k)
    out[name] = dict(n=len(y), chi2_isotropic=round(c0, 2), chi2_model=round(cm, 2), extra_params=k, dchi2=round(dchi, 2),
                     p=float(p), sigma=float(norm.isf(p/2)), chi2_model_per_dof=round(cm/(len(y)-k-1), 2))
dc = out['A']['dchi2']+out['B']['dchi2']; kk = 6+3; p = chi2.sf(dc, kk)
out['A+B'] = dict(dchi2=round(dc, 2), extra_params=kk, p=float(p), sigma=float(norm.isf(p/2)) if p > 0 else '>8')
print(json.dumps(out, indent=1)); json.dump(out, open(PKG + '/coe_review/18_rebuild/chi2_198.json', 'w'), indent=1)
