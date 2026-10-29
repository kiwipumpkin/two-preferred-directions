import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root
# Exact-LTB Migkas fit (relfree.ev, as universe_rel_mcfree.json) with D FIXED at the AoE value (D/R = 3.564, D = 50.97 Gpc).
# Free: mcc = Mc/Mcrit(D), kin, kout (shell edges), l0, Amp, phi - as in relfree, minus D. Start: the stored optimum's mcc, kin, kout.
import sys,os,json,numpy as np
B=PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble/gr_ltb/refit_mass';sys.path.insert(0,B);sys.path.insert(0,PKG + '/kiwipumpkin_work/cosmology/papers/cmb_redshift');sys.path.insert(0,PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble/gr_ltb');os.chdir(B)
import relfree2 as F
from scipy.optimize import minimize
D=3.564*14.30048287260589;L=np.log10(D)
def f(v):o=F.evs(np.r_[L,v]);return -o.get('R2_s',-5.0)
best=None
for v0 in ([np.log10(0.0835),np.log(0.0404),np.log(0.1709)],[np.log10(0.06),np.log(0.05),np.log(0.3)],[np.log10(0.1),np.log(0.1),np.log(0.1)]):
    r=minimize(f,np.array(v0),method='Nelder-Mead',options=dict(xatol=0.01,fatol=2e-4,maxiter=120))
    o=F.evs(np.r_[L,r.x]);print('start',np.round(v0,3),'->',{k:o.get(k) for k in ('D','mcc','R_in','R_out','R2_s','l0_s','age_Gyr')},flush=True)
    if 'R2_s' in o and (best is None or o['R2_s']>best['R2_s']):best=o
json.dump(best,open(PKG + '/coe_review/06_consistent/fig4/ltb_fixedD.json','w'),indent=1,default=float)
