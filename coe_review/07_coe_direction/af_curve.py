import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
# Axisymmetric Newtonian directional expansion af(psi) (psi from the anti-CoE direction) at the AoE-fixed D, from your newt_omb model
# (fitted mcc, kin, kout of the D = 50.97 Gpc refit; no longitude sine term).
import sys,os,json,numpy as np
B=PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble';sys.path.insert(0,B);os.chdir(B)
import newt_omb as N
d=json.load(open(PKG + '/coe_review/06_consistent/fig4/newt_fixedD.json'))
af,rr=N.af_of(N.geo(d['D'],d['mcc'],d['kin'],d['kout'],N.RV))
th=np.arange(len(af))*360/len(af);print(len(af),'max/min',af.max(),af.min(),'at',th[np.argmax(af)],th[np.argmin(af)])
json.dump(dict(theta=th.tolist(),af=af.tolist(),src='newt_fixedD.json'),open(PKG + '/coe_review/07_coe_direction/af_D51.json','w'))
