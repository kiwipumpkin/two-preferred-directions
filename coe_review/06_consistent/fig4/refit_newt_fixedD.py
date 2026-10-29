import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root
# Newtonian CoE model (newt_omb.py, Omega_s forced 0.049) refit to Migkas A/A_all with D FIXED by the Axis of Evil:
# D = (D/R) * R = 3.564 * 14.30 Gpc. Free: mcc, kin, kout, l0, (Amp, phi) exactly as in newt_omb.fit, minus D.
import sys,os,json,time,numpy as np
B=PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble';sys.path.insert(0,B);os.chdir(B)
import newt_omb as N
from scipy.optimize import minimize
DR=json.load(open(PKG + '/coe_review/06_consistent/geometry_from_aoe.json'))['Land & Magueijo 2005 (260, +60)']['DR'];D=DR*N.RV
t=time.time()
def obj(v):return N.objective(np.r_[np.log10(D),v],N.RV,True)
rng=np.random.default_rng(0);U=np.c_[rng.uniform(-6,1.5,200),rng.uniform(-4,2.5,200),rng.uniform(-4,2.5,200)]
vals=np.array([obj(u) for u in U]);print('screen done %.0fs best R2 %.3f'%(time.time()-t,1-vals.min()),flush=True)
best=(np.inf,None)
for i in np.argsort(vals)[:5]:
    r=minimize(obj,U[i],method='Nelder-Mead',options=dict(xatol=1e-3,fatol=1e-5,maxiter=400))
    if r.fun<best[0]:best=(r.fun,r.x)
d=N.describe(*N.unpack(np.r_[np.log10(D),best[1]]),N.RV,True)
print({k:(float('%.4g'%v) if isinstance(v,float) else v) for k,v in d.items() if k!='A'},flush=True)
json.dump(d,open(PKG + '/coe_review/06_consistent/fig4/newt_fixedD.json','w'),indent=1,default=float)
