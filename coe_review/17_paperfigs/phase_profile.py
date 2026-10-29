import numpy as np, json
from scipy.optimize import minimize
exec(open('sheet_AB.py').read().split('# 1) sheet inputs')[0])
DR=3.564; j=json.load(open('sheet_joint356.json'))
def tot(p,ph):
    Lr,lM,amp,l0=p
    if not (0.02<Lr<2 and 0<amp<0.2): return 1e9
    A,B=sheet(D=10,R=10/DR,M=10**lM,L=Lr*10,amp=-amp,ph=ph)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))): return 1e9
    return fit(yA,eA,A,l0)[0]+fit(yB,eB,B,l0)[0]
x=[j['L_over_D'],np.log10(j['M']),j['amp'],j['l0']]; out=[]
for dph in np.arange(-60,61,5):
    ph=j['phase']+np.radians(dph); best=None
    for l0s in (x[3]-15,x[3],x[3]+15):
        r=minimize(tot,[x[0],x[1],x[2],l0s],args=(ph,),method='Nelder-Mead',options=dict(maxiter=2000,xatol=1e-4,fatol=1e-5))
        if best is None or r.fun<best.fun: best=r
    lmax=(np.degrees(np.pi/2-ph)+best.x[3])%360   # sky longitude where A factor peaks
    out.append((dph,best.fun,lmax,best.x[3]%360)); print(dph,round(best.fun,2),round(lmax,1),round(best.x[3]%360,1))
c0=min(o[1] for o in out); ok=[o for o in out if o[1]-c0<=1]; print('dchi2<=1 Amax longitude range',min(o[2] for o in ok),max(o[2] for o in ok))
ok=[o for o in out if o[1]-c0<=4]; print('dchi2<=4 range',min(o[2] for o in ok),max(o[2] for o in ok))
