"""Chris's spreadsheet model ("density w rotation"), formulas as traced. One set of inputs drives both A and B.
Fit: shared orientation l0 (A and B), per-curve linear normalisation (as the sheet's LINEST); physics inputs scanned jointly on A only;
then B is evaluated with those inputs, sharing l0, with only its normalisation fitted."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import numpy as np, pandas as pd, json
from scipy.stats import chi2, norm
m=pd.read_csv(PKG + '/workspace/cosmology/migkas/fig6_digitized.csv'); l=m.l_deg.values
yA,eA=m.A_over_Aall.values,(m.A_err_low+m.A_err_high).values/2
yB,eB=m.B_over_Ball.values,(m.B_err_low+m.B_err_high).values/2
def sheet(D=10,R=3,M=50,L=1.5,k=1,G=1,dt=1,amp=-0.055,ph=0.15):
    th=np.radians(np.arange(360.)); px=D+R*np.cos(th); py=R*np.sin(th); r=np.hypot(px,py); phi=np.arctan2(py,px)
    Mt=M/(1-np.exp(-D/L)*(1+D/L+D**2/(2*L**2))); mr=Mt*(1-np.exp(-r/L)*(1+r/L+r**2/(2*L**2)))
    Vo=k*D; Ax=-G*M/D**2; a=-G*mr/r**2
    dVx=k*r*np.cos(phi)-Vo; dVy=k*r*np.sin(phi); dAx=a*np.cos(phi)-Ax; dAy=a*np.sin(phi)
    A=np.hypot(dVx+dAx*dt,dAy+dVy*dt)/np.hypot(dVx,dVy)
    deg=np.arange(360.); A=(1-amp*np.sin(np.radians(deg)+ph))*A; B=(1+amp*np.sin(np.radians(deg)+ph))*mr/M
    return A,B
def fit(y,e,model,l0):
    f=np.interp((l-l0)%360,np.arange(360.),model,period=360); X=np.column_stack([np.ones_like(l),f])
    b=np.linalg.lstsq(X/e[:,None],y/e,rcond=None)[0]; return np.sum(((y-X@b)/e)**2),b,f
iso=lambda y,e:(lambda w:np.sum(w*(y-np.sum(w*y)/np.sum(w))**2))(1/e**2)
cA0,cB0=iso(yA,eA),iso(yB,eB)
# 1) sheet inputs exactly, scan l0 shared (fit on A)
A,B=sheet(); best=min(((fit(yA,eA,A,l0)[0],l0) for l0 in range(360)))
l0=best[1]; cA=best[0]; cB=fit(yB,eB,B,l0)[0]
out={'sheet_inputs':dict(l0_from_A=l0,chi2A=cA,chi2B=cB,dA=cA0-cA,dB=cB0-cB)}
# 2) scan physics inputs on A (D/R, L, M, amp, ph, l0), then B fixed
res=[]
for DR in (3.0,3.33,3.56,4.0):
  for Lr in (0.1,0.15,0.25):
    for M in (20,50,100):
      for amp in (-0.04,-0.055,-0.07):
        for ph in np.linspace(0,2*np.pi,13)[:-1]:
          A,B=sheet(D=10,R=10/DR,M=M,L=Lr*10,amp=amp,ph=ph)
          if not np.all(np.isfinite(A)): continue
          c,l0=min(((fit(yA,eA,A,x)[0],x) for x in range(0,360,2)))
          res.append((c,DR,Lr,M,amp,ph,l0))
res.sort(); c,DR,Lr,M,amp,ph,l0=res[0]
A,B=sheet(D=10,R=10/DR,M=M,L=Lr*10,amp=amp,ph=ph); cB,bB,_=fit(yB,eB,B,l0)
pB=chi2.sf(cB0-cB,1)
out['scanned_on_A']=dict(DR=DR,L_over_D=Lr,M=M,amp=amp,phase=ph,l0=l0,chi2A=c,chi2B=cB,dB=cB0-cB,pB=pB,sigmaB=norm.isf(pB/2),
                        R2B_w=1-cB/cB0,slopeB=bB[1])
for k,v in out.items(): print(k,{kk:(round(vv,4) if isinstance(vv,float) else vv) for kk,vv in v.items()})
json.dump(out,open('sheet_AB.json','w'),indent=1,default=float)
