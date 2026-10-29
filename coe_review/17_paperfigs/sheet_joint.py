"""Joint fit of Chris's spreadsheet model to A and B: one shared set of inputs (D/R, L/D, M, rotation amp, phase, l0);
each curve gets only a linear normalisation (offset + scale, as the sheet's LINEST). B is never fitted separately."""
import numpy as np, json, os
AMPMAX = float(os.environ.get('AMPMAX', 0.6))   # rotation-amplitude bound: 0.2 -> D/R 2.83 (Appendix B); 0.6 -> flat to D/R ~1.65
from scipy.optimize import minimize
from scipy.stats import chi2, norm
exec(open('sheet_AB.py').read().split('# 1) sheet inputs')[0])
def tot(p):
    DR,Lr,lM,amp,ph,l0=p
    if not (1.2<DR<12 and 0.02<Lr<2 and 0<amp<AMPMAX): return 1e9
    A,B=sheet(D=10,R=10/DR,M=10**lM,L=Lr*10,amp=-amp,ph=ph)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))): return 1e9
    return fit(yA,eA,A,l0)[0]+fit(yB,eB,B,l0)[0]
starts=[[3.33,0.15,np.log10(50),0.055,0.15,8]]+[[dr,lr,lm,a,ph,l0] for dr in (2.5,3.5,5) for lr in (0.1,0.3) for lm in (1.3,1.7,2.0) for a in (0.03,0.07) for ph in (0.15,2.0,4.0) for l0 in (0,40,320)]
best=None
for s in starts:
    if tot(s)>1e8: continue
    r=minimize(tot,s,method='Nelder-Mead',options=dict(maxiter=3000,xatol=1e-4,fatol=1e-5))
    if best is None or r.fun<best.fun: best=r
DR,Lr,lM,amp,ph,l0=best.x
A,B=sheet(D=10,R=10/DR,M=10**lM,L=Lr*10,amp=-amp,ph=ph)
cA,bA,fA=fit(yA,eA,A,l0); cB,bB,fB=fit(yB,eB,B,l0)
d=(cA0+cB0)-(cA+cB); k=8; p=chi2.sf(d,k)
pB=chi2.sf(cB0-cB,7); pA=chi2.sf(cA0-cA,7)
out=dict(DR=DR,L_over_D=Lr,M=10**lM,amp=amp,phase=ph%(2*np.pi),l0=l0%360,chi2A=cA,chi2B=cB,chi2A_iso=cA0,chi2B_iso=cB0,
         R2wA=1-cA/cA0,R2wB=1-cB/cB0,dchi2_joint=d,extra_params=k,p_joint=p,sigma_joint=norm.isf(p/2),
         sigma_A_strict=norm.isf(pA/2),sigma_B_strict=norm.isf(pB/2),
         sheet_inputs_ref=dict(DR=10/3,chi2A=10.25,chi2B=7.28))
print(json.dumps(out,indent=1,default=float)); json.dump(out,open('sheet_joint.json','w'),indent=1,default=float)
np.savez('sheet_joint_curves.npz',A=A,B=B,bA=bA,bB=bB,l0=l0)
