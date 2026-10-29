import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import numpy as np, json
from scipy.optimize import minimize
from scipy.stats import chi2, norm
exec(open('sheet_AB.py').read().split('# 1) sheet inputs')[0])
DR=3.564
def tot(p):
    Lr,lM,amp,ph,l0=p
    if not (0.02<Lr<2 and 0<amp<0.2): return 1e9
    A,B=sheet(D=10,R=10/DR,M=10**lM,L=Lr*10,amp=-amp,ph=ph)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))): return 1e9
    return fit(yA,eA,A,l0)[0]+fit(yB,eB,B,l0)[0]
best=None
for s in [[lr,lm,a,ph,l0] for lr in (0.1,0.15,0.3) for lm in (1.3,1.7,2.2,2.5) for a in (0.05,0.12) for ph in (0.15,1.5,3,4.5) for l0 in (0,10,30,340)]:
    if tot(s)>1e8: continue
    r=minimize(tot,s,method='Nelder-Mead',options=dict(maxiter=3000,xatol=1e-4,fatol=1e-5))
    if best is None or r.fun<best.fun: best=r
Lr,lM,amp,ph,l0=best.x
A,B=sheet(D=10,R=10/DR,M=10**lM,L=Lr*10,amp=-amp,ph=ph); A0,B0=sheet(D=10,R=10/DR,M=10**lM,L=Lr*10,amp=0,ph=ph)
cA,bA,_=fit(yA,eA,A,l0); cB,bB,_=fit(yB,eB,B,l0)
d=(cA0+cB0)-(cA+cB); p=chi2.sf(d,7)
o=dict(DR=DR,L_over_D=Lr,M=10**lM,amp=amp,phase=ph%(2*np.pi),l0=l0%360,chi2A=cA,chi2B=cB,R2wA=1-cA/cA0,R2wB=1-cB/cB0,
       dchi2_joint=d,extra_params=7,sigma_joint=norm.isf(p/2),p_joint=p,
       sigma_A_strict=norm.isf(chi2.sf(cA0-cA,6)/2),sigma_B_strict=norm.isf(chi2.sf(cB0-cB,6)/2),dA=cA0-cA,dB=cB0-cB)
print({k:(round(v,4) if isinstance(v,float) else v) for k,v in o.items()}); json.dump(o,open('sheet_joint356.json','w'),indent=1,default=float)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
lg=np.arange(360.)
fig,ax=plt.subplots(4,1,figsize=(9,10.5),sharex=True,gridspec_kw=dict(height_ratios=[3,1.2,3,1.2],hspace=0.08))
for i,(y,e,mod,mod0,b,lab,col,c) in enumerate(((yA,eA,A,A0,bA,r'$A/A_{\rm all}$','#c0392b',cA),(yB,eB,B,B0,bB,r'$B/B_{\rm all}$','#2b6cb0',cB))):
    k=2*i; f=lambda M: b[0]+b[1]*np.interp((lg-l0)%360,np.arange(360.),M,period=360)
    ax[k].errorbar(l,y,yerr=e,fmt='o',ms=3.5,color='0.25',ecolor='0.75',lw=1,label='Migkas et al. (71 bins)')
    ax[k].plot(lg,f(mod0),color=col,lw=1.2,ls=':',label='geometry only (no rotation term)')
    ax[k].plot(lg,f(mod),color=col,lw=2.4,label=f'CoE model with rotation term: $\\chi^2$ = {c:.1f}, weighted $R^2$ = {1-c/(cA0 if i==0 else cB0):.2f}')
    ax[k].set_ylabel(lab,fontsize=12); ax[k].legend(fontsize=8,loc='lower left'); ax[k].grid(alpha=.3); ax[k].set_ylim(0.78,1.2)
    res=100*(y-np.interp(l,lg,f(mod))); ax[k+1].bar(l,res,width=4,color=np.where(res>0,'#e8a29b','#9bb8e8')); ax[k+1].axhline(0,color='k',lw=.8)
    ax[k+1].set_ylabel('data − model\n(%)',fontsize=9); ax[k+1].set_ylim(-8,8); ax[k+1].grid(alpha=.3)
ax[3].set_xlabel('Galactic longitude ℓ (deg)'); ax[3].set_xlim(0,360); ax[3].set_xticks(range(0,361,45))
ax[0].set_title('One set of inputs (D/R = 3.56, density scale, mass, rotation, orientation) for both curves; each curve gets only a normalization.\nA: expansion boost; B: enclosed mass along the line of sight; rotation term with opposite sign.',fontsize=8.5)
fig.savefig(PKG + '/coe_review/14_paper/figs/fig_AB.png',dpi=180,bbox_inches='tight'); print('fig ok')
