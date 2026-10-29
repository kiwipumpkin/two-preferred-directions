import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import sys, os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
B=PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble'; sys.path.insert(0,B); os.chdir(B)
import newt_omb as N
from bubble_core import shifted, A_from, TH1
d=json.load(open(PKG + '/coe_review/06_consistent/fig4/newt_fixedD.json'))
af,_=N.af_of(N.geo(d['D'],d['mcc'],d['kin'],d['kout'],N.RV)); af=af/af.mean()
Amod=np.asarray(A_from(shifted(af,d['l0']),TH1-d['l0'],d['amp'],d['phi']),float)
m=pd.read_csv(PKG + '/workspace/cosmology/migkas/fig6_digitized.csv'); l=m.l_deg.values; y=m.A_over_Aall.values; e=(m.A_err_low.values+m.A_err_high.values)/2
fA=np.interp(l,TH1,Amod,period=360); s=np.sum(fA*y/e**2)/np.sum(fA**2/e**2); cC=np.sum(((y-s*fA)/e)**2)
L=np.radians(l); X=np.column_stack([np.ones_like(L),np.sin(L),np.cos(L)]); b=np.linalg.lstsq(X/e[:,None],y/e,rcond=None)[0]; cD=np.sum(((y-X@b)/e)**2)
w=1/e**2; cI=np.sum(w*(y-np.sum(w*y)/np.sum(w))**2)
dch=cD-cC; p=chi2.sf(dch,4); sg=norm.isf(p/2)
lg=np.arange(0,360.5,0.5); Lg=np.radians(lg); dip=b[0]+b[1]*np.sin(Lg)+b[2]*np.cos(Lg)
fig,ax=plt.subplots(2,1,figsize=(9,6.6),sharex=True,gridspec_kw=dict(height_ratios=[3,1.3],hspace=.08))
ax[0].errorbar(l,y,yerr=e,fmt='o',ms=3.5,color='0.25',ecolor='0.75',lw=1,label='Migkas et al. $A/A_{\\rm all}$ (71 bins)')
ax[0].axhline(np.sum(w*y)/np.sum(w),color='0.5',ls=':',lw=1.5,label=f'isotropy (constant): $\\chi^2$ = {cI:.1f}')
ax[0].plot(lg,dip,color='#2b6cb0',lw=2,ls='--',label=f'best-fit dipole (one high, one low): $\\chi^2$ = {cD:.1f}')
ax[0].plot(TH1,s*Amod,color='#c0392b',lw=2.4,label=f'CoE geometry (two highs, two lows): $\\chi^2$ = {cC:.1f}')
ax[0].set_ylabel('$A/A_{\\rm all}$'); ax[0].legend(fontsize=8.5,loc='lower left'); ax[0].grid(alpha=.3); ax[0].set_ylim(0.78,1.2)
ax[0].set_title(f'CoE geometry vs dipole: $\\Delta\\chi^2$ = {dch:.1f} for 4 extra parameters, $p$ = {p:.1e} ({sg:.1f}$\\sigma$)',fontsize=10)
ax[1].bar(l-1.2,100*(y-np.interp(l,lg,dip)),width=2.4,color='#2b6cb0',alpha=.7,label='data − dipole')
ax[1].bar(l+1.2,100*(y-s*fA),width=2.4,color='#c0392b',alpha=.8,label='data − CoE')
ax[1].axhline(0,color='k',lw=.8); ax[1].set_ylabel('residual (%)'); ax[1].legend(fontsize=8,ncol=2,loc='lower right'); ax[1].grid(alpha=.3)
ax[1].set_xlabel('Galactic longitude ℓ (deg)'); ax[1].set_xlim(0,360); ax[1].set_xticks(range(0,361,45))
fig.savefig(PKG + '/coe_review/14_paper/figs/fig_dipole_cmp.png',dpi=180,bbox_inches='tight')
print(dict(chi2_iso=cI,chi2_dip=cD,chi2_coe=cC,dchi2=dch,p=p,sigma=sg))
