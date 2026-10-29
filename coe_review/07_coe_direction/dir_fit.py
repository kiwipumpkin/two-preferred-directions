import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
# CoE direction from Migkas cones, exact 313-cluster OLS estimator with the digitised-bin covariance (Team A mfit, 'D full C').
# Model: the letter's Newtonian CoE at the AoE-fixed D (axisymmetric af(psi), psi from anti-CoE), A/A_all = af (linear, as in the
# letter's fit), i.e. H_eff = sqrt(af) in mfit's 2 log10 H convention. Scan the axis over HEALPix nside 32; chi2 map -> best CoE and
# Delta chi2 regions. Also a free-amplitude version (beta multiplies log A).
import sys,json,numpy as np,healpy as hp
sys.path.insert(0,PKG + '/coe_review/rerun/teamA');import mfit
d=json.load(open('af_D51.json'));af=np.array(d['af']);PS=np.arange(0.5,180,1.0);h=np.sqrt(np.interp(PS,np.arange(360),af))
J=dict(PS=PS.tolist(),ZS=[0.003,0.46],H=np.c_[h,h].tolist())
NS=32;pl,pb=hp.pix2ang(NS,np.arange(hp.nside2npix(NS)),lonlat=True);pv=mfit.cm.lb2vec(pl,pb)
chi=np.empty(len(pv));chib=np.empty(len(pv));beta=np.empty(len(pv))
Ad=mfit.Ad;Ci=mfit.Ci;W=Ci*np.outer(Ad*np.log(10),Ad*np.log(10));Y=np.log10(Ad)
for k,v in enumerate(pv):
    m,g=mfit.curve(J,v);chi[k]=mfit.chi(Ad-m)
    b=(g@W@Y)/(g@W@g);beta[k]=b;chib[k]=mfit.chi(Ad-10**(b*g))
k=np.argmin(chi);kb=np.argmin(chib)
anti=lambda i:(float(pl[i]),float(pb[i]));coe=lambda i:(float((pl[i]+180)%360),float(-pb[i]))
print('isotropy chi2 %.2f'%mfit.c_iso)
print('fixed amplitude: chi2_min %.2f, anti-CoE %s, CoE %s'%(chi[k],anti(k),coe(k)))
print('free amplitude : chi2_min %.2f (beta %.2f), anti-CoE %s, CoE %s'%(chib[kb],beta[kb],anti(kb),coe(kb)))
for name,c in (('fixed',chi),('free',chib)):
    for dc in (1.0,2.3,6.18):
        s=c<=c.min()+dc;print('  %s amp: Delta chi2 <= %.2f region: %d pixels, area %.0f deg2, equiv radius %.1f deg'%(name,dc,s.sum(),s.sum()*hp.nside2pixarea(NS,degrees=True),np.degrees(np.sqrt(s.sum()*hp.nside2pixarea(NS)/np.pi))))
np.savez('dir_fit.npz',pl=pl,pb=pb,chi=chi,chib=chib,beta=beta,iso=mfit.c_iso)
