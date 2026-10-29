import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root
# Forward model of Migkas Fig.6 (90-deg all-latitude longitude cones, repo-style OLS L-T fit, 313 actual clusters):
# each cluster's inferred log10 L shifts by 2 log10 H_eff(psi_i, z_i) (fiducial D_L ~ 1/H0); cone log10(A/A_all) = K . delta
# with K the exact OLS-intercept kernel (cone minus full sample). Axis orientation scanned over the sphere (HEALPix nside 16).
# chi2 vs digitised curve with C = R_boot * sig_dig sig_dig ('D full C'), plus diag chi2, plus best free amplitude beta.
import os
os.environ['OMP_NUM_THREADS']='1'
import sys,json,glob,numpy as np,healpy as hp
from scipy.interpolate import RegularGridInterpolator as RGI
sys.path.insert(0,PKG + '/coe_review/hmap');import common as cm
T=PKG + '/coe_review/rerun/teamA/';d=np.load(T+'migkas_data.npz')
ld,Ad,sd,Cd,z,l,b,x=d['ld'],d['Ad'],d['sd'],d['Cd'],d['z'],d['l'],d['b'],d['x'];Mem=d['M'].astype(float)
Ci=np.linalg.inv(Cd);A0,Cb=d['A0'],d['Cb'];CiS=np.linalg.inv(Cb);n=cm.lb2vec(l,b)
K=cm.ols_kernel(dict(x=x),Mem)                 # (71,313)
NS=16;pl,pb=hp.pix2ang(NS,np.arange(hp.nside2npix(NS)),lonlat=True);pv=cm.lb2vec(pl,pb)
chi=lambda r:float(r@Ci@r)
one=np.ones(71);c_iso=chi(Ad-1);c_iso_diag=float(np.sum(((Ad-1)/sd)**2));c_iso_S=float((A0-1)@CiS@(A0-1));cfree=(one@Ci@Ad)/(one@Ci@one);c_iso_free=chi(Ad-cfree)
def curve(J,axis_vec):
    PS=np.array(J['PS']);ZS=np.array(J['ZS']);H=np.array(J['H']);H[H<0]=np.nan
    lnH=np.log(H);lnH=np.where(np.isfinite(lnH),lnH,np.nanmean(lnH,0)[None,:])
    f=RGI((PS,ZS),lnH,bounds_error=False,fill_value=None)
    psi=np.degrees(np.arccos(np.clip(n@axis_vec,-1,1)));psi=np.clip(psi,PS[0],PS[-1])
    dl=2*f(np.c_[psi,np.clip(z,ZS[0],ZS[-1])])/np.log(10)   # delta log10 L
    return 10**(K@dl),K@dl
def fit(J):
    best=None;allc=np.empty(len(pv));cS=np.empty(len(pv));cD=np.empty(len(pv))
    for k,v in enumerate(pv):
        m,_=curve(J,v);c=chi(Ad-m);allc[k]=c;cS[k]=(A0-m)@CiS@(A0-m);cD[k]=np.sum(((Ad-m)/sd)**2)
        if best is None or c<best[0]:best=(c,k)
    c,k=best;m,lg=curve(J,pv[k])
    # free amplitude on log A (how much larger/smaller the anisotropy would need to be), at the best orientation for free beta
    bb=None;bp=None;bn=None
    for kk,v in enumerate(pv):
        _,g=curve(J,v);Y=np.log10(Ad)
        # minimise (Y-beta g)^T Ci' (Y-beta g) in log space with Ci in linear approx (dA ~ A ln10 dlog)
        W=Ci*np.outer(Ad*np.log(10),Ad*np.log(10));beta=(g@W@Y)/(g@W@g);cc=chi(Ad-10**(beta*g))
        if bb is None or cc<bb[0]:bb=(cc,beta,kk)
        if beta>0 and (bp is None or cc<bp[0]):bp=(cc,beta,kk)
        if beta<0 and (bn is None or cc<bn[0]):bn=(cc,beta,kk)
    return dict(chi2_full=c,l0=float(pl[k]),b0=float(pb[k]),anti_l0=float((pl[k]+180)%360),anti_b0=float(-pb[k]),
                chi2_diag=float(np.sum(((Ad-m)/sd)**2)),model_curve=m.tolist(),p2p=float(m.max()-m.min()),
                chi2_beta=float(bb[0]),chi2_bpos=float(bp[0]) if bp else None,beta_pos=float(bp[1]) if bp else None,l0_bpos=float(pl[bp[2]]) if bp else None,b0_bpos=float(pb[bp[2]]) if bp else None,chi2_bneg=float(bn[0]) if bn else None,beta_neg=float(bn[1]) if bn else None,l0_bneg=float(pl[bn[2]]) if bn else None,b0_bneg=float(pb[bn[2]]) if bn else None,beta=float(bb[1]),l0_beta=float(pl[bb[2]]),b0_beta=float(pb[bb[2]]),
                chi2_iso=c_iso,chi2_iso_diag=c_iso_diag,chi2_S_min=float(cS.min()),chi2_iso_S=c_iso_S,l0_S=float(pl[cS.argmin()]),b0_S=float(pb[cS.argmin()]),chi2_diag_min=float(cD.min()),l0_diag=float(pl[cD.argmin()]),b0_diag=float(pb[cD.argmin()]),chi2_iso_cfree=c_iso_free,chi2_map_min=float(allc.min()),chi2_map_median=float(np.median(allc)))
if __name__=='__main__':
    fs=sys.argv[1:] or sorted(glob.glob(T+'runs/lc_*.json'));res={}
    if os.path.exists(T+'mfit.json'):res=json.load(open(T+'mfit.json'))
    for f in fs:
        key=os.path.basename(f)[3:-5]
        if key in res and os.path.getmtime(f)<res[key].get('_mtime',0):continue
        J=json.load(open(f));r=fit(J);r['_mtime']=os.path.getmtime(f)+1;res[key]=r
        print(f"{key:40s} chi2={r['chi2_full']:7.1f} (iso {c_iso:.1f}, iso c-free {c_iso_free:.1f}) diag(best-orient)={r['chi2_diag_min']:6.1f}/iso {c_iso_diag:.1f} S={r['chi2_S_min']:6.1f}/iso {c_iso_S:.1f} p2p={r['p2p']:.3f} axis(CoE)=({r['l0']:.0f},{r['b0']:.0f}) | free amp beta={r['beta']:.2f} chi2={r['chi2_beta']:.1f} | beta>0: {r['beta_pos']} chi2 {r['chi2_bpos']}",flush=True)
    json.dump(res,open(T+'mfit.json','w'),indent=1)
