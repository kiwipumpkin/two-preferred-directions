import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root
# One fixed-setup CoE build -> (1) low-z H(psi,z) and the Migkas cone fit (Team A's mfit, exact estimator, full covariance),
# (2) full-sky CMB: every Gauss-Legendre direction traced to its recombination shell (grid extended as needed; any failure is an
# error, not a gap), Legendre multipoles of dT/T, D_A at psi = 90, theta* from CAMB (local and emitting-shell omega_m), age, D.
# usage: P='{"FRAC":0.3,"U":4,"PMF":0.5,"XC":0.2}' TAG=name [NG=16] [CMB=1] python3 run.py
import os,sys,json,time,numpy as np;sys.path.insert(0,PKG + '/coe_review/rerun/fixed');import coe_ltb as C
from numpy.polynomial import legendre as LG
from scipy.integrate import solve_ivp,quad
t_=time.time();P=json.loads(os.environ['P']);P['FEAT']=[tuple(f) for f in P.get('FEAT',[])];TAG=os.environ.get('TAG','run');NG=int(os.environ.get('NG','16'))
OUT=PKG + '/coe_review/rerun/fixed/res/';os.makedirs(OUT,exist_ok=True);out=dict(P=P,TAG=TAG)
def save():json.dump(out,open(OUT+TAG+'.json','w'),indent=1,default=float)
try:M=C.build(P)
except C.InvalidBuild as e:out['invalid']=str(e);save();print('INVALID',TAG,e);sys.exit(0)
xs=np.geomspace(0.01,min(M['XMAX'],40.),120);trec=__import__('scipy.optimize',fromlist=['brentq']).brentq(lambda t:M['S'](t,1.)[0,0]-M['R1']/1100.,M['T'][0]*1.001,M['t0'])
dp=M['density_profile'](xs,[M['t0'],trec]);out['density']=dict(x=xs.tolist(),rho_today_rel=(dp[0]/M['rho'](M['t0'],1.)).tolist(),rho_rec_rel=(dp[1]/M['rho'](trec,1.)).tolist())
out.update(Omega_local=M['Om'],Hr_over_Ht=M['Hr']/M['Ht'],q_min=M['q_min'],x_qmin=M['x_qmin'],t0=M['t0'])
# low-z + Migkas fit
J=C.lowz(M);out['lowz']=J;Hm=np.array(J['H'])
w=np.sin(np.radians(J['PS']));Hbar=(w[:,None]*Hm).sum(0)/w.sum()       # sky-mean H_eff by z
Dg=C.c_kms*Hbar[4]/70/1e3                                               # D [Gpc] pinned: sky-mean H_eff(z=0.05) = 70
out.update(D_Gpc=Dg,age_Gyr=M['t0']*Dg*3.26156,H_t_kms=M['Ht']*C.c_kms/(Dg*1e3),H_r_kms=M['Hr']*C.c_kms/(Dg*1e3),omega_m_local=M['Om']*(M['Ht']*C.c_kms/(Dg*1e3)/100)**2)
sys.path.insert(0,PKG + '/coe_review/rerun/teamA');import mfit
r=mfit.fit(J);out['migkas']={k:r[k] for k in ('chi2_full','chi2_iso','l0','b0','anti_l0','anti_b0','p2p','chi2_diag','chi2_S_min','chi2_iso_S','beta','chi2_beta')}
print(f"{TAG}: Omega_local={M['Om']:.3f} Hr/Ht={M['Hr']/M['Ht']:.3f} D={Dg:.2f} Gpc age={out['age_Gyr']:.2f} Gyr | Migkas chi2={r['chi2_full']:.1f} (iso {r['chi2_iso']:.1f}) axis=({r['l0']:.0f},{r['b0']:.0f}) p2p={r['p2p']:.3f}  [{time.time()-t_:.0f}s]",flush=True)
save()
if os.environ.get('CMB','1')=='1':
    mu,wt=LG.leggauss(NG);mu=mu[::-1];wt=wt[::-1];PS=np.degrees(np.arccos(mu))
    try:M,res=C.build_full_sky(P,np.r_[PS,90.,90.02])
    except C.RayFailure as e:out['cmb_error']=str(e);save();print('RAYFAIL',TAG,e);sys.exit(1)
    Z=np.array([-y[3] for ps,s,y in res[:NG]]);T=1/Z;T0=(wt*T).sum()/2;dT=T/T0-1
    al=[(2*l+1)/2*(wt*dT*LG.legval(mu,[0]*l+[1])).sum() for l in range(min(11,NG))]
    s1=res[NG][1];s2=res[NG+1][1];DA=C.DA_pair(M,s1,s2,90.,0.02,s1.t_events[0][0]);opz90=-res[NG][2][3];xe=res[NG][2][1]
    rhoe=M['rho'](res[NG][2][0],xe);rho_us_rec=M['rho'](res[NG][2][0],xe)  # emitting-shell density at emission
    om_emit=out['omega_m_local']*float(M['rho'](M['t0'],xe)/M['rho'](M['t0'],1.))
    out['cmb']=dict(XMAX=M['XMAX'],PS=PS.tolist(),opz=Z.tolist(),x_emit=[float(y[1]) for ps,s,y in res[:NG]],a_l=al,opz_90=opz90,x_emit_90=float(xe),
                    DA_phys_Mpc=DA*Dg*1e3,omega_m_emit=om_emit)
    import camb
    def theta(om):
        oc=max(om-0.0224,1e-5);p=camb.CAMBparams();p.set_cosmology(H0=100*np.sqrt(0.0224+oc),ombh2=0.0224,omch2=oc,omk=0,mnu=0,tau=0.054)
        if om>3:p.Reion.Reionization=False
        r=camb.get_background(p);d=r.get_derived_params();return 100*d['rstar']/(1+d['zstar'])/(DA*Dg*1e3)
    try:out['cmb'].update(theta100_local=theta(out['omega_m_local']),theta100_emit=theta(om_emit))
    except Exception as e:out['cmb']['theta_error']=str(e)
    save()
    print(f"{TAG}: CMB all {NG} directions recombine (XMAX={M['XMAX']:g}); 1+z* {Z.min():.1f}-{Z.max():.1f}; a1..a4 = "+' '.join('%.2e'%a for a in al[1:5])+f"; D_A,phys={DA*Dg*1e3:.2f} Mpc; 100theta*={out['cmb'].get('theta100_local',float('nan')):.3f}/{out['cmb'].get('theta100_emit',float('nan')):.3f}; omega_m local/emit={out['omega_m_local']:.4f}/{om_emit:.4f}  [{time.time()-t_:.0f}s]",flush=True)
