import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root
# P3 blind test (pre-registered in blind_predictions.md): local Hubble-rate anisotropy in Pantheon+ supernovae.
# Template (no free direction, no free amplitude): build C* exact light-cone delta(psi,z) = H_eff(psi,z)/<H_eff>(z) - 1,
# psi measured from the pre-registered anti-CoE axis (0, +15). Distance-modulus shift: dmu = -(5/ln10) * delta.
# Sample: Hubble-flow SNe 0.0233 < z < 0.15 (no calibrators); full STAT+SYS covariance; M (absolute magnitude) marginalised.
# Reported: chi2 improvement of the fixed template over isotropy, best-fit amplitude beta (prediction beta = 1), and a null built
# by placing the same template on 20000 random axes (how often a random axis fits as well).
import os,json,numpy as np
from astropy.coordinates import SkyCoord;import astropy.units as u
from scipy.interpolate import RegularGridInterpolator as RGI
D=PKG + '/coe_review/data/pantheon/';ZCOL=os.environ.get('ZCOL','zHD')
names=open(D+'Pantheon+SH0ES.dat').readline().split();dat=np.genfromtxt(D+'Pantheon+SH0ES.dat',names=True,dtype=None,encoding=None)
C=np.loadtxt(D+'Pantheon+SH0ES_STAT+SYS.cov',skiprows=1).reshape(1701,1701)
z=dat[ZCOL].astype(float);sel=(z>0.0233)&(z<0.15)&(dat['IS_CALIBRATOR']==0);idx=np.where(sel)[0]
z=z[idx];m=dat['m_b_corr'][idx].astype(float);Cs=C[np.ix_(idx,idx)];Ci=np.linalg.inv(Cs)
g=SkyCoord(ra=dat['RA'][idx]*u.deg,dec=dat['DEC'][idx]*u.deg).galactic;l,b=g.l.deg,g.b.deg
def vec(l,b):l,b=np.radians(l),np.radians(b);return np.stack([np.cos(b)*np.cos(l),np.cos(b)*np.sin(l),np.sin(b)],-1)
n=vec(l,b)
# fiducial distance modulus shape (flat, Om 0.3; only its z-shape matters, H0 and M are degenerate and marginalised)
from scipy.integrate import quad
dl=np.array([(1+zz)*quad(lambda x:1/np.sqrt(0.3*(1+x)**3+0.7),0,zz)[0] for zz in z]);mu0=5*np.log10(dl)
r0=m-mu0
def chi2(res):                       # marginalise the constant M analytically
    o=np.ones_like(res);a=(o@Ci@res)/(o@Ci@o);rr=res-a;return float(rr@Ci@rr)
def template(build):
    J=json.load(open(PKG + '/coe_review/rerun/fixed/res/'+build+'.json'))['lowz'];PS=np.array(J['PS']);ZS=np.array(J['ZS']);H=np.array(J['H'])
    w=np.sin(np.radians(PS));Hb=(w[:,None]*H).sum(0)/w.sum();dlt=H/Hb[None,:]-1
    return RGI((PS,ZS),dlt,bounds_error=False,fill_value=None)
def dmu(f,axis):
    psi=np.degrees(np.arccos(np.clip(n@axis,-1,1)));psi=np.clip(psi,2.5,177.5);return -5/np.log(10)*f(np.c_[psi,np.clip(z,0.003,0.46)])
c_iso=chi2(r0);print(f'{ZCOL}: N = {len(z)} SNe (0.0233 < z < 0.15, no calibrators); chi2 isotropic = {c_iso:.1f}')
rng=np.random.default_rng(3);NR=int(os.environ.get('NR','20000'));rand=rng.normal(size=(NR,3));rand/=np.linalg.norm(rand,axis=1)[:,None]
o=np.ones_like(r0)
def fitbeta(t):                      # best amplitude (with M marginalised)
    P=Ci-np.outer(Ci@o,o@Ci)/(o@Ci@o);be=(t@P@r0)/(t@P@t);return be,1/np.sqrt(t@P@t)
for build,lab in (('s1_U1_F0.2_P0.9_X0.6','C*'),('s1_U1_F0.3_P0.8_X0.8','A*')):
    f=template(build)
    for axname,ax in (('pre-registered frame axis (0,+15)',vec(0,15)),('exact Migkas-fit axis (17,-7)',vec(17,-7))):
        t=dmu(f,ax);c1=chi2(r0-t);be,se=fitbeta(t)
        nul=np.array([chi2(r0-dmu(f,a)) for a in rand]);p=np.mean(nul<=c1)
        print(f'  {lab} template, {axname}: dchi2(beta=1) = {c1-c_iso:+.1f}; best beta = {be:+.2f} +- {se:.2f}; fraction of {NR} random axes doing as well: {p:.3f}',flush=True)
# context: free dipole + quadrupole in delta (8 params) best direction
X=np.column_stack([n,np.stack([n[:,0]*n[:,1],n[:,0]*n[:,2],n[:,1]*n[:,2],n[:,0]**2-n[:,1]**2,3*n[:,2]**2-1],1)])*(-5/np.log(10))
P=Ci-np.outer(Ci@o,o@Ci)/(o@Ci@o);cf=np.linalg.solve(X.T@P@X,X.T@P@r0);dv=cf[:3];dd=np.linalg.norm(dv)
lg=np.degrees(np.arctan2(dv[1],dv[0]))%360;bg=np.degrees(np.arcsin(dv[2]/dd))
print(f'  context: free dipole+quadrupole fit: dchi2 = {chi2(r0-X@cf)-c_iso:+.1f} (8 params); dipole |delta| = {dd:.3f} toward (l,b)=({lg:.0f},{bg:+.0f})')
