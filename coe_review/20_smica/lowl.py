"""Low-l axes and in-plane orientation of Planck SMICA (user upload, nside 64, 3 deg, uK)."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import json, numpy as np, healpy as hp
from scipy.special import sph_harm_y
d=json.load(open(PKG + '/data/smica_I_nside64_fwhm3deg_uK.json'))
T=np.array(d['T_uK']); ns=d['nside']
LMAX=6; alm=hp.map2alm(T,lmax=LMAX,iter=5)
cl=hp.alm2cl(alm); print('C_l (uK^2) l=2..6:',np.round(cl[2:],1))
def v(l,b): l,b=np.radians(l),np.radians(b); return np.array([np.cos(b)*np.cos(l),np.cos(b)*np.sin(l),np.sin(b)])
def lb(x): x=x/np.linalg.norm(x); return np.degrees(np.arctan2(x[1],x[0]))%360, np.degrees(np.arcsin(x[2]))
ang=lambda a,b: np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1)))
# l-only maps evaluated by sampling on an nside-16 grid in a rotated frame
NS=16; xp=np.array(hp.pix2vec(NS,np.arange(hp.nside2npix(NS)))).T; th_,ph_=hp.vec2ang(xp)
Tl={l:None for l in (2,3)}
def lmap(l):
    a=np.zeros_like(alm)
    for m in range(l+1): a[hp.Alm.getidx(LMAX,l,m)]=alm[hp.Alm.getidx(LMAX,l,m)]
    return a
A={l:lmap(l) for l in (2,3)}
def frame(z,x0=None):
    z=z/np.linalg.norm(z)
    if x0 is None: x0=np.array([0,0,1.]) if abs(z[2])<.9 else np.array([1.,0,0])
    x=x0-(x0@z)*z; x/=np.linalg.norm(x); y=np.cross(z,x); return np.array([x,y,z])   # rows: frame axes in Galactic
def alm_in(l,R):
    g=xp@R      # Galactic coords of frame-grid points
    th,ph=hp.vec2ang(g); vals=hp.alm2map(A[l],ns,lmax=LMAX,verbose=False) if False else None
    t=np.array([hp.sphtfunc.alm2map(A[l],ns,lmax=LMAX)[0]]) if False else None
    vals=hp.get_interp_val(Lmap[l],th,ph)
    w=4*np.pi/len(xp)
    return {m:np.sum(vals*np.conj(sph_harm_y(l,m,th_,ph_)))*w for m in range(-l,l+1)}
Lmap={l:hp.alm2map(A[l],ns,lmax=LMAX) for l in (2,3)}
def disp(l,z): a=alm_in(l,frame(z)); return sum(m*m*abs(a[m])**2 for m in a)/sum(abs(a[m])**2 for m in a)
dirs=np.array(hp.pix2vec(16,np.arange(hp.nside2npix(16)))).T; dirs=dirs[dirs[:,2]>=0]
res={}
for key,ls in (('l2',(2,)),('l3',(3,)),('l23',(2,3))):
    s=np.array([sum(disp(l,n)*1 for l in ls) for n in dirs]); n=dirs[np.argmax(s)]
    # refine
    from scipy.optimize import minimize
    f=lambda p: -sum(disp(l,v(p[0],p[1])) for l in ls); r=minimize(f,lb(n),method='Nelder-Mead',options=dict(xatol=.05,fatol=1e-6))
    res[key]=v(*r.x); print(key,'axis (l,b)=',np.round(lb(res[key]),1),' dispersion',round(-r.fun,3))
AoE=v(260,60); P12=v(245.7,56.93); CoE=v(180,-15); PoEn=v(98.44,28.71); CoE2=v(198,-4)
for k,n in res.items():
    a=lambda x: min(ang(n,x),180-ang(n,x))
    print(f'{k}: to published AoE {a(AoE):.1f}, to P1-P2 axis {a(P12):.1f}, CoE off its plane {abs(90-ang(n,CoE)):.1f}, (198,-4) off plane {abs(90-ang(n,CoE2)):.1f}, to plane-of-evil normal {a(PoEn):.1f}')
json.dump({k:list(lb(n)) for k,n in res.items()},open('axes.json','w'))
# in-plane orientation: frame z = axis, x = CoE projected into the plane
out={}
for zname,z in (('P1-P2 axis',P12),('SMICA l2+l3 axis',res['l23']),('published AoE',AoE)):
    R=frame(z,CoE)
    for l in (2,3):
        a=alm_in(l,R)
        # azimuth of hot lobes of the m=l sectoral part: Re(a_ll e^{i l phi}) max
        phl=(-np.angle(a[l])/l)%(360/l*np.pi/180); phl=np.degrees(phl)
        frac=2*abs(a[l])**2/sum(abs(a[m])**2 for m in a)
        lob=np.arange(l)*360/l+phl; hot=[(x+180)%360-180 for x in lob]; cold=[((x+180/l)+180)%360-180 for x in lob]
        dh=min(abs(x) for x in hot); dc=min(abs(x) for x in cold); dcoe=min(dh,dc); half=180/(2*l)
        print(f'[{zname}] l={l}: sectoral power fraction {frac:.2f}; hot lobes at {np.round(sorted(hot),1)} deg, cold at {np.round(sorted(cold),1)} deg from CoE direction; nearest extremum {dcoe:.1f} deg (range 0-{half:.0f}, chance P={dcoe/half:.2f}); CoE sits on {"hot" if dh<dc else "cold"}')
