import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import json, numpy as np, healpy as hp
from scipy.special import sph_harm_y
from scipy.optimize import minimize
d=json.load(open(PKG + '/data/smica_I_nside64_fwhm3deg_uK.json'))
T=np.array(d['T_uK']); ns=d['nside']; th,ph=hp.pix2ang(ns,np.arange(len(T))); b=90-np.degrees(th)
def v(l,b_): l,b_=np.radians(l),np.radians(b_); return np.array([np.cos(b_)*np.cos(l),np.cos(b_)*np.sin(l),np.sin(b_)])
def lb(x): x=x/np.linalg.norm(x); return np.degrees(np.arctan2(x[1],x[0]))%360, np.degrees(np.arcsin(x[2]))
ang=lambda a,c: np.degrees(np.arccos(np.clip(a@c/np.linalg.norm(a)/np.linalg.norm(c),-1,1)))
LF=10
def realY(l,m,t,p):
    y=sph_harm_y(l,abs(m),t,p)
    return np.sqrt(2)*(y.real if m>0 else y.imag) if m!=0 else y.real
CoE=v(180,-15); P12=v(245.7,56.93); AoE=v(260,60)
NS=16; xp=np.array(hp.pix2vec(NS,np.arange(hp.nside2npix(NS)))).T; th_,ph_=hp.vec2ang(xp); w=4*np.pi/len(xp)
dirs=np.array(hp.pix2vec(16,np.arange(hp.nside2npix(16)))).T; dirs=dirs[dirs[:,2]>=0]
def run(cut,lmax_fit=LF):
    sel=np.abs(b)>cut
    cols=[(l,m) for l in range(lmax_fit+1) for m in range(-l,l+1)]
    X=np.column_stack([realY(l,m,th[sel],ph[sel]) for l,m in cols])
    c=np.linalg.lstsq(X,T[sel],rcond=None)[0]
    coef={lm:c[i] for i,lm in enumerate(cols)}
    def Tl(l,g):  # l-only field at Galactic unit vectors g
        t,p=hp.vec2ang(g); return sum(coef[(l,m)]*realY(l,m,t,p) for m in range(-l,l+1))
    def disp(ls,z):
        z=z/np.linalg.norm(z); x0=np.array([0,0,1.]) if abs(z[2])<.9 else np.array([1.,0,0]); x=x0-(x0@z)*z; x/=np.linalg.norm(x); R=np.array([x,np.cross(z,x),z])
        g=xp@R; s=0
        for l in ls:
            vals=Tl(l,g); a={m:np.sum(vals*np.conj(sph_harm_y(l,m,th_,ph_)))*w for m in range(-l,l+1)}
            s+=sum(m*m*abs(a[m])**2 for m in a)/sum(abs(a[m])**2 for m in a)
        return s
    out={}
    for key,ls in (('l2',(2,)),('l3',(3,)),('l23',(2,3))):
        s=[disp(ls,n) for n in dirs]; n=dirs[int(np.argmax(s))]
        r=minimize(lambda p:-disp(ls,v(*p)),lb(n),method='Nelder-Mead',options=dict(xatol=.05,fatol=1e-6)); n=v(*r.x)
        out[key]=dict(axis=[round(x,1) for x in lb(n)],to_P12=round(min(ang(n,P12),180-ang(n,P12)),1),to_AoE=round(min(ang(n,AoE),180-ang(n,AoE)),1),CoE_off_plane=round(abs(90-ang(n,CoE)),1))
    return out
R={}
for cut in (0,10,20,30):
    R[cut]=run(cut); print('|b| >',cut,R[cut])
json.dump(R,open('masked.json','w'),indent=1)
