# Continuous-plasma CoE CMB: exact LTB (point core mass + test plasma), common bang, null geodesics off-centre.
import json,sys,numpy as np
from scipy.integrate import solve_ivp,quad
from scipy.interpolate import RectBivariateSpline
f=0.015426145753569655; uD=10**0.021927885054513485   # point-mass best fit; units D=1,c=G=1
def setup(M):
    E0=0.5*uD**2-M; wo=np.sqrt(2*E0)
    t0=quad(lambda r:1/np.sqrt(2*M/r+2*E0),0,1)[0] if M>0 else 1/uD
    return wo,t0
def Rfun(tau,w,M):
    E=np.sign(w)*w*w/2
    if M==0: return np.abs(w)*tau
    if abs(E)<1e-9: return (4.5*M)**(1/3)*tau**(2/3)
    a=M/(2*abs(E)); b=a/np.sqrt(2*abs(E)); x=tau/b
    if E>0:
        eta=np.cbrt(6*x) if x<1 else np.log(2*x+1)+1
        for _ in range(60): eta-= (np.sinh(eta)-eta-x)/(np.cosh(eta)-1+1e-300)
        return a*(np.cosh(eta)-1)
    if x>=2*np.pi: return 0.0
    lo,hi=0.,2*np.pi
    for _ in range(80):
        m=(lo+hi)/2
        if m-np.sin(m)<x: lo=m
        else: hi=m
    return a*(1-np.cos(lo))
def build(M,t0,wmin,wmax=25):
    T=np.linspace(2e-3,t0,500); W=np.r_[np.linspace(wmin,1.5,500),np.linspace(1.5,wmax,200)[1:]]
    G=np.array([[Rfun(t,w,M) for w in W] for t in T]); return RectBivariateSpline(T,W,G,kx=5,ky=5),T,W
def trace(M,Rrec,psis):
    wo,t0=setup(M); wmin=-np.sqrt(2*M/Rrec)*0.98 if M>0 else 1e-3
    S,T,W=build(M,t0,wmin); E=lambda w:np.sign(w)*w*w/2; Ep=lambda w:np.abs(w)
    def AB(t,w):
        R=S(t,w)[0,0];Rp=S(t,w,dy=1)[0,0];Rt=S(t,w,dx=1)[0,0];Rpp=S(t,w,dy=2)[0,0];Rtp=S(t,w,dx=1,dy=1)[0,0];q=1+2*E(w)
        A=Rp**2/q; At=2*Rp*Rtp/q; Aw=2*Rp*Rpp/q-Rp**2*2*Ep(w)/q**2
        return R,A,At,Aw,R*R,2*R*Rt,2*R*Rp
    def rhs(l,y):
        t,w,ph,pt,pw,pp=y; R,A,At,Aw,B,Bt,Bw=AB(t,w)
        return [-pt,pw/A,pp/B,0.5*At/A**2*pw**2+0.5*Bt/B**2*pp**2,0.5*Aw/A**2*pw**2+0.5*Bw/B**2*pp**2,0]
    hit=lambda l,y:AB(y[0],y[1])[0]-Rrec; hit.terminal=True
    edge=lambda l,y:min(y[0]-T[0]-1e-3,y[1]-W[0]-1e-3,W[-1]-y[1]-1e-2); edge.terminal=True
    out=[]
    _,A0,*_=AB(t0,wo); B0=1.0
    for ps in psis:
        p=np.radians(ps); y0=[t0,wo,0.,-1.,-np.sqrt(A0)*np.cos(p),-np.sqrt(B0)*np.sin(p)]
        s=solve_ivp(rhs,(0,-50),[y0[0],y0[1],y0[2],y0[3],-y0[4]*0-np.sqrt(A0)*np.cos(p)*1,y0[5]],events=[hit,edge],rtol=1e-9,atol=1e-11)
        if s.t_events[0].size:
            y=s.y_events[0][0]; out.append((-y[3],y[0],t0-y[0],y[1]))   # 1+z, emission proper time, time before now, element label
        else: out.append((np.nan,)*4)
    return np.array(out),wo,t0
if __name__=='__main__':
    M=float(sys.argv[1]); Rrec=float(sys.argv[2]); psis=np.arange(90,180.5,1.0)
    r,wo,t0=trace(M,Rrec,psis); np.save(f'h_M{M:.5f}_R{Rrec}.npy',np.c_[psis,r]); print(wo,t0,np.isfinite(r[:,0]).sum(),flush=True)
