import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root
# Fixed CoE LTB toolkit (replaces the teamA/teamB tracers for all new runs).
#
# Model: exact Lemaitre-Tolman-Bondi dust, common bang (all shells start at R = 0 at t = 0), shells labelled x = today's
# areal radius in units of D (our distance to the CoE). Units D = 1, c = G = 1; our outward speed today is u.
#   metric   ds^2 = -dt^2 + R'^2/(1+2E(x)) dx^2 + R^2 dOmega^2           (exact LTB; base.py v11 form)
#   dynamics R_t^2 = 2M(x)/R + 2E(x),  E(x) fixed by equal age t0 (common bang)
# Fixes relative to the earlier runs:
#   1. No matter cutoff. The author's profile shape0 already clamps itself to a constant beyond x = Rb (min(|x|, Rb));
#      the earlier code multiplied it by (x <= Rb), i.e. set the density to zero there. Matter now fills the whole grid.
#   2. Outer grid extended automatically until every CMB ray reaches its recombination shell.
#   3. The CoE concentration is an EXTENDED core (Gaussian, radius XC), not a point mass. A build is accepted only if it is a
#      regular GR solution everywhere: every shell still outgoing today (no collapsed shells), 1+2E > 0, no shell crossing.
#      Invalid builds raise InvalidBuild; nothing is silently dropped.
#   4. Every CMB ray must reach recombination; otherwise RayFailure is raised.
# Recombination: each shell recombines when its local density falls to rho_c = density of our shell at R = R(t0)/1100
# (author's criterion, cmbq.py / ltb.py).
import os,sys,numpy as np
from scipy.integrate import solve_ivp,cumulative_trapezoid,quad
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline,make_interp_spline,interp1d
BD=PKG + '/phase1C_v11/'
sys.path.insert(0,BD);import base as V
c_kms=2.998e5
DR=42/14.30048287260589;Rb=1.5*(1+1/DR);CC=[-0.041,-0.035,0.1353,0.0]
class InvalidBuild(Exception):pass
class RayFailure(Exception):pass
def shape0(x):return np.exp(sum(c*(np.minimum(np.abs(x),Rb)-1)**(k+1) for k,c in enumerate(CC)))   # author's profile (clamped beyond Rb)
def age_out(r,M,E):
    if M==0:return r/np.sqrt(2*E)
    return quad(lambda y:1/np.sqrt(max(2*M/y+2*E,1e-300)),0,r,limit=200)[0]

def build(p,XMAX=40.,nT=500):
    """p: dict(FRAC, U, PMF, XC, FA, FL, FEAT=[(A,x0,w),...], UNIFORM). FRAC: Mtot = FRAC u^2/2 = M(<1)."""
    u=p['U'];FRAC=p['FRAC'];PMF=p.get('PMF',0.);XC=p.get('XC',0.2)
    def shape(x):
        if p.get('UNIFORM'):return np.ones_like(x)
        # author's profile inside x <= Rb; beyond it the density keeps varying: rho ~ x^-TAILN (TAILN = 2: equal mass per shell,
        # the author's 'same amount of mass per shell' option), continuous at Rb. TAILN = 0 reproduces the old constant clamp.
        n=p.get('TAILN',2.);s=np.where(x<=Rb,shape0(x),shape0(Rb)*(Rb/np.maximum(x,Rb))**n)
        s=s*(1+p.get('FA',0.)*np.tanh((x-1)/p.get('FL',1.)))
        for A,x0,w in p.get('FEAT',[]):s=s*(1+A*np.exp(-0.5*((x-x0)/w)**2))
        return s
    XT=1.5*XMAX;X=np.concatenate([[0.],np.geomspace(1e-6,XT,12000)])
    ms=cumulative_trapezoid(4*np.pi*X**2*shape(np.maximum(X,1e-12)),X,initial=0.);ms/=np.interp(1.,X,ms)
    if PMF>0:
        mc=cumulative_trapezoid(4*np.pi*X**2*np.exp(-(X/XC)**2),X,initial=0.);mc/=np.interp(1.,X,mc)
        mt=(1-PMF)*ms+PMF*mc
    else:mt=ms
    Mtot=FRAC*0.5*u*u;Mx=lambda x:Mtot*np.interp(x,X,mt)
    # bang time t_B(x) <= 0 (inner shells first): t_B = -TAU exp(-(x/XB)^2); TAU = 0 is the common bang. Cosmic time today t0 =
    # age of our shell + t_B(1); every shell's age today is t0 - t_B(x).
    TAU=p.get('TAU',0.);XB=p.get('XB',1.);tB=lambda x:-TAU*np.exp(-(np.asarray(x,float)/XB)**2)
    E0=0.5*u*u-Mtot;t0=age_out(1.,Mtot,E0)+float(tB(1.))
    nout=max(2,int(round(140*np.log10(XMAX/1.5))))
    XL=np.r_[np.geomspace(1e-3,1.5,600),np.geomspace(1.5,XMAX,nout)[1:]]
    Mv=np.array([Mx(x) for x in XL])
    collapsed=[x for x,m in zip(XL,Mv) if m>0 and age_out(x,m,-m/x*(1-1e-12))<t0-float(tB(x))]
    if collapsed:raise InvalidBuild('shells already collapsed today (x<=%.3g): CoE core too dense for a common bang'%max(collapsed))
    E=np.empty_like(XL)
    for i,(x,m) in enumerate(zip(XL,Mv)):
        if m==0:E[i]=0.5*(x/(t0-float(tB(x))))**2;continue
        E[i]=brentq(lambda e:age_out(x,m,e)-(t0-float(tB(x))),-m/x*(1-1e-12),50*(x/t0)**2+10*m/x+10,xtol=1e-15,rtol=1e-13)
    if np.any(1+2*E<=1e-3):raise InvalidBuild('1+2E<=0 at x=%.3g (not a Lorentzian LTB solution)'%XL[np.argmin(1+2*E)])
    T=np.geomspace(float(os.environ.get('TMIN','1e-9')),t0,nT);W=np.sign(E)*np.sqrt(2*np.abs(E))
    TBv=tB(XL);G=np.array([[V.Rfun(t-b,w,m) for w,m,b in zip(W,Mv,TBv)] for t in T])
    bad=np.diff(G[1:],axis=1)<=0;rows=np.where(bad.any(1))[0]+1;tcross=float(T[rows.max()]) if len(rows) else 0.
    # inner-first bang times cross shells at early times (Hellaby-Lake: dust needs t_B' <= 0). A build is accepted only if the
    # crossing has ended before p['CROSS_OK_BEFORE'] (set below our shell's recombination); light is traced only after that.
    if tcross>p.get('CROSS_OK_BEFORE',0.):raise InvalidBuild('shell crossing (until t=%.3g)'%tcross)
    if tcross>0:
        k0=rows.max()+2;T=T[k0:];G=G[k0:]
    S=RectBivariateSpline(T,XL,G,kx=5,ky=5)
    Es=make_interp_spline(XL,E,k=5);Esd=Es.derivative()
    dmt=np.gradient(mt,X);dm=lambda x:Mtot*np.interp(x,X,dmt)
    def AB(t,w):
        Rr=S(t,w)[0,0];Rp=S(t,w,dy=1)[0,0];Rt=S(t,w,dx=1)[0,0];Rpp=S(t,w,dy=2)[0,0];Rtp=S(t,w,dx=1,dy=1)[0,0]
        q=1+2*float(Es(w));qp=2*float(Esd(w))
        return Rr,Rp*Rp/q,2*Rp*Rtp/q,2*Rp*Rpp/q-Rp*Rp*qp/q**2,Rr*Rr,2*Rr*Rt,2*Rr*Rp
    def rhs(l,y):
        t,w,ph,pt,pw,pp=y;Rr,A,At,Aw,B_,Bt,Bw=AB(t,w)
        return [-pt,pw/A,pp/B_,0.5*At/A**2*pw**2+0.5*Bt/B_**2*pp**2,0.5*Aw/A**2*pw**2+0.5*Bw/B_**2*pp**2,0]
    rho=lambda t,x:dm(x)/(4*np.pi*S(t,x)[0,0]**2*S(t,x,dy=1)[0,0])
    R1=S(t0,1.)[0,0];Ht=S(t0,1.,dx=1)[0,0]/R1;Hr=S(t0,1.,dx=1,dy=1)[0,0]/S(t0,1.,dy=1)[0,0]
    Om=2*dm(1.)/(R1**2*S(t0,1.,dy=1)[0,0])/3/Ht**2
    A0=AB(t0,1.)[1];B0=AB(t0,1.)[4]
    def y0(ps):pr=np.radians(ps);return [t0,1.,0.,-1.,-np.sqrt(A0)*np.cos(pr),-np.sqrt(B0)*np.sin(pr)]
    def density_profile(xs,ts):return np.array([[rho(t,x) for x in xs] for t in ts])
    return dict(tcross=tcross,tB=tB,density_profile=density_profile,p=p,XMAX=XMAX,XL=XL,E=E,T=T,S=S,t0=t0,Mtot=Mtot,AB=AB,rhs=rhs,rho=rho,dm=dm,Es=Es,R1=R1,Ht=Ht,Hr=Hr,Om=Om,y0=y0,
                q_min=float((1+2*E).min()),x_qmin=float(XL[np.argmin(1+2*E)]))

def DA_pair(M,s1,s2,ps,d,lsel):
    """angular-diameter distance (code units) from rays psi and psi+d, at the time of point lsel on ray 1"""
    y1=s1.sol(lsel);t1,w1,ph1,pt1,pw1,pp1=y1
    ev=lambda l,y:y[0]-t1;ev.terminal=True
    if s2.sol(s2.t[-1])[0]<t1:l=brentq(lambda l:s2.sol(l)[0]-t1,s2.t[-1],0.);y2=s2.sol(l)
    else:r2=solve_ivp(M['rhs'],(0,-1e5),M['y0'](ps+d),events=[ev],rtol=1e-11,atol=1e-15);y2=r2.y_events[0][0]
    Rr,A,_,_,B_,_,_=M['AB'](t1,w1)
    sep=np.array([np.sqrt(A)*(y2[1]-w1),np.sqrt(B_)*(y2[2]-ph1)]);n=np.array([pw1/np.sqrt(A),pp1/np.sqrt(B_)]);n/=np.linalg.norm(n)
    return np.sqrt(np.linalg.norm(sep-(sep@n)*n)/abs(np.radians(d))*abs(Rr*np.sin(ph1)/np.sin(np.radians(ps))))

LZ_PS=np.r_[2.5,np.arange(7.5,180,7.5)[:-1],177.5];LZ_ZS=np.array([0.003,0.01,0.02,0.035,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.375,0.46])
def lowz(M,PS=LZ_PS,ZS=LZ_ZS,d=0.05):
    """H_eff(psi, z) = z(1+z/2)/D_L on Team A's grid (psi = 0 looks away from the CoE). Raises RayFailure on any gap."""
    stop=lambda l,y:-y[3]-(1+ZS[-1]+0.01);stop.terminal=True
    ray=lambda ps:solve_ivp(M['rhs'],(0,-1e4),M['y0'](ps),events=[stop],rtol=1e-11,atol=1e-14,dense_output=True)
    H=np.empty((len(PS),len(ZS)))
    for i,ps in enumerate(PS):
        s1=ray(ps);dd=d if ps<90 else -d;s2=ray(ps+dd)
        if not s1.t_events[0].size:raise RayFailure('low-z ray psi=%g did not reach z=%g'%(ps,ZS[-1]))
        for j,z in enumerate(ZS):
            l=brentq(lambda l:-s1.sol(l)[3]-(1+z),s1.t[-1],0.);H[i,j]=z*(1+z/2)/((1+z)**2*DA_pair(M,s1,s2,ps,dd,l))
    return dict(PS=PS.tolist(),ZS=ZS.tolist(),H=H.tolist())

def recomb_setup(M):
    S,t0,T,XL,rho=M['S'],M['t0'],M['T'],M['XL'],M['rho']
    XR=M['p'].get('XREC',1100.)   # our shell recombines at R = R(t0)/XREC; XREC fixes the recombination density (matter per photon)
    tus=brentq(lambda t:S(t,1.)[0,0]-M['R1']/XR,T[0]*1.001,t0);rc=rho(tus,1.)
    XX=np.geomspace(XL[0]*1.01,XL[-1]*0.99,int(150*np.log10(XL[-1]/XL[0])));te=[]
    for x in XX:
        f=lambda t:np.log(rho(t,x)/rc)
        te.append(T[0]*1.0001 if f(T[0]*1.0001)<0 else brentq(f,T[0]*1.0001,t0,xtol=1e-16,rtol=1e-14))   # shell recombined before the grid starts (head start): clamp
    lt=interp1d(np.log(XX),np.log(te),kind='cubic')
    return lambda w:float(np.exp(lt(np.log(min(max(w,XX[0]),XX[-1]))))),(XX[0],XX[-1])

def cmb_rays(M,PS,rtol=1e-10):
    """trace each direction to its recombination shell; returns list of event states, or raises RayFailure / NeedsLargerGrid"""
    tf,(xlo,xhi)=recomb_setup(M)
    hit=lambda l,y:y[0]-tf(y[1]);hit.terminal=True
    outer=lambda l,y:0.97*xhi-y[1];outer.terminal=True
    inner=lambda l,y:y[1]-1.05*xlo;inner.terminal=True
    res=[]
    for ps in PS:
        s=solve_ivp(M['rhs'],(0,-1e5),M['y0'](ps),events=[hit,outer,inner],rtol=rtol,atol=rtol*1e-4,dense_output=True)
        if s.t_events[0].size:res.append((ps,s,s.y_events[0][0]))
        elif s.t_events[1].size:return 'EXTEND',ps
        else:raise RayFailure('ray psi=%g stopped without recombining: %s (inner=%s, x_end=%.3g)'%(ps,s.message,bool(s.t_events[2].size),s.y[1,-1]))
    return 'OK',res

def build_full_sky(p,PS,XMAX0=40.,XCAP=1e8):
    """size the outer grid with the most outward ray (smallest psi) first, then trace every ray in PS on that grid;
    any ray that still fails to recombine raises (no gaps)"""
    XMAX=XMAX0;ps0=float(np.min(PS))
    while True:                                   # probe
        M=build(p,XMAX);st,r=cmb_rays(M,[ps0])
        if st=='OK':break
        if XMAX>=XCAP:raise RayFailure('ray psi=%g still outside recombination at XMAX=%g'%(ps0,XMAX))
        XMAX*=10
    while True:
        st,r=cmb_rays(M,PS)
        if st=='OK':return M,r
        if XMAX>=XCAP:raise RayFailure('ray psi=%g still outside recombination at XMAX=%g'%(r,XMAX))
        XMAX*=4;M=build(p,XMAX)
