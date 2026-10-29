import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import sys, json, numpy as np
sys.argv=['x']; exec(open(PKG + '/coe_review/joint/spotmc.py').read().split("F0=np.array")[0])
exec(open(PKG + '/coe_review/13_combined/combined.py').read().split("def frame2")[1].split("Rm0 =")[0].join(["def frame2",""]) if False else "")
def frame2(C, A):
    if C @ A < 0: A = -A
    a = A - (A @ C)*C; a /= np.linalg.norm(a)
    G = np.column_stack([C, a, np.cross(C, a)]); M = np.column_stack([[-1,0,0],[0,1,0],np.cross([-1,0,0],[0,1,0])]); return G @ M.T
gal=lambda x:(round(float(np.degrees(np.arctan2(x[1],x[0]))%360),1),round(float(np.degrees(np.arcsin(np.clip(x[2],-1,1)))),1))
C=vec(198,-4); A=vec(80,-60); th0=np.degrees(np.arccos(abs(C@A))); DR=0.5/np.cos(np.radians(th0))
Rm=frame2(C,A); t=np.radians(th0)
P1=Rm@np.array([-np.cos(t),np.sin(t),0]); P2=Rm@np.array([-np.cos(t),-np.sin(t),0]); N=np.cross(P1,P2); N/=np.linalg.norm(N)
# roll relative to letter convention: tilt -4 about y then roll about CoE axis? report azimuth of P1 about CoE from the b-increasing direction
F0=np.array([f[2] for f in feat])
def T1(Rm,F=F0):
    s=0; rows=[]
    for (K,nm,v,c),vv in zip(feat,F):
        m=vv@Rm; lm=np.degrees(np.arctan2(m[1],m[0]))%360; val=pi(MM['l'],MM[K],lm); ok=(val>0)==(c=='hot'); s+=ok; rows.append((K,nm,round(float(val),2),bool(ok)))
    return s,rows
L4=vec(115.3,31.4)
T2=lambda C,A:(lambda n:np.degrees(np.arccos(min(1,abs(n@L4)/np.linalg.norm(n)))))(np.cross(C,A))
s,rows=T1(Rm); t2=T2(C,A)
out=dict(CoE=(198,-4),theta0=round(th0,2),DR=round(DR,3),D_Gpc=round(DR*14.30,1),P1=gal(P1),P2=gal(P2),plane_normal=gal(N),T1=int(s),T1_rows=rows,T2=round(t2,1))
rng=np.random.default_rng(11)
def randrot():
    q=rng.normal(size=4);q/=np.linalg.norm(q);a,b,c,d=q
    return np.array([[a*a+b*b-c*c-d*d,2*(b*c-a*d),2*(b*d+a*c)],[2*(b*c+a*d),a*a-b*b+c*c-d*d,2*(c*d-a*b)],[2*(b*d-a*c),2*(c*d+a*b),a*a-b*b-c*c+d*d]])
NS=10000
for kind in 'AB':
    t1=np.zeros(NS); tt=np.zeros(NS)
    for i in range(NS):
        if kind=='A': Q=randrot(); Cc,Aa=Q@C,Q@A
        else: Cc=rng.normal(size=3); Cc/=np.linalg.norm(Cc); Aa=A
        t1[i]=T1(frame2(Cc,Aa))[0]; tt[i]=T2(Cc,Aa)
    p1=(t1>=s).mean(); p2=(tt<=t2).mean()
    P1d={x:(t1>=x).mean() for x in np.unique(t1)}; q1=np.array([P1d[x] for x in t1]); q2=np.searchsorted(np.sort(tt),tt,side='right')/NS
    F=-2*(np.log(q1)+np.log(q2)); Fo=-2*(np.log(max(p1,1/NS))+np.log(max(p2,1/NS))); pj=(F>=Fo).mean()
    from scipy.stats import norm; out['null_'+kind]=dict(p_T1=p1,p_T2=p2,p_joint=pj,sigma_joint=float(norm.isf(pj/2)))
print(json.dumps(out,indent=1,default=float)); json.dump(out,open('frame198.json','w'),indent=1,default=float)
