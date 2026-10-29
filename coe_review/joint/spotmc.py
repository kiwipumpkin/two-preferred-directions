import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
# Significance of the feature-sign matches (figure book 23_sky_mapA_al.py scoring), with the feature set's own correlations kept.
# Residual curves: Migkas A and B minus (CoE model + shared n = 1..3 harmonics), in %, boxcar-smoothed +-25 deg (23_dumpM.py).
# Score: a feature matches if the residual at its model-frame longitude has the feature's sign (hot > 0, cold < 0).
# Null: rotate the WHOLE feature catalogue rigidly by a uniformly random rotation (keeps antipodal pairs and the clustering of the
# CMB/NVSS/CatWISE dipoles), then give the null the same freedom the model had, and count how often it scores >= the real one.
import json,csv,numpy as np
R=PKG + '/kiwipumpkin_work/coepdf/book/recovered/tmp_rescue/rnx/'
d=json.load(open(R+'l50/curves.json'));g0=np.array(d['grid'])
rows=list(csv.DictReader(open(PKG + '/workspace/cosmology/migkas/fig6_digitized.csv')));l=np.array([float(r['l_deg']) for r in rows]);th=np.radians(l)
res=np.genfromtxt(R+'fr/residuals.csv',delimiter=',',names=True);sA=res['sigma'];sB=np.array([(float(r['B_err_low'])+float(r['B_err_high']))/2 for r in rows])
pi=lambda x0,y0,x:np.interp(x,x0,y0,period=360)
A0,B0,AFl,BLl=[pi(g0,np.array(d[k]),l) for k in ('A0','B0','AF','BL')]
rA=A0-AFl;rB=B0-BLl;o=np.ones_like(th);Z=0*o;steps=[]
for n in range(1,4):
    h=np.column_stack([np.cos(n*th),np.sin(n*th)]);X=np.vstack([np.column_stack([o,Z,h]),np.column_stack([Z,o,h])]);s=np.r_[sA,sB]
    p=np.linalg.lstsq(X/s[:,None],np.r_[rA,rB]/s,rcond=None)[0];steps.append((n,p));rA=rA-X[:71]@p;rB=rB-X[71:]@p
T=lambda n,p,x,k:p[k]+p[2]*np.cos(n*x)+p[3]*np.sin(n*x)
MM={'l':l}
for nm,Y,mb,k in (('A',A0,AFl,0),('B',B0,BLl,1)):
    m=mb.copy()
    for n,p in steps:m=m+T(n,p,th,k)
    r=100*(Y-m)/Y;MM[nm]=np.array([np.mean(r[np.abs((l-c+180)%360-180)<=25]) for c in l])
SP={'A':[('CMB dipole cold',84,-48,'cold'),('Bulk flow away',107,-8,'cold'),('CMB Cold Spot',209,-57,'cold'),('CMB dipole hot',264,48,'hot'),('Bulk flow toward',287,8,'hot')],
    'B':[('NVSS cold',16,-46,'cold'),('Hemi. asym. cold',50,20,'cold'),('CatWISE cold',58,-31,'cold'),('NVSS hot',196,46,'hot'),('Hemi. asym. hot',230,-20,'hot'),('CatWISE hot',238,31,'hot'),('Loop I / NPS',329,17.5,'hot')]}
def vec(l,b):l,b=np.radians(l),np.radians(b);return np.stack([np.cos(b)*np.cos(l),np.cos(b)*np.sin(l),np.sin(b)],-1)
def rot(ax,a):
    ax=ax/np.linalg.norm(ax);a=np.radians(a);K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]]);return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
def frame(tilt,roll):R1=rot(np.array([0,1.,0]),tilt);return rot(R1@vec(180,0),roll)@R1
feat=[(K,nm,vec(sl,sb),c) for K in 'AB' for nm,sl,sb,c in SP[K]]
def score(F,Rm,skip=()):
    sc=[]
    for (K,nm,v,c),vv in zip(feat,F):
        if nm in skip:continue
        m=vv@Rm;lm=np.degrees(np.arctan2(m[1],m[0]))%360;val=pi(MM['l'],MM[K],lm);sc.append((val>0)==(c=='hot'))
    return int(np.sum(sc)),len(sc)
F0=np.array([f[2] for f in feat])
print('validation (tilt -15, roll 45):',score(F0,frame(-15,45)),' roll 80.8:',score(F0,frame(-15,80.8)))
rolls=np.arange(0,360,2.5)
def best(F,rr,tilts=(-15,),skip=()):return max(score(F,frame(t,r),skip)[0] for t in tilts for r in rr)
rng=np.random.default_rng(7);N=int(__import__('os').environ.get('N','4000'))
def randrot():
    q=rng.normal(size=4);q/=np.linalg.norm(q);a,b,c,dd=q
    return np.array([[a*a+b*b-c*c-dd*dd,2*(b*c-a*dd),2*(b*dd+a*c)],[2*(b*c+a*dd),a*a-b*b+c*c-dd*dd,2*(c*dd-a*b)],[2*(b*dd-a*c),2*(c*dd+a*b),a*a-b*b-c*c+dd*dd]])
import os
V=os.environ.get('VAR','all')
for label,rr,skip in [('roll fixed in the independent 45-80 deg window',np.arange(45,80.1,2.5),()),('roll free 0-360',rolls,()),
                      ('roll 45-80, Loop I excluded',np.arange(45,80.1,2.5),('Loop I / NPS',))]+[('TILT FREE -30..30 & roll free',rolls,())]:
    if V!='all' and V not in label:continue
    tl=tuple(range(-30,31,5)) if 'TILT' in label else (-15,);nn=N if 'TILT' not in label else N//10
    obs=best(F0,rr,tl,skip=skip);ntot=12-len(skip);ge=0
    for i in range(nn):
        Q=randrot();ge+=best(F0@Q.T,rr,tl,skip=skip)>=obs
    N_=nn
    print(f'{label}: observed best {obs}/{ntot}; null P(>= observed) = {ge/N_:.4f} ({N_} random rigid rotations of the catalogue)',flush=True)
json.dump({k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in MM.items()},open('M25_rebuilt.json','w'))
