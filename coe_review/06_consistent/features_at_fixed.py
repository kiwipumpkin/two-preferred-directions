import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
# Feature signs at the AoE-fixed orientation (tilt 15, roll 60.18; D/R plays no role in the sign maps). No orientation freedom.
# Null: rigid random rotation of the feature catalogue (keeps antipodes and the dipole clustering), orientation held fixed.
import sys,json,numpy as np;sys.argv=['x'];exec(open(PKG + '/coe_review/joint/spotmc.py').read().split("F0=np.array")[0])
F0=np.array([f[2] for f in feat]);out={}
for roll in (60.18,240.18):
    Rm=frame(-15,roll);rows=[]
    for (K,nm,v,c) in feat:
        m=v@Rm;lm=np.degrees(np.arctan2(m[1],m[0]))%360;val=float(pi(MM['l'],MM[K],lm));rows.append((K,nm,round(val,3),(val>0)==(c=='hot')))
    sc=sum(r[3] for r in rows);out[roll]=dict(score=int(sc),rows=rows);print('roll %.2f: %d/12'%(roll,sc));[print('   ',r) for r in rows]
rng=np.random.default_rng(5);N=100000;Rm=frame(-15,60.18);ge=0
def randrot():
    q=rng.normal(size=4);q/=np.linalg.norm(q);a,b,c,dd=q
    return np.array([[a*a+b*b-c*c-dd*dd,2*(b*c-a*dd),2*(b*dd+a*c)],[2*(b*c+a*dd),a*a-b*b+c*c-dd*dd,2*(c*dd-a*b)],[2*(b*dd-a*c),2*(c*dd+a*b),a*a-b*b-c*c+dd*dd]])
obs=out[60.18]['score'];cnt=np.zeros(13,int)
for i in range(N):
    Q=randrot();F=F0@Q.T;s=0
    for (K,nm,v,c),vv in zip(feat,F):
        m=vv@Rm;lm=np.degrees(np.arctan2(m[1],m[0]))%360;s+=(pi(MM['l'],MM[K],lm)>0)==(c=='hot')
    cnt[s]+=1
p=cnt[obs:].sum()/N;print('null (orientation fixed by the AoE; catalogue rotated rigidly, %d draws): P(score >= %d) = %.5f'%(N,obs,p))
out['null']=dict(N=N,counts=cnt.tolist(),p=p);json.dump(out,open('features_at_fixed.json','w'),indent=1,default=bool)
