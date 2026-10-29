# The Axis of Evil fixes the geometry: with the CoE direction set (180, -15), the AoE's angle from the CoE axis fixes the
# critical-point circle theta0 = arccos(R/2D) -> D/R, and its azimuth about the CoE axis fixes the roll. Two numbers in, two out.
import json,numpy as np
from scipy.optimize import brentq,minimize_scalar
def vec(l,b):l,b=np.radians(l),np.radians(b);return np.array([np.cos(b)*np.cos(l),np.cos(b)*np.sin(l),np.sin(b)])
def gal(v):return float(np.degrees(np.arctan2(v[1],v[0]))%360),float(np.degrees(np.arcsin(np.clip(v[2],-1,1))))
def rot(ax,a):
    ax=ax/np.linalg.norm(ax);a=np.radians(a);K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]]);return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
def frame(tilt,roll):R1=rot(np.array([0,1.,0]),tilt);return rot(R1@vec(180,0),roll)@R1
out={}
for name,aoe in (('Land & Magueijo 2005 (260, +60)',(260,60)),):
    A=vec(*aoe);coe=frame(-15,0)@vec(180,0);ang=np.degrees(np.arccos(A@coe));th0=min(ang,180-ang);DR=0.5/np.cos(np.radians(th0))
    # roll: make a critical point (model (180 -+ th0, 0)) coincide with the AoE axis (headless); scan 0..360 and refine
    def err(r,s):p=frame(-15,r)@vec(180-s*th0,0);return np.degrees(np.arccos(min(1,abs(p@A))))
    sols=[]
    for s in (+1,-1):
        rs=np.arange(0,360,0.25);e=[err(r,s) for r in rs];
        for i in np.argsort(e)[:40]:
            m=minimize_scalar(lambda r:err(r,s),bounds=(rs[i]-1,rs[i]+1),method='bounded')
            if m.fun<0.05 and not any(abs(m.x-x[0])<1 for x in sols):sols.append((round(float(m.x)%360,2),s,round(float(m.fun),3)))
    print(name,': angle from CoE axis %.2f deg -> theta0 %.2f deg -> D/R = %.3f'%(ang,th0,DR))
    for r,s,e in sorted(sols):
        Rm=frame(-15,r);print('   roll %.2f deg (critical point %s): offset %.3f deg; CoE %s, P1 %s, P2 %s'%(r,'P1' if s>0 else 'P2',e,gal(Rm@vec(180,0)),gal(Rm@vec(180-th0,0)),gal(Rm@vec(180+th0,0))))
    out[name]=dict(DR=DR,theta0=th0,rolls=sorted(sols))
json.dump(out,open('geometry_from_aoe.json','w'),indent=1)
for R in (14.3,):print('with R = %.1f Gpc: D = %.1f Gpc'%(R,R*DR))
