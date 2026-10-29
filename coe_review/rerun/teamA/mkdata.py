import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root
# cache digitised Migkas Fig.6 curve, covariances and the 313-cluster catalogue (same code path as significance/template_sig.py)
import io,contextlib,sys,os,numpy as np
os.environ['OMP_NUM_THREADS']='1'
P='/tmp/claude-0/-home-user/f456509a-71ff-5e95-b1fb-466a922b58cd/scratchpad/phase1B/';sys.path.insert(0,P+'in')
__file__=P+'cov.py';src=open(__file__).read().split('# (1) self-consistent')[0]
with contextlib.redirect_stdout(io.StringIO()):exec(src)
sys.path.insert(0,PKG + '/coe_review/hmap');import common as cm;cat=cm.load_catalogue()
assert np.allclose(cat['z'],z) and np.allclose(cat['l'],l)
Cd=Rb*np.outer(sd,sd)
np.savez(PKG + '/coe_review/rerun/teamA/migkas_data.npz',ld=ld,Ad=Ad,sd=sd,Cd=Cd,A0=A0,Cb=Cb,z=z,l=l,b=cat['b'],x=x,y=y,M=M)
print('ok',A0[:5],Ad[:5])
