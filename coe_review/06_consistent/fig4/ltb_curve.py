import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root
# A(l) curve of the exact-LTB Migkas fit, built the same way for the stored D=32.94 optimum (validation against the stored
# ltb_mcfree_z03_curve.npy) and for the AoE-fixed D = 50.97 Gpc refit.
import sys,os,json,numpy as np
B=PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble/gr_ltb/refit_mass'
sys.path[:0]=[B,PKG + '/kiwipumpkin_work/cosmology/papers/cmb_redshift',PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble/gr_ltb',PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble'];os.chdir(B)
import relfree as R
from bubble_core import A_from,shifted,TH1
def curve(u):
    o=R.ev(*R.unpack(u),full=True);Ad=np.array(o['Ad']);print('Ad len',len(Ad),'R2_s',o['R2_s'],'l0',o['l0_s'],'amp',o['amp'],'phi',o['phi'])
    l0=o['l0_s'];idx=(((TH1-l0-180.0)%360)/0.5).astype(int);A=Ad[idx];A=A/A.mean()   # exactly as rm_core.lfits
    return np.asarray(A_from(A,TH1-l0,o['amp'],o['phi']),float),o
st=json.load(open(PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble/universe_rel_mcfree.json'))
u0=[np.log10(st['D']),np.log10(st['mcc']),np.log(st['kin']-1),np.log(st['kout']-1)]
A0,o0=curve(u0);ref=np.load('/home/user/kiwipumpkin/coe-cosmology/draft/current/figure_sources/src/cosmology/papers/spreadsheet_model/bubble/gr_ltb/refit_mass/ltb_mcfree_z03_curve.npy');print('validation vs stored curve: max |diff| =',float(np.max(np.abs(A0-ref))))
n=json.load(open(PKG + '/coe_review/06_consistent/fig4/ltb_fixedD.json'));A1,o1=curve(n['u'])
np.save(PKG + '/coe_review/06_consistent/fig4/ltb_fixedD_curve.npy',A1);print('saved; R2',o1['R2_s'])
