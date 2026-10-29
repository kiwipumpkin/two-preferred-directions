import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import sys, os, json, time, itertools, numpy as np
B = PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble/gr_ltb/refit_mass'
sys.path[:0] = [B, PKG + '/kiwipumpkin_work/cosmology/papers/cmb_redshift', PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble/gr_ltb', PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble']; os.chdir(B)
import relfree as R
exec(open(PKG + '/coe_review/15_knobs/sweep.py').read().split("def r2mig")[0].split("from bubble_core import shifted")[1])  # feats()
RV = 14.30048287260589
n0 = json.load(open(PKG + '/coe_review/06_consistent/fig4/ltb_fixedD.json'))
rows = []; t0 = time.time()
for DR, mcc, (kin, kout) in itertools.product([2.0, 3.0, 3.564, 4.5, 6.0, 8.0], [0.03, 0.09, 0.3], [(1.094, 1.107), (1.5, 2.0)]):
    D = DR*RV
    if kin*RV >= 0.99*D: continue
    u = [np.log10(D), np.log10(mcc), np.log(kin-1), np.log(kout-1)]
    try:
        o = R.ev(*R.unpack(u), full=True); Ad = np.array(o['Ad'], float); ok = bool(o.get('ok', True)) and np.all(np.isfinite(Ad))
    except Exception as e:
        ok = False; o = {}; print('fail', DR, mcc, kin, e, flush=True)
    kn = f'mcc={mcc},kin={kin},kout={kout}'
    if not ok: rows.append(dict(model='LTB point+shell', DR=DR, knobs=kn, ok=False)); continue
    # LTB Ad index 0 maps to sky l0+180 (rm_core.lfits): theta from anti-CoE = index angle + 180
    A = np.roll(Ad, len(Ad)//2); f = feats(A)
    rows.append(dict(model='LTB point+shell', DR=DR, knobs=kn, ok=True, R2_ns=float(o.get('R2_ns', np.nan)), **f))
    print(DR, kn, f['theta_max'], f['theta_min'], f['n_interior_extrema'], '%.0fs' % (time.time()-t0), flush=True)
json.dump(rows, open(PKG + '/coe_review/15_knobs/sweep_ltb.json', 'w'), indent=1, default=float)
