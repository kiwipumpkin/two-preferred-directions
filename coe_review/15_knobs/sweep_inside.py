"""R >= D: CoE inside (or at the edge of) the sampling sphere. D/R = 0.55 ... 1.9, smooth Newtonian profiles."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import sys, os, json, itertools, numpy as np
B = PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble'; sys.path.insert(0, B); os.chdir(B)
import newt_omb as N, prof_core as P, ptshell
exec(open(PKG + '/coe_review/15_knobs/sweep.py').read().split("def r2mig")[0].split("from bubble_core import shifted")[1])
rows = []
PROF = [('gauss', dict(LD=0.18)), ('cored', dict(LD=0.05)), ('exp', dict(LD=0.1)), ('pl', dict(LD=0.05, n=3.0)), ('st53', dict(rst=0.24))]
for (kind, pars), DR, mc, Ob in itertools.product(PROF, [0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 1.9], [0.003, 0.05], [0.05, 0.3]):
    try:
        rr = P.run(kind, dict(pars, DR=DR), mc, Ob); af = rr['af']; ok = bool(np.all(np.isfinite(af)) and rr['vrmin'] > 0)
        rough = float(np.std(np.diff(af, 2))/(af.max()-af.min())) if ok else None; ok = ok and rough < 0.01
    except Exception as e: ok = False
    row = dict(kind=kind, DR=DR, mc=mc, Ob=Ob, ok=ok)
    if ok: row.update(feats(af)); row['theta_crit'] = float(np.degrees(np.arccos(-1/(2*DR))))
    rows.append(row)
json.dump(rows, open(PKG + '/coe_review/15_knobs/sweep_inside.json', 'w'), indent=1, default=float)
import collections
for DR in sorted(set(r['DR'] for r in rows)):
    L = [r for r in rows if r['DR'] == DR and r['ok']]
    if not L: print(f'D/R {DR}: no valid runs'); continue
    print(f"D/R {DR}: valid {len(L)}/{sum(r['DR']==DR for r in rows)}  crit {L[0]['theta_crit']:.1f}  theta_min {min(r['theta_min'] for r in L):.0f}-{max(r['theta_min'] for r in L):.0f}  maxima at {sorted(set(r['theta_max'] for r in L))}  interior extrema {dict(collections.Counter(r['n_interior_extrema'] for r in L))}")
