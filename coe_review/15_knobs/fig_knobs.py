import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import sys, os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble'; sys.path.insert(0, B); os.chdir(B)
import newt_omb as N, prof_core as P, ptshell
Bl = B+'/gr_ltb/refit_mass'; sys.path[:0] = [Bl, PKG + '/kiwipumpkin_work/cosmology/papers/cmb_redshift', B+'/gr_ltb']; os.chdir(Bl)
import relfree as R
th = np.arange(360.)
def norm(a): a = np.asarray(a, float); return (a-a.min())/(a.max()-a.min())
cur = []
af, _ = N.af_of(N.geo(3.5*N.RV, 1.7e-3, 1.04, 1.66, N.RV)); cur.append(("Newton point mass + shell", norm(af)))
for kind, pars, lab in [('exp', dict(LD=0.01), 'Newton compact exponential core'), ('gauss', dict(LD=0.18), 'Newton Gaussian core'),
                        ('shell', dict(rs=0.25, w=0.1), 'Newton inner shell'), ('st53', dict(rst=0.24), 'Newton Sedov-Taylor blast'),
                        ('exp', dict(LD=0.3), 'Newton very extended core (max flips)')]:
    rr = P.run(kind, dict(pars, DR=3.5), 0.003 if pars.get('LD') == 0.01 else 0.05, 0.05); assert rr['vrmin'] > 0, lab; cur.append((lab, norm(rr['af'])))
o = R.ev(*R.unpack([np.log10(3.564*14.30048), np.log10(0.09), np.log(0.094), np.log(0.107)]), full=True)
Ad = np.roll(np.array(o['Ad']), len(o['Ad'])//2); cur.append(("exact GR (LTB) point mass + shell", norm(Ad[::2])))
Nw = [r for r in json.load(open(PKG + '/coe_review/15_knobs/sweep_newton.json')) if r['ok']]
Lt = [r for r in json.load(open(PKG + '/coe_review/15_knobs/sweep_ltb.json')) if r['ok']]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for i, (lab, a) in enumerate(cur):
    ax[0].plot(th[:181], a[:181], lw=2.2 if 'LTB' in lab else 1.6, ls='--' if 'LTB' in lab else '-', label=lab)
tc = np.degrees(np.arccos(-1/(2*3.5))); ax[0].axvline(tc, color='k', ls=':', lw=1.2); ax[0].text(tc+1, 0.92, 'equal-distance\ncircle (D/R 3.5)', fontsize=8)
ax[0].set_xlabel('angle from the anti-CoE direction (deg)   [0 = away from CoE, 180 = toward CoE]'); ax[0].set_ylabel('A, scaled 0–1')
ax[0].set_title('Same shape in every setup: maxima at both poles, one minimum ring', fontsize=10); ax[0].legend(fontsize=7, loc='upper left', bbox_to_anchor=(0.0, 0.62)); ax[0].set_xlim(0, 180); ax[0].grid(alpha=.3)
dr = np.linspace(1.8, 8.3, 200)
ax[1].plot(dr, np.degrees(np.arccos(-1/(2*dr))), 'k-', lw=2, label='equal-distance (critical-point) circle')
jit = lambda x: x + np.random.default_rng(1).normal(0, 0.04, len(x))
ax[1].scatter(jit(np.array([r['DR'] for r in Nw])), [r['theta_min'] for r in Nw], s=12, alpha=.5, label=f'Newtonian minimum ({len(Nw)} setups)')
ax[1].scatter([r['DR'] for r in Lt], [r['theta_min'] for r in Lt], s=40, marker='s', facecolor='none', edgecolor='#c0392b', lw=1.5, label=f'exact-GR (LTB) minimum ({len(Lt)} setups)')
ax[1].axhline(100.0, color='#1b7f3b', ls='--', lw=1.5, label='Axis of Evil, 100° from data-fit anti-CoE')
ax[1].axhspan(97.0, 104.5, color='#3b6fb0', alpha=.12, label='Migkas MIN points (97°, 104.5°)')
ax[1].set_xlabel('D/R'); ax[1].set_ylabel('angle of the A minimum from anti-CoE (deg)'); ax[1].set_ylim(88, 132); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
ax[1].set_title('Where the minimum ring sits', fontsize=10)
fig.tight_layout(); fig.savefig(PKG + '/coe_review/15_knobs/knobs.png', dpi=170); print('ok')
