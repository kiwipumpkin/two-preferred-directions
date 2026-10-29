import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.coordinates import SkyCoord
import astropy.units as u
v = lambda l, b: np.array([np.cos(np.radians(b))*np.cos(np.radians(l)), np.cos(np.radians(b))*np.sin(np.radians(l)), np.sin(np.radians(b))])
lb = lambda x: (np.degrees(np.arctan2(x[..., 1], x[..., 0])) % 360, np.degrees(np.arcsin(np.clip(x[..., 2], -1, 1))))
X = lambda l: -np.radians(((np.asarray(l)+180) % 360)-180)
def gc_through(a, b, n=721):     # great circle through directions a and b
    a = a/np.linalg.norm(a); c = b-(b@a)*a; c /= np.linalg.norm(c); t = np.linspace(0, 2*np.pi, n)
    return np.outer(np.cos(t), a)+np.outer(np.sin(t), c)
def plot_gc(ax, P, **kw):
    l, b = lb(P); x = X(l); y = np.radians(b); br = np.where(np.abs(np.diff(x)) > 1)[0]+1
    for i, (xs, ys) in enumerate(zip(np.split(x, br), np.split(y, br))): ax.plot(xs, ys, **(kw if i == 0 else {k: w for k, w in kw.items() if k != 'label'}))
def small_circle(ax, l0, b0, rad, **kw):
    c = v(l0, b0); a = np.cross(c, [0, 0, 1.]); a /= np.linalg.norm(a); e = np.cross(c, a); t = np.linspace(0, 2*np.pi, 400)
    P = np.cos(np.radians(rad))*c+np.sin(np.radians(rad))*(np.outer(np.cos(t), a)+np.outer(np.sin(t), e)); plot_gc(ax, P, **kw)
COE = v(180, -15); AOE = v(260, 60); P2 = v(234, 52.5)
D = [('CMB dipole', 264, 48.3, '#d7301f', 'o'), ('CatWISE dipole', 238.2, 28.8, '#a6216e', 's'), ('NVSS dipole', 196, 46, '#7a1fa2', 'D'),
     ('Bulk flow', 287, 8, '#1b7f3b', '^'), ('Hemispherical asymmetry', 230, -20, '#0f8b8d', 'v'), ('Axis of Evil', 260, 60, 'k', '*')]
fig = plt.figure(figsize=(13, 7.4)); ax = fig.add_subplot(111, projection='mollweide'); ax.grid(alpha=.35)
plot_gc(ax, gc_through(COE, AOE), color='#c98a0b', lw=2.2, label='plane of evil')
for n, l, b, c, mk in D:
    plot_gc(ax, gc_through(v(l, b), COE), color=c, lw=1, ls=':', alpha=.7)
    ax.plot(X(l), np.radians(b), mk, color=c, mec='k', ms=15 if mk != '*' else 22, label=f'{n} ({l:g}°, {b:+g}°)', zorder=5)
    al, ab = lb(-v(l, b)); ax.plot(X(al), np.radians(ab), mk, mfc='white', mec=c, mew=2, ms=13 if mk != '*' else 20, zorder=5)
for (l, b), txt, col in (((240, 42), 'HIGH ends\n(more sources / hotter)', '#d7301f'), ((60, -42), 'LOW ends', '#2b6cb0')):
    small_circle(ax, l, b, 28, color=col, lw=3, ls='--', zorder=4)
    tl, tb = ((205, 66) if b > 0 else (112, -58)); ax.text(X(tl), np.radians(tb), txt, color=col, ha='center', fontsize=11, weight='bold')
ax.plot(X(180), np.radians(-15), 'P', color='gold', mec='k', ms=20, zorder=6, label='CoE (180°, −15°), in Taurus')
ax.plot(X(0), np.radians(15), 'P', color='white', mec='goldenrod', mew=2, ms=17, zorder=6, label='anti-CoE (0°, +15°)')
ax.plot(X(234), np.radians(52.5), 'X', color='#1b7f3b', mec='k', ms=13, zorder=6, label='P2 (234°, +52.5°)')
ax.set_xticks(np.radians([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])); ax.set_xticklabels(['150°', '120°', '90°', '60°', '30°', '0°', '330°', '300°', '270°', '240°', '210°'])
ax.set_title('Published dipoles and axes: filled = high/hot end, open = low/cold end. Dotted: each dipole\'s great circle through the CoE. Galactic, ℓ increases left.', fontsize=10, pad=12)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.06), ncol=3, fontsize=9, frameon=False)
fig.savefig('fig_dipoles_connected.png', dpi=170, bbox_inches='tight'); plt.close(fig)
# ---- where to look: star chart of Taurus / Orion
stars = {'Aldebaran': (4.598, 16.51, 0.9), 'Elnath': (5.438, 28.61, 1.7), 'Pleiades': (3.79, 24.11, 2.9), 'Ain': (4.477, 19.18, 3.5), 'γ Tau': (4.33, 15.63, 3.7),
         'ζ Tau': (5.627, 21.14, 3.0), 'Betelgeuse': (5.92, 7.41, 0.5), 'Bellatrix': (5.418, 6.35, 1.6), 'Rigel': (5.242, -8.20, 0.1), 'Saiph': (5.796, -9.67, 2.1),
         'Mintaka': (5.533, -0.30, 2.2), 'Alnilam': (5.603, -1.20, 1.7), 'Alnitak': (5.679, -1.94, 1.8), 'Meissa': (5.585, 9.93, 3.4),
         'Capella': (5.278, 46.0, 0.1), 'Castor': (7.577, 31.89, 1.6), 'Pollux': (7.755, 28.03, 1.1), 'Alhena': (6.629, 16.40, 1.9), 'Procyon': (7.655, 5.22, 0.4)}
fig, ax = plt.subplots(figsize=(9, 7.2)); ax.set_facecolor('#0b1530')
for n, (ra, de, mag) in stars.items():
    ax.scatter(ra*15, de, s=max(8, 140-40*mag), color='#fff7d6', zorder=3)
    ax.text(ra*15-1.0, de+1.0, n, color='#cfd8ff', fontsize=8)
for a, b in [('Rigel', 'Mintaka'), ('Betelgeuse', 'Alnitak'), ('Mintaka', 'Alnilam'), ('Alnilam', 'Alnitak'), ('Betelgeuse', 'Bellatrix'), ('Bellatrix', 'Mintaka'), ('Alnitak', 'Saiph'), ('Betelgeuse', 'Meissa'), ('Bellatrix', 'Meissa'),
             ('Aldebaran', 'γ Tau'), ('γ Tau', 'Ain'), ('Ain', 'Elnath'), ('Aldebaran', 'ζ Tau'), ('Castor', 'Pollux')]:
    ax.plot([stars[a][0]*15, stars[b][0]*15], [stars[a][1], stars[b][1]], color='#5f74b5', lw=1, zorder=2)
for (l, b), lab, col in (((198, -4), 'CoE (best fit)\n(l, b) = (198°, −4°)', 'gold'), ((180, -15), 'reference: (180°, −15°)', '#9fb3d9')):
    c = SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic').icrs
    best = 'best' in lab
    ax.scatter(c.ra.deg, c.dec.deg, marker='P', s=380 if best else 160, color=col, edgecolor='k', zorder=5)
    if best: ax.add_patch(Circle((c.ra.deg, c.dec.deg), 5, fill=False, ec=col, lw=1.8, ls='--', zorder=4))
    ax.text(c.ra.deg+(0 if best else 2), c.dec.deg+(-9.5 if best else 3.2), lab, color=col, fontsize=10 if best else 8.5, weight='bold', ha='center')
ax.set_xlim(125, 45); ax.set_ylim(-15, 50); ax.set_aspect(1/np.cos(np.radians(18)))
ax.set_xlabel('Right ascension (deg; east to the left, as seen in the sky)'); ax.set_ylabel('Declination (deg)')
ax.set_title('Where the center of expansion lies: in Orion, 5° from Betelgeuse (winter evening sky, northern hemisphere)\nDashed circle: 5° around the best-fit CoE. The CoE itself lies far beyond any visible star.', fontsize=10)
for s in ax.spines.values(): s.set_color('#5f74b5')
fig.savefig('fig_where_to_look.png', dpi=170, bbox_inches='tight'); print('ok')
