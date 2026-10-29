"""Frame geometry and every quoted angle (Sec. 4 Table 1, Sec. 9 'Other axes', Sec. 10 'Where to look', Sec. 11, Sec. 12, Appendix D).
Inputs: CoE (180, -15), published Axis of Evil (260, +60) [Land & Magueijo 2005], D/R and roll from geometry_from_aoe.json."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import json, numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
def v(l, b): l, b = np.radians(l), np.radians(b); return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])
def lb(x): x = x/np.linalg.norm(x); return round(float(np.degrees(np.arctan2(x[1], x[0])) % 360), 2), round(float(np.degrees(np.arcsin(x[2]))), 2)
ang = lambda a, c: float(np.degrees(np.arccos(np.clip(a @ c/np.linalg.norm(a)/np.linalg.norm(c), -1, 1))))
hl = lambda a, c: min(ang(a, c), 180 - ang(a, c))            # headless
off_plane = lambda x, n: abs(90 - ang(x, n))                 # angle from x to the plane with normal n
g = json.load(open(PKG + "/coe_review/06_consistent/geometry_from_aoe.json"))["Land & Magueijo 2005 (260, +60)"]
DR, th0 = g["DR"], g["theta0"]
coe, aoe = v(180, -15), v(260, 60)
P1 = v(80, -60)                                              # the end of the published axis at theta0 from the CoE
# P2: mirror of P1 across the CoE axis within the plane of evil
c = coe; p1 = P1 - (P1 @ c)*c; P2 = (P1 @ c)*c - p1
n_poe = np.cross(coe, aoe); p12 = P2 - P1
print(f"D/R = {DR:.3f}, theta0 = {th0:.2f} deg, cos(theta0) = R/2D -> D/R = {1/(2*np.cos(np.radians(th0))):.3f}")
print("P1", lb(P1), " P2", lb(P2), "/ antipode", lb(-P2), f" P1-P2 separation {ang(P1, P2):.1f} (2 theta0 = {2*th0:.1f})")
print("plane-of-evil normal", lb(n_poe), "  P1-P2 axis", lb(p12), f"  P1-P2 axis to CoE {ang(p12, coe):.3f}")
print(f"P1-P2 axis to published axis {hl(p12, aoe):.1f};  CoE off plane normal to published axis {off_plane(coe, aoe):.1f}")
print(f"mutual angles: PoE normal/P1-P2 {ang(n_poe, p12):.4f}, CoE/P1-P2 {ang(coe, p12):.4f}, CoE/PoE normal {ang(coe, n_poe):.4f}")
for name, l, b in [("CMB dipole", 264, 48), ("Copi quadrupole normal", 231.7, 63), ("hemispherical asymmetry", 230, -20), ("Bianchi VIIh (Jaffe)", 222, -62), ("Bao & Bao centre", 286, -43)]:
    x = v(l, b)
    print(f"{name:26s} ({l},{b}): to CoE {ang(x, coe):5.1f}, to AoE {hl(x, aoe):5.1f}, off plane of evil {off_plane(x, n_poe):5.1f}, "
          f"off multipole plane (P1-P2) {off_plane(x, p12):5.1f}, off plane normal to published axis {off_plane(x, aoe):5.1f}, to PoE normal {hl(x, n_poe):5.1f}")
for name, (l, b) in [("best-fit CoE", (198, -4)), ("frame CoE", (180, -15)), ("anti-CoE (best fit)", (18, 4))]:
    s = SkyCoord(l=l*u.deg, b=b*u.deg, frame="galactic").icrs
    print(f"{name}: RA {s.ra.to_string(u.hour, precision=0)}, Dec {s.dec.deg:+.1f}; constellation {s.get_constellation()}")
bet = SkyCoord(ra="05h55m10.3s", dec="+07d24m25s")
print(f"best-fit CoE to Betelgeuse {SkyCoord(l=198*u.deg, b=-4*u.deg, frame='galactic').separation(bet).deg:.1f} deg; "
      f"frame CoE to Aldebaran {SkyCoord(l=180*u.deg, b=-15*u.deg, frame='galactic').separation(SkyCoord(ra='04h35m55.2s', dec='+16d30m33s')).deg:.1f} deg")
anti = v(18, 4)   # anti-CoE of the 2-D best fit (198, -4)
for name, (l, b) in [("Migkas MIN (115, 0)", (115, 0)), ("Migkas MIN (274, -22)", (274, -22)), ("Axis of Evil (headless)", (260, 60))]:
    x = v(l, b)
    print(f"{name}: {ang(x, anti):.1f} deg from the anti-CoE (18, +4)" + (f" (other end {ang(-x, anti):.1f})" if "Axis" in name else ""))
for DR in (2, 3.564, 8): print(f"critical circle at D/R = {DR}: {180 - np.degrees(np.arccos(1/(2*DR))):.1f} deg from the anti-CoE")
