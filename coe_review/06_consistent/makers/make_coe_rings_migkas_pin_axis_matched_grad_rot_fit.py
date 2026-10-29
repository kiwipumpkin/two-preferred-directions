#!/usr/bin/env python3
"""
CoE rings + Migkas A(ell) extrema loops in the Galactic frame.

Concept:
  Pure CoE axisymmetry → constant A on small-circle rings about x = CoE↔anti-CoE.
  Scan A vs angle from +x (anti-CoE) around 360°. After-2ℓ A(ℓ) has TWO maxima
  and TWO minima (from best_fit_2ell_multiplier.npz A_after):
    MAX at θ = 30° and 185° from +x,
    MIN at θ = 115° and 265° from +x.

  Exactly FOUR loops — each ONE transparent highlight band, 18° end-to-end
  (= ±9° = 5% of 360°) about the center θ:
    MAX: 30±9 → [21°, 39°] and 185±9 → [176°, 194°]
    MIN: 115±9 → [106°, 124°] and 265±9 → [256°, 274°]
  Drawing: ONE smooth transparent filled ribbon between θ−9° and θ+9°
  + ONE solid mid-stroke at the exact center θ. No multi-stroke wash,
  no dense linspace hairball, no dual strong boundaries.
  Four distinct colors (two reds + two blues). No orange, no 283° companion.

  Galactic orientation: anti-CoE (+x) at ℓ=0°, b=+15°; CoE (−x) at ℓ=180°, b=−15°
  (axis tilted +15° so features near ℓ=0 move north on Mollweide; Migkas markers fixed).
  Sky markers (ℓ, b):
    Maxes: map extrema (17°, −9°) for ~30° max; strip (185°, 0°) for second max.
    Mins:  strip (115°, 0°) for ~115° min; map extrema (274°, −22°) for ~265° min.

Outputs:
  coe_rings_migkas_pin_axis.png
  coe_rings_migkas_pin_axis.json
"""

from __future__ import annotations
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, Patch, Polygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = Path(PKG + "/coe_review/06_consistent/figs/loop_threepanel_matched_grad_rotfit_rot60.png")
JSON_PATH = Path(PKG + "/coe_review/06_consistent/figs/loop_threepanel_matched_grad_rotfit_rot60.json")

# 5% of 360° = 18° end-to-end → half-width ±9° about each center
BAND_HALF = 9.0
BAND_FULL = 2.0 * BAND_HALF  # 18° = 5% of 360°
BAND_TOTAL = BAND_FULL  # alias
STRIP_THETA_MAX = [29.5, 187.5]
STRIP_THETA_MIN = [122.6, 264.4]

MIGKAS_MAXES = [(17.0, -9.0), (185.0, 0.0)]
MIGKAS_MINS = [(115.0, 0.0), (274.0, -22.0)]

C_COE = "#6b3fa0"
C_ANTI = "#8e1b1b"  # dark red marker; distinct from loop reds
# Four distinct loop colors: two reds (MAX) + two blues (MIN); no orange/yellow
C_MAX_30 = "#0097b2"    # cyan — MAX at 29.5°
C_MAX_185 = "#e0107a"   # magenta — MAX at 187.5°
C_MIN_115 = "#6fa800"   # green — MIN at 122.6°
C_MIN_265 = "#2166d4"   # blue — MIN at 264.4°
C_AXIS = "#5b2c6f"
C_EQ = "#1e8449"

# Per-loop color maps (order matches STRIP_THETA_MAX / STRIP_THETA_MIN)
COLORS_MAX = [C_MAX_30, C_MAX_185]
COLORS_MIN = [C_MIN_115, C_MIN_265]


def wrap_lon_deg(lon: np.ndarray) -> np.ndarray:
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def lon_disp_from_ell(ell_deg: np.ndarray) -> np.ndarray:
    """Migkas Galactic: ℓ increases LEFT → lon_disp = −ℓ_wrapped."""
    return -wrap_lon_deg(ell_deg)


def lb_to_n(l_deg: float, b_deg: float) -> np.ndarray:
    l = np.deg2rad(l_deg)
    b = np.deg2rad(b_deg)
    return np.array(
        [np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)], dtype=float
    )


def n_to_lb(n: np.ndarray) -> tuple[float, float]:
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    b = float(np.rad2deg(np.arcsin(np.clip(n[2], -1.0, 1.0))))
    l = float(np.rad2deg(np.arctan2(n[1], n[0])) % 360.0)
    return l, b


def celestial_equator_lb(n: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """Dec=0 (ICRS) as Galactic (ℓ, b). IAU J2000 NGP convention, no astropy."""
    ra_ngp = np.deg2rad(192.85948)
    dec_ngp = np.deg2rad(27.12825)
    l_ncp = np.deg2rad(122.93192)
    ra = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    dec = 0.0
    sin_b = np.sin(dec) * np.sin(dec_ngp) + np.cos(dec) * np.cos(dec_ngp) * np.cos(ra - ra_ngp)
    sin_b = np.clip(sin_b, -1.0, 1.0)
    b = np.arcsin(sin_b)
    y = np.cos(dec) * np.sin(ra - ra_ngp)
    x = np.sin(dec) * np.cos(dec_ngp) - np.cos(dec) * np.sin(dec_ngp) * np.cos(ra - ra_ngp)
    l = (l_ncp - np.arctan2(y, x)) % (2.0 * np.pi)
    return np.rad2deg(l), np.rad2deg(b)


def great_circle_through_poles(D: np.ndarray, n: int = 361) -> tuple[np.ndarray, np.ndarray]:
    """Great circle through ±D (any meridian of the CoE axis)."""
    a = D / np.linalg.norm(D)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(a, tmp)) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(a, tmp)
    e2 /= np.linalg.norm(e2)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    pts = np.outer(np.cos(t), a) + np.outer(np.sin(t), e2)
    ls = np.array([n_to_lb(p)[0] for p in pts])
    bs = np.array([n_to_lb(p)[1] for p in pts])
    return ls, bs


def small_circle_about_axis(axis: np.ndarray, mu: float, n: int = 361) -> np.ndarray:
    """Unit vectors on the small circle n·axis = mu (|mu|<1)."""
    a = axis / np.linalg.norm(axis)
    mu = float(np.clip(mu, -1.0 + 1e-9, 1.0 - 1e-9))
    tmp = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    rho = np.sqrt(max(0.0, 1.0 - mu * mu))
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return (mu * a)[None, :] + rho * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))


def split_lon_segments(lon_rad: np.ndarray, lat_rad: np.ndarray, gap: float = np.pi):
    """Split polyline where Mollweide lon jumps (wrap discontinuity)."""
    d = np.diff(lon_rad)
    cuts = np.where(np.abs(d) > gap)[0] + 1
    idx = np.concatenate([[0], cuts, [len(lon_rad)]])
    segs = []
    for i0, i1 in zip(idx[:-1], idx[1:]):
        if i1 - i0 >= 2:
            segs.append((lon_rad[i0:i1], lat_rad[i0:i1]))
    return segs


def plot_lb_curve_mollweide(ax, l_deg, b_deg, **kwargs):
    lon = np.deg2rad(lon_disp_from_ell(l_deg))
    lat = np.deg2rad(b_deg)
    for xr, yr in split_lon_segments(lon, lat):
        ax.plot(xr, yr, **kwargs)


def band_edges(centers: list[float], half: float = BAND_HALF) -> list[tuple[float, float]]:
    """Return (theta_lo, theta_hi) for each band center (±half about center)."""
    return [(c - half, c + half) for c in centers]


def mu_pair_for_band(lo: float, hi: float) -> tuple[float, float]:
    """Edge μ=cos(θ) for a scan-angle band (two boundary cones only)."""
    return float(np.cos(np.deg2rad(lo))), float(np.cos(np.deg2rad(hi)))


def migkas_scan_stops() -> dict:
    th_max = [float(t) for t in STRIP_THETA_MAX]
    th_min = [float(t) for t in STRIP_THETA_MIN]
    max_bands = band_edges(th_max)
    min_bands = band_edges(th_min)
    return {
        "theta_max_from_plus_x_deg": th_max,
        "theta_min_from_plus_x_deg": th_min,
        "max_bands_deg": max_bands,
        "min_bands_deg": min_bands,
        "band_half_width_deg": BAND_HALF,
        "band_full_width_deg": BAND_FULL,
        "band_total_width_deg": BAND_TOTAL,
        "band_width_note": "5% of 360° = 18° end-to-end = ±9° about each MAX/MIN center",
        "mu_max_centers": [float(np.cos(np.deg2rad(t))) for t in th_max],
        "mu_min_centers": [float(np.cos(np.deg2rad(t))) for t in th_min],
        "mu_max_bands": [mu_pair_for_band(lo, hi) for lo, hi in max_bands],
        "mu_min_bands": [mu_pair_for_band(lo, hi) for lo, hi in min_bands],
        "source": (
            "best_fit_2ell_multiplier.npz A_after extrema (two max, two min); "
            f"each drawn as one transparent {BAND_FULL:.0f}° end-to-end highlight "
            f"(±{BAND_HALF:.0f}° = 5% of 360°) + one solid mid-stroke"
        ),
    }


def compute_geometry(st: dict) -> dict:
    anti_l, anti_b = 0.0, 15.0
    coe_l, coe_b = 180.0, -15.0

    n_coe = lb_to_n(coe_l, coe_b)
    n_anti = lb_to_n(anti_l, anti_b)
    D = n_anti  # +D = anti-CoE = +x

    th_max = [float(t) for t in st["theta_max_from_plus_x_deg"]]
    th_min = [float(t) for t in st["theta_min_from_plus_x_deg"]]
    max_bands = list(st["max_bands_deg"])
    min_bands = list(st["min_bands_deg"])

    n_maxes = [lb_to_n(*lb) for lb in MIGKAS_MAXES]
    n_mins = [lb_to_n(*lb) for lb in MIGKAS_MINS]

    sep = float(
        np.rad2deg(
            np.arccos(np.clip(np.dot(n_maxes[0], n_mins[1]), -1.0, 1.0))
        )
    )

    return {
        "D": D,
        "n_coe": n_coe,
        "n_anti": n_anti,
        "n_maxes": n_maxes,
        "n_mins": n_mins,
        "anti_lb": (anti_l, anti_b),
        "coe_lb": (coe_l, coe_b),
        "theta_max_from_plus_x_deg": th_max,
        "theta_min_from_plus_x_deg": th_min,
        "max_bands_deg": max_bands,
        "min_bands_deg": min_bands,
        "mu_max_bands": list(st["mu_max_bands"]),
        "mu_min_bands": list(st["mu_min_bands"]),
        "sep_deg": sep,
        "band_half_width_deg": st["band_half_width_deg"],
    }


def draw_wire_sphere(ax, n_mer=12, n_par=6, **kwargs):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(-np.pi / 2, np.pi / 2, 40)
    xs = np.outer(np.cos(u), np.cos(v))
    ys = np.outer(np.sin(u), np.cos(v))
    zs = np.outer(np.ones_like(u), np.sin(v))
    ax.plot_surface(xs, ys, zs, color="#dfe6e9", alpha=0.15, linewidth=0, shade=False)
    kw = dict(color="#95a5a6", lw=0.4, alpha=0.45)
    kw.update(kwargs)
    for i in range(n_mer):
        phi = 2 * np.pi * i / n_mer
        t = np.linspace(-np.pi / 2, np.pi / 2, 80)
        ax.plot(np.cos(t) * np.cos(phi), np.cos(t) * np.sin(phi), np.sin(t), **kw)
    for j in range(1, n_par):
        b = -np.pi / 2 + np.pi * j / n_par
        t = np.linspace(0, 2 * np.pi, 80)
        r = np.cos(b)
        ax.plot(r * np.cos(t), r * np.sin(t), np.full_like(t, np.sin(b)), **kw)


def foreshortened_ellipse_xy(mu: float, n: int = 200):
    """Side-view foreshortened ellipse of a 3D ring at n·x = mu."""
    rho = np.sqrt(max(0.0, 1.0 - mu * mu))
    phi = np.linspace(0, 2 * np.pi, n)
    ex = mu + 0.045 * rho * np.cos(phi)
    ey = rho * np.sin(phi)
    return ex, ey, rho



# ---------------------------------------------------------------------------
# Opacity gradient along each loop (Chris 2026-09-26), same method as
# make_coe_rings_gradient_loops.py: loop drawn as runs of short elements with
# per-element alpha = base_alpha * (1 + cos psi)/2, psi = ring azimuth from the
# θ-ray point (ring point most along sign(sin θ_mid)·Ŷ; Ŷ = Galactic +z
# projected ⊥ D, i.e. model-frame "up"; no roll). Darkest at the θ-ray point,
# fully transparent at the diametrically opposite point.
# ---------------------------------------------------------------------------
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def theta_ray_sign(theta_mid_deg):
    return 1.0 if np.sin(np.deg2rad(float(theta_mid_deg))) >= 0 else -1.0


ROLL_BC_DEG = 60.18  # fixed by the Axis of Evil (was 45 / 80.8)


def ring_basis(D):
    a = np.asarray(D, dtype=float)
    a = a / np.linalg.norm(a)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(a, tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    y = tmp - np.dot(tmp, a) * a
    y /= np.linalg.norm(y)
    z = np.cross(a, y)
    z /= np.linalg.norm(z)
    # +60.2° roll (Panels B/C only), same convention as coe_half_loop_geometry.ring_basis
    # / earlier "+60.2° CCW about CoE" figures: Y' = c Y - s Z, Z' = s Y + c Z
    r = np.deg2rad(float(ROLL_BC_DEG))
    c, s_ = np.cos(r), np.sin(r)
    return c * y - s_ * z, s_ * y + c * z


def grad_phi(theta_mid_deg, n):
    """phi from antipode of θ-ray point round to antipode (psi = -180..+180)."""
    phi0 = 0.5 * np.pi if theta_ray_sign(theta_mid_deg) > 0 else 1.5 * np.pi
    return phi0, phi0 + np.linspace(-np.pi, np.pi, n)


def grad_scale(phi, phi0):
    return 0.5 * (1.0 + np.cos(np.asarray(phi) - phi0))


def small_circle_grad(D, mu, theta_mid_deg, n=721):
    a = np.asarray(D, dtype=float)
    a = a / np.linalg.norm(a)
    mu = float(np.clip(mu, -1.0 + 1e-9, 1.0 - 1e-9))
    rho = np.sqrt(max(0.0, 1.0 - mu * mu))
    y, z = ring_basis(a)
    phi0, phi = grad_phi(theta_mid_deg, n)
    pts = (mu * a)[None, :] + rho * (np.outer(np.sin(phi), y) + np.outer(np.cos(phi), z))
    return pts, grad_scale(phi, phi0)


def segment_scale(sc):
    sc = np.asarray(sc, dtype=float)
    return 0.5 * (sc[:-1] + sc[1:])


def rgba_array(color, alphas):
    r, g, b, _ = to_rgba(color)
    al = np.clip(np.asarray(alphas, dtype=float), 0.0, 1.0)
    out = np.empty((al.size, 4))
    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = r, g, b, al
    return out


def quantized_runs(seg_sc, valid=None, n_levels=64):
    s = np.asarray(seg_sc, dtype=float)
    lev = np.round(np.clip(s, 0.0, 1.0) * n_levels) / n_levels
    ok = lev > 0 if valid is None else (lev > 0) & np.asarray(valid, dtype=bool)
    runs, i, n = [], 0, s.size
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i + 1
        while j < n and ok[j] and lev[j] == lev[i]:
            j += 1
        runs.append((i, j, float(lev[i])))
        i = j
    return runs


def _strip_polys_2d(xa, ya, xb, yb, runs):
    return [
        np.column_stack([
            np.concatenate([xa[i0:i1 + 1], xb[i0:i1 + 1][::-1]]),
            np.concatenate([ya[i0:i1 + 1], yb[i0:i1 + 1][::-1]]),
        ])
        for i0, i1, _ in runs
    ]


def _polylines_2d(x, y, runs):
    return [np.column_stack([x[i0:i1 + 1], y[i0:i1 + 1]]) for i0, i1, _ in runs]


def draw_panelA_ribbon(ax, theta_lo: float, theta_hi: float, theta_mid: float, color: str):
    """
    Panel-A highlight (edge-on chord): same vertical strip between
    x=cos(θ_hi)..cos(θ_lo) clipped to the unit circle, base alpha 0.28, and the
    same foreshortened mid-ellipse (lw 2.8, alpha 0.95) — now with the opacity
    gradient along the chord: alpha *= (1 + sgn·y/ρ(x))/2 = (1+cos psi)/2,
    darkest at the θ-ray end, transparent at the opposite end.
    """
    mu_lo = float(np.cos(np.deg2rad(theta_lo)))
    mu_hi = float(np.cos(np.deg2rad(theta_hi)))
    x0, x1 = (min(mu_lo, mu_hi), max(mu_lo, mu_hi))
    sgn = theta_ray_sign(theta_mid)
    nx, ny = 64, 1200
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys)
    rho = np.sqrt(np.clip(1.0 - X * X, 1e-6, None))
    sc = np.clip(0.5 * (1.0 + sgn * Y / rho), 0.0, 1.0)
    img = np.zeros((ny, nx, 4))
    img[..., :3] = to_rgba(color)[:3]
    img[..., 3] = 0.28 * sc
    im = ax.imshow(img, extent=(x0, x1, -1.0, 1.0), origin="lower",
                   interpolation="bilinear", zorder=2, aspect="auto")
    im.set_clip_path(Circle((0.0, 0.0), 1.0, transform=ax.transData))

    mu_mid = float(np.cos(np.deg2rad(theta_mid)))
    rho_m = np.sqrt(max(0.0, 1.0 - mu_mid * mu_mid))
    phi0, phi = grad_phi(theta_mid, 721)
    ex_m = mu_mid + 0.045 * rho_m * np.cos(phi)
    ey_m = rho_m * np.sin(phi)
    runs = quantized_runs(segment_scale(grad_scale(phi, phi0)))
    ax.add_collection(
        LineCollection(
            _polylines_2d(ex_m, ey_m, runs),
            colors=rgba_array(color, [0.95 * r[2] for r in runs]),
            linewidths=2.8, zorder=4, capstyle="butt", joinstyle="round",
        )
    )


def draw_panelA_theta_rays(ax, geo):
    """
    Illustrate θ: from the observer (origin), one ray per loop out to the
    ring at that loop's center θ. θ is measured from +x (anti-CoE, right)
    up to the ring. Emphasize MAX θ=30° (thick semi-transparent red to the
    top of that foreshortened ring), matching Chris's markup.
    """
    items = list(zip(geo["theta_max_from_plus_x_deg"], COLORS_MAX)) + list(
        zip(geo["theta_min_from_plus_x_deg"], COLORS_MIN)
    )
    for mid, color in items:
        t = float(np.deg2rad(mid))
        # Endpoint = top/rim point of the ring in side view = (cos θ, sin θ)
        x1, y1 = float(np.cos(t)), float(np.sin(t))
        is_max30 = abs(mid - 29.5) < 0.5
        lw = 3.2 if is_max30 else 1.8
        alpha = 0.70 if is_max30 else 0.45
        ax.plot(
            [0.0, x1], [0.0, y1],
            color=color, lw=lw, alpha=alpha, zorder=3,
            solid_capstyle="round",
        )
        # Small tick at the ring hit
        ax.plot([x1], [y1], "o", color=color, ms=4.5 if is_max30 else 3.5,
                alpha=0.9, zorder=5, markeredgecolor="none")

    # Angle arc at θ=30° from +x axis up to the ray (makes the measure obvious)
    arc_r = 0.38
    phi = np.linspace(0.0, np.deg2rad(29.5), 60)
    ax.plot(
        arc_r * np.cos(phi), arc_r * np.sin(phi),
        color=C_MAX_30, lw=1.6, alpha=0.85, zorder=3,
    )
    ax.text(
        arc_r * np.cos(np.deg2rad(15.0)) + 0.06,
        arc_r * np.sin(np.deg2rad(15.0)) + 0.04,
        r"$\theta$",
        color=C_MAX_30, fontsize=10, fontweight="bold",
        ha="left", va="bottom", zorder=6,
    )


def panel_A_model_frame(ax, geo):
    """Side view: exactly four thick foreshortened loop ribbons about x."""
    ax.set_title(
        "A. Model frame (no Galactic yet)\n"
        r"axisymmetric about $x$ = CoE$\leftrightarrow$anti-CoE"
        "\n"
        r"two MAX + two MIN, each $18^\circ$ end-to-end ($\pm9^\circ$)",
        fontsize=10,
        pad=8,
    )
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="#7f8c8d", lw=1.2, zorder=1)
    ax.plot(0, 0, "ko", ms=6, zorder=5)
    ax.text(0.08, -0.12, "observer", fontsize=8, color="k", ha="left")

    ax.annotate(
        "",
        xy=(1.15, 0),
        xytext=(-1.15, 0),
        arrowprops=dict(arrowstyle="<->", color=C_AXIS, lw=2.5),
        zorder=4,
    )
    ax.text(1.18, 0.08, "anti-CoE\n(+x)", color=C_ANTI, fontsize=8, ha="left", va="bottom")
    ax.text(-1.18, 0.08, "CoE\n(−x)", color=C_COE, fontsize=8, ha="right", va="bottom")

    for (lo, hi), mid, color in zip(
        geo["max_bands_deg"], geo["theta_max_from_plus_x_deg"], COLORS_MAX
    ):
        draw_panelA_ribbon(ax, lo, hi, mid, color)
    for (lo, hi), mid, color in zip(
        geo["min_bands_deg"], geo["theta_min_from_plus_x_deg"], COLORS_MIN
    ):
        draw_panelA_ribbon(ax, lo, hi, mid, color)

    # Illustrate θ from +x (anti-CoE) to each loop (Chris markup)
    draw_panelA_theta_rays(ax, geo)

    label_r = 1.30
    for mid, color in list(zip(geo["theta_max_from_plus_x_deg"], COLORS_MAX)) + list(
        zip(geo["theta_min_from_plus_x_deg"], COLORS_MIN)
    ):
        t = np.deg2rad(mid)
        ax.text(
            label_r * np.cos(t),
            label_r * np.sin(t),
            rf"$\theta\!=\!{mid:g}^\circ$",
            color=color, fontsize=7.5, ha="center", va="center", fontweight="bold",
        )

    ax.text(
        0.0,
        -1.38,
        (
            r"four transparent highlight bands about $x$  ·  each $18^\circ$ end-to-end ($\pm9^\circ$ = 5% of $360^\circ$)"
            + "\n"
            + r"(after-2$\ell$ $A(\ell)$: two MAX + two MIN; one mid-stroke each)"
            + "\n"
            + r"$y,z$ free (roll) — model does not pick celestial equator"
        ),
        fontsize=7.5,
        ha="center",
        va="top",
        style="italic",
        color="#34495e",
    )
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")


def _ring_lb(D, mu, n_pts=721):
    pts = small_circle_about_axis(D, mu, n=n_pts)
    rl = np.array([n_to_lb(p)[0] for p in pts])
    rb = np.array([n_to_lb(p)[1] for p in pts])
    return rl, rb


def _ring_lonlat_rad(D, mu, n_pts=361):
    """Small-circle → Mollweide display (lon_disp, lat) in radians."""
    pts = small_circle_about_axis(D, float(mu), n=n_pts)
    ls = np.array([n_to_lb(p)[0] for p in pts])
    bs = np.array([n_to_lb(p)[1] for p in pts])
    lon = np.deg2rad(lon_disp_from_ell(ls))
    lat = np.deg2rad(bs)
    return lon, lat


def fill_mollweide_ribbon(ax, D, theta_lo, theta_hi, theta_mid, color, n_pts=721):
    """
    Mollweide highlight band (±9° edges, base alpha 0.26) + mid-stroke
    (lw 2.6, alpha 0.95), both with the (1+cos psi)/2 opacity gradient.
    Elements crossing the antimeridian seam are dropped.
    """
    mu_a, mu_b = mu_pair_for_band(theta_lo, theta_hi)
    mu_mid = float(np.cos(np.deg2rad(theta_mid)))

    def lonlat(mu, n):
        pts, sc = small_circle_grad(D, mu, theta_mid, n=n)
        ls = np.array([n_to_lb(p)[0] for p in pts])
        bs = np.array([n_to_lb(p)[1] for p in pts])
        return np.deg2rad(lon_disp_from_ell(ls)), np.deg2rad(bs), sc

    lon_a, lat_a, sc = lonlat(mu_a, n_pts)
    lon_b, lat_b, _ = lonlat(mu_b, n_pts)
    gap = 0.9 * np.pi
    quad = np.stack([lon_a[:-1], lon_a[1:], lon_b[1:], lon_b[:-1]])
    valid = (quad.max(axis=0) - quad.min(axis=0)) <= gap
    runs = quantized_runs(segment_scale(sc), valid)
    if runs:
        ax.add_collection(
            PolyCollection(
                _strip_polys_2d(lon_a, lat_a, lon_b, lat_b, runs),
                facecolors=rgba_array(color, [0.26 * r[2] for r in runs]),
                edgecolors="none", zorder=2, linewidths=0,
            )
        )
    lon_m, lat_m, sc_m = lonlat(mu_mid, max(n_pts, 721))
    valid = np.abs(np.diff(lon_m)) <= gap
    runs = quantized_runs(segment_scale(sc_m), valid)
    if runs:
        ax.add_collection(
            LineCollection(
                _polylines_2d(lon_m, lat_m, runs),
                colors=rgba_array(color, [0.95 * r[2] for r in runs]),
                linewidths=2.6, zorder=4, capstyle="butt", joinstyle="round",
            )
        )


def panel_B_mollweide(ax, geo):
    ax.set_title(
        r"B. Migkas $A(\ell)$ loops on sky (anti-CoE at $(0^\circ,+15^\circ)$)"
        "\n"
        r"(Galactic Mollweide; CoE axis tilted $+15^\circ$; $\ell$ increases LEFT; "
        r"two MAX + two MIN, each $18^\circ$ end-to-end ($\pm9^\circ$))",
        fontsize=10,
        pad=8,
    )
    for b in [-60, -30, 0, 30, 60]:
        ell = np.linspace(0, 360, 361)
        plot_lb_curve_mollweide(ax, ell, np.full_like(ell, b), color="#bdc3c7", lw=0.5, zorder=0)
    for ell0 in range(0, 360, 30):
        b = np.linspace(-90, 90, 181)
        plot_lb_curve_mollweide(ax, np.full_like(b, ell0), b, color="#bdc3c7", lw=0.5, zorder=0)

    D = geo["D"]
    ls, bs = great_circle_through_poles(D, n=721)
    plot_lb_curve_mollweide(ax, ls, bs, color=C_AXIS, lw=2.0, zorder=2)

    for (lo, hi), mid, color in zip(
        geo["max_bands_deg"], geo["theta_max_from_plus_x_deg"], COLORS_MAX
    ):
        fill_mollweide_ribbon(ax, D, lo, hi, mid, color)
    for (lo, hi), mid, color in zip(
        geo["min_bands_deg"], geo["theta_min_from_plus_x_deg"], COLORS_MIN
    ):
        fill_mollweide_ribbon(ax, D, lo, hi, mid, color)

    ell_eq, b_eq = celestial_equator_lb(721)
    plot_lb_curve_mollweide(ax, ell_eq, b_eq, color=C_EQ, lw=1.8, ls="--", zorder=2)

    ax.plot(
        np.deg2rad(lon_disp_from_ell(geo["anti_lb"][0])),
        np.deg2rad(geo["anti_lb"][1]),
        marker="o", ms=8, color=C_ANTI,
        markerfacecolor=C_ANTI, markeredgecolor="k", markeredgewidth=0.6,
        zorder=6, linestyle="None",
    )
    ax.plot(
        np.deg2rad(lon_disp_from_ell(geo["coe_lb"][0])),
        np.deg2rad(geo["coe_lb"][1]),
        marker="*", ms=14, color=C_COE,
        markerfacecolor=C_COE, markeredgecolor="k", markeredgewidth=0.6,
        zorder=6, linestyle="None",
    )
    for (l, b), color in zip(MIGKAS_MAXES, COLORS_MAX):
        ax.plot(
            np.deg2rad(lon_disp_from_ell(l)),
            np.deg2rad(b),
            marker="^",
            ms=10,
            color=color,
            markeredgecolor="k",
            markeredgewidth=0.7,
            zorder=7,
            linestyle="None",
        )
    for (l, b), color in zip(MIGKAS_MINS, COLORS_MIN):
        ax.plot(
            np.deg2rad(lon_disp_from_ell(l)),
            np.deg2rad(b),
            marker="s",
            ms=10,
            color=color,
            markeredgecolor="k",
            markeredgewidth=0.7,
            zorder=7,
            linestyle="None",
        )

    xticks = np.deg2rad(lon_disp_from_ell(np.array([180, 90, 0, 270])))
    ax.set_xticks(xticks)
    ax.set_xticklabels([r"$180^\circ$", r"$90^\circ$", r"$0^\circ$", r"$270^\circ$"], fontsize=8)
    ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
    ax.set_yticklabels([r"$-60^\circ$", r"$-30^\circ$", r"$0^\circ$", r"$+30^\circ$", r"$+60^\circ$"], fontsize=8)
    ax.grid(False)
    ax.set_xlabel(r"Galactic $\ell$ (increases LEFT)", fontsize=8)
    ax.set_ylabel(r"Galactic $b$", fontsize=8)


def fill_sphere_ribbon(ax, D, theta_lo, theta_hi, theta_mid, color, n_pts=160):
    """
    3D highlight belt (±9° edges, base alpha 0.25) + mid-stroke (lw 2.6,
    alpha 0.95), both with the (1+cos psi)/2 opacity gradient.
    """
    mu_a, mu_b = mu_pair_for_band(theta_lo, theta_hi)
    mu_mid = float(np.cos(np.deg2rad(theta_mid)))
    n_g = 361
    p0, sc = small_circle_grad(D, float(mu_a), theta_mid, n=n_g)
    p1, _ = small_circle_grad(D, float(mu_b), theta_mid, n=n_g)
    seg = segment_scale(sc)
    idx = np.where(seg > 1e-4)[0]
    verts = [[p0[j], p0[j + 1], p1[j + 1], p1[j]] for j in idx]
    coll = Poly3DCollection(
        verts, facecolors=rgba_array(color, 0.25 * seg[idx]),
        edgecolors="none", linewidths=0.0, zorder=2,
    )
    ax.add_collection3d(coll)
    pts, scm = small_circle_grad(D, mu_mid, theta_mid, n=n_g)
    runs = quantized_runs(segment_scale(scm))
    ax.add_collection3d(
        Line3DCollection(
            [pts[i0:i1 + 1] for i0, i1, _ in runs],
            colors=rgba_array(color, [0.95 * r[2] for r in runs]),
            linewidths=2.6, zorder=5, capstyle="butt", joinstyle="round",
        )
    )


def panel_C_sphere(ax, geo):
    th_max = geo["theta_max_from_plus_x_deg"]
    th_min = geo["theta_min_from_plus_x_deg"]
    ax.set_title(
        r"C. Sphere 3D — two MAX + two MIN, each $18^\circ$ end-to-end ($\pm9^\circ$)"
        "\n"
        r"(axis line emphasized; celestial equator dashed)",
        fontsize=10,
        pad=6,
    )
    draw_wire_sphere(ax)
    D = geo["D"]
    ell_eq, b_eq = celestial_equator_lb(400)
    eq_xyz = np.array([lb_to_n(l, b) for l, b in zip(ell_eq, b_eq)])
    ax.plot(eq_xyz[:, 0], eq_xyz[:, 1], eq_xyz[:, 2], color=C_EQ, lw=1.8, ls="--", zorder=3)
    ax.plot(
        [-D[0], D[0]],
        [-D[1], D[1]],
        [-D[2], D[2]],
        color=C_AXIS,
        lw=3.0,
        zorder=5,
    )
    ax.scatter(*(-D), color=C_COE, s=120, marker="*", edgecolors="k", linewidths=0.6,
               depthshade=False, zorder=8)
    ax.scatter(*(D), color=C_ANTI, s=80, edgecolors="k", linewidths=0.6, depthshade=False, zorder=8)
    ax.text(*(-D * 1.22), "CoE", color=C_COE, fontsize=8, fontweight="bold")
    ax.text(*(D * 1.18), "anti-CoE", color=C_ANTI, fontsize=8, fontweight="bold")
    ax.scatter([0], [0], [0], color="k", s=25, depthshade=False, zorder=9)

    for (lo, hi), mid, color in zip(
        geo["max_bands_deg"], geo["theta_max_from_plus_x_deg"], COLORS_MAX
    ):
        fill_sphere_ribbon(ax, D, lo, hi, mid, color)
    for (lo, hi), mid, color in zip(
        geo["min_bands_deg"], geo["theta_min_from_plus_x_deg"], COLORS_MIN
    ):
        fill_sphere_ribbon(ax, D, lo, hi, mid, color)

    for n, color in zip(geo["n_maxes"], COLORS_MAX):
        ax.scatter(
            [n[0]], [n[1]], [n[2]],
            c=[color], marker="^", s=70,
            edgecolors="k", linewidths=0.7, depthshade=False, zorder=10,
        )
        ax.plot([0, n[0]], [0, n[1]], [0, n[2]], color=color, lw=0.8, ls=":", alpha=0.7)
    for n, color in zip(geo["n_mins"], COLORS_MIN):
        ax.scatter(
            [n[0]], [n[1]], [n[2]],
            c=[color], marker="s", s=60,
            edgecolors="k", linewidths=0.7, depthshade=False, zorder=10,
        )
        ax.plot([0, n[0]], [0, n[1]], [0, n[2]], color=color, lw=0.8, ls=":", alpha=0.7)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_zlim(-1.25, 1.25)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-55)
    ax.set_axis_off()
    ax.text2D(
        0.5,
        -0.02,
        rf"MAX $\theta\!=\!{th_max[0]:g}^\circ,{th_max[1]:g}^\circ$; "
        rf"MIN $\theta\!=\!{th_min[0]:g}^\circ,{th_min[1]:g}^\circ$  "
        rf"($18^\circ$ end-to-end = $\pm9^\circ$ = 5% of $360^\circ$)",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.5,
        color="#34495e",
    )


def main():
    st = migkas_scan_stops()
    geo = compute_geometry(st)

    th_max = geo["theta_max_from_plus_x_deg"]
    th_min = geo["theta_min_from_plus_x_deg"]
    max_bands = geo["max_bands_deg"]
    min_bands = geo["min_bands_deg"]

    fig = plt.figure(figsize=(14.5, 5.6), dpi=160)
    fig.suptitle(
        r"Pure CoE axisymmetry: two MAX + two MIN highlight bands  (gradient: darkest at $\theta$ ray, transparent opposite; +45$^\circ$ CCW about CoE axis (B/C))"
        "\n"
        rf"(each $18^\circ$ end-to-end = $\pm9^\circ$ = 5% of $360^\circ$; "
        rf"MAX $\theta\!=\!{th_max[0]:g}^\circ/{th_max[1]:g}^\circ$; "
        rf"MIN $\theta\!=\!{th_min[0]:g}^\circ/{th_min[1]:g}^\circ$; "
        r"anti-CoE $(0^\circ,+15^\circ)$, CoE axis tilted $+15^\circ$)",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    axA = fig.add_subplot(1, 3, 1)
    panel_A_model_frame(axA, geo)

    axB = fig.add_subplot(1, 3, 2, projection="mollweide")
    panel_B_mollweide(axB, geo)

    axC = fig.add_subplot(1, 3, 3, projection="3d")
    panel_C_sphere(axC, geo)

    handles = [
        Line2D([0], [0], color=C_AXIS, lw=2.5, label=r"CoE$\leftrightarrow$anti-CoE axis"),
        Patch(
            facecolor=C_MAX_30, alpha=0.45, edgecolor=C_MAX_30,
            label=(
                rf"MAX $\theta\!=\!{th_max[0]:g}^\circ$ band "
                rf"[{max_bands[0][0]:g},{max_bands[0][1]:g}]$^\circ$ "
                rf"($18^\circ$=±$9^\circ$=5% of $360^\circ$)"
            ),
        ),
        Patch(
            facecolor=C_MAX_185, alpha=0.45, edgecolor=C_MAX_185,
            label=(
                rf"MAX $\theta\!=\!{th_max[1]:g}^\circ$ band "
                rf"[{max_bands[1][0]:g},{max_bands[1][1]:g}]$^\circ$ "
                rf"($18^\circ$=±$9^\circ$)"
            ),
        ),
        Patch(
            facecolor=C_MIN_115, alpha=0.45, edgecolor=C_MIN_115,
            label=(
                rf"MIN $\theta\!=\!{th_min[0]:g}^\circ$ band "
                rf"[{min_bands[0][0]:g},{min_bands[0][1]:g}]$^\circ$ "
                rf"($18^\circ$=±$9^\circ$)"
            ),
        ),
        Patch(
            facecolor=C_MIN_265, alpha=0.45, edgecolor=C_MIN_265,
            label=(
                rf"MIN $\theta\!=\!{th_min[1]:g}^\circ$ band "
                rf"[{min_bands[1][0]:g},{min_bands[1][1]:g}]$^\circ$ "
                rf"($18^\circ$=±$9^\circ$)"
            ),
        ),
        Line2D([0], [0], color=C_EQ, lw=1.8, ls="--", label="celestial equator (Dec$=0$)"),
        Line2D(
            [0], [0], marker="^", color="w", markerfacecolor=C_MAX_30, markeredgecolor="k",
            ms=9,
            label=rf"MAX $({MIGKAS_MAXES[0][0]:.0f}^\circ,{MIGKAS_MAXES[0][1]:+.0f}^\circ)$",
        ),
        Line2D(
            [0], [0], marker="^", color="w", markerfacecolor=C_MAX_185, markeredgecolor="k",
            ms=9,
            label=rf"MAX $({MIGKAS_MAXES[1][0]:.0f}^\circ,{MIGKAS_MAXES[1][1]:+.0f}^\circ)$",
        ),
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=C_MIN_115, markeredgecolor="k",
            ms=8,
            label=rf"MIN $({MIGKAS_MINS[0][0]:.0f}^\circ,{MIGKAS_MINS[0][1]:+.0f}^\circ)$",
        ),
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=C_MIN_265, markeredgecolor="k",
            ms=8,
            label=rf"MIN $({MIGKAS_MINS[1][0]:.0f}^\circ,{MIGKAS_MINS[1][1]:+.0f}^\circ)$",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=C_ANTI, markeredgecolor="k",
            ms=8, label=rf"anti-CoE $({geo['anti_lb'][0]:.1f}^\circ,{geo['anti_lb'][1]:+.1f}^\circ)$",
        ),
        Line2D(
            [0], [0], marker="*", color="w", markerfacecolor=C_COE, markeredgecolor="k",
            ms=11, label=rf"CoE $({geo['coe_lb'][0]:.1f}^\circ,{geo['coe_lb'][1]:+.1f}^\circ)$",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=7.0,
        frameon=True,
        fancybox=True,
        bbox_to_anchor=(0.5, -0.10),
    )

    fig.tight_layout()
    # --- Chris 2026-09-26: panel C sphere drawn at the same on-page diameter as
    # panel A's model-frame circle; panel C axes matched to A's vertical
    # position/height. Nothing else changes.
    from mpl_toolkits.mplot3d import proj3d
    pa, pc = axA.get_position(), axC.get_position()
    axC.set_position([pc.x0, pa.y0, pc.width, pa.height])

    def _diam_A():
        fig.canvas.draw()
        p0 = axA.transData.transform((-1.0, 0.0)); p1 = axA.transData.transform((1.0, 0.0))
        return float(p1[0] - p0[0])

    def _diam_C():
        fig.canvas.draw()
        u = np.random.default_rng(0).normal(size=(20000, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        xs, ys, _ = proj3d.proj_transform(u[:, 0], u[:, 1], u[:, 2], axC.get_proj())
        d = axC.transData.transform(np.column_stack([xs, ys]))
        return float(d[:, 0].max() - d[:, 0].min()), float(d[:, 1].max() - d[:, 1].min())

    zoom = 1.0
    for _ in range(8):
        dA = _diam_A(); dCx, dCy = _diam_C()
        zoom *= dA / dCx
        axC.set_box_aspect((1, 1, 1), zoom=zoom)
    dA = _diam_A(); dCx, dCy = _diam_C()
    print(f"PANEL_DIAM A={dA:.1f}px C={dCx:.1f}x{dCy:.1f}px zoom={zoom:.3f}")
    fig.savefig(PNG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    payload = {
        "png": str(PNG_PATH),
        "script": str(OUT_DIR / "make_coe_rings_migkas_pin_axis_matched_grad_rot_fit.py"),
        "migkas_max_lb_deg": [list(lb) for lb in MIGKAS_MAXES],
        "migkas_min_lb_deg": [list(lb) for lb in MIGKAS_MINS],
        "strip_theta_max_deg": list(STRIP_THETA_MAX),
        "strip_theta_min_deg": list(STRIP_THETA_MIN),
        "band_half_width_deg": BAND_HALF,
        "band_full_width_deg": BAND_FULL,
        "band_total_width_deg": BAND_TOTAL,
        "band_width_note": "5% of 360° = 18° end-to-end = ±9° about each MAX/MIN center",
        "max_bands_deg": [list(b) for b in max_bands],
        "min_bands_deg": [list(b) for b in min_bands],
        "band_intervals": {
            "MAX_30": [float(max_bands[0][0]), float(max_bands[0][1])],
            "MAX_185": [float(max_bands[1][0]), float(max_bands[1][1])],
            "MIN_115": [float(min_bands[0][0]), float(min_bands[0][1])],
            "MIN_265": [float(min_bands[1][0]), float(min_bands[1][1])],
        },
        "scan_stop_source": st["source"],
        "theta_max_from_plus_x_deg": [round(t, 4) for t in th_max],
        "theta_min_from_plus_x_deg": [round(t, 4) for t in th_min],
        "mu_max_bands": [[round(a, 8), round(b, 8)] for a, b in geo["mu_max_bands"]],
        "mu_min_bands": [[round(a, 8), round(b, 8)] for a, b in geo["mu_min_bands"]],
        "CoE_lb_deg": [round(geo["coe_lb"][0], 4), round(geo["coe_lb"][1], 4)],
        "anti_CoE_lb_deg": [round(geo["anti_lb"][0], 4), round(geo["anti_lb"][1], 4)],
        "D_hat_galactic_xyz": geo["D"].tolist(),
        "angular_sep_migkas_max0_min1_deg": round(geo["sep_deg"], 6),
        "axis_tilt_b_deg": 15,
        "axis_tilt_reason": (
            "Chris 2026-09-22 — markers fixed, rings rotate up 30°"
        ),
        "orientation": (
            "Galactic orientation: anti-CoE at (ell=0°, b=+15°), CoE at "
            "(ell=180°, b=−15°); +D = anti-CoE = +x; axis tilted +15° "
            "(Chris 2026-09-22 — markers fixed, rings rotate up 30°)"
        ),
        "drawing": (
            "Exactly 4 features. Each = ONE smooth transparent highlight band "
            "between θ−9° and θ+9° (18° end-to-end = 5% of 360°) + ONE solid "
            "mid-stroke at the exact center θ. No multi-stroke wash / hairball. "
            "Four distinct colors (two reds + two blues); no orange; no 283° "
            "companion. Panel A: clean vertical strip clipped to unit circle; theta rays from observer to each loop (emphasized at MAX 30°). Mollweide "
            "seam-split honestly. Sphere: single Poly3D strip (no n_wash stack)."
        ),
        "note": (
            "Stops are after-2ℓ A(ℓ) extrema from best_fit_2ell_multiplier.npz "
            f"A_after: two MAX at θ={th_max[0]:g}°/{th_max[1]:g}° and "
            f"two MIN at θ={th_min[0]:g}°/{th_min[1]:g}°, each drawn as "
            f"one {BAND_FULL:.0f}° end-to-end highlight (±{BAND_HALF:.0f}° = 5% of 360°). "
            "Rings use μ=cos(θ); sky markers are map extrema (17°,−9°) and "
            "(274°,−22°) plus strip stand-ins (185°,0°) and (115°,0°)."
        ),
        "plot_convention": "Migkas Galactic Mollweide: lon_disp=-ell_wrapped (ell increases LEFT)",
        "legend_summary": (
            "two MAX + two MIN highlight bands, each 18° end-to-end "
            "(±9° = 5% of 360°); four distinct colors"
        ),
        "colors": {
            "MAX_30": C_MAX_30,
            "MAX_185": C_MAX_185,
            "MIN_115": C_MIN_115,
            "MIN_265": C_MIN_265,
            "no_orange": True,
        },
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {PNG_PATH}")
    print(f"Wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
