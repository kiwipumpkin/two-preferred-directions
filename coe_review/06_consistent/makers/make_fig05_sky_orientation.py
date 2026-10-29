#!/usr/bin/env python3
"""Fig05 — Four-row sky orientation (untilted → tilt → gradient → roll).

Letter-matching rebuild from Chris's make_coe_rings_* three-panel scripts,
composed with make_stack_clean-style row crops + row labels.

NOT the simplified coe-figures analytic remake (rejected by Chris).

Gold target: ~1370×1420 (letter PDF embed / money_shot_four_row_only.jpg).
"""
from __future__ import annotations
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root

import importlib.util
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import OUT  # noqa: E402

OUT_NAME = "Fig05_sky_orientation_four_row_rot60.jpg"
FIG_DIR = Path(PKG + "/workspace/cosmology/figures")
PANEL_DIR = Path(PKG + "/coe_review/06_consistent/figs")  # panels made by ../makers/make_coe_rings_*.py (roll 60.18)

# Chris letter-era row generators (in order)
RING_SCRIPTS = [
    "make_coe_rings_notilt.py",
    "make_coe_rings_migkas_pin_axis_matched.py",
    "make_coe_rings_migkas_pin_axis_matched_grad.py",
    "make_coe_rings_migkas_pin_axis_matched_grad_rot_fit.py",
]

# PNG outputs written by those scripts (see PNG_PATH inside each)
PANEL_PNGS = [
    PANEL_DIR / "loop_threepanel_notilt.png",
    PANEL_DIR / "loop_threepanel_matched.png",
    PANEL_DIR / "loop_threepanel_matched_grad.png",
    PANEL_DIR / "loop_threepanel_matched_grad_rotfit_rot60.png",
]

LAB = [
    "Untilted, no roll",
    "+15° tilt",
    "+15° tilt, gradient",
    "+15° tilt, gradient, +60.2° roll (fixed by the Axis of Evil)",
]

# make_stack_clean.py row_content crops (on 1700-wide web images)
WEB_W = 1700
LH = 32          # row-label band height (tuned so final @1370 → 1420)
GAP = 2
ROW0_CROP = (0, 110, WEB_W, 628)   # keep A/B/C headings + footnotes
ROWI_CROP = (0, 222, WEB_W, 572)   # drop suptitle / A-C headings / footnotes / legend
B_WHITEN = (385, 12, 1212, 57)     # wipe residual B heading inside ROWI crop
LEGEND_CROP = (0, 712, WEB_W, 768)
TARGET_W = 1370


def _load_module(script_name: str):
    path = FIG_DIR / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    # Ensure figures/ is on path for sibling imports inside the scripts
    if str(FIG_DIR) not in sys.path:
        sys.path.insert(0, str(FIG_DIR))
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def regenerate_panels(force: bool = True) -> list[Path]:
    """Run Chris's four ring makers; return PNG paths."""
    outs: list[Path] = []
    for script, png in zip(RING_SCRIPTS, PANEL_PNGS):
        if (not force) and png.exists():
            print(f"reuse {png.name}")
            outs.append(png)
            continue
        print(f"running {script} …")
        mod = _load_module(script)
        mod.main()
        if not png.exists():
            raise FileNotFoundError(f"{script} did not write {png}")
        print(f"  wrote {png} {png.stat().st_size}")
        outs.append(png)
    return outs


def _to_web(png: Path) -> Image.Image:
    im = Image.open(png).convert("RGB")
    h = round(im.height * WEB_W / im.width)
    return im.resize((WEB_W, h), Image.LANCZOS)


def _row_content(i: int, web: Image.Image) -> Image.Image:
    if i == 0:
        return web.crop(ROW0_CROP)
    c = web.crop(ROWI_CROP).copy()
    ImageDraw.Draw(c).rectangle(B_WHITEN, fill="white")
    return c


def compose_four_row(panel_pngs: list[Path]) -> Image.Image:
    """Stack four three-panel rows with labels + one shared legend (letter layout)."""
    webs = [_to_web(p) for p in panel_pngs]
    for w in webs:
        if w.size != (WEB_W, 772):
            # Accept near-miss heights from dpi jitter; crops assume ~772
            print(f"note: web size {w.size} (expected ({WEB_W}, 772))")

    blocks = [_row_content(i, webs[i]) for i in range(4)]
    legend = webs[3].crop(LEGEND_CROP)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
    except OSError:
        font = ImageFont.load_default()

    H = sum(LH + b.height + GAP for b in blocks) + legend.height + GAP
    canvas = Image.new("RGB", (WEB_W, H), "white")
    dr = ImageDraw.Draw(canvas)
    y = 0
    for i, rc in enumerate(blocks):
        dr.text((WEB_W // 2, y + LH // 2), LAB[i], font=font, fill="black", anchor="mm")
        y += LH
        canvas.paste(rc, (0, y))
        y += rc.height + GAP
    canvas.paste(legend, (0, y))
    y += legend.height + GAP
    canvas = canvas.crop((0, 0, WEB_W, y))

    out_h = round(canvas.height * TARGET_W / canvas.width)
    return canvas.resize((TARGET_W, out_h), Image.LANCZOS)


def main(force_regen: bool = True):
    OUT.mkdir(parents=True, exist_ok=True)
    panels = regenerate_panels(force=False)
    composed = compose_four_row(panels)

    png_sib = OUT / "_Fig05_sky_orientation_four_row.png"
    composed.save(png_sib)
    dest = OUT / OUT_NAME
    composed.convert("RGB").save(dest, quality=92)
    print("wrote", dest, dest.stat().st_size, composed.size)
    print("wrote", png_sib, png_sib.stat().st_size)
    return dest


if __name__ == "__main__":
    force = "--reuse-panels" not in sys.argv
    main(force_regen=force)
