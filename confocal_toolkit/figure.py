"""Build an editable per-eye PDF figure from the exported panels.

Layout mirrors on eye (S-opsin = left, M-opsin = right):
  left eye : [ D1 D2 D3 / V1 V2 V3 ] [ low-mag overlay ] [ compass T..N ]
  right eye: [ compass N..T ] [ low-mag overlay ] [ D1 D2 D3 / V1 V2 V3 ]

The high-power D/V crops are embedded at full resolution (for figure making),
text/vectors stay editable (pdf.fonttype 42 -> opens in Illustrator with live
text). The retina image pixels are never flipped; only the layout and the
nasal/temporal labels mirror between eyes.
"""

from __future__ import annotations

import numpy as np

PANEL_ORDER = ["D1", "D2", "D3", "V1", "V2", "V3"]


def _mpl():
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42   # editable text in Illustrator
    matplotlib.rcParams["ps.fonttype"] = 42
    return plt


def _scalebar(ax, img_w_px, um_per_px, bar_um, color="white"):
    """Draw a scale bar (lower-right) in an imshow axes (data coords = pixels)."""
    if not um_per_px or um_per_px <= 0:
        return
    bar_px = bar_um / um_per_px
    h, w = ax.get_ylim()[0], img_w_px
    x1 = w * 0.95
    x0 = x1 - bar_px
    y = h * 0.93
    ax.plot([x0, x1], [y, y], color=color, lw=3, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y * 0.985, f"{bar_um:g} µm", color=color,
            ha="center", va="bottom", fontsize=7)


def _compass(fig, rect, eye_right: bool):
    """D up / V down; horizontal T..N (left eye) or N..T (right eye)."""
    ax = fig.add_axes(rect)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.4)
    ax.axis("off")
    ax.set_aspect("equal")
    ar = dict(arrowstyle="-|>", lw=1.5, color="black")
    ax.annotate("", xy=(0, 1), xytext=(0, -1), arrowprops=ar)   # up arrow (D)
    ax.annotate("", xy=(0, -1), xytext=(0, 1), arrowprops=ar)   # down arrow (V)
    ax.annotate("", xy=(1, 0), xytext=(-1, 0), arrowprops=ar)
    ax.annotate("", xy=(-1, 0), xytext=(1, 0), arrowprops=ar)
    left_lbl, right_lbl = ("N", "T") if eye_right else ("T", "N")
    ax.text(0, 1.12, "D", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(0, -1.12, "V", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(-1.12, 0, left_lbl, ha="right", va="center", fontsize=11, fontweight="bold")
    ax.text(1.12, 0, right_lbl, ha="left", va="center", fontsize=11, fontweight="bold")


def _place_panels(fig, block, panels, crop_um_per_px, scalebar_um):
    """2x3 grid of the high-power crops inside the figure-fraction rect `block`."""
    bx, by, bw, bh = block
    gx, gy = 0.02 * bw, 0.10 * bh          # gaps
    cw = (bw - 2 * gx) / 3
    ch = (bh - gy) / 2
    for i, name in enumerate(PANEL_ORDER):
        r, c = divmod(i, 3)
        x = bx + c * (cw + gx)
        y = by + bh - (r + 1) * ch - r * gy
        ax = fig.add_axes([x, y, cw, ch])
        ax.axis("off")
        arr = panels.get(name)
        if arr is None:
            ax.text(0.5, 0.5, f"{name}\n(missing)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="0.5")
            continue
        ax.imshow(arr, interpolation="none")
        ax.set_title(name, fontsize=10, fontweight="bold", pad=2)
        if i == 0:
            _scalebar(ax, arr.shape[1], crop_um_per_px, scalebar_um)


def build_eye_figure(panels: dict, overlay, meta: dict, out_pdf,
                     box_um=100.0, overlay_um_per_px=None,
                     panel_scalebar_um=25.0, overlay_scalebar_um=500.0,
                     std_name: str | None = None):
    """Write one editable PDF for an eye. `panels` maps D1..V3 -> RGB arrays."""
    plt = _mpl()
    eye = (meta.get("eye") or "").lower()
    eye_right = eye.startswith("r")

    # crop pixel size (crops are box_um across their pixel width)
    any_crop = next((v for v in panels.values() if v is not None), None)
    crop_um_per_px = (box_um / any_crop.shape[1]) if any_crop is not None else None

    fig = plt.figure(figsize=(14, 7.5))

    # title band
    title = std_name or meta.get("slide", "")
    sub = "   ".join(
        f"{k}: {meta.get(k)}" for k in ("model", "samd7", "eye", "stain", "age")
        if meta.get(k))
    fig.text(0.5, 0.955, title, ha="center", va="top", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.915, f"{sub}\n{meta.get('slide','')}", ha="center", va="top",
             fontsize=8.5, color="0.25")

    # content band y range
    y0, h = 0.06, 0.80
    # column rects (figure fraction): panels block, overlay, compass
    if eye_right:
        compass_rect = [0.02, y0 + h * 0.30, 0.10, h * 0.45]
        overlay_rect = [0.15, y0, 0.36, h]
        panels_block = [0.55, y0, 0.43, h]
    else:
        panels_block = [0.02, y0, 0.43, h]
        overlay_rect = [0.49, y0, 0.36, h]
        compass_rect = [0.88, y0 + h * 0.30, 0.10, h * 0.45]

    _place_panels(fig, panels_block, panels, crop_um_per_px, panel_scalebar_um)

    ax_ov = fig.add_axes(overlay_rect)
    ax_ov.axis("off")
    if overlay is not None:
        ax_ov.imshow(overlay, interpolation="none")
        ax_ov.set_title("low-mag (fields marked)", fontsize=9, pad=3)
        _scalebar(ax_ov, overlay.shape[1], overlay_um_per_px, overlay_scalebar_um)

    _compass(fig, compass_rect, eye_right)

    fig.savefig(str(out_pdf))
    plt.close(fig)
    return out_pdf
