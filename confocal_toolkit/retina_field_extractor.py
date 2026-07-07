#!/usr/bin/env python3
"""
Retina Field Extractor
======================

Interactive + batch tool for sampling standardized dorsal/ventral fields from
whole-retina flat-mount scans (M- or S-opsin stained, EDF projections from a
slide scanner, stored as .czi — possibly multi-scene).

Per CZI file, per scene:
  1. Load the scene (bioio) and read the physical pixel size, so field boxes
     are a true 100 x 100 um regardless of source magnification.
  2. Open a napari viewer with the channels overlaid (additive, colorized),
     contrast set from the command line, a live ROTATION slider to bring
     dorsal up, and six draggable 100 um boxes: D1 D2 D3 (dorsal), V1 V2 V3
     (ventral).
  3. On window close, capture the rotation angle and the six box centres.
  4. Orient the full-resolution data to the chosen angle and export:
        <base>_oriented_fullres.tif   full-res oriented multi-channel composite
        <base>_oriented_ds{f}.tif     downsampled composite
        <base>_fields_overlay.tif     downsampled RGB with the 6 boxes burned in
        <base>_<FIELD>.tif            per-field full-res composite crop (100 um)
        <base>_<FIELD>_rgb.tif        per-field colorized RGB crop
        <base>_params.json           angle, box coords, contrast (reproducibility)
     where <base> = "<stem>_s<scene>" for multi-scene files, else "<stem>".

Contrast / color is set by terminal inputs (percentiles or absolute limits and
gamma); rotation and box placement are interactive.

Usage
-----
    python retina_field_extractor.py IMAGE.czi
    python retina_field_extractor.py FOLDER/            # batch every .czi
    python retina_field_extractor.py img.czi --downsample 0.1 --pmin 1 --pmax 99.7
    python retina_field_extractor.py img.czi --clim 120,4000 0,2000  # per channel
    python retina_field_extractor.py img.czi --box-um 100 --gamma 0.8

Run with -h for all options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Works both as an installed package module and when run as a loose script.
try:
    from . import geometry as G
except ImportError:
    import geometry as G

FIELD_NAMES = ["D1", "D2", "D3", "V1", "V2", "V3"]
DEFAULT_BOX_UM = 100.0

DEFAULT_COLORMAPS = ["green", "magenta", "cyan", "yellow", "red", "bop blue"]
FLUOR_COLORMAP_HINTS = [
    ("dapi", "blue"), ("hoechst", "blue"), ("405", "blue"),
    ("egfp", "green"), ("gfp", "green"), ("488", "green"), ("fitc", "green"),
    ("opsin", "green"), ("mopsin", "green"), ("sopsin", "magenta"),
    ("mcher", "red"), ("rfp", "red"), ("dsred", "red"), ("561", "red"),
    ("tritc", "red"), ("555", "red"),
    ("af647", "magenta"), ("cy5", "magenta"), ("647", "magenta"),
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def scene_count(path: str) -> int:
    from bioio import BioImage
    return len(list(BioImage(path).scenes))


def load_scene(path: str, scene: int = 0, timepoint: int = 0):
    """Return (stack[C,H,W], um_per_px, channel_names, scene_name, n_scenes)."""
    from bioio import BioImage

    img = BioImage(path)
    scenes = list(img.scenes)
    img.set_scene(scene)
    data = np.asarray(img.get_image_data("CZYX", T=timepoint))  # (C, Z, Y, X)

    # EDF projections are single-plane; collapse Z defensively.
    if data.shape[1] > 1:
        data = data.max(axis=1, keepdims=True)
    stack = data[:, 0]                                           # (C, H, W)

    px = img.physical_pixel_sizes
    um_per_px = px.X or px.Y
    if not um_per_px:
        raise ValueError(f"{path}: no physical pixel size in metadata; "
                         "cannot size 100 um boxes. Pass --um-per-px.")
    try:
        names = [str(n) for n in img.channel_names]
    except Exception:
        names = [f"Ch{i}" for i in range(stack.shape[0])]
    return stack, float(um_per_px), names, scenes[scene], len(scenes)


def guess_colormaps(channel_names, n):
    out = []
    for i in range(n):
        name = channel_names[i].lower() if i < len(channel_names) else ""
        match = next((cm for key, cm in FLUOR_COLORMAP_HINTS if key in name), None)
        out.append(match or DEFAULT_COLORMAPS[i % len(DEFAULT_COLORMAPS)])
    return out


# --------------------------------------------------------------------------- #
# Contrast
# --------------------------------------------------------------------------- #
def compute_climits(stack, pmin, pmax, clim_overrides):
    """Per-channel (lo, hi). Overrides (absolute) win where provided."""
    lims = []
    for c in range(stack.shape[0]):
        if clim_overrides and c < len(clim_overrides) and clim_overrides[c]:
            lims.append(tuple(map(float, clim_overrides[c])))
        else:
            lo = float(np.percentile(stack[c], pmin))
            hi = float(np.percentile(stack[c], pmax))
            if hi <= lo:
                hi = lo + 1.0
            lims.append((lo, hi))
    return lims


def stretch(img, lo, hi, gamma=1.0):
    out = (img.astype(np.float32) - lo) / max(hi - lo, 1e-9)
    out = np.clip(out, 0.0, 1.0)
    if gamma != 1.0:
        out = out ** (1.0 / gamma)
    return out


def colorize(unit_img, colormap_name):
    from napari.utils.colormaps import ensure_colormap
    cmap = ensure_colormap(colormap_name)
    rgba = cmap.map(unit_img.ravel()).reshape(*unit_img.shape, 4)
    return (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)


def additive_rgb(stack, climits, colormaps, gamma=1.0):
    """Additive colour blend of all channels -> RGB uint8."""
    acc = None
    for c in range(stack.shape[0]):
        lo, hi = climits[c]
        unit = stretch(stack[c], lo, hi, gamma)
        rgb = colorize(unit, colormaps[c % len(colormaps)]).astype(np.uint16)
        acc = rgb if acc is None else acc + rgb
    return np.clip(acc, 0, 255).astype(np.uint8)


def additive_rgb_tiled(stack, climits, colormaps, gamma=1.0,
                       tile_pixels=8_000_000):
    """Memory-safe additive RGB for very large (full-res) images.

    Identical result to ``additive_rgb`` but processes in row-blocks so the
    transient float arrays from colormap mapping stay bounded (colorizing a
    28k x 28k image in one shot would allocate tens of GB). Output is
    (H, W, 3) uint8.
    """
    C, H, W = stack.shape
    out = np.empty((H, W, 3), np.uint8)
    step = max(1, int(tile_pixels // max(1, W)))
    for r0 in range(0, H, step):
        r1 = min(H, r0 + step)
        out[r0:r1] = additive_rgb(stack[:, r0:r1, :], climits, colormaps, gamma)
    return out


# --------------------------------------------------------------------------- #
# Interactive orientation + field placement
# --------------------------------------------------------------------------- #
class FieldPicker:
    def __init__(self, stack, um_per_px, channel_names, box_um,
                 climits, colormaps, gamma=1.0, init_angle=0.0):
        self.stack = stack
        self.C, self.H, self.W = stack.shape
        self.um_per_px = um_per_px
        self.channel_names = channel_names
        self.box_um = box_um
        self.box_px = box_um / um_per_px
        self.climits = climits
        self.colormaps = colormaps
        self.gamma = gamma

        self.viewer = None
        self.box_layer = None
        self.image_layers = []
        self.angle = float(init_angle)

        # captured on close
        self.box_centers = None    # list of (row, col) world coords, D1..V3

    def _initial_boxes(self):
        """Six axis-aligned squares: D row near top, V row near bottom.

        Columns are spread so the three boxes in a row don't overlap when the
        physical box size is small relative to the image; if the boxes are
        large they may still overlap (the user repositions them anyway).
        """
        s = self.box_px
        pitch = min(self.W / 3.0, max(s * 1.15, self.W * 0.22))
        cx0 = self.W / 2.0
        cols = [cx0 - pitch, cx0, cx0 + pitch]
        margin = min(self.H * 0.22, s * 0.75 + 10)
        rows = {"D": margin, "V": self.H - margin}
        rects, centers = [], []
        for name in FIELD_NAMES:
            cy = rows[name[0]]
            cx = cols[int(name[1]) - 1]
            centers.append((cy, cx))
            rects.append(np.array([
                [cy - s / 2, cx - s / 2], [cy - s / 2, cx + s / 2],
                [cy + s / 2, cx + s / 2], [cy + s / 2, cx - s / 2],
            ]))
        return rects, centers

    @staticmethod
    def _pyramid(img2d, max_side=4096):
        """Multiscale levels (2x downsampling) so images larger than the GPU
        texture limit still display. Level 0 is full res, so world coordinates
        stay in full-resolution pixels and the crop geometry is unaffected."""
        levels = [img2d]
        while max(levels[-1].shape) > max_side:
            levels.append(levels[-1][::2, ::2])
        return levels

    def _apply_rotation(self, angle):
        self.angle = float(angle)
        A = G.preview_affine((self.H, self.W), self.angle)
        for lyr in self.image_layers:
            lyr.affine = A

    def build_viewer(self):
        import napari
        from magicgui import magicgui

        self.viewer = napari.Viewer(
            title="Retina Field Extractor — rotate dorsal-up, place D/V boxes, close")

        big = max(self.H, self.W) > 4096
        for c in range(self.C):
            lo, hi = self.climits[c]
            data = self._pyramid(self.stack[c]) if big else self.stack[c]
            self.image_layers.append(self.viewer.add_image(
                data,
                multiscale=big or None,
                name=self.channel_names[c] if c < len(self.channel_names) else f"Ch{c}",
                colormap=self.colormaps[c % len(self.colormaps)],
                blending="additive",
                contrast_limits=(lo, hi),
                gamma=self.gamma,
            ))

        rects, centers = self._initial_boxes()
        # Edge width in data pixels; scaled to box size so the outline is
        # visible when zoomed out over a large whole-retina mosaic.
        edge_w = max(4.0, self.box_px * 0.12)
        self.box_layer = self.viewer.add_shapes(
            rects, shape_type="rectangle", name="fields",
            edge_color="yellow", face_color="transparent", edge_width=edge_w,
            text={"string": FIELD_NAMES, "size": 12, "color": "yellow",
                  "anchor": "upper_left"},
        )
        self.box_layer.mode = "select"
        self.viewer.layers.selection.active = self.box_layer

        @magicgui(auto_call=True,
                  angle={"widget_type": "FloatSlider", "min": -180, "max": 180,
                         "step": 0.5, "label": "Rotate (deg)"})
        def rotate_ctl(angle: float = self.angle):
            self._apply_rotation(angle)

        self.viewer.window.add_dock_widget(rotate_ctl, area="left",
                                           name="Orientation")
        self._apply_rotation(self.angle)
        self._rotate_ctl = rotate_ctl
        return self.viewer

    def run(self):
        import napari
        self.build_viewer()
        print("\nViewer open. Use the Rotate slider to bring dorsal UP, then drag "
              "the six boxes (D1-3 dorsal, V1-3 ventral). Close the window to export.\n")
        napari.run()
        self._capture()

    def _capture(self):
        # self.angle is kept current by _apply_rotation on every slider move;
        # the magicgui Qt widget is destroyed when the window closes, so we must
        # NOT read it back here (that raises "QSlider has been deleted").
        centers = []
        for verts in self.box_layer.data:
            v = np.asarray(verts)
            centers.append((float(v[:, 0].mean()), float(v[:, 1].mean())))
        self.box_centers = centers


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def rescale_stack(stack, factor):
    """Anti-aliased downsample that stays memory-safe on huge (20k+ px) images.

    Uses integer block-mean (never converts the full array to float64 as
    skimage.rescale would). Returns ``(downsampled_stack, actual_factor)``.
    """
    from skimage.measure import block_reduce
    block = max(1, int(round(1.0 / factor)))
    chans = [block_reduce(stack[c], (block, block), func=np.mean).astype(stack.dtype)
             for c in range(stack.shape[0])]
    return np.stack(chans, 0), 1.0 / block


def save_composite(path, stack, channel_names):
    import tifffile
    tifffile.imwrite(
        str(path), stack, imagej=True,
        metadata={"axes": "CYX", "mode": "composite",
                  "Labels": list(channel_names)},
        compression="zlib")


def burn_in_overlay(rgb, boxes_px, factor, out_path):
    """Draw the six labelled boxes on a (downsampled) RGB image."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    for name, (y0, y1, x0, x1) in boxes_px.items():
        y0, y1 = y0 * factor, y1 * factor
        x0, x1 = x0 * factor, x1 * factor
        color = (255, 220, 0)
        d.rectangle([x0, y0, x1, y1], outline=color, width=max(2, int(3)))
        d.text((x0 + 3, y0 + 3), name, fill=color)
    im.save(str(out_path))


def export(picker: FieldPicker, base: str, outdir: Path, downsample: float,
           fullres_rgb: bool = False):
    import tifffile
    outdir.mkdir(parents=True, exist_ok=True)
    stack = picker.stack
    angle = picker.angle
    names = picker.channel_names
    cmaps = picker.colormaps
    clim = picker.climits
    gamma = picker.gamma

    print(f"\n=== {base} ===")
    print(f"  rotation angle : {angle:.2f} deg")
    print(f"  box size       : {picker.box_um} um = {picker.box_px:.1f} px")

    # 1. orient full-res
    oriented, offset = G.orient_image(stack, angle)
    save_composite(outdir / f"{base}_oriented_fullres.tif", oriented, names)

    # 2. downsampled composite
    ds, ds_factor = rescale_stack(oriented, downsample)
    save_composite(outdir / f"{base}_oriented_ds{downsample:g}.tif", ds, names)

    # 3. field boxes -> output-pixel bounds (full-res oriented frame)
    boxes_px = {}
    for name, ctr in zip(FIELD_NAMES, picker.box_centers):
        boxes_px[name] = G.world_box_to_output(ctr, picker.box_px, offset,
                                               oriented.shape[1:])

    # 4. colored whole retina (oriented, downsampled RGB) -- with and without boxes
    ds_rgb = additive_rgb(ds, clim, cmaps, gamma)
    tifffile.imwrite(str(outdir / f"{base}_whole_rgb.tif"), ds_rgb,
                     photometric="rgb", compression="zlib")   # clean, no burn-in
    burn_in_overlay(ds_rgb, boxes_px, ds_factor,
                    outdir / f"{base}_fields_overlay.tif")     # boxes burned in

    # 4b. optional full-resolution colored whole retina (no boxes), tiled so it
    # doesn't blow memory on 20k+ px images.
    if fullres_rgb:
        full_rgb = additive_rgb_tiled(oriented, clim, cmaps, gamma)
        tifffile.imwrite(str(outdir / f"{base}_whole_rgb_fullres.tif"), full_rgb,
                         photometric="rgb", tile=(1024, 1024), compression="zlib")
        print(f"  wrote full-res colored RGB {full_rgb.shape}")

    # 5. per-field crops: raw composite + colorized RGB
    for name, (y0, y1, x0, x1) in boxes_px.items():
        crop = oriented[:, y0:y1, x0:x1]
        save_composite(outdir / f"{base}_{name}.tif", crop, names)
        rgb = additive_rgb(crop, clim, cmaps, gamma)
        tifffile.imwrite(str(outdir / f"{base}_{name}_rgb.tif"), rgb,
                         photometric="rgb", compression="zlib")
        print(f"  {name}: crop [{y0}:{y1}, {x0}:{x1}]")

    # 6. reproducibility sidecar
    params = {
        "base": base, "rotation_deg": angle,
        "um_per_px": picker.um_per_px, "box_um": picker.box_um,
        "box_px": picker.box_px, "downsample": downsample, "gamma": gamma,
        "channels": names, "colormaps": cmaps,
        "contrast_limits": {names[i]: list(clim[i]) for i in range(len(clim))},
        "field_centers_world": {n: list(c) for n, c in
                                zip(FIELD_NAMES, picker.box_centers)},
        "field_bounds_oriented_px": {n: list(b) for n, b in boxes_px.items()},
        "oriented_shape": list(oriented.shape),
    }
    (outdir / f"{base}_params.json").write_text(json.dumps(params, indent=2))
    print(f"  wrote outputs to {outdir}/")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def process_scene(path, scene, args, outroot, base_override=None, csv_row=None):
    stack, um_per_px, names, scene_name, n_scenes = load_scene(path, scene)
    if args.um_per_px:
        um_per_px = args.um_per_px
    stem = Path(path).stem
    n_ch = stack.shape[0]
    base = base_override or (f"{stem}_s{scene}" if n_scenes > 1 else stem)

    # Per-image settings: CSV row overrides CLI defaults where present.
    if csv_row is not None:
        clim_over = row_clim(csv_row, n_ch)
        cmaps = [c.strip() for c in str(csv_row.get("colormap", "")).split(";")
                 if c.strip()] or guess_colormaps(names, n_ch)
        gamma = _num(csv_row.get("gamma"), args.gamma)
        rotate = _num(csv_row.get("rotate"), args.rotate)
        box_um = _num(csv_row.get("box_um"), args.box_um)
    else:
        clim_over = args.clim
        cmaps = args.colormaps or guess_colormaps(names, n_ch)
        gamma = args.gamma
        rotate = args.rotate
        box_um = args.box_um

    climits = compute_climits(stack, args.pmin, args.pmax, clim_over)

    print(f"\nLoaded {stem}  scene {scene+1}/{n_scenes} ({scene_name})")
    print(f"  shape {stack.shape}  {um_per_px:.4f} um/px  channels {names}")
    print(f"  box {box_um} um  colormaps {cmaps}  gamma {gamma}  "
          f"clim {[tuple(round(x, 1) for x in c) for c in climits]}")

    picker = FieldPicker(stack, um_per_px, names, box_um, climits, cmaps,
                         gamma=gamma, init_angle=rotate)
    picker.run()
    if picker.box_centers is None:
        print("  (window closed without capture; skipping)")
        return
    export(picker, base, outroot / stem, args.downsample,
           fullres_rgb=getattr(args, "fullres_rgb", False))


def gather_inputs(inp):
    p = Path(inp)
    if p.is_dir():
        return sorted(p.glob("*.czi"))
    return [p]


def parse_clim(values):
    if not values:
        return None
    out = []
    for v in values:
        lo, hi = v.split(",")
        out.append((float(lo), float(hi)))
    return out


def _num(val, default):
    """Parse a CSV cell as float, falling back to default if blank/missing."""
    try:
        s = str(val).strip()
        return float(s) if s != "" else default
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# CSV batch configuration
# --------------------------------------------------------------------------- #
CSV_FIELDS = ["file", "scene", "n_scenes", "base", "channels", "colormap",
              "clim_lo", "clim_hi", "gamma", "box_um", "rotate", "skip"]

# When the channel name is uninformative (e.g. every file is "AF647"), guess a
# per-opsin colormap from the file name instead. Editable in the CSV.
OPSIN_FILE_HINTS = [
    ("m-opsin", "green"), ("mopsin", "green"), ("m opsin", "green"),
    ("s-opsin", "magenta"), ("sopsin", "magenta"), ("s opsin", "magenta"),
    ("s-647", "magenta"), ("s647", "magenta"),
]


def guess_colormaps_for_file(stem, names):
    """Colormap per channel, preferring a per-opsin guess from the file name."""
    low = stem.lower()
    for key, cm in OPSIN_FILE_HINTS:
        if key in low:
            return [cm] * max(1, len(names))
    return guess_colormaps(names, len(names)) if names else ["gray"]


def scene_channel_info(path):
    """(list of (scene_idx, scene_name, [channel names]), n_scenes) — metadata
    only, no pixel data loaded (fast even for 20k px files)."""
    from bioio import BioImage
    img = BioImage(path)
    scenes = list(img.scenes)
    out = []
    for i, sc in enumerate(scenes):
        img.set_scene(i)
        try:
            names = [str(n) for n in img.channel_names]
        except Exception:
            names = []
        out.append((i, str(sc), names))
    return out, len(scenes)


def write_config_csv(files, out_csv, box_um=DEFAULT_BOX_UM):
    """Generate a per-(file, scene) template CSV: one row per extracted scene,
    with a guessed colormap and blank contrast (blank = auto percentile)."""
    import csv
    rows = []
    for f in files:
        try:
            info, n = scene_channel_info(str(f))
        except Exception as e:
            print(f"  skip {f}: {e}", file=sys.stderr)
            continue
        stem = Path(f).stem
        for i, sc, names in info:
            base = f"{stem}_s{i}" if n > 1 else stem
            cmaps = guess_colormaps_for_file(stem, names)
            rows.append({
                "file": str(f), "scene": i, "n_scenes": n, "base": base,
                "channels": ";".join(names), "colormap": ";".join(cmaps),
                "clim_lo": "", "clim_hi": "", "gamma": 1.0, "box_um": box_um,
                "rotate": 0, "skip": 0,
            })
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows (one per file+scene) to {out_csv}")
    print("Edit 'colormap', 'clim_lo', 'clim_hi', 'gamma' as needed "
          "(blank clim = auto), then run with --csv.")


def read_config_csv(path):
    import csv
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def row_clim(row, n_ch):
    """Per-channel absolute (lo, hi) from a CSV row's ';'-separated clim cells.
    Returns None (all-auto) if empty; per-channel None where a value is blank."""
    los = str(row.get("clim_lo", "")).split(";")
    his = str(row.get("clim_hi", "")).split(";")
    if all(x.strip() == "" for x in los) and all(x.strip() == "" for x in his):
        return None
    out = []
    for i in range(n_ch):
        lo = los[i].strip() if i < len(los) else ""
        hi = his[i].strip() if i < len(his) else ""
        out.append((float(lo), float(hi)) if lo and hi else None)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Interactive dorsal/ventral field extractor for retina scans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", nargs="?",
                   help="A .czi file or a folder of .czi files "
                        "(optional when using --csv).")
    p.add_argument("--make-csv", metavar="PATH", default=None,
                   help="Scan the input folder and write a batch-config CSV "
                        "(one row per file+scene), then exit.")
    p.add_argument("--csv", metavar="PATH", default=None,
                   help="Run using a batch-config CSV (per-file color/brightness).")
    p.add_argument("--outdir", default=None, help="Output root (default <input>_fields).")
    p.add_argument("--box-um", type=float, default=DEFAULT_BOX_UM,
                   help="Field box side length in microns.")
    p.add_argument("--downsample", type=float, default=0.1,
                   help="Scale factor for the downsampled outputs (0-1).")
    p.add_argument("--fullres-rgb", action="store_true",
                   help="Also write a full-resolution colored RGB whole retina "
                        "(no boxes); tiled so it is memory-safe on huge images.")
    p.add_argument("--rotate", type=float, default=0.0,
                   help="Initial rotation angle (deg); still adjustable in the GUI.")
    p.add_argument("--pmin", type=float, default=1.0,
                   help="Low percentile for auto per-channel contrast.")
    p.add_argument("--pmax", type=float, default=99.7,
                   help="High percentile for auto per-channel contrast.")
    p.add_argument("--clim", nargs="+", default=None,
                   help='Absolute per-channel limits "lo,hi" (overrides percentiles).')
    p.add_argument("--gamma", type=float, default=1.0, help="Display/output gamma.")
    p.add_argument("--colormaps", nargs="+", default=None,
                   help="Per-channel colormap names (else guessed from channel names).")
    p.add_argument("--um-per-px", type=float, default=None,
                   help="Override physical pixel size (microns/pixel).")
    p.add_argument("--scene", type=int, default=None,
                   help="Process only this scene index (default: all scenes).")
    return p.parse_args(argv)


def _outroot_for(files, args):
    if args.outdir:
        return Path(args.outdir)
    parent = Path(files[0]).parent
    return parent / f"{parent.name}_fields"


def main(argv=None):
    args = parse_args(argv)
    args.clim = parse_clim(args.clim)

    # Mode 1: generate a batch-config CSV from a folder and exit.
    if args.make_csv:
        if not args.input:
            print("error: --make-csv needs an input file/folder.", file=sys.stderr)
            return 2
        files = gather_inputs(args.input)
        if not files:
            print(f"error: no .czi found at {args.input}", file=sys.stderr)
            return 2
        write_config_csv(files, args.make_csv, box_um=args.box_um)
        return 0

    # Mode 2: run from a batch-config CSV (per-row color/brightness).
    if args.csv:
        rows = read_config_csv(args.csv)
        if not rows:
            print(f"error: no rows in {args.csv}", file=sys.stderr)
            return 2
        outroot = _outroot_for([r["file"] for r in rows], args)
        for row in rows:
            if str(row.get("skip", "0")).strip().lower() in ("1", "true", "yes"):
                print(f"skip (csv): {row.get('base')}")
                continue
            process_scene(row["file"], int(row["scene"]), args, outroot,
                          base_override=(row.get("base") or None), csv_row=row)
        print("\nAll done.")
        return 0

    # Mode 3: direct file/folder run.
    if not args.input:
        print("error: provide an input file/folder, --csv, or --make-csv.",
              file=sys.stderr)
        return 2
    files = gather_inputs(args.input)
    if not files:
        print(f"error: no .czi found at {args.input}", file=sys.stderr)
        return 2
    outroot = _outroot_for(files, args)
    for f in files:
        try:
            n = scene_count(str(f))
        except Exception as e:
            print(f"error reading {f}: {e}", file=sys.stderr)
            continue
        scenes = [args.scene] if args.scene is not None else range(n)
        for s in scenes:
            process_scene(str(f), s, args, outroot)
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
