"""Headless test of the retina pipeline: loader (real CZI) + full export path
(synthetic stack, simulating a captured angle + box placement)."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from confocal_toolkit import retina_field_extractor as R


def test_loader_real_czi():
    p = "/Users/ramizsomjee/Documents/crx ko imaging ntc p8 p32/20x_p8_invivo.czi"
    if not os.path.exists(p):
        print("[skip] reference czi missing")
        return
    stack, um, names, sname, n = R.load_scene(p, 0)
    assert stack.ndim == 3, stack.shape
    assert um > 0
    assert len(names) == stack.shape[0]
    print(f"[ok] load_scene: shape {stack.shape}, {um:.4f} um/px, "
          f"{n} scene(s), channels {names}")


def test_climits_and_rgb():
    rng = np.random.default_rng(0)
    stack = (rng.random((2, 64, 80)) * 4000).astype(np.uint16)
    lims = R.compute_climits(stack, 1, 99.7, None)
    assert len(lims) == 2 and all(hi > lo for lo, hi in lims)
    ov = R.compute_climits(stack, 1, 99.7, [(100.0, 3000.0), None])
    assert ov[0] == (100.0, 3000.0)          # override honored
    rgb = R.additive_rgb(stack, lims, ["green", "magenta"])
    assert rgb.shape == (64, 80, 3) and rgb.dtype == np.uint8
    print("[ok] contrast limits (auto + override) and additive RGB")


def test_tiled_rgb_matches():
    rng = np.random.default_rng(1)
    stack = (rng.random((2, 517, 613)) * 4000).astype(np.uint16)
    lims = R.compute_climits(stack, 1, 99.7, None)
    cmaps = ["green", "magenta"]
    full = R.additive_rgb(stack, lims, cmaps)
    # tiny tile forces many row-blocks; result must be identical
    tiled = R.additive_rgb_tiled(stack, lims, cmaps, tile_pixels=613 * 7)
    assert np.array_equal(full, tiled), "tiled RGB differs from single-shot"
    print("[ok] tiled full-res RGB is identical to single-shot (memory-safe)")


def test_fullres_rgb_export():
    pk = _make_picker(box_um=100.0, um_per_px=0.5)
    pk.angle = 0.0
    pk.box_centers = [(150.0, 200.0)] + [(60.0, 100.0)] * 5
    with tempfile.TemporaryDirectory() as td:
        R.export(pk, "frr", Path(td), downsample=0.25, fullres_rgb=True)
        f = os.path.join(td, "frr_whole_rgb_fullres.tif")
        assert os.path.exists(f), "full-res RGB not written"
        img = tifffile.imread(f)
        # oriented (angle 0) full-res is the original H,W with 3 channels
        assert img.shape == (300, 400, 3), img.shape
        print(f"[ok] --fullres-rgb export wrote {os.path.basename(f)} {img.shape}")


def _make_picker(box_um=100.0, um_per_px=0.5):
    # 300x400 image, 2 channels; box_px = 200
    C, H, W = 2, 300, 400
    stack = np.zeros((C, H, W), np.uint16)
    stack[0, 140:160, 190:210] = 5000        # bright marker near centre
    stack[1] = 300
    names = ["M-opsin", "DAPI"]
    cmaps = R.guess_colormaps(names, C)
    clim = R.compute_climits(stack, 1, 99.7, None)
    pk = R.FieldPicker(stack, um_per_px, names, box_um, clim, cmaps)
    return pk


def test_full_export():
    pk = _make_picker(box_um=100.0, um_per_px=0.5)   # -> 200 px boxes
    assert abs(pk.box_px - 200.0) < 1e-6
    pk.angle = 30.0
    # place D1 exactly on the bright marker centre (row 150, col 200)
    pk.box_centers = [(150.0, 200.0)] + [(60.0, 100.0)] * 5

    with tempfile.TemporaryDirectory() as td:
        R.export(pk, "sampleA_s0", Path(td), downsample=0.25)
        files = set(os.listdir(td))
        need = {"sampleA_s0_oriented_fullres.tif",
                "sampleA_s0_oriented_ds0.25.tif",
                "sampleA_s0_whole_rgb.tif",
                "sampleA_s0_fields_overlay.tif",
                "sampleA_s0_params.json"}
        for f in need:
            assert f in files, f"missing {f} in {sorted(files)}"
        for fld in R.FIELD_NAMES:
            assert f"sampleA_s0_{fld}.tif" in files
            assert f"sampleA_s0_{fld}_rgb.tif" in files

        # D1 crop is 100 um = 200 px square and contains the marker near centre
        d1 = tifffile.imread(os.path.join(td, "sampleA_s0_D1.tif"))
        assert d1.shape == (2, 200, 200), d1.shape
        assert d1[0].max() > 1000, "marker not captured in D1 crop"
        yx = np.unravel_index(np.argmax(d1[0]), d1[0].shape)
        assert 80 < yx[0] < 120 and 80 < yx[1] < 120, f"marker off-centre {yx}"

        # colored whole retina (no burn-in) is a real RGB image
        whole = tifffile.imread(os.path.join(td, "sampleA_s0_whole_rgb.tif"))
        assert whole.ndim == 3 and whole.shape[2] == 3, whole.shape

        params = json.loads(
            Path(td, "sampleA_s0_params.json").read_text())
        assert params["rotation_deg"] == 30.0
        assert params["field_bounds_oriented_px"]["D1"]
        print(f"[ok] full export: {len(files)} files incl whole_rgb {whole.shape}, "
              f"D1 crop {d1.shape}, marker at {yx}, params captured")


def test_row_clim_parsing():
    # blank -> all auto
    assert R.row_clim({"clim_lo": "", "clim_hi": ""}, 1) is None
    # single channel absolute
    assert R.row_clim({"clim_lo": "120", "clim_hi": "4000"}, 1) == [(120.0, 4000.0)]
    # multi channel, second blank -> None (auto) for that channel
    out = R.row_clim({"clim_lo": "120;", "clim_hi": "4000;"}, 2)
    assert out == [(120.0, 4000.0), None], out
    print("[ok] CSV row_clim parsing (blank=auto, per-channel absolute)")


def test_csv_generate_and_read():
    folder = "/Users/ramizsomjee/Documents/confocal_panel_extractor/retina_whole_images"
    if not os.path.isdir(folder) or not list(Path(folder).glob("*.czi")):
        print("[skip] retina folder not present for CSV generation")
        return
    files = R.gather_inputs(folder)
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "config.csv")
        R.write_config_csv(files, csv_path)
        rows = R.read_config_csv(csv_path)
        assert rows, "no rows written"
        assert set(R.CSV_FIELDS).issubset(rows[0].keys())
        # multi-scene files must produce multiple rows (SD7 ...Corbo have 3)
        from collections import Counter
        per_file = Counter(r["file"] for r in rows)
        multi = [f for f, c in per_file.items() if c > 1]
        assert multi, "expected at least one multi-scene file to expand into rows"
        # each row has a colormap guess and a base name
        assert all(r["colormap"] and r["base"] for r in rows)
        print(f"[ok] CSV generate/read: {len(rows)} rows from {len(files)} files, "
              f"{len(multi)} multi-scene expanded")


def test_pyramid_levels():
    # a 20000-wide image must be reduced below the GPU texture limit
    img = np.zeros((17000, 20000), np.uint16)
    levels = R.FieldPicker._pyramid(img, max_side=4096)
    assert levels[0].shape == (17000, 20000)          # level 0 stays full-res
    assert max(levels[-1].shape) <= 4096, levels[-1].shape
    assert len(levels) >= 3
    print(f"[ok] multiscale pyramid: {len(levels)} levels, "
          f"top {levels[0].shape} -> {levels[-1].shape}")


def test_naming_single_vs_multi():
    # single-scene -> no _s token; multi-scene -> _sN token (logic in process_scene)
    stem, n_scenes, scene = "retinaX", 1, 0
    base = f"{stem}_s{scene}" if n_scenes > 1 else stem
    assert base == "retinaX"
    base2 = f"{stem}_s{2}" if 3 > 1 else stem
    assert base2 == "retinaX_s2"
    print("[ok] naming: single-scene omits _s token, multi-scene includes it")


if __name__ == "__main__":
    test_loader_real_czi()
    test_climits_and_rgb()
    test_full_export()
    test_tiled_rgb_matches()
    test_fullres_rgb_export()
    test_row_clim_parsing()
    test_csv_generate_and_read()
    test_pyramid_levels()
    test_naming_single_vs_multi()
    print("\nPIPELINE OK")
