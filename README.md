# confocal-toolkit

Interactive + batch tools for pulling figure panels and quantification ROIs out
of confocal / slide-scanner microscopy (`.czi`). Works on Apple Silicon **and
Intel Macs**.

**Tools**
- `retina-fields` — sample standardized dorsal/ventral 100 µm fields (D1–D3,
  V1–V3) from whole-retina flat-mounts: rotate dorsal-up, drag six boxes, export
  oriented images + crops. Handles multi-scene files and batch CSVs.

More tools will be added to this same package over time; updating (below) brings
them all.

---

## First-time setup

You need **conda** (Miniforge). If you don't have it:

**Apple Silicon (M1/M2/M3) Mac:**
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh -b
~/miniforge3/bin/conda init zsh && exec zsh
```

**Intel Mac:**
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
bash Miniforge3-MacOSX-x86_64.sh -b
~/miniforge3/bin/conda init zsh && exec zsh
```

Then get the code and build the environment (same on both architectures):
```bash
git clone https://github.com/ramizsomjee/confocal-toolkit.git
cd confocal-toolkit
conda env create -f environment.yml
```

That creates a conda env named `confocal-toolkit` with napari, the `.czi`
reader, and the tools installed.

Test it (should print `GEOMETRY OK`):
```bash
conda activate confocal-toolkit
python tests/test_geometry.py
```

---

## Updating (when we push new versions)

```bash
cd confocal-toolkit
git pull
conda env update -f environment.yml --prune   # only needed if dependencies changed
```

The toolkit is installed in editable mode, so **code updates take effect on the
next `git pull`** — you only need the `conda env update` line when we add/upgrade
a dependency (we'll say so in the release notes).

---

## Running `retina-fields`

```bash
conda activate confocal-toolkit

# one file (opens a napari window per scene):
retina-fields "/path/to/retina.czi"

# a whole folder (batch every .czi; multi-scene files split automatically):
retina-fields "/path/to/folder/"
```

In the viewer: use the **Rotate** slider (left dock) to bring dorsal up, then
drag the six yellow boxes (D1–D3 dorsal, V1–V3 ventral). **Rotate first, then
place boxes.** Close the window to export.

### Contrast / color (command line)
- `--pmin 1 --pmax 99.7` — auto per-channel contrast from percentiles.
- `--clim "lo,hi" ...` — absolute per-channel limits (raw intensity units).
- `--gamma 0.8`, `--colormaps green` (m-opsin→green, s-opsin/S-647→magenta, dapi→blue guessed).
- `--downsample 0.1` — scale factor for downsampled outputs.
- `--fullres-rgb` — also write a full-resolution colored RGB (tiled, memory-safe).

### Per-file color + brightness via CSV (for consistent batches)
```bash
retina-fields "/path/to/folder/" --make-csv batch_config.csv   # generate a template (1 row per file+scene)
# edit colormap / clim_lo / clim_hi / gamma in the CSV, then:
retina-fields --csv batch_config.csv
```

### Outputs (per file, in `<parent>_fields/<stem>/`)
`<base>` = `<stem>_s<scene>` for multi-scene files, else `<stem>`.

| File | Description |
|------|-------------|
| `<base>_oriented_fullres.tif` | Full-res oriented multi-channel composite (16-bit) |
| `<base>_oriented_ds<f>.tif` | Downsampled composite |
| `<base>_whole_rgb.tif` | Colored whole retina, downsampled RGB — no boxes |
| `<base>_whole_rgb_fullres.tif` | Full-res colored RGB (only with `--fullres-rgb`) |
| `<base>_fields_overlay.tif` | Colored whole retina with the 6 boxes burned in |
| `<base>_<FIELD>.tif` / `_<FIELD>_rgb.tif` | Per-field 100 µm crops (raw composite + RGB) |
| `<base>_params.json` | Rotation angle, box coords, contrast — reproducibility |

---

## Intel Mac / older macOS notes

- Install the **x86_64** Miniforge on Intel (see above); conda-forge then pulls
  `osx-64` builds of napari/Qt/scipy automatically.
- conda-forge binaries target older macOS, so this generally works back to
  macOS ~10.15. If the environment fails to solve or napari won't launch on a
  very old OS, tell us the macOS version — we can pin an older napari/Qt.
- If the napari window is black on a machine with an old GPU, run with
  `LIBGL_ALWAYS_SOFTWARE=1 retina-fields ...` (software rendering).

## Troubleshooting
- `command not found: retina-fields` → run `conda activate confocal-toolkit` first.
- `.czi` won't load → it's read by `bioio-czi`; make sure the env built without
  pip errors (`conda env update -f environment.yml --prune`).
- Big whole-retina images pause for 15–60 s on close while the full-res oriented
  TIFF is written — that's normal, not a hang.
