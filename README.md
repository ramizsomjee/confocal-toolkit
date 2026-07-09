# confocal-toolkit

Interactive + batch tools for pulling figure panels and quantification ROIs out
of confocal / slide-scanner microscopy (`.czi`). Runs on **Apple Silicon and
Intel Macs**.

**Tools included**
- **`retina-fields`** — sample standardized dorsal/ventral **100 µm fields**
  (D1–D3 dorsal, V1–V3 ventral) from whole-retina flat-mounts. You rotate the
  retina dorsal-up, drag six boxes onto the tissue, and it exports the oriented
  whole retina plus a labeled overlay and one crop per field. Handles files with
  multiple scenes and whole-folder batches.

More tools will land in this same package over time; [updating](#updating-to-new-versions)
brings them all.

---

# Part 1 — One-time setup

You'll do this once. Budget ~15 minutes (most of it is the environment building
itself while you wait). Everything below is meant to be **copy-pasted into
Terminal** (open it from Applications → Utilities → Terminal).

## Step 1 — Which Mac do you have?

Some steps differ for Apple Silicon (M1/M2/M3) vs Intel. Find out:

```bash
uname -m
```

- `arm64` → **Apple Silicon** (use the arm64 commands below)
- `x86_64` → **Intel** (use the Intel commands below)

## Step 2 — Install conda (Miniforge), if you don't have it

Check first:

```bash
conda --version
```

If that prints a version, skip to Step 3. If it says "command not found",
install Miniforge:

**Apple Silicon (arm64):**
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh -b
~/miniforge3/bin/conda init zsh
exec zsh
```

**Intel (x86_64):**
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
bash Miniforge3-MacOSX-x86_64.sh -b
~/miniforge3/bin/conda init zsh
exec zsh
```

After `exec zsh`, your prompt should start with `(base)`. That means conda is active.

## Step 3 — Download the code

```bash
git clone https://github.com/ramizsomjee/confocal-toolkit.git
cd confocal-toolkit
```

> First time you use `git`, macOS may pop up "install command line developer
> tools" — click **Install**, wait, then run the two lines again.

## Step 4 — Build the environment

This installs napari, the `.czi` reader, and the tools. It downloads a few
hundred MB and takes a few minutes — let it run:

```bash
conda env create -f environment.yml
```

When it finishes you'll see a line like `To activate this environment, use
$ conda activate confocal-toolkit`.

## Step 5 — Check it worked

```bash
conda activate confocal-toolkit
python tests/test_geometry.py
```

You should see `GEOMETRY OK`. If so, you're done setting up. 🎉

---

# Part 2 — Using `retina-fields`

**Every time** you open a new Terminal to use the tool, first activate the
environment:

```bash
conda activate confocal-toolkit
```

(Your prompt will change to start with `(confocal-toolkit)`.)

## Quick start — one file

```bash
retina-fields "/full/path/to/your retina.czi"
```

Tip: type `retina-fields ` (with a space), then **drag the .czi file from Finder
into the Terminal window** — it pastes the full path with quotes for you. Press
Enter.

A napari window opens. Here's the workflow:

### The interactive window, step by step
0. **Confirm metadata** — before the window opens, the tool parses the slide
   name and asks you to confirm/edit the fields (see [Metadata & naming](#metadata--naming)).
1. **Set the brightness** (optional) — done on the command line, see
   [Brightness & color](#brightness--color). By default it auto-scales.
2. **Rotate dorsal-up** — on the left there's a **Rotate (deg)** slider. Drag it
   until the retina is oriented the way you want (dorsal at the top).
   **Do this first, before placing boxes.**
3. **Place the six boxes** — six yellow squares labeled D1 D2 D3 (dorsal) and
   V1 V2 V3 (ventral) are already on the image and ready to drag. Click the
   middle of a box and drag it onto the tissue region you want. Each box is a
   true 100 × 100 µm. Zoom with the scroll wheel / trackpad to place precisely.
4. **Close the window** when all six are where you want them. Closing is what
   triggers the export — there's no separate save button.

> Whatever positions the boxes are in at the moment you close are what gets
> cropped. If you rotate *after* placing boxes, the boxes won't follow the
> tissue — so rotate first.

### Metadata & naming
The tool reads each slide name and fills in: degeneration **model** (RD10 / P23H
/ RhoKO), **Samd7** genotype (KO/WT), **eye** (S-opsin → left, M-opsin → right),
**stain**, and **age**. Missing age defaults by model (P23H→p30, RD10→p60,
RhoKO→p90) or pass `--age p60`. You confirm/edit each field at a prompt; skip it
with `--no-confirm`. The confirmed fields form a standardized name like
`RD10_Samd7-KO_L_S-opsin_p60`.

### The per-eye PDF figure (editable)
Closing the window also writes `<name>_figure.pdf`, laid out by eye —
**left (S-opsin):** panels on the left, low-mag overlay, compass (T↔N);
**right (M-opsin):** the mirror (compass N↔T, overlay, panels on the right). It
embeds the six D/V crops at **full resolution** for figure-making, adds scale
bars, and stays **editable in Illustrator** (vector text, `pdf.fonttype 42`). The
retina pixels are never flipped — only the layout and nasal/temporal labels
mirror between eyes. Use `--no-figure` to skip.

### Rebuild figures from an existing folder of panels
```bash
retina-fields --build-figure "/path/to/<something>_fields/"
```
Rebuilds the PDFs from already-exported panels (reads each `*_params.json`, or
falls back to any `*_fields_overlay.tif` it finds and prompts for metadata).

### Files with multiple scenes
Some `.czi` contain several scenes (e.g. multiple retinas on one slide). The tool
splits them automatically: after you close the window for scene 1, a new window
opens for scene 2, and so on. Do rotate + place + close for each.

## Quick start — a whole folder (batch)

```bash
retina-fields "/full/path/to/folder-of-czi/"
```

It walks every `.czi` in the folder (and every scene), opening a window for each.

**Memory:** each file/scene is processed in its **own worker process**, so all
memory (the big image, napari, the GPU) is fully released before the next one
starts. A 50-file batch uses the same peak memory as a single file — a batch
can't gradually eat all your RAM. (Add `--in-process` to force everything into
one process; not recommended for large images.)

---

## Brightness & color

These are set on the command line (they apply to the whole run):

| Option | What it does | Example |
|--------|--------------|---------|
| `--pmin` / `--pmax` | Auto-contrast percentiles (default 1 / 99.7) | `--pmin 2 --pmax 99.5` |
| `--clim "lo,hi"` | Exact brightness limits, in raw intensity units | `--clim "120,4000"` |
| `--gamma` | Brighten/darken midtones (`<1` brightens) | `--gamma 0.8` |
| `--colormaps` | Channel color (guessed from the file name otherwise) | `--colormaps green` |
| `--box-um` | Field box side length in **microns** (default 100) | `--box-um 150` |
| `--downsample` | Scale for the downsampled outputs (default 0.1 = 10%) | `--downsample 0.15` |
| `--fullres-rgb` | Also save the colored whole retina at **full** resolution | `--fullres-rgb` |

Box size is a true physical size — the tool converts microns to pixels using each
file's own pixel size, so `--box-um 150` gives 150 × 150 µm regardless of
magnification. In the batch CSV it's the per-file **`box_um`** column, so
different images can use different box sizes.

Example:
```bash
retina-fields "/path/to/retina.czi" --clim "150,5000" --colormaps green --fullres-rgb
```

To keep two retinas on the **same brightness scale**, give them the same
`--clim` values (or use the CSV below).

---

## Batch with per-file color + brightness (CSV)

When a batch mixes, say, S-opsin and M-opsin retinas that need different colors
or brightness, use a config CSV — one row per file (and per scene).

**1. Generate a template:**
```bash
retina-fields "/path/to/folder/" --make-csv batch_config.csv
```
This makes a spreadsheet you can open in Excel/Numbers. Columns:

| column | meaning |
|--------|---------|
| `file`, `scene`, `n_scenes` | which image (don't edit) |
| `base` | output name (edit if you like) |
| `channels` | channel names (info) |
| `colormap` | color per channel (e.g. `green`); guessed from the file name |
| `clim_lo` / `clim_hi` | brightness limits; **leave blank = auto** |
| `gamma` | midtone adjust (default 1.0) |
| `box_um` | field box size in microns for this image (default 100) |
| `model`, `samd7`, `eye`, `stain`, `age`, `animal_id` | metadata (pre-filled from the name; edit as needed) — this is your **images metadata spreadsheet** |
| `rotate` | starting angle (you can still change it in the window) |
| `skip` | put `1` to skip that row |

**2. Edit** `colormap`, `clim_lo`, `clim_hi`, `gamma` as needed, then **save**.

**3. Run from it:**
```bash
retina-fields --csv batch_config.csv
```
Each row still opens the window for rotation + box placement; the CSV just
supplies the color/brightness so they're consistent across the batch.

---

## What you get (outputs)

For each image, a folder is created next to your data at
`<parent-folder>_fields/<filename>/`. `<base>` = the file name (plus `_s0`,
`_s1`… for multi-scene files).

| File | What it is |
|------|-----------|
| `<base>_oriented_fullres.tif` | Full-resolution oriented whole retina, all channels (16-bit) |
| `<base>_oriented_ds0.1.tif` | Downsampled version of the above |
| `<base>_whole_rgb.tif` | Colored whole retina (downsampled), **no boxes** |
| `<base>_whole_rgb_fullres.tif` | Colored whole retina at full res (only with `--fullres-rgb`) |
| `<base>_fields_overlay.tif` | Colored whole retina with the 6 labeled boxes drawn on |
| `<base>_D1.tif` … `_V3.tif` | The six 100 µm field crops (raw, for quantification) |
| `<base>_D1_rgb.tif` … `_V3_rgb.tif` | The six crops as colored images (for figures) |
| `<std-name>_figure.pdf` | **Editable per-eye figure** (panels + overlay + compass + metadata) |
| `<base>_params.json` | Rotation, box positions, contrast, and metadata (for your records / figure rebuilds) |

---

# Updating to new versions

When new features/fixes are pushed, update with:

```bash
cd confocal-toolkit
git pull
```

That's usually all you need (the code updates in place). **Only if the release
note says dependencies changed**, also run:

```bash
conda env update -f environment.yml --prune
```

---

# Troubleshooting

**`conda: command not found`** — Close and reopen Terminal, or re-run
`~/miniforge3/bin/conda init zsh && exec zsh`. Your prompt should show `(base)`.

**`retina-fields: command not found`** — You forgot to activate the environment:
`conda activate confocal-toolkit`.

**The napari window is black / nothing shows** (older Intel GPUs) — run with
software rendering:
```bash
LIBGL_ALWAYS_SOFTWARE=1 retina-fields "/path/to/retina.czi"
```

**It seems to freeze after I close the window** — for big whole-retina images it
pauses 15–60 s while writing the full-resolution output. That's normal; wait for
`wrote outputs to …`.

**The whole Mac froze / had to restart during a batch** — that's memory
exhaustion. Each file now runs in its own process (memory is released between
files), which prevents batch build-up. If a *single* huge file still does it,
close other apps first (especially browsers) and process fewer files/scenes at a
time; if it persists, tell Ramiz your file's pixel dimensions and your RAM.

**`.czi` won't open** — rebuild the environment to be sure the reader installed:
`conda env update -f environment.yml --prune`.

**`conda env create` fails to "solve" on an older macOS** — tell Ramiz your macOS
version (Apple menu → About This Mac); we can pin an older napari that supports it.

**Git asks for a username/password when cloning** — the repo is public, so a plain
`git clone https://github.com/ramizsomjee/confocal-toolkit.git` should just work.
If it still prompts, press Enter past it or install the GitHub CLI (`gh`).

---

# For developers (Ramiz)

- Package: `confocal_toolkit/` (modules `retina_field_extractor`, `geometry`),
  installed editable via `environment.yml` so `git pull` = live code update.
- Tests: `python tests/test_geometry.py` and `python tests/test_pipeline.py`
  (the latter's real-`.czi` tests skip when the files aren't present).
- Push updates: edit, then `git add -A && git commit -m "…" && git push`.
- To add a tool: drop a module in `confocal_toolkit/` and add a `[project.scripts]`
  entry point in `pyproject.toml`.
