# %% [markdown]
# # Interface coherency — Colab
#
# One atomic-resolution image of an epitaxial interface, in: **is the film in registry with the
# substrate?** out. Same pipeline as `03_streamlined.ipynb`, which is the version to use on a local
# kernel; this one adds a setup cell and writes its output somewhere that survives the session.
#
# **What works here and what does not.** Loading, drift correction, detection, measurement, the
# figures and the CSV are plain numpy/scipy/matplotlib and behave normally. The two *interactive*
# cells — `SiteEditor` and the explorer sliders — need ipympl canvas events, which are unreliable on
# Colab. They are left in and they will not crash, but treat them as unverified: if clicking does
# nothing, set `INTERACTIVE = False` in the imports cell and work from the static figures.

# %% [markdown]
# ## Setup
#
# Installs quantem from the fork that carries `drift_single_image` and `interface_coherency`, and
# puts `zbkit` (the simulation side) on the path from the same clone. Idempotent — safe to re-run.
#
# This is not a small install: quantem declares torch, torchvision, tensorboard, torchmetrics and
# optuna. Colab's preinstalled torch normally satisfies it, and the cell prints the version it found
# so you can see whether pip is about to fetch a 2 GB wheel instead. **If pip reports that it
# upgraded numpy or torch, restart the runtime** (Runtime → Restart session) and re-run this cell
# before going on.

# %%
# ruff: noqa: E402
# This cell pip-installs quantem, so every import in the cells below it is necessarily "late".

import importlib.util
import subprocess
import sys
from pathlib import Path

QUANTEM_REPO = "https://github.com/wwmills/quantem.git"
QUANTEM_REF = "defect_processing_branch"
CLONE = Path("/content/quantem")
IN_COLAB = importlib.util.find_spec("google.colab") is not None


def _run(*cmd, **kw):
    print("$", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=False, **kw)


if IN_COLAB:
    try:
        import torch

        print(f"preinstalled torch {torch.__version__} (quantem needs >= 2.7.0)")
    except ImportError:
        print("no preinstalled torch -- pip will fetch one, which is slow")

    # Arial-metric text in the PDFs.  Must happen before zbkit.style is imported, or matplotlib's
    # font list is already built; zbkit.style.refresh_fonts() below covers the other order too.
    _run("apt-get", "-qq", "install", "-y", "fonts-liberation",
         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not (CLONE / "pyproject.toml").exists():
        _run("git", "clone", "--depth", "1", "--branch", QUANTEM_REF, QUANTEM_REPO, CLONE)
    # The two modules this notebook needs are only on this branch; `pip install quantem` from PyPI
    # does not have them.  Installing the clone gets the package and them in one step.
    _run(sys.executable, "-m", "pip", "install", "-q", CLONE, "ncempy", "ipympl")

# zbkit is not part of the quantem package -- it ships beside it in examples/ and is imported from
# the clone rather than installed.
ZBKIT_DIR = CLONE / "examples" / "zb_drift"
if (ZBKIT_DIR / "zbkit").exists():
    sys.path.insert(0, str(ZBKIT_DIR))
elif (Path.cwd() / "zbkit").exists():        # running from the project folder, not Colab
    sys.path.insert(0, str(Path.cwd()))
else:
    raise RuntimeError(f"zbkit not found -- looked in {ZBKIT_DIR} and {Path.cwd()}")
print("zbkit path:", sys.path[0])

# %% [markdown]
# ## Where output goes
#
# `/content` is wiped when the session ends. Mount Drive to keep the figures and the CSV, and to
# read a real image file — a Drive path is the only sane way to get a multi-hundred-MB `.emd` in
# here, since `files.upload()` through the browser is slow and drops large files.

# %%
SAVE_TO_DRIVE = False        # True mounts Drive and writes there instead of /content
DRIVE_SUBDIR = "zb_drift"    # under MyDrive/

if SAVE_TO_DRIVE and IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive")
    OUT = Path("/content/drive/MyDrive") / DRIVE_SUBDIR
else:
    OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)
print("output ->", OUT.resolve())

# %%
import csv
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from quantem.core.datastructures import Dataset2d
from quantem.imaging import Lattice
from quantem.imaging.drift_single_image import SingleImageDrift
from quantem.imaging.interface_coherency import (
    InterfaceCoherency,
    SiteEditor,
    estimate_interface_row,
    find_all_sites,
    peak_report,
)

import zbkit as zb
from zbkit import style
from zbkit.widgets import coherency_explorer, interactive_available, site_editor_panel, use_backend

INTERACTIVE = interactive_available()   # set False by hand if canvas clicks do nothing
use_backend(INTERACTIVE)
print("sans face:", style.refresh_fonts())   # picks up fonts-liberation installed above
style.use_style(7.0)

# %% [markdown]
# ## Load
#
# `IMAGE_PATH = None` simulates. `RUMPLE` gives the film layer-by-layer structure — see the
# docstring of `zbkit.structures.cubic_film_100`.

# %%
IMAGE_PATH = None          # .emd .npy .tif .dm3 .dm4 .h5 ... or None to simulate
PIXEL_SIZE = 0.12          # A/px; a file's own metadata wins if it has any
CASE = dict(relaxation=1.0)                    # coherent: relaxation=0.0
RUMPLE = None              # e.g. dict(x=0.35, decay=12.0) -- alternating row registry


def load_image(path=None, pixel_size=PIXEL_SIZE, rumple=RUMPLE, **case):
    """
    Return ``(image, pixel_size_A, ground_truth)``; ``ground_truth`` is None for real data.

    Readers by extension: ncempy for ``.emd``/``.dm3``/``.dm4``/``.h5``, numpy for ``.npy``,
    tifffile or skimage for ``.tif``.  A file's own pixel size overrides the argument, and is
    assumed to be in **nm** (ncempy's convention) and converted -- check it against a known
    spacing before trusting any length that comes out of this notebook.

    hyperspy is the last-resort reader and is *not* installed by the setup cell above: it is a
    large dependency and every format this project has actually met is covered by ncempy.  If you
    need it, ``pip install hyperspy`` and re-run.
    """
    if path is None:
        cols = zb.epitaxial_stack(fov=(120.0, 140.0), film_zone="100", d_film_relaxed=4.15,
                                  film_fraction=0.45, interface_roughness=0.30,
                                  rumple=rumple, **case)
        shape = (int(cols.meta["ground_truth"]["fov"][1] / pixel_size),
                 int(cols.meta["ground_truth"]["fov"][0] / pixel_size))
        im, _ = zb.simulate(
            cols, shape, pixel_size,
            kernel=zb.GaussianMixtureKernel.from_probe(zb.Probe(), column_sigma=0.45),
            scan=zb.ScanDrift(s=-0.030, k=0.985, jitter_amplitude=0.05, seed=1),
            mean_counts=45.0, ripple=0.05, seed=0)
        return im, pixel_size, cols.meta["ground_truth"]

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} -- mount Drive above and give the path under MyDrive/")
    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path).astype(float), pixel_size, None
    if ext in (".tif", ".tiff"):
        try:
            from tifffile import imread
        except ImportError:
            from skimage.io import imread
        return np.asarray(imread(path), float), pixel_size, None
    if ext in (".emd", ".dm3", ".dm4", ".h5", ".hdf5", ".ser", ".mrc"):
        import ncempy.io as nio

        d = nio.read(str(path))
        px = d.get("pixelSize")
        return (np.asarray(d["data"], float).squeeze(),
                float(px[-1]) * 10.0 if px is not None else pixel_size, None)
    import hyperspy.api as hs  # last resort: whatever hyperspy knows

    sig = hs.load(str(path))
    ax = sig.axes_manager[-1]
    scale = float(ax.scale) * (10.0 if str(getattr(ax, "units", "")).lower().startswith("nm") else 1.0)
    return np.asarray(sig.data, float), scale, None


image, PIXEL_SIZE, gt = load_image(IMAGE_PATH, **CASE)

# %% [markdown]
# ## Drift correction
#
# `set_reference_band` says which rows are the known crystal. `find_sites` detects the sites
# *inside that band only* and refines them — its own local-maxima pass, independent of the
# detection further down, because the drift fit needs the substrate's dumbbell centroids and
# nothing else. `fit` then solves for the shear `s` and the slow-axis scale `k`.
#
# Flyback cropping is off. On real data the first 10–20 rows carry the post-flyback settling
# transient; pass `flyback_rows=N` below if that matters for your frame.

# %%
with warnings.catch_warnings():   # the no-flyback-crop warning; off deliberately, see above
    warnings.filterwarnings("ignore", message=".*flyback crop.*")
    sid = SingleImageDrift.from_data(image, pixel_size=PIXEL_SIZE, flyback_rows=0)
INTERFACE_ROW = estimate_interface_row(sid.image)
sid.set_reference_band(0, INTERFACE_ROW - 20)
_ = sid.find_sites()          # `_ =` because these return self, and a notebook would echo it
_ = sid.fit()

# %%
corrected_raw, apply_info = sid.apply_to_image(order=3)
VALID = apply_info["valid"]      # marks the shear's fill wedge; keep before nan_to_num
corrected = np.nan_to_num(corrected_raw, nan=float(np.nanmedian(corrected_raw)))
INTERFACE_ROW_CORR = INTERFACE_ROW / sid.solution["k"]

# %% [markdown]
# Indexed sites, the FFT before and after, the angle residual, and the row-wise trajectory `d(t)`.
# The FFT panels are the check that matters: after correction the substrate's spots should sit on
# an orthogonal net with the reference axial ratio.

# %%
fig, _ = sid.plot(corrected_image=corrected)
fig.savefig(OUT / "03_drift.pdf")

# %% [markdown]
# ## Detect sites

# %%
def detect_peaks(image, min_spacing=12.0, min_rel_intensity=0.2, edge_boundary=10,
                 lowpass_q=0.11, lowpass_order=8, smooth_px=0.8, subpixel="poly",
                 max_num_peaks=10000):
    """
    Find atomic columns with ``Lattice.get_maxima_2D``.  Returns ``(peaks_xy, filtered_image)``.

    ``min_spacing`` (px)
        Minimum separation between accepted peaks, applied greedily from the brightest.  **The
        knob that decides whether interface sites survive**: rows straddling an interface sit
        closer together than rows within either phase, so a value tied to the bulk spacing
        deletes them.  Independent here -- set it below the closest real separation.
    ``min_rel_intensity``
        Floor relative to the brightest peak.  At 0 this admits noise maxima one ``min_spacing``
        from a real column; 0.2 removed ten of them on the test frame.
    ``edge_boundary`` (px)
        Rectangular border to ignore.  The *slanted* fill wedge left by the shear needs
        ``valid_mask`` instead, which is a different exclusion.
    ``lowpass_q`` (cycles/px), ``lowpass_order``, ``smooth_px``
        Butterworth low-pass then a Gaussian, applied before the search.  Removes the slowly
        varying background so a dim phase is not thresholded away near a bright one.
    ``subpixel``
        ``'pixel'``, ``'poly'`` (parabolic), or ``'multicorr'`` (Fourier upsampling).
    """
    ny, nx = image.shape
    q = np.sqrt(np.fft.fftfreq(ny)[:, None] ** 2 + np.fft.fftfreq(nx)[None, :] ** 2)
    filt = np.real(np.fft.ifft2(np.fft.fft2(image) / (1 + (q / lowpass_q) ** (2 * lowpass_order))))
    filt = gaussian_filter(filt, smooth_px)
    host = Lattice.from_data(image=Dataset2d.from_array(filt), normalize_min=False)
    m = host.get_maxima_2D(filt, subpixel=subpixel, sigma=0, minSpacing=min_spacing,
                          minRelativeIntensity=min_rel_intensity, edgeBoundary=edge_boundary,
                          maxNumPeaks=max_num_peaks)
    return np.column_stack([m["y"], m["x"]]).astype(float), filt


# %%
MIN_SPACING, EDGE_PX = 12.0, 10
site_spacing_px = min(sid.geometry.site_spacing("centroid"),
                      (gt["d_film_parallel"] if gt else sid.geometry.ax)) / PIXEL_SIZE

peaks, filtered = detect_peaks(
    corrected,
    min_spacing=MIN_SPACING,        # px between peaks; lower it if interface sites go missing
    min_rel_intensity=0.2,          # 0 admits noise maxima
    edge_boundary=EDGE_PX,          # px border ignored
    lowpass_q=0.11,                 # Butterworth cutoff, cycles/px
    lowpass_order=8,
    smooth_px=0.8,                  # Gaussian after the low-pass
    subpixel="poly",                # 'pixel' | 'poly' | 'multicorr'
)
sites = find_all_sites(
    corrected, site_spacing_px,
    peaks=peaks,                    # skip the built-in search, refine these instead
    valid_mask=VALID,               # excludes the shear's slanted fill wedge
    min_edge_dist_px=EDGE_PX,       # a floor; the margin used is max(this, spacing/2)
    refine="com",                   # 'com' | 'gaussian' | 'parabolic'
)

# %% [markdown]
# Green: kept. Red ×: too dim. Magenta +: inside `min_spacing` of a brighter peak. Yellow outline:
# the valid region, `edge_boundary` inside the frame and the fill wedge. Blue diamond: dropped for
# a reason this figure cannot measure — a few are normal, a pattern is not.

# %%
fig, pinfo = peak_report(filtered, site_spacing_px, peaks=peaks, valid_mask=VALID, smooth=0.0,
                         min_spacing_px=MIN_SPACING, min_edge_dist_px=EDGE_PX,
                         interface_y=INTERFACE_ROW_CORR)
fig.savefig(OUT / "03_peaks.pdf")

# %% [markdown]
# ## Fix sites by hand
#
# Left-click adds, right-click removes the nearest, `f` refines, `u` undoes. Site indices along a
# row come from rank order, so one spurious or missing column shifts every index after it.
#
# **This is the cell that may not work on Colab.** If the figure appears but clicking does nothing,
# ipympl is not delivering canvas events; skip it — the sites from the cell above are already in
# `sites` and the rest of the notebook runs on them unchanged.

# %%
editor = SiteEditor(corrected, sites, figsize=(7.5, 8.0), interface_y=INTERFACE_ROW_CORR)
if INTERACTIVE:
    panel, accepted = site_editor_panel(editor)
    display(panel)      # noqa: F821

# %%
sites = editor.sites

# %% [markdown]
# ## Measure
#
# Groups the sites into atomic rows, numbers the sites along each row `0, 1, 2, …`, and does a
# **least-squares straight-line fit of `x` against that index** for every row: the slope is that
# row's in-plane spacing, the intercept its `x` offset. The two phases' slopes are then compared.
# `coh.rows[i]` keeps `.spacing`, `.intercept`, `.x`, `.index`, `.n_missing`, `.phase`.
#
# `n_rows_each_side` is the only argument that changes the numbers; the rest is display.

# %%
N_ROWS_SIDE = 4      # rows of substrate and of film, counted outward from the interface
coh = InterfaceCoherency.from_sites(sites, INTERFACE_ROW_CORR, PIXEL_SIZE, label="corrected")
coh.measure(n_rows_each_side=N_ROWS_SIDE)

# %% [markdown]
# ## Coherency plot
#
# Left: `x` against site index along the row. Right: the same minus `index × d_ref`, where
# **`d_ref` is the substrate's measured in-plane site spacing** (for zincblende ⟨110⟩ centroids
# that is `a/√2 ≈ 4.00 Å`, not `a`). Subtracting it flattens the substrate, so what is left is the
# **disregistry** — the accumulated lateral mismatch. One line per atomic row, blues for substrate
# and oranges for film, numbered outward from the interface.
#
# `remove_row_offset` subtracts each row's own fitted `x` offset so it starts at the origin.
# `"substrate"` does that for the substrate only, leaving the film's rows where they sit against
# the flattened ruler, referenced to the mean substrate offset.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
coh.plot(remove_row_offset="substrate", stack_offset=0.0, axes=axes)
fig.savefig(OUT / "03_coherency.pdf")

# %% [markdown]
# The explorer below re-runs that same measurement live on every slider move, so it is the place to
# choose `rows` and `skip`. It does not write back: the figure above and the CSV below come from the
# `coh` in the measure cell, so once the sliders settle, put those two numbers there and re-run.
#
# Sliders are ipywidgets rather than canvas events, so this has a better chance of working on Colab
# than the editor above. Still unverified there.

# %%
if INTERACTIVE:
    display(coherency_explorer(corrected, sites, INTERFACE_ROW_CORR, drift=sid,   # noqa: F821
                               pixel_size=PIXEL_SIZE))

# %% [markdown]
# ## Export

# %%
csv_path = OUT / "03_x_vs_index.csv"
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["phase", "row", "row_y_px", "index", "x_px", "x_A", "disregistry_A"])
    counts = {"substrate": 0, "film": 0}
    for r in coh.rows:
        counts[r.phase] += 1
        for i, x in zip(r.index, r.x):
            w.writerow([r.phase, counts[r.phase], f"{r.y_centre:.3f}", int(i), f"{x:.4f}",
                        f"{x * PIXEL_SIZE:.4f}",
                        f"{(x - r.intercept - coh.reference_spacing * i) * PIXEL_SIZE:.4f}"])
print(csv_path)

# %% [markdown]
# If you did not mount Drive, the cell below is the only way the PDFs and the CSV leave the
# session.

# %%
if IN_COLAB and not SAVE_TO_DRIVE:
    import shutil

    from google.colab import files

    archive = shutil.make_archive("/content/zb_drift_output", "zip", OUT)
    files.download(archive)
    print(archive)
