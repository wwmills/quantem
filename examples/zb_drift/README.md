# Single-image drift correction and interface coherency

One atomic-resolution image of an epitaxial interface in, an answer to *is the film in registry with
the substrate?* out. The correction lives in `quantem.imaging.drift_single_image`, the measurement in
`quantem.imaging.interface_coherency`; this directory holds the simulation side and a runnable
notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wwmills/quantem/blob/defect_processing_branch/examples/zb_drift/03_colab.ipynb)

| file | what it is |
|---|---|
| `03_colab.ipynb` | the workflow, with a setup cell. Generated — edit `scripts/03_colab.py` |
| `scripts/03_colab.py` | percent-format source of the above |
| `zbkit/` | the simulation side: zincblende ⟨110⟩ substrate, cubic ⟨100⟩ film, drifting scan, figure style |

`zbkit` is deliberately **not** part of the `quantem` package — it is a forward model for validating
the correction, not analysis anyone should import. `pip install` does not ship it (the sdist
`only-include` covers `/src` and `/tests`); the notebook picks it up off the clone instead.

## Setup

The notebook's first cell does this for you. By hand:

```bash
git clone --depth 1 --branch defect_processing_branch https://github.com/wwmills/quantem.git
pip install ./quantem ncempy ipympl
export PYTHONPATH=$PWD/quantem/examples/zb_drift
```

`drift_single_image` and `interface_coherency` only exist on this branch — `pip install quantem`
from PyPI does not have them, which is why the install is from the clone.

The install is not small: quantem declares torch, torchvision, tensorboard, torchmetrics and optuna.
Colab's preinstalled torch normally satisfies `torch>=2.7.0` and pip leaves it alone; the setup cell
prints the version it found so you can see if it is about to fetch a wheel instead. Only
`Lattice.get_maxima_2D`, used by the notebook's `detect_peaks`, needs any of that — the two modules
themselves are numpy/scipy/matplotlib, and `find_all_sites` has its own detector if you ever want to
run them without the package.

## What is verified on Colab and what is not

**Verified.** The numerical pipeline was run end to end in a clean environment with only numpy,
scipy, matplotlib and ipywidgets: simulate, drift solve, `apply_to_image`, detection, refinement,
`peak_report`, the row fits, the coherency figure, the CSV. No heavy dependency was touched.

**Not verified.** Nobody has run this in an actual Colab session. In particular the two interactive
cells — `SiteEditor` (click to add/remove sites) and the explorer sliders — depend on ipympl
delivering canvas events, which is unreliable there. They are left in, they will not crash, and
`use_backend` enables Colab's custom widget manager and falls back to the inline backend if ipympl
is missing. If clicking does nothing, set `INTERACTIVE = False` and work from the static figures:
the measurement does not need the editor.

**Fonts.** The figures are laid out against Arial. A bare Linux box has neither Arial nor a metric
clone, so the setup cell installs `fonts-liberation` and `zbkit.style.refresh_fonts()` re-scans
after the fact — without that, matplotlib silently falls back to DejaVu Sans, whose different text
extents shift the panel spacing.

**Output.** `/content` is wiped when the session ends. Set `SAVE_TO_DRIVE = True` to write the PDFs
and the CSV to Drive, or use the download cell at the bottom.
