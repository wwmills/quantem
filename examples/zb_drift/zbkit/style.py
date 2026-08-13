"""
Figure style: vector PDF, embedded Arial, journal column widths.

Same style module used across these projects -- a matplotlib style that embeds real
TrueType fonts in the PDF (``fonttype=42``, so text stays selectable/editable in
Illustrator rather than being outlined), plus helpers for panel furniture.

One deliberate difference from the ``ffkit`` copy: ``imshow_field`` here does **not**
transpose.  Everything in ``zbkit`` and in ``quantem.imaging.drift_single_image``
uses the standard image convention ``image[iy, ix]`` -- row index ``iy`` is the slow
scan axis, column index ``ix`` is the fast scan axis -- because the whole drift
argument is about which axis is which, and silently flipping it for display is how
you end up correcting the wrong one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# Nature Portfolio column widths, in inches (89 mm and 183 mm).
COL_SINGLE = 89.0 / 25.4
COL_DOUBLE = 183.0 / 25.4
MAX_HEIGHT = 247.0 / 25.4

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def _first_available(candidates):
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    return "DejaVu Sans"


#: Whichever of these is installed wins.  Arial is what the journal's own figures use;
#: the rest are metrically compatible stand-ins, so a figure laid out against one of
#: them keeps its text extents when the PDF is later opened somewhere Arial exists.
#: ``Liberation Sans`` is the one a bare Linux box (Colab) can get with
#: ``apt-get install -y fonts-liberation``.
SANS_CANDIDATES = ["Arial", "Helvetica", "TeX Gyre Heros", "Liberation Sans",
                   "Nimbus Sans", "DejaVu Sans"]

SANS = _first_available(SANS_CANDIDATES)


def refresh_fonts():
    """
    Re-scan the system for fonts installed *after* matplotlib was imported, and return
    the best available sans face.

    Needed on a machine where the font arrives during the session -- ``apt-get install
    fonts-liberation`` in a Colab cell, say.  matplotlib builds ``fontManager.ttflist``
    once at import and caches it on disk, so without this the new font is invisible and
    every figure silently falls back to DejaVu Sans: readable, but not the metrics the
    figures were laid out against, so text extents and panel spacing shift.
    """
    known = {f.fname for f in font_manager.fontManager.ttflist}
    for path in font_manager.findSystemFonts():
        if path not in known:
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass  # a broken or unsupported font file is not worth failing a figure over
    global SANS
    SANS = _first_available(SANS_CANDIDATES)
    return SANS


def use_style(base_size: float = 7.0, dpi: float = 150.0, savefig_dpi: float = 600.0,
              retina: bool = True):
    """
    Apply the figure style.  Call once at the top of a notebook.

    ``dpi`` sets the *inline* resolution -- what a figure looks like in the notebook -- and
    ``retina`` asks IPython for 2x PNGs on top of that, so the effective inline resolution is
    ~300 dpi.  Worth the bytes for dense micrograph panels at journal column width, where
    matplotlib's default 100 dpi turns the atomic columns to mush on screen even though the
    saved PDF is fine.

    Don't push ``dpi`` much past this.  150 x 2 is already ~300 dpi; at 220 x 2 the eight
    figures in notebook 01 came to a 9.6 MB ``.ipynb``, which is a nuisance to open and to
    diff for no visible gain.  ``savefig_dpi`` is separate and set high, because the PNG
    proofs in ``figures/`` are the ones that get looked at closely -- and the PDF beside each
    one is vector anyway.
    """
    # Re-resolved here rather than taken from the module-level ``SANS``, so a font
    # installed after import (see :func:`refresh_fonts`) is still picked up.
    sans = _first_available(SANS_CANDIDATES)
    mpl.rcParams.update(
        {
            # --- fonts -------------------------------------------------------
            "font.family": "sans-serif",
            "font.sans-serif": [sans, "DejaVu Sans"],
            "font.size": base_size,
            "axes.titlesize": base_size,
            "axes.labelsize": base_size,
            "xtick.labelsize": base_size - 0.5,
            "ytick.labelsize": base_size - 0.5,
            "legend.fontsize": base_size - 0.5,
            "figure.titlesize": base_size + 1,
            # Math set in the same sans face so `\epsilon_{xx}` matches the labels.
            "mathtext.fontset": "custom",
            "mathtext.rm": sans,
            "mathtext.it": f"{sans}:italic",
            "mathtext.bf": f"{sans}:bold",
            # --- embed text as text, not as outlines -------------------------
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 6,
            "svg.fonttype": "none",
            # --- lines / axes ------------------------------------------------
            "axes.linewidth": 0.6,
            "axes.labelpad": 2.0,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.borderpad": 0.2,
            "legend.labelspacing": 0.25,
            # --- images --------------------------------------------------------
            "image.interpolation": "nearest",
            "image.origin": "lower",
            "image.cmap": "gray",
            # --- output --------------------------------------------------------
            "figure.dpi": dpi,
            "savefig.dpi": savefig_dpi,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            # 0.01 inch is a quarter of a millimetre, and matplotlib's tight bbox is computed
            # from the artists' own extents -- anything it slightly under-estimates (rotated
            # tick labels, a legend hanging off an axes, the last marker of a scatter sitting
            # on the spine) then gets shaved off the edge.  0.05 costs nothing and stops it.
            "savefig.pad_inches": 0.05,
        }
    )
    if retina:
        # Only meaningful inside IPython with the inline backend; a no-op anywhere else.
        try:
            from IPython import get_ipython

            ip = get_ipython()
            if ip is not None:
                ip.run_line_magic("config", "InlineBackend.figure_formats = ['retina']")
                # The inline backend does not inherit savefig.bbox, so set it here too or the
                # figure you see in the notebook is cropped differently from the one on disk.
                ip.run_line_magic(
                    "config",
                    "InlineBackend.print_figure_kwargs = "
                    "{'bbox_inches': 'tight', 'pad_inches': 0.05}",
                )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# colour maps
# ---------------------------------------------------------------------------
#: Blue--white--red for strain and other signed fields.  Perceptually symmetric
#: about zero, and the mid-tone is a true white so zero reads as "no signal".
CMAP_STRAIN = LinearSegmentedColormap.from_list(
    "strain",
    ["#2166ac", "#67a9cf", "#d1e5f0", "#ffffff", "#fddbc7", "#ef8a62", "#b2182b"],
)
#: Cyclic map for wrapped phase; endpoints match so the 2*pi wrap is invisible.
CMAP_PHASE = plt.get_cmap("twilight_shifted")
#: Slightly warm grey for micrographs -- keeps them distinct from plots on the page.
CMAP_IMAGE = "gray"


# ---------------------------------------------------------------------------
# panel furniture
# ---------------------------------------------------------------------------
def panel_label(ax, letter, dx=0.0, dy=0.0, color=None, inside=True, box=True, **kw):
    """
    Bold panel letter.

    Defaults to *inside* the top-left corner, which is what micrograph panels want:
    an outside label collides with a centred title, and image panels have no
    margin to spare on a two-column figure.  Pass ``inside=False`` for line plots,
    where the label belongs above the axes.
    """
    if inside:
        x, y, ha, va = 0.03 + dx, 0.97 + dy, "left", "top"
        color = "w" if color is None else color
        if box:
            kw.setdefault("bbox", dict(fc="k", ec="none", alpha=0.35, pad=1.0))
    else:
        x, y, ha, va = -0.14 + dx, 1.02 + dy, "left", "bottom"
        color = "k" if color is None else color
    ax.text(
        x, y, letter, transform=ax.transAxes, fontsize=mpl.rcParams["font.size"] + 1,
        fontweight="bold", ha=ha, va=va, color=color, zorder=6, **kw,
    )


def title(ax, text, pad=2.5, **kw):
    """Panel title sized to leave room for an inside panel label."""
    ax.set_title(text, pad=pad, fontsize=mpl.rcParams["font.size"] - 0.5, **kw)


def row_label(ax, text, pad=4):
    """Italic left-hand label for a row of image panels."""
    ax.set_ylabel(text, labelpad=pad, style="italic",
                  fontsize=mpl.rcParams["font.size"] - 0.5)


def imshow_field(ax, data, extent=None, cmap=CMAP_IMAGE, vlim=None, sym=False,
                 percentile=None, **kw):
    """
    ``imshow`` with the defaults every micrograph panel here wants.

    ``data`` is ``[iy, ix]`` and is shown unmodified with ``origin="lower"``, so the
    slow scan axis runs up the page and the fast axis runs across it.
    ``percentile=(0.5, 99.5)`` clips display levels, which matters once shot noise is
    on the image and a single hot pixel would otherwise set the white point.
    """
    if sym:
        v = np.nanpercentile(np.abs(data), 99.5) if vlim is None else vlim
        kw.update(vmin=-v, vmax=v)
    elif vlim is not None:
        lo, hi = (vlim, vlim) if np.isscalar(vlim) else vlim
        kw.update(vmin=lo, vmax=hi)
    elif percentile is not None:
        lo, hi = np.nanpercentile(np.asarray(data, float), percentile)
        kw.update(vmin=lo, vmax=hi)
    im = ax.imshow(np.asarray(data), extent=extent, cmap=cmap, origin="lower",
                   interpolation="nearest", **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def scalebar(ax, length, label=None, unit="nm", loc="lower right", color="w",
             pad=0.05, height=0.018, fontsize=None):
    """Draw a scale bar in *data* units on an axes whose extent is in those units."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    bx = x1 - pad * w - length if "right" in loc else x0 + pad * w
    by = y0 + pad * h if "lower" in loc else y1 - pad * h - height * h
    ax.add_patch(Rectangle((bx, by), length, height * h, fc=color, ec="none", zorder=5))
    if label is None:
        label = f"{length:g} {unit}"
    if label:
        ax.text(bx + length / 2, by + height * h * 1.5, label, ha="center", va="bottom",
                color=color, zorder=5,
                fontsize=fontsize or mpl.rcParams["font.size"] - 0.5)


def cbar(fig, im, ax, label=None, width=0.02, pad=0.012, ticks=None, side="right"):
    """Slim colour bar hugging the right edge of ``ax`` (works with fixed-aspect images)."""
    pos = ax.get_position()
    if side == "right":
        cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
        c = fig.colorbar(im, cax=cax, orientation="vertical", ticks=ticks)
        c.ax.tick_params(length=1.8, pad=1.2)
    else:
        cax = fig.add_axes([pos.x0, pos.y0 - pad - width, pos.width, width])
        c = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
        c.ax.tick_params(length=1.8, pad=1.2)
    c.outline.set_linewidth(0.4)
    if label:
        c.set_label(label, labelpad=2)
    return c


def save(fig, name, directory=None, png=True, pdf=True, png_dpi=600):
    """Write ``name.pdf`` (vector, embedded fonts) and a PNG proof next to it."""
    directory = directory or FIG_DIR
    os.makedirs(directory, exist_ok=True)
    out = []
    if pdf:
        p = os.path.join(directory, f"{name}.pdf")
        fig.savefig(p)
        out.append(p)
    if png:
        p = os.path.join(directory, f"{name}.png")
        fig.savefig(p, dpi=png_dpi)
        out.append(p)
    return out


@dataclass
class Extent:
    """
    Turn an image shape + pixel size into an ``imshow`` extent.

    ``shape`` is ``(ny, nx)`` -- i.e. ``image.shape`` -- matching the ``[iy, ix]``
    convention used throughout.
    """

    shape: tuple
    pixel_size: float  # Angstrom / pixel
    unit: str = "nm"

    @property
    def scale(self):
        return {"nm": 0.1, "A": 1.0, "um": 1e-4}[self.unit]

    @property
    def extent(self):
        ny, nx = self.shape[:2]
        s = self.pixel_size * self.scale
        return (0.0, nx * s, 0.0, ny * s)
