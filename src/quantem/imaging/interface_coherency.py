"""
Interactive site QC and interface-coherency measurement.

Companion to :mod:`quantem.imaging.drift_single_image`.  That module removes scan drift
using a known substrate as an internal ruler; this one answers the question the drift
correction was a prerequisite for: **is the film coherent with the substrate?**

The measurement
---------------
Take a row of atomic columns running parallel to the interface, index the columns along it
``0, 1, 2, ...``, and plot ``x`` against that index.  A perfect row gives a straight line
whose slope is the in-plane spacing.  Then do it for several rows above and below the
interface and compare:

======================  ==================================================================
coherent                film and substrate rows share one slope
uniformly relaxed       film rows take a different slope and walk away from the substrate
misfit dislocations     film rows keep the substrate slope but *step* by one lattice vector
======================  ==================================================================

Plotted raw, all three look like the same straight line, because a 4% slope difference is
invisible next to a 4 Angstrom slope.  So the informative plot is the **residual**,
``x - index * d_reference``, which flattens the substrate and leaves the film's behaviour
as the only structure on the axes.  :meth:`InterfaceCoherency.plot` shows both.

What the drift correction does and does not change
--------------------------------------------------
Worth stating precisely, because the intuitive answer is wrong.  Linear drift is
``M = [[1, s], [0, k]]``, and along a row of *constant y* the shear term ``s * y`` is a
constant: every site in the row moves sideways by the same amount.  So the in-plane
**spacing** along a row -- the slope of ``x`` versus index, and therefore the coherent /
relaxed verdict -- is **exactly invariant** under linear drift.  Correcting for drift first
does not change it, and not correcting does not corrupt it.

What linear drift does corrupt, and where this analysis needs the correction:

* **The out-of-plane row pitch**, which is scaled directly by ``k`` and is the film's
  out-of-plane strain -- the other half of the strain state.  Reported here as
  ``row_pitch``, and wrong by ``1/k - 1`` if uncorrected.
* **Row-to-row registry.**  Successive rows slide sideways by ``s`` times their separation,
  so any question of the form "do the film columns line up with the substrate columns" is
  meaningless before correction.  This is what ``remove_row_offset=False`` looks at.
* **Rows that are not horizontal.**  If the crystal is rotated by ``theta`` with respect to
  the scan, a row direction has a ``y`` component and the shear does reach the spacing --
  by -0.03% at 0.5 deg and -0.27% at 5 deg for ``s = -0.03``.  Mostly common-mode between
  film and substrate, so it largely cancels in the ratio.
* **Nonlinear drift**, which by definition is *not* constant along a row.

:func:`compare_with_and_without_correction` runs the measurement both ways and reports which
numbers moved, so the invariance is demonstrated rather than asserted.

Interactive tools
-----------------
:class:`RegionSelector` drags out the reference region.  :class:`SiteEditor` shows detected
sites over the image and lets you left-click to add one and right-click to delete the
nearest -- because automatic detection is good but never perfect near an interface, and a
single spurious site in a row shifts every index after it by one.  A click identifies a
column to within a few pixels; :meth:`SiteEditor.refine` then takes the position from the data,
fitting a Gaussian in a window around each site (:func:`snap_sites`), which is also usable on
its own for a set of sites from anywhere.  Both tools need an interactive
matplotlib backend (``%matplotlib widget`` in Jupyter, ``macosx``/``qtagg`` in a script).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from quantem.imaging.drift_single_image import find_peaks, fit_peaks_gaussian, refine_peaks

__all__ = [
    "InterfaceLine",
    "RegionSelector",
    "SiteEditor",
    "estimate_interface_row",
    "InterfaceCoherency",
    "RowFit",
    "find_all_sites",
    "group_rows",
    "median_nn_spacing",
    "peak_report",
    "snap_sites",
    "compare_with_and_without_correction",
]


# ===========================================================================
# site detection over a whole image
# ===========================================================================


def find_all_sites(
    image,
    spacing_px,
    threshold_rel=0.12,
    smooth=None,
    refine="com",
    refine_radius=None,
    edge_margin=None,
    min_edge_dist_px=10.0,
    valid_mask=None,
    threshold_mode="prominence",
    peaks=None,
):
    """
    Detect and refine atomic sites over an entire image, both phases at once.

    ``spacing_px`` should be the *smaller* of the two phases' site spacings: the peak
    finder's exclusion radius has to fit inside the tightest lattice present, and using the
    substrate's spacing on a finer-spaced film silently merges the film's columns in pairs.

    ``min_edge_dist_px`` is a floor under the edge exclusion, so no site is reported closer than
    that to the frame boundary however small ``spacing_px`` is.  A site near the edge has an
    incomplete window on one side, which biases its refined position outward and gives it no
    neighbour to be indexed against -- and unlike most bad sites it looks perfectly ordinary.

    ``valid_mask`` is where the image came from real data.  **Pass it for a drift-corrected
    image**: ``SingleImageDrift.apply_to_image`` returns it as ``info["valid"]``, and without it
    the half-columns along the slanted fill wedge are detected and refined like real sites, since
    ``min_edge_dist_px`` can only exclude a rectangular border.

    ``peaks`` accepts an ``(N, 2)`` array of ``(x, y)`` positions from somewhere else -- your own
    detector, a hand-built list -- and skips the search, keeping only the refinement and the edge
    and validity exclusions.  Use :func:`peak_report` to see what the built-in search did and why
    before deciding you need this.

    Returns ``(N, 2)`` array of ``(x, y)`` in pixels.
    """
    smooth = 0.25 * spacing_px if smooth is None else smooth
    refine_radius = 0.42 * spacing_px if refine_radius is None else refine_radius
    edge_margin = 0.5 * spacing_px if edge_margin is None else edge_margin
    edge_margin = max(edge_margin, float(min_edge_dist_px))
    im = np.nan_to_num(np.asarray(image, float), nan=float(np.nanmedian(np.asarray(image, float))))
    if peaks is None:
        xy = find_peaks(
            im,
            min_distance=spacing_px,
            threshold_rel=threshold_rel,
            smooth=smooth,
            edge_margin=edge_margin,
            threshold_mode=threshold_mode,
            valid_mask=valid_mask,
        )
    else:
        xy = np.asarray(peaks, float).reshape(-1, 2)
        xy = xy[_keep_away_from_invalid(im.shape, xy, edge_margin, valid_mask)]
    if len(xy) == 0:
        return xy
    return refine_peaks(im, xy, radius=refine_radius, mode=refine)


def _keep_away_from_invalid(shape, xy, margin, valid_mask=None):
    """Mask of sites at least ``margin`` px from the frame border and from any invalid pixel."""
    ny, nx = shape
    keep = (
        (xy[:, 0] >= margin)
        & (xy[:, 0] <= nx - 1 - margin)
        & (xy[:, 1] >= margin)
        & (xy[:, 1] <= ny - 1 - margin)
    )
    if valid_mask is not None and margin > 0:
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(np.asarray(valid_mask, bool))
        ix = np.clip(np.rint(xy[:, 0]).astype(int), 0, nx - 1)
        iy = np.clip(np.rint(xy[:, 1]).astype(int), 0, ny - 1)
        keep &= dist[iy, ix] >= margin
    return keep


def peak_report(
    image,
    spacing_px,
    threshold_rel=0.12,
    smooth=None,
    min_edge_dist_px=10.0,
    valid_mask=None,
    threshold_mode="prominence",
    interface_y=None,
    ax=None,
    figsize=(9.0, 9.0),
    ms=5.0,
    peaks=None,
    min_spacing_px=None,
):
    """
    Show every candidate maximum in the image and why each one was kept or dropped.

    For when detection is missing sites that are plainly there.  Guessing between the possible
    causes is slow, and they need different fixes, so this separates them: candidates are collected
    at a deliberately *fine* exclusion radius (0.3 of ``spacing_px``) and then each of the real
    filter's tests is applied one at a time, so a dropped site can be attributed.

    ==========================  =====================================================  ============
    marker                      why it was dropped                                     what to do
    ==========================  =====================================================  ============
    green circle                kept                                                   --
    red x                       contrast below ``threshold_rel`` of the frame scale     lower it, or
                                                                                       try the other
                                                                                       threshold_mode
    magenta +                   a brighter maximum within ``0.7 * spacing_px``          lower spacing_px
    yellow square               within ``min_edge_dist_px`` of the border or of         expected
                                invalid data
    ==========================  =====================================================  ============

    The magenta and red cases are the two that look identical on screen and are usually confused.
    Magenta means the *spacing* is too large -- the exclusion radius of the local-maximum test is
    swallowing a real site next to a brighter one, which is what happens at an interface where the
    rows across the boundary sit closer together than the rows within either phase.  Red means the
    *threshold* is too high for that site's contrast, which is what happens on the dimmer side of an
    interface between phases of different brightness.

    **Pass ``peaks`` to diagnose the detector you are actually using.**  Without it this describes
    :func:`~quantem.imaging.drift_single_image.find_peaks`, which is only the right thing to look at
    if that is what produced your sites.  Diagnosing one detector while a different one supplies the
    data is worse than no diagnostic: the report disagrees with the picture and neither is wrong.

    With ``peaks`` given, those are the kept set -- from ``Lattice.get_maxima_2D``, your own code,
    anywhere -- and the categories become properties measured off the image rather than assumptions
    about the detector's internals, so they hold for any finder.  Each candidate that the finder did
    *not* keep is attributed to whichever is true of it, and each maps onto one knob:

    ==============================  =========================================  ==================
    category                        measured as                                knob
    ==============================  =========================================  ==================
    ``below_threshold``             dimmer than 0.6x the median kept site      intensity floor
    ``suppressed_by_spacing``       within ``min_spacing_px`` of a *brighter*  minimum spacing
                                    kept site
    ``near_edge_or_invalid``        inside the border or the fill wedge        edge margin, mask
    ``unexplained``                 none of the above                          none -- look at it
    ==============================  =========================================  ==================

    ``unexplained`` is the interesting one: a candidate the finder dropped for a reason this report
    cannot see.  A handful is normal (noise maxima on a peak's shoulder); a systematic pattern of
    them means the finder is doing something you have not accounted for.

    Returns ``(fig, info)``; ``info`` holds the ``(N, 2)`` arrays plus counts, and, when
    ``interface_y`` is given, the kept fraction binned by distance from the interface -- which is
    the number to look at when the complaint is "it misses sites near the interface".
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import maximum_filter, minimum_filter

    im = np.asarray(image, float)
    im = np.nan_to_num(im, nan=float(np.nanmedian(im)))
    spacing_px = float(spacing_px)
    smooth = 0.25 * spacing_px if smooth is None else float(smooth)
    margin = max(0.5 * spacing_px, float(min_edge_dist_px))
    work = gaussian_filter(im, smooth) if smooth > 0 else im

    fine = max(int(round(spacing_px * 0.3)), 3)
    coarse = max(int(round(spacing_px * 0.7)), 3)
    cand = work == maximum_filter(work, size=fine, mode="nearest")
    survives = work == maximum_filter(work, size=coarse, mode="nearest")
    if threshold_mode == "prominence":
        contrast = work - minimum_filter(
            work, size=max(int(round(spacing_px * 1.2)), 3), mode="nearest"
        )
    else:
        contrast = work - gaussian_filter(im, max(spacing_px * 1.5, 2.0))
    scale = np.percentile(contrast, 99.5) - np.percentile(contrast, 50.0)
    above = contrast > threshold_rel * scale

    yx = np.argwhere(cand)
    xy = yx[:, ::-1].astype(float)
    ok_edge = _keep_away_from_invalid(im.shape, xy, margin, valid_mask)
    min_spacing_px = 0.7 * spacing_px if min_spacing_px is None else float(min_spacing_px)
    unexplained = np.zeros(len(xy), bool)

    if peaks is None:
        ok_thr = above[yx[:, 0], yx[:, 1]]
        ok_excl = survives[yx[:, 0], yx[:, 1]]
        # Attributed in a fixed order, so each candidate is counted once and the counts add up.
        kept = ok_edge & ok_thr & ok_excl
        edge_drop = ~ok_edge
        thr_drop = ok_edge & ~ok_thr
        excl_drop = ok_edge & ok_thr & ~ok_excl
    else:
        from scipy.spatial import cKDTree

        given = np.asarray(peaks, float).reshape(-1, 2)
        bright = work[yx[:, 0], yx[:, 1]]
        # A candidate counts as kept if the finder returned a peak on it.  Matching radius is a
        # small fraction of the spacing: subpixel refinement moves a peak by well under a pixel,
        # so anything further away is a different feature, not the same one relocated.
        match = max(2.0, 0.15 * spacing_px)
        if len(given):
            tree = cKDTree(given)
            kept = tree.query(xy)[0] <= match
            gy = np.clip(np.rint(given[:, 1]).astype(int), 0, work.shape[0] - 1)
            gx = np.clip(np.rint(given[:, 0]).astype(int), 0, work.shape[1] - 1)
            kept_bright = work[gy, gx]
            ref = float(np.median(kept_bright)) if len(kept_bright) else 0.0
            # Nearest *brighter* kept site: the greedy brightest-first spacing filter every finder
            # of this family uses drops a candidate that sits inside one of those.
            crowded = np.zeros(len(xy), bool)
            order = np.argsort(kept_bright)[::-1]
            for i, (px, py) in enumerate(xy):
                brighter = given[order[kept_bright[order] > bright[i]]]
                if len(brighter):
                    crowded[i] = (
                        np.hypot(*(brighter - np.array([px, py])).T).min() < min_spacing_px
                    )
        else:
            kept = np.zeros(len(xy), bool)
            crowded = np.zeros(len(xy), bool)
            ref = 0.0
        dim = bright < 0.6 * ref
        edge_drop = ~ok_edge & ~kept
        thr_drop = ~kept & ok_edge & dim
        excl_drop = ~kept & ok_edge & ~dim & crowded
        unexplained = ~kept & ok_edge & ~dim & ~crowded
        kept = kept & ok_edge

    info = {
        "kept": xy[kept],
        "below_threshold": xy[thr_drop],
        "suppressed_by_spacing": xy[excl_drop],
        "near_edge_or_invalid": xy[edge_drop],
        "unexplained": xy[unexplained],
        "n_candidates": len(xy),
        "n_given": 0 if peaks is None else len(np.asarray(peaks).reshape(-1, 2)),
        "threshold_value": float(threshold_rel * scale),
        "spacing_px": spacing_px,
        "min_spacing_px": min_spacing_px,
        "margin_px": margin,
        "diagnosed": "find_peaks" if peaks is None else "the peaks you passed in",
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    lo, hi = np.nanpercentile(im, [0.5, 99.5])
    ax.imshow(im, cmap="gray", origin="lower", vmin=lo, vmax=hi)
    # The valid region as one outline rather than a marker per excluded site.  Those markers are
    # the most numerous thing on the plot and the least informative -- they say "the border is the
    # border" a few hundred times, and they hide the sites that were dropped for a real reason.
    if valid_mask is not None:
        from scipy.ndimage import distance_transform_edt

        inside = distance_transform_edt(np.asarray(valid_mask, bool)) >= margin
        ax.contour(inside.astype(float), levels=[0.5], colors="#ffea00", linewidths=1.0)
    else:
        ny_i, nx_i = im.shape
        ax.add_patch(
            plt.Rectangle(
                (margin, margin),
                nx_i - 1 - 2 * margin,
                ny_i - 1 - 2 * margin,
                fill=False,
                ec="#ffea00",
                lw=1.0,
            )
        )
    ax.plot(
        [],
        [],
        "-",
        color="#ffea00",
        lw=1.0,
        # Stated with its provenance: the margin is max(min_edge_dist_px, 0.5 * spacing), because a
        # site closer to the edge than the refinement radius has a truncated window.  The requested
        # value is a floor, so printing only the result invites "why is that not the 10 I asked for".
        label=f"valid region ({margin:.0f} px = max(edge {float(min_edge_dist_px):.0f}, "
        f"spacing/2), {len(info['near_edge_or_invalid'])} outside)",
    )

    for key, marker, colour, label in (
        ("kept", "o", "#00e676", "kept"),
        ("below_threshold", "x", "#ff5252", "dimmer than the kept sites"),
        ("suppressed_by_spacing", "+", "#e040fb", f"within {min_spacing_px:.0f} px of a brighter"),
        ("unexplained", "d", "#40c4ff", "dropped, reason not measurable"),
    ):
        p = info[key]
        ax.plot(
            p[:, 0] if len(p) else [],
            p[:, 1] if len(p) else [],
            marker,
            ms=ms,
            mfc="none",
            mec=colour,
            mew=0.9,
            ls="none",
            label=f"{label} ({len(p)})",
        )
    if interface_y is not None:
        ax.axhline(float(interface_y), color="#ffffff", lw=0.6, ls=":", alpha=0.7)
        # Binned over candidates that are *in play* -- edge and fill-wedge candidates are excluded
        # from the denominator, since a border ring full of them would otherwise read as a
        # detection failure in the interior.  Reported as (lo, hi, n_in_play, n_kept, fraction).
        edges = np.array([-np.inf, -60, -40, -20, 0, 20, 40, 60, np.inf])
        rows, dy = [], xy[:, 1] - float(interface_y)
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            m = ((dy >= lo_e) & (dy < hi_e)) & ~edge_drop
            rows.append(
                (
                    float(lo_e),
                    float(hi_e),
                    int(m.sum()),
                    int(kept[m].sum()),
                    float(kept[m].mean()) if m.sum() else float("nan"),
                )
            )
        info["by_distance_from_interface"] = rows
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        fontsize=7,
        loc="lower right",
        frameon=True,  # a style with legend.frameon False would otherwise drop the box entirely
        facecolor="white",
        edgecolor="0.6",
        framealpha=0.55,
        labelcolor="black",
    )
    ax.set_title(
        f"{len(xy)} candidate maxima at spacing {spacing_px:.1f} px -- "
        f"diagnosing {info['diagnosed']}",
        fontsize=8,
    )
    return fig, info


def median_nn_spacing(xy):
    """Median nearest-neighbour distance over a set of sites; ``nan`` for fewer than two."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) < 2:
        return float("nan")
    from scipy.spatial import cKDTree

    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


def infer_site_spacing(image, xy, min_sites=8, max_spread=0.25):
    """
    Site spacing in pixels, from the sites if they can support it and from the image if not.

    Returns ``(spacing_px, source)`` with ``source`` in ``{"sites", "image"}``.

    The median nearest-neighbour distance of the sites is the right answer when the sites are a
    *sample of the lattice*, and nonsense otherwise -- a handful of hand-placed sites scattered
    over a frame report their own sparseness as the lattice constant.  That is worth guarding
    carefully because every use of the spacing fails hard in the same direction: the peak finder's
    exclusion radius, the smoothing length, the fit window and the duplicate threshold all
    degrade gently when the spacing is too small and catastrophically when it is too large.  Ten
    clicks on a 14 px lattice gave 42 px, which left 8 detectable peaks in the whole image instead
    of 144 and merged two distinct clicks as duplicates.

    The test is whether the nearest-neighbour distances are *tight*, which is exactly the property
    being relied on, rather than a count (10 clicks along one row are a perfectly good sample, and
    3 clicks anywhere are not).  Measured ``IQR/median``: 0.03 for a full synthetic lattice, 0.05
    for ten clicks along a row, 0.16 for 1061 real detections with their missing and spurious
    sites -- against 0.39-0.46 for scattered clicks.  The 0.25 default sits in that gap; it is a
    judgement call, not a derived number.

    Falling back to :func:`~quantem.imaging.drift_single_image.estimate_site_spacing` (the
    autocorrelation's shortest lattice repeat) is safe on a dumbbell image, which is not obvious:
    it returns 29 px on the test frame, the lattice repeat, not the 10.1 px intra-dumbbell
    separation, because that separation is not a lattice translation.
    """
    from quantem.imaging.drift_single_image import estimate_site_spacing

    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) >= max(int(min_sites), 2):
        from scipy.spatial import cKDTree

        d, _ = cKDTree(xy).query(xy, k=2)
        nn = d[:, 1]
        med = float(np.median(nn))
        q1, q3 = np.percentile(nn, [25, 75])
        if med > 0 and (q3 - q1) / med <= max_spread:
            return med, "sites"
    try:
        return float(estimate_site_spacing(np.nan_to_num(np.asarray(image, float)))), "image"
    except RuntimeError:  # featureless image; the sites are all there is to go on
        med = median_nn_spacing(xy)
        return (med, "sites") if np.isfinite(med) else (float("nan"), "unknown")


def snap_sites(
    image,
    sites,
    max_move=8.0,
    fit_radius=None,
    mode="gaussian",
    dedupe=True,
    indices=None,
    min_edge_dist_px=10.0,
    max_passes=4,
    tol=0.01,
    snap_to_peaks=False,
    spacing_px=None,
    smooth=None,
    threshold_rel=0.06,
):
    """
    Fit each site in a window around where it is now, and move it to the fitted centre.

    Written for hand-placed sites.  A click lands within a few pixels of the column you meant,
    which is close enough to see and not close enough to measure: the site indices along a row
    come from rank order, so a click is only *identifying* a column, and the position still has
    to come from the data.

    The operation is deliberately **local**.  For each site, take a window of ``fit_radius``
    pixels (default ``max_move``) centred on it, fit a 2D Gaussian on a planar background inside
    that window (:func:`~quantem.imaging.drift_single_image.fit_peaks_gaussian`), and move the
    site to the fitted centre.  Repeated up to ``max_passes`` times, until the median step falls
    below ``tol``: the first window is centred on the click rather than on the column, and the
    second pass sees one centred on the answer.  It converges in two or three passes (~0.06 px on
    the first, 0.000 on the second), so a second call is a no-op and can be used as a check.
    ``mode="com"`` uses the windowed centroid instead.  Nothing here
    needs to know the lattice spacing, and nothing needs a list of the image's peaks -- which is
    the point.  An earlier version snapped to the nearest detected peak first, which meant it
    needed both, and a handful of scattered clicks cannot supply either: their median
    nearest-neighbour distance was 42 px on a 14 px lattice, so the peak finder's exclusion radius
    left 8 findable peaks in the whole image and every click reported "no peak within 8 px" with
    the peaks plainly there.  A window around the click has no such failure mode.

    ``snap_to_peaks=True`` restores that first stage, for a *dense* set of sites where the peak
    list is meaningful and the discrete which-column choice is worth making explicitly; it needs
    ``spacing_px``, inferred by :func:`infer_site_spacing` when not given.

    A fit that ends up further than ``max_move`` from where the site started is **reverted and
    flagged**, not clamped to the bound.  Clamping is the trap: the fit converges, returns a
    plausible number, and nothing tells you the position came from the bound rather than the data
    -- and since the clamped fraction grows with how bad the data is, it flatters the worst image
    in a set.

    **On a dumbbell image a single Gaussian will move sites onto one atom of the pair.**  Not
    hypothetical: measured 5.15 px on the test frame, half a dumbbell (0.72 A) to two decimal
    places.  If a site is a dumbbell *centroid* -- a maximum only in the smoothed image, with no
    peak there at all in the raw one -- then a Gaussian does not refine it, it redefines it, and
    the result looks perfect: sites on atoms, in rows, at the right spacing.  This is reported
    rather than prevented, because which one you want is your call: ``info["text"]`` says how far
    the Gaussian centres sit from the centroids of the same windows, and ``mode="com"`` gives the
    estimator :func:`find_all_sites` uses.  It is one undo step either way.

    ``indices`` restricts the work to a subset (e.g. only the sites just added); everything else
    passes through untouched but still takes part in duplicate detection.  Sites within
    ``min_edge_dist_px`` of the frame boundary are left alone and counted: their window is
    incomplete on one side, which pulls the fitted centre inward while looking entirely ordinary.

    Returns ``(sites, info)``.  ``info["keep"]`` is a mask over the *input* sites, so any parallel
    per-site array can be carried through the dedupe with ``arr[info["keep"]]``.

    >>> cleaned, info = snap_sites(image, editor.sites, max_move=8.0)
    >>> print(info["text"])
    """
    im = np.nan_to_num(np.asarray(image, float))
    xy = np.asarray(sites, float).reshape(-1, 2)
    n = len(xy)
    max_move = float(max_move)
    fit_radius = max_move if fit_radius is None else float(fit_radius)
    info: dict = {
        "refined": np.zeros(n, bool),
        "moved": np.zeros(n),
        "too_far": np.zeros(n, bool),
        "on_edge": np.zeros(n, bool),
        "passes": 0,
        "last_shift": float("nan"),
        "n_unconverged": 0,
        "keep": np.ones(n, bool),
        "n_merged": 0,
        "mode_used": mode,
        "n_gaussian": 0,
        "n_centroid": 0,
        "sigma_px": float("nan"),
        "fit": {},
        "centroid_offset": float("nan"),
        "spacing_px": float("nan"),
        "spacing_source": "not needed",
        "text": "no sites to refine",
    }
    if mode not in ("gaussian", "com"):
        raise ValueError(f"unknown refine mode {mode!r} (gaussian, com)")
    if n == 0:
        return xy.copy(), info

    sel = np.arange(n) if indices is None else np.unique(np.asarray(indices, int))
    out = xy.copy()
    seeds = xy[sel].copy()
    started = np.ones(len(sel), bool)
    no_peak = np.zeros(len(sel), bool)  # only the peak-snap path can set this

    if snap_to_peaks:
        if spacing_px is None:
            spacing_px, source = infer_site_spacing(im, xy)
        else:
            spacing_px, source = float(spacing_px), "given"
        info["spacing_px"], info["spacing_source"] = spacing_px, source
        smooth = 0.25 * spacing_px if smooth is None else float(smooth)
        work = gaussian_filter(im, smooth) if smooth > 0 else im
        peaks = find_peaks(
            work,
            min_distance=spacing_px,
            threshold_rel=threshold_rel,
            smooth=0.0,
            edge_margin=0.0,
        )
        nearest = np.full(len(sel), np.inf)
        if len(peaks):
            from scipy.spatial import cKDTree

            # Queried unbounded, then thresholded, so a site that finds nothing can still say how
            # far away the nearest peak was.  "No peak within 8 px" alone reads as a fault in the
            # image; the distance says where to look.
            nearest, j = cKDTree(peaks).query(xy[sel])
            started = nearest <= max_move
            seeds[started] = peaks[j[started]]
        no_peak = ~started
        info["nearest_peak"] = nearest
        info["n_peaks"] = len(peaks)

    # Sites whose window would hang off the frame: not refined, and said so.  Their window is
    # incomplete on one side, which pulls the fitted centre inward, and the result looks ordinary.
    ny, nx = im.shape
    edge = min(float(min_edge_dist_px), fit_radius)
    on_edge = (
        (xy[sel, 0] < edge)
        | (xy[sel, 0] > nx - 1 - edge)
        | (xy[sel, 1] < edge)
        | (xy[sel, 1] > ny - 1 - edge)
    )
    info["on_edge"][sel] = on_edge
    started &= ~on_edge

    if started.any():
        target = sel[started]
        # Iterate to convergence, in one call.  A single pass leaves a little on the table because
        # the window starts centred on the click rather than on the column; a second pass sees a
        # window centred on the answer.  Measured: pass 1 moves ~0.06 px, pass 2 onward 0.000 px, so
        # this converges in two or three and the cap is a backstop, not the usual exit.  Pressing
        # refine repeatedly used to be how you got here, which is a state the button should not
        # have -- and it means a second press is now a genuine no-op check.
        cur = seeds[started].copy()
        com0 = None
        shift = np.zeros(len(cur))
        for step in range(int(max_passes)):
            com = refine_peaks(im, cur, radius=fit_radius, mode="com")
            if com0 is None:
                com0 = com.copy()
            if mode == "gaussian":
                fitted, fit_info = fit_peaks_gaussian(
                    im, cur, radius=fit_radius, max_move=fit_radius
                )
                ok = fit_info["ok"]
                # How far the Gaussian centres sit from the centroids of the same windows.  On
                # single columns these agree; a large offset is the dumbbell signature.
                if ok.any():
                    info["centroid_offset"] = float(np.median(np.hypot(*(fitted[ok] - com[ok]).T)))
                    info["sigma_px"] = float(np.median(fit_info["sigma"][ok]))
                # Per site, not per image: where a Gaussian does not settle, use the centroid for
                # that site rather than leaving it where the click landed.  A window holding two
                # atoms of a dumbbell cannot be fitted by one Gaussian -- its width runs to the
                # bound -- and on such an image most sites take this path, which is why the counts
                # are reported.  Refusing the whole set instead leaves a click on a position that
                # came from a mouse rather than from the data, which is worse than a centroid.
                info["fit"] = fit_info
                info["n_gaussian"], info["n_centroid"] = int(ok.sum()), int((~ok).sum())
                nxt = np.where(ok[:, None], fitted, com)
            else:
                nxt = com
                info["n_gaussian"], info["n_centroid"] = 0, len(nxt)
            shift = np.hypot(*(nxt - cur).T)
            cur = nxt
            info["passes"] = step + 1
            info["last_shift"] = float(np.median(shift))
            if np.median(shift) < tol:
                break
        # A site still moving when the passes run out has not been measured, it has been sampled
        # mid-oscillation: on a dumbbell with a window wide enough to hold both atoms the fit hops
        # between them and never settles, and the position you get depends on which pass you
        # stopped on.  Those fall back to the centroid of the *first* window, which is defined and
        # repeatable, and are counted.
        # Deliberately much looser than the loop's own break tolerance, which is a *median* over
        # sites: an individual site sitting at 0.02 px when the median has reached 0.005 px is
        # converged, not oscillating.  This is here to catch the multi-pixel hopping a two-atom
        # window produces (6.3 px per pass, measured), so the bar is half a pixel.
        unconverged = shift > max(0.5, 10.0 * tol)
        if unconverged.any() and com0 is not None:
            cur = np.where(unconverged[:, None], com0, cur)
            info["n_unconverged"] = int(unconverged.sum())
        if mode == "gaussian":
            # Counted from what each site actually ended up using, so the two add up: a site can
            # fail the width bound *and* fail to settle, and must not be subtracted twice.
            used_gaussian = ok & ~unconverged
            info["n_gaussian"] = int(used_gaussian.sum())
            info["n_centroid"] = int(len(cur) - used_gaussian.sum())
        moved = np.hypot(*(cur - xy[target]).T)
        good = moved <= max_move
        out[target[good]] = cur[good]
        info["refined"][target[good]] = True
        info["moved"][target[good]] = moved[good]
        info["too_far"][target[~good]] = True

    if dedupe and len(out) > 1:
        from scipy.spatial import cKDTree

        # Two clicks on one column is the failure this editor exists to prevent: a duplicate in a
        # row shifts every site index after it by one.  After fitting, two sites on one column land
        # within a fraction of a pixel of each other, so the threshold is absolute -- no lattice
        # spacing needed, and no risk of merging genuinely adjacent columns.  Keep the lower index,
        # which is the original detection when the duplicate came from a click.
        pairs = cKDTree(out).query_pairs(min(2.0, 0.25 * fit_radius))
        drop = {max(a, b) for a, b in pairs}
        if drop:
            keep = np.ones(len(out), bool)
            keep[sorted(drop)] = False
            info["keep"] = keep
            info["n_merged"] = int(len(drop))
            out = out[keep]

    moved_ok = info["moved"][info["refined"]]
    how = f"{info['n_gaussian']} gaussian" + (
        f", {info['n_centroid']} centroid" if info["n_centroid"] else ""
    )
    bits = [f"refined {int(info['refined'].sum())}/{len(sel)} ({how})"]
    if len(moved_ok):
        bits.append(f"median move {np.median(moved_ok):.2f} px, max {moved_ok.max():.2f} px")
    if info["passes"]:
        settled = "settled" if info["last_shift"] < tol else "still moving"
        bits.append(
            f"{info['passes']} pass{'es' if info['passes'] != 1 else ''}, {settled} "
            f"({info['last_shift']:.3f} px on the last)"
        )
    if info["n_unconverged"]:
        bits.append(
            f"{info['n_unconverged']} never settled (the fit oscillates in this window) -- "
            "those used the centroid instead"
        )
    n_edge = int(info["on_edge"].sum())
    if n_edge:
        bits.append(f"{n_edge} within {edge:.0f} px of the frame edge (left alone)")
    if no_peak.any():
        near = float(np.median(info["nearest_peak"][no_peak]))
        bits.append(
            f"{int(no_peak.sum())} with no peak within {max_move:.1f} px (left alone; nearest of "
            f"the {info['n_peaks']} peaks found was {near:.1f} px away)"
        )
    n_rejected = int(info["too_far"].sum())
    if n_rejected:
        bits.append(
            f"{n_rejected} fit did not settle within {max_move:.1f} px (left alone, not clamped)"
        )
    if info["n_merged"]:
        bits.append(f"{info['n_merged']} duplicate(s) merged")
    # The window is the one number to tune, and the fit measures the column width it should be
    # matched to, so report both rather than leaving it to be guessed.  Around 3 sigma is the broad
    # optimum: measured on a synthetic lattice at four noise levels, the error rises steeply below
    # 2.5 sigma (the window clips the peak, and cannot hold the peak *and* the click offset at
    # once) and gently above ~4.5 sigma as the neighbours enter the window.
    sig = info["sigma_px"]
    too_wide = False
    if np.isfinite(sig) and sig > 0:
        ratio = fit_radius / sig
        too_wide = ratio > 5.0
        bits.append(f"fitted column sigma {sig:.1f} px (window = {ratio:.1f} sigma)")
        # A fitted width that is a fixed fraction of its own window is not a column width -- the
        # Gaussian is describing the whole window rather than a peak inside it, which is what
        # happens when the window holds more than one column.  Advising "go wider" from that number
        # chases its own tail: 8 px gave sigma 5.2, 16 px gave 10.3, and the answers got worse
        # (1.12 -> 3.50 px against the detector).  So say what is actually wrong instead.
        if sig > 0.55 * fit_radius:
            bits.append(
                f"note: the fitted width {sig:.1f} px is set by the {fit_radius:.1f} px window, "
                "not by a column -- this window holds more than one peak (a dumbbell, typically). "
                "Narrow it, or use mode='com'"
            )
        elif ratio < 2.5:
            bits.append(
                f"note: a {fit_radius:.1f} px window is tight for a {sig:.1f} px column -- try "
                f"{3.0 * sig:.0f} px, since the window has to hold the peak and the click offset"
            )
        elif ratio > 5.0:
            bits.append(
                f"note: a {fit_radius:.1f} px window is wide for a {sig:.1f} px column, so it is "
                f"mostly background and may reach the neighbours -- try {3.0 * sig:.0f} px"
            )
    offset = info["centroid_offset"]
    # Skipped when the window is already flagged as too wide: that alone pulls the centroid off the
    # peak, so reporting it as a second finding would be two notes for one cause.
    if np.isfinite(offset) and offset > 1.0 and not too_wide:
        bits.append(
            f"note: the gaussian centres sit {offset:.1f} px from the centroids of the same "
            "windows -- if these sites are dumbbell centroids the fit has moved onto single "
            "atoms; u to undo, or mode='com'"
        )
    info["text"] = "; ".join(bits)
    return out, info


def estimate_interface_row(image, exclude_fraction=0.08, smooth_rows=None):
    """
    First guess at the interface row: the steepest gradient in the row-mean intensity.

    The profile is smoothed before differentiating, and the smoothing length is taken from the
    profile's *own* dominant periodicity rather than being a fixed number.  That matters more
    than it sounds.  A fixed-width step filter resonates whenever its width happens to be a
    near-multiple of the lattice row pitch: measured on synthetic data, widths of 8, 15, 25 and
    60 rows all landed within +3 to +10 rows of the truth, while 40 rows -- about 2.2 film
    periods -- came out 15 rows off *on the other side*.  Sizing the kernel from the fringes it
    has to suppress removes the failure mode instead of tuning around it.

    Still only a starting point.  Two phases of similar mean intensity will defeat it, and it
    knows nothing about where the *lattice* changes, which is the real definition of the
    interface.  It exists to put :class:`InterfaceLine` somewhere sensible so you can drag it the
    rest of the way.

    Note also what it returns: the *contrast midpoint*, which lies between the last row of one
    phase and the first row of the other and so does not have to coincide with any nominal
    interface coordinate.  On the synthetic stack used for testing it comes out ~13 rows below
    the value the simulator calls ``y_interface``, which is the tiling boundary -- the difference
    is roughly the interfacial gap.  Either separates the two phases correctly, which is all the
    downstream measurement needs.

    ``exclude_fraction`` ignores that fraction of rows at each end, where the derivative of a
    smoothed profile is dominated by the edge.
    """
    im = np.nan_to_num(np.asarray(image, float))
    prof = im.mean(axis=1)
    prof = prof - prof.mean()
    n = len(prof)
    if smooth_rows is None:
        # dominant period in the profile = the lattice row pitch; smooth over one of them
        spec = np.abs(np.fft.rfft(prof * np.hanning(n)))
        spec[: max(4, n // 200)] = 0.0
        peak = int(np.argmax(spec))
        smooth_rows = n / peak if peak > 0 else 8.0
    sigma = float(np.clip(smooth_rows, 2.0, n / 8.0))
    grad = np.abs(np.gradient(gaussian_filter1d(prof, sigma, mode="nearest")))
    guard = max(int(exclude_fraction * n), int(2 * sigma))
    if n - 2 * guard <= 0:
        return n // 2
    return int(np.argmax(grad[guard:-guard]) + guard)


# ===========================================================================
# row grouping
# ===========================================================================


def group_rows(xy, expected_spacing=None, smooth_frac=0.18, min_sites=4, prominence_frac=0.12):
    """
    Assign each site to a horizontal atomic row.

    Rows are found as peaks in the smoothed histogram of site ``y``, then each site is
    attached to its nearest row centre.  Peak-finding on the projection rather than
    clustering on ``y`` directly, because it copes with the two phases having different row
    spacings and with the rows nearest the interface being pulled off their lattice by
    interface roughness -- both of which break a fixed gap threshold.

    Relies on rows being horizontal in the image, which holds for this distortion model even
    before correction: shear displaces ``x`` as a function of ``y``, so a row at constant
    ``y`` stays at constant ``y`` and only slides sideways.  That is precisely why the
    sideways slide is confusable with relaxation.

    Returns ``(row_index, row_centres)`` with ``row_index = -1`` for sites in rows that ended
    up with fewer than ``min_sites`` members.
    """
    from scipy.signal import find_peaks as _find_peaks

    xy = np.asarray(xy, float)
    y = xy[:, 1]
    if len(y) < 2 * min_sites:
        raise RuntimeError(f"only {len(y)} sites -- too few to group into rows")
    if expected_spacing is None:
        # median nearest-neighbour distance is a decent proxy for the row pitch and
        # needs no prior knowledge of either phase
        from scipy.spatial import cKDTree

        d, _ = cKDTree(xy).query(xy, k=2)
        expected_spacing = float(np.median(d[:, 1]))

    # Pad the histogram by a full row pitch each side and smooth against zeros.  With the
    # range ending at the data and ``mode="nearest"``, the outermost row sits on the boundary
    # and edge replication turns it into a plateau -- it loses its prominence and gets merged
    # into its neighbour.  That silently drops the first and last rows, which are the ones
    # furthest from the interface and therefore the cleanest reference available.
    lo, hi = y.min() - expected_spacing, y.max() + expected_spacing
    nbins = max(int((hi - lo) / max(expected_spacing * 0.12, 0.5)), 32)
    hist, edges = np.histogram(y, bins=nbins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    sigma = max(smooth_frac * expected_spacing / (edges[1] - edges[0]), 0.8)
    prof = gaussian_filter1d(hist.astype(float), sigma, mode="constant", cval=0.0)
    distance = max(0.55 * expected_spacing / (edges[1] - edges[0]), 1.0)
    idx, _ = _find_peaks(prof, distance=distance, prominence=prominence_frac * prof.max())
    if len(idx) < 2:
        raise RuntimeError("could not identify atomic rows from the site positions")
    row_y = centres[idx]

    row = np.argmin(np.abs(y[:, None] - row_y[None, :]), axis=1)
    # Replace the histogram-bin centres with the mean y of each row's actual members.  The
    # bins are ~0.1 of a row pitch wide, and a centre pinned to a bin carries that much
    # quantisation -- enough to put a 2% error on the out-of-plane pitch, which is the same
    # size as the strain the pitch is being measured to find.  Assignment is unaffected
    # (nothing moves by half a pitch), so this is a free refinement.
    for j in range(len(row_y)):
        members = row == j
        if members.sum():
            row_y[j] = float(y[members].mean())

    counts = np.bincount(row, minlength=len(row_y))
    row = np.where(counts[row] >= min_sites, row, -1)
    return row, row_y


@dataclass
class RowFit:
    """One atomic row: its sites, their indices along it, and the straight-line fit."""

    row_index: int
    y_centre: float
    x: NDArray  # site x, sorted
    index: NDArray  # 0, 1, 2, ... along the row
    spacing: float  # fitted slope, pixels per site
    intercept: float
    residual: NDArray  # x - (intercept + spacing * index)
    phase: str = "?"  # "substrate" | "film"
    n_missing: int = 0  # index gaps implied by outlier spacings

    @property
    def rms_residual(self) -> float:
        return float(np.sqrt(np.mean(self.residual**2)))


# ===========================================================================
# the measurement
# ===========================================================================


@dataclass
class InterfaceCoherency:
    """
    Coherency of an interface, from ``x``-versus-site-index along rows either side of it.

    Attributes are filled by :meth:`measure`.  ``pixel_size`` is only needed to report in
    Angstrom; everything works in pixels without it.
    """

    sites: NDArray
    interface_y: float
    pixel_size: float | None = None
    label: str = "coherency"
    # Smallest film/substrate mismatch worth calling relaxation.  A significance test alone
    # is not enough: it answers "can I distinguish these two numbers", not "does the
    # difference mean anything".  With a few clean rows the standard error on the mean
    # spacing gets down to ~1e-4, so a 0.03% mismatch -- about 0.001 Angstrom, far below any
    # honest strain precision on a real instrument -- comes out "significant" and a perfectly
    # coherent interface gets reported as relaxed.  0.3% is a defensible floor for STEM
    # strain measurement; raise it if your data are worse than that.
    min_mismatch: float = 0.003

    rows: list = field(default_factory=list)
    row_centres: NDArray | None = None
    reference_spacing: float | None = None
    reference: str = "substrate"
    warnings_raised: list = field(default_factory=list)

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_sites(
        cls, sites, interface_y, pixel_size=None, label="coherency", min_mismatch=0.003
    ):
        return cls(
            sites=np.asarray(sites, float),
            interface_y=float(interface_y),
            pixel_size=pixel_size,
            label=label,
            min_mismatch=min_mismatch,
        )

    def _warn(self, msg):
        self.warnings_raised.append(msg)
        warnings.warn(f"[{self.label}] {msg}", stacklevel=3)

    # ------------------------------------------------------------------ measure
    def measure(
        self,
        n_rows_each_side=4,
        expected_spacing=None,
        min_sites=6,
        reference="substrate",
        max_spacing_outlier=0.35,
        skip_rows=0,
    ):
        """
        Group sites into rows, index each row, and fit its spacing.

        ``n_rows_each_side`` rows of substrate and of film are kept, counted outward from the
        interface.  ``skip_rows`` drops that many rows immediately either side of it; the default is
        0, so nothing is discarded unless you ask.  Raising it to 1 is defensible when the last
        substrate row and the first film row sit in a chemically intermixed, structurally relaxed
        region -- but it is a claim about the sample, and on the test frame it moves the film's
        in-plane spacing, so it should be a deliberate choice rather than a silent default.

        ``reference`` chooses whose spacing flattens the residual plot -- ``"substrate"``
        (the ruler, and the default), ``"film"``, or a number in pixels.

        ``max_spacing_outlier`` guards the index assignment.  Indices come from *rank order*
        along the row, so one missed column shifts every index after it by one and tilts the
        fitted spacing.  Gaps wider than ``(1 + max_spacing_outlier)`` times the row's median
        step are therefore counted as missing sites and the index is advanced accordingly,
        rather than being silently absorbed.
        """
        row, row_y = group_rows(self.sites, expected_spacing=expected_spacing, min_sites=min_sites)
        self.row_centres = row_y

        below = np.nonzero(row_y < self.interface_y)[0]
        above = np.nonzero(row_y >= self.interface_y)[0]
        # Out-of-plane pitch from *all* rows on each side, not just the ones picked for the
        # in-plane fit: it is the one quantity here that linear drift genuinely corrupts
        # (``k`` scales it directly), so measure it on every row available.
        self._row_pitch = {}
        for side, idx in (("substrate", below), ("film", above)):
            ys = np.sort(row_y[idx])
            self._row_pitch[side] = float(np.median(np.diff(ys))) if len(ys) > 2 else np.nan
        below = below[np.argsort(-row_y[below])]  # nearest the interface first
        above = above[np.argsort(row_y[above])]
        pick = [(i, "substrate") for i in below[skip_rows : skip_rows + n_rows_each_side]] + [
            (i, "film") for i in above[skip_rows : skip_rows + n_rows_each_side]
        ]
        if not pick:
            raise RuntimeError("no rows left after skipping; check interface_y and skip_rows")

        self.rows = []
        for ridx, phase in pick:
            sel = row == ridx
            xs = np.sort(self.sites[sel, 0])
            if len(xs) < min_sites:
                continue
            step = np.diff(xs)
            med = float(np.median(step))
            # rank order -> index, advancing across gaps that are whole multiples of the step
            jumps = np.rint(step / med).astype(int)
            jumps = np.clip(jumps, 1, None)
            jumps[np.abs(step / med - jumps) > max_spacing_outlier] = np.rint(
                step[np.abs(step / med - jumps) > max_spacing_outlier] / med
            ).astype(int)
            index = np.concatenate([[0], np.cumsum(jumps)])
            n_missing = int(index[-1] - (len(xs) - 1))

            A = np.column_stack([np.ones(len(xs)), index])
            (intercept, spacing), *_ = np.linalg.lstsq(A, xs, rcond=None)
            self.rows.append(
                RowFit(
                    row_index=int(ridx),
                    y_centre=float(row_y[ridx]),
                    x=xs,
                    index=index.astype(float),
                    spacing=float(spacing),
                    intercept=float(intercept),
                    residual=xs - (intercept + spacing * index),
                    phase=phase,
                    n_missing=n_missing,
                )
            )

        sub = [r for r in self.rows if r.phase == "substrate"]
        film = [r for r in self.rows if r.phase == "film"]
        if not sub or not film:
            self._warn(
                "rows were found on only one side of the interface; the comparison "
                "the measurement rests on is not available"
            )
        #: which phase's spacing flattens the residual panel, kept so the plot can name it
        self.reference = str(reference)
        if reference == "substrate":
            self.reference_spacing = (
                float(np.median([r.spacing for r in sub]))
                if sub
                else float(np.median([r.spacing for r in self.rows]))
            )
        elif reference == "film":
            self.reference_spacing = float(np.median([r.spacing for r in film]))
        else:
            self.reference_spacing = float(reference)

        total_missing = sum(r.n_missing for r in self.rows)
        if total_missing > 0.05 * sum(len(r.x) for r in self.rows):
            self._warn(
                f"{total_missing} sites appear to be missing from the indexed rows.  "
                "Index assignment is rank-based, so gaps matter: check the sites with "
                "SiteEditor before trusting the spacings"
            )
        return self.summary()

    # ------------------------------------------------------------------ results
    def summary(self) -> dict:
        """Per-phase spacings and the coherency verdict, in pixels and Angstrom."""
        px = self.pixel_size

        def stats(rows):
            if not rows:
                return dict(n_rows=0)
            sp = np.array([r.spacing for r in rows])
            out = dict(
                n_rows=len(rows),
                spacing_px=float(sp.mean()),
                spacing_std_px=float(sp.std(ddof=1)) if len(sp) > 1 else 0.0,
                rms_residual_px=float(np.mean([r.rms_residual for r in rows])),
                n_sites=int(sum(len(r.x) for r in rows)),
            )
            if px:
                out["spacing_A"] = out["spacing_px"] * px
                out["spacing_std_A"] = out["spacing_std_px"] * px
                out["rms_residual_pm"] = out["rms_residual_px"] * px * 100.0
            return out

        def add_pitch(out, side):
            pitch = getattr(self, "_row_pitch", {}).get(side, np.nan)
            out["row_pitch_px"] = float(pitch)
            if px and np.isfinite(pitch):
                out["row_pitch_A"] = float(pitch) * px
            return out

        sub = add_pitch(stats([r for r in self.rows if r.phase == "substrate"]), "substrate")
        film = add_pitch(stats([r for r in self.rows if r.phase == "film"]), "film")
        out = dict(
            substrate=sub,
            film=film,
            reference_spacing_px=self.reference_spacing,
            n_rows=len(self.rows),
            warnings=list(self.warnings_raised),
        )
        if sub.get("n_rows") and film.get("n_rows"):
            mismatch = film["spacing_px"] / sub["spacing_px"] - 1.0
            # significance: is the difference larger than the row-to-row scatter allows?
            err = np.hypot(
                sub["spacing_std_px"] / max(np.sqrt(sub["n_rows"]), 1),
                film["spacing_std_px"] / max(np.sqrt(film["n_rows"]), 1),
            )
            out["mismatch"] = float(mismatch)
            out["mismatch_err"] = float(err / sub["spacing_px"]) if err else None
            out["significant"] = bool(err > 0 and abs(mismatch * sub["spacing_px"]) > 3 * err)
            out["above_threshold"] = bool(abs(mismatch) >= self.min_mismatch)
            out["min_mismatch"] = float(self.min_mismatch)
            # A dislocated film has large row residuals, because a step is not a straight
            # line.  The residual ratio therefore tests something the average spacing cannot.
            out["residual_ratio"] = float(
                film["rms_residual_px"] / max(sub["rms_residual_px"], 1e-9)
            )
            # Residuals are checked *before* the slope, and the ordering is the whole point.
            # A dislocation array also shifts the mean spacing -- one step across a 12 nm row
            # moves it by b/L, which at these sizes is the same few percent that uniform
            # relaxation gives -- so a slope-first classifier reports "RELAXED" for a film
            # that is locally in perfect registry with the substrate.  A row that is not a
            # straight line has no meaningful single slope, so say that first.
            # The classification, kept separate from the sentence.  Comparing the sentences is a
            # trap: they quote the mismatch to two decimals, so two runs that classify identically
            # read as disagreeing when one says +3.83% and the other +3.82%.
            if out["residual_ratio"] > 3.0:
                out["verdict_class"] = "INHOMOGENEOUS"
            elif out["significant"] and out["above_threshold"]:
                out["verdict_class"] = "RELAXED"
            else:
                out["verdict_class"] = "COHERENT"

            if out["residual_ratio"] > 3.0:
                verdict = (
                    f"INHOMOGENEOUS: film row residuals are {out['residual_ratio']:.1f}x the "
                    "substrate's, so the film rows are not a single uniform lattice "
                    "(steps/misfit dislocations or local strain -- read the residual plot, "
                    f"not the slope).  Mean spacing mismatch {mismatch:+.2%} is an average "
                    "over that structure and not a relaxation state"
                )
            elif out["significant"] and out["above_threshold"]:
                verdict = (
                    f"RELAXED: film spacing differs from the substrate by {mismatch:+.2%}, "
                    "uniformly along the rows"
                )
            elif out["significant"]:
                verdict = (
                    f"COHERENT: the {mismatch:+.2%} mismatch is statistically resolved but "
                    f"below the {self.min_mismatch:.1%} threshold for meaning anything"
                )
            else:
                verdict = f"COHERENT: film spacing matches the substrate to {mismatch:+.2%}"
            out["verdict"] = verdict
        return out

    def summary_text(self) -> str:
        s = self.summary()
        px = self.pixel_size
        unit = "A" if px else "px"

        def fmt(d):
            if not d.get("n_rows"):
                return "   (no rows)"
            v = d["spacing_A"] if px else d["spacing_px"]
            e = d["spacing_std_A"] if px else d["spacing_std_px"]
            r = d.get("rms_residual_pm", d["rms_residual_px"])
            ru = "pm" if px else "px"
            pitch = d.get("row_pitch_A", d.get("row_pitch_px", float("nan")))
            return (
                f"   {d['n_rows']} rows, {d['n_sites']} sites, in-plane spacing "
                f"{v:.4f} +/- {e:.4f} {unit}, row residual {r:.1f} {ru}\n"
                f"   out-of-plane row pitch {pitch:.4f} {unit}   "
                f"(scaled by k -- unlike the in-plane spacing, this one needs the "
                f"drift correction)"
            )

        L = [
            f"=== {self.label}: interface coherency ===",
            f"interface at y = {self.interface_y:.1f} px",
            "substrate",
            fmt(s["substrate"]),
            "film",
            fmt(s["film"]),
        ]
        if "mismatch" in s:
            L += [
                "",
                f"film/substrate spacing mismatch = {s['mismatch']:+.3%}"
                + (f" +/- {s['mismatch_err']:.3%}" if s["mismatch_err"] else ""),
                f"film/substrate row-residual ratio  = {s['residual_ratio']:.2f}",
                "",
                f"verdict: {s['verdict']}",
            ]
        if self.warnings_raised:
            L += ["", "warnings"] + [f"  ! {w}" for w in self.warnings_raised]
        return "\n".join(L)

    # ------------------------------------------------------------------ plots
    def plot(
        self,
        remove_row_offset=True,
        stack_offset=0.0,
        figsize=(10.0, 4.2),
        axes=None,
        in_angstrom=True,
        title=None,
        panels=("raw", "residual"),
        legend=True,
    ):
        """
        The coherency plot: ``x`` versus site index, and the same data with the reference
        spacing subtracted.

        The ``"raw"`` panel is the literal thing -- position against index.  The
        ``"residual"`` panel is ``x - index * reference_spacing``: the **disregistry**, the
        accumulated lateral mismatch between the film and a substrate-spaced reference.  That
        is where the answer shows up, because on the raw panel a 4% slope difference is a
        barely-perceptible fanning of lines whose slope is 4 Angstrom.

        ``panels`` selects which to draw, so a figure comparing several samples can ask for
        ``panels=("residual",)`` and get one axes per sample.  When passing ``axes``, give as
        many as there are panels.

        ``remove_row_offset`` takes ``True``, ``False``, or a phase -- ``"substrate"``,
        ``"film"``, or a pair of them -- so the intercept can be removed from the reference rows
        only.  That setting is worth knowing about: it overlays the substrate rows into one clean
        ruler while leaving the film's rows where they actually sit against it, which *is* the
        disregistry.  It subtracts each row's own fitted intercept so the rows overlay
        instead of being separated by wherever each row's first column happened to fall.  That
        is the option to think about: with it off you see the absolute registry of each row
        (and any residual drift between rows); with it on you see only the *spacing*, which is
        what coherency is about.  ``stack_offset`` spreads the rows vertically for legibility.
        """
        import matplotlib.pyplot as plt

        if not self.rows:
            raise RuntimeError("call measure() first")
        panels = tuple(panels)
        for name in panels:
            if name not in ("raw", "residual"):
                raise ValueError(f"unknown panel {name!r}; use 'raw' and/or 'residual'")
        scale = self.pixel_size if (in_angstrom and self.pixel_size) else 1.0
        unit = "$\\AA$" if scale != 1.0 else "px"

        if axes is None:
            fig, axarr = plt.subplots(1, len(panels), figsize=figsize, squeeze=False)
            axarr = axarr.ravel()
        else:
            axarr = np.atleast_1d(axes)
            if len(axarr) < len(panels):
                raise ValueError(f"{len(panels)} panels requested but {len(axarr)} axes given")
            fig = axarr[0].figure
        ax = {name: axarr[i] for i, name in enumerate(panels)}

        colours = {"substrate": plt.get_cmap("Blues"), "film": plt.get_cmap("Oranges")}
        counts = {"substrate": 0, "film": 0}
        n_each = {p: sum(1 for r in self.rows if r.phase == p) for p in counts}

        # Which phases get their row intercept subtracted.  ``"substrate"`` is the informative
        # middle setting: it overlays the reference rows, so they read as one clean ruler, while
        # leaving the film's rows where they actually sit relative to it -- which is the
        # disregistry.  Removing both hides that; removing neither buries it under the row-to-row
        # stagger.
        if remove_row_offset is True:
            offset_phases = {"substrate", "film"}
        elif not remove_row_offset:
            offset_phases = set()
        elif isinstance(remove_row_offset, str):
            offset_phases = {remove_row_offset}
        else:
            offset_phases = set(remove_row_offset)

        # Rows whose offset is *kept* are shifted to a common datum: the mean intercept of the rows
        # whose offset was removed.  Without it, a kept row is plotted against the image origin, and
        # that number is arbitrary -- it depends on which column happened to get index 0, so the
        # film landed 8.3 A up the axis and squashed the slope that the panel exists to show.
        # Against the substrate's own intercept it is the disregistry at index 0, which is the
        # quantity meant.
        datum = (
            float(np.mean([r.intercept for r in self.rows if r.phase in offset_phases]))
            if offset_phases
            else 0.0
        )

        def keep_datum(r):
            """The datum for a row whose own offset is kept, moved by whole reference spacings.

            Site indices come from rank order along the row, so a row whose site 0 lands on a
            different column is displaced by a whole lattice vector -- which read as the film row
            nearest the interface sitting a full period below its neighbours, i.e. as physics.
            Registry is only defined modulo one period, so wrapping is not cosmetic: it removes an
            artefact of where each row's indexing happened to start and keeps every row on a
            comparable footing.  Slopes, and therefore every measured spacing, are untouched.
            """
            d = self.reference_spacing
            off = r.intercept - datum
            return datum + (d * np.round(off / d) if d else 0.0)

        for r in self.rows:
            j = counts[r.phase]
            counts[r.phase] += 1
            c = colours[r.phase](0.35 + 0.55 * j / max(n_each[r.phase] - 1, 1))
            off = j * stack_offset
            x0 = r.intercept if r.phase in offset_phases else keep_datum(r)
            # Every row labelled, not just the first of each phase.  Eight shades of two colours is
            # not something anyone can read off a colourbar-free plot, and the row's distance from
            # the interface is exactly what you want to know when one of them misbehaves.
            lab = f"{r.phase} row {j + 1}" + (" (nearest)" if j == 0 else "")
            if "raw" in ax:
                ax["raw"].plot(
                    r.index, (r.x - x0) * scale + off, "o-", ms=2.0, lw=0.6, color=c, label=lab
                )
            if "residual" in ax:
                dev = (r.x - x0 - self.reference_spacing * r.index) * scale + off
                ax["residual"].plot(r.index, dev, "o-", ms=2.0, lw=0.6, color=c, label=lab)

        s = self.summary()
        if "raw" in ax:
            a = ax["raw"]
            a.set_xlabel("site index along row")
            a.set_ylabel(f"$x$ position ({unit})")
            if legend:
                a.legend(fontsize=6)
            a.set_title("$x$ vs site index", fontsize=8)
        if "residual" in ax:
            a = ax["residual"]
            a.axhline(0, lw=0.5, color="k")
            a.set_xlabel("site index along row")
            ref = self.reference_spacing * scale
            # Name the reference rather than printing a bare number in the label: it is the
            # *measured in-plane site spacing of the reference phase*, so a reader can tell it
            # apart from a lattice constant (for zincblende <110> centroids it is a/sqrt(2), not a).
            a.set_ylabel(f"disregistry  $x - i\\,d_\\mathrm{{ref}}$  ({unit})")
            if legend:
                a.legend(fontsize=6)
            sub = f"{s.get('mismatch', float('nan')):+.2%} mismatch" if "mismatch" in s else ""
            a.set_title(
                f"$d_\\mathrm{{ref}}$ = {ref:.3f} {unit}, the {self.reference} in-plane "
                f"spacing — {sub}",
                fontsize=8,
            )
        if title:
            fig.suptitle(title, fontsize=9)
        # Skipped when the caller's figure already manages its own layout: calling tight_layout on
        # a constrained-layout figure switches the engine and warns, which is noise in a notebook
        # that passes in its own axes.
        if not fig.get_constrained_layout():
            # tight_layout does not reserve room for a suptitle, so make room explicitly --
            # otherwise the figure title lands on top of the panel titles.
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92) if title else None)
        return fig, axarr

    def plot_rows_on_image(self, image, ax=None, figsize=(6.0, 6.0)):
        """Show which sites went into which row, over the image."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        im = np.asarray(image, float)
        lo, hi = np.nanpercentile(im, [0.5, 99.5])
        ax.imshow(im, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        ax.axhline(self.interface_y, color="#5ac8fa", lw=0.8, ls=":")
        cmap = {"substrate": "tab:cyan", "film": "tab:orange"}
        for r in self.rows:
            ax.plot(r.x, np.full(len(r.x), r.y_centre), ".", ms=2.5, color=cmap[r.phase])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("rows used in the measurement", fontsize=8)
        return ax


def compare_with_and_without_correction(
    image_raw, sites_raw, drift, interface_y, pixel_size=None, **measure_kw
):
    """
    Run the coherency measurement on raw and drift-corrected site positions.

    ``drift`` is a fitted :class:`~quantem.imaging.drift_single_image.SingleImageDrift`.
    Corrected coordinates come from ``apply_to_coordinates`` -- no interpolation, so this
    compares the measurement rather than two different resamplings of the image.

    Returns ``(raw, corrected, delta)``.  ``delta`` reports what actually moved, which is the
    reason to run both: the in-plane spacings and the coherency verdict should come out
    *identical*, because shear is constant along a row of constant ``y`` and so cannot change
    a spacing measured along one.  The out-of-plane row pitch should move by ``1/k - 1``.

    If the two verdicts do disagree, something other than linear drift is involved -- rows
    that are not horizontal, or nonlinear drift, which is not constant along a row -- and the
    corrected one is the one to trust.
    """
    corrected_sites = drift.apply_to_coordinates(np.asarray(sites_raw, float))
    k = drift.solution["k"]
    raw = InterfaceCoherency.from_sites(sites_raw, interface_y, pixel_size, label="uncorrected")
    cor = InterfaceCoherency.from_sites(
        corrected_sites, interface_y / k, pixel_size, label="drift-corrected"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        raw.measure(**measure_kw)
        cor.measure(**measure_kw)

    a, b = raw.summary(), cor.summary()
    delta = dict(
        k=k,
        expected_pitch_change=1.0 / k - 1.0,
        verdicts_agree=a.get("verdict_class") == b.get("verdict_class"),
        verdict_class=b.get("verdict_class"),
    )
    for side in ("substrate", "film"):
        for key in ("spacing_px", "row_pitch_px"):
            u, v = a[side].get(key), b[side].get(key)
            if u and v and np.isfinite(u) and np.isfinite(v):
                delta[f"{side}_{key}_change"] = float(v / u - 1.0)
    if "mismatch" in a and "mismatch" in b:
        delta["mismatch_change"] = float(b["mismatch"] - a["mismatch"])
    return raw, cor, delta


# ===========================================================================
# interactive tools
# ===========================================================================


class InterfaceLine:
    """
    A horizontal line you drag onto the interface.  Its row lands in ``.y``.

    Click anywhere in the image to put the line there and keep dragging; release to set it.  The
    arrow keys nudge it by a pixel, shift-arrow by ten -- worth having, because the interface is
    usually easier to place to the pixel by nudging than by dragging.

    The right-hand panel is the row-mean intensity on the same vertical axis, which is where the
    interface is usually most obvious: a contrast step between two phases shows up there even
    when it is hard to see in the image.  Pass ``profile=False`` to drop it.

    Beats typing a row number into a variable for the reason any direct-manipulation control
    does: the number is meaningless without the image next to it, and an interface a few rows out
    silently moves rows from one phase into the other.

    >>> iface = InterfaceLine(image, y=estimate_interface_row(image))
    >>> # ... drag ...
    >>> INTERFACE_ROW = iface.y
    """

    def __init__(
        self,
        image,
        y=None,
        ax=None,
        figsize=(7.5, 6.0),
        title=None,
        on_change=None,
        colour="#ffb000",
        cmap="gray",
        profile=True,
        profile_width=0.22,
    ):
        import matplotlib.pyplot as plt

        self.image = np.nan_to_num(np.asarray(image, float))
        ny, nx = self.image.shape
        self._y = float(ny // 2 if y is None else y)
        #: called with the new row on every move; assignable after construction,
        #: because whatever it drives usually does not exist yet at that point
        self.on_change = on_change
        self._dragging = False

        if ax is None:
            if profile:
                self.fig, (self.ax, self.ax_prof) = plt.subplots(
                    1,
                    2,
                    figsize=figsize,
                    sharey=True,
                    gridspec_kw=dict(
                        width_ratios=(1.0 - profile_width, profile_width), wspace=0.04
                    ),
                )
            else:
                self.fig, self.ax = plt.subplots(figsize=figsize)
                self.ax_prof = None
        else:
            self.ax, self.fig, self.ax_prof = ax, ax.figure, None

        lo, hi = np.nanpercentile(self.image, [0.5, 99.5])
        self.ax.imshow(self.image, cmap=cmap, origin="lower", vmin=lo, vmax=hi)
        self.ax.set_xlim(0, nx)
        self.ax.set_ylim(0, ny)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self._lines = [self.ax.axhline(self._y, color=colour, lw=1.2)]
        if self.ax_prof is not None:
            prof = self.image.mean(axis=1)
            self.ax_prof.plot(prof, np.arange(ny), lw=0.7, color="0.35")
            self.ax_prof.set_xticks([])
            self.ax_prof.tick_params(labelleft=False)
            self.ax_prof.set_xlabel("row mean", fontsize=7)
            self._lines.append(self.ax_prof.axhline(self._y, color=colour, lw=1.2))

        self._title = title or "click / drag to place the interface;  arrows nudge, shift x10"
        self._status = self.ax.set_title("", fontsize=8)
        self._cids = [
            self.fig.canvas.mpl_connect("button_press_event", self._on_press),
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.fig.canvas.mpl_connect("button_release_event", self._on_release),
            self.fig.canvas.mpl_connect("key_press_event", self._on_key),
        ]
        self._refresh()

    # -- state -----------------------------------------------------------
    @property
    def y(self) -> float:
        """Interface position, in rows of the image the line was drawn on."""
        return self._y

    @y.setter
    def y(self, value):
        self._set(value)

    @property
    def row(self) -> int:
        return int(round(self._y))

    def _set(self, value, notify=True):
        self._y = float(np.clip(value, 0, self.image.shape[0] - 1))
        for line in self._lines:
            line.set_ydata([self._y, self._y])
        self._refresh()
        if notify and self.on_change:
            self.on_change(self._y)

    def _refresh(self):
        self._status.set_text(f"{self._title}\nrow {self._y:.0f}")
        self.fig.canvas.draw_idle()

    # -- events ----------------------------------------------------------
    def _busy(self):
        """True while a pan/zoom tool is armed, so dragging the view does not move the line."""
        tb = getattr(self.fig.canvas, "toolbar", None)
        return bool(getattr(tb, "mode", "")) if tb is not None else False

    def _on_press(self, event):
        if event.inaxes not in (self.ax, self.ax_prof) or event.ydata is None:
            return
        if event.button != 1 or self._busy():
            return
        self._dragging = True
        self._set(event.ydata)

    def _on_motion(self, event):
        if not self._dragging or event.ydata is None:
            return
        if event.inaxes not in (self.ax, self.ax_prof):
            return
        self._set(event.ydata)

    def _on_release(self, event):
        self._dragging = False

    def _on_key(self, event):
        step = {"up": 1, "down": -1, "shift+up": 10, "shift+down": -10}.get(event.key)
        if step is None:
            return
        self._set(self._y + step)

    def disconnect(self):
        for cid in self._cids:
            self.fig.canvas.mpl_disconnect(cid)


class RegionSelector:
    """
    Drag a rectangle on an image to choose a region; the result lands in ``.bounds``.

    ``bounds`` is ``(y0, y1, x0, x1)`` in pixels, ready to hand to
    ``SingleImageDrift.set_reference_band``.  Needs an interactive backend.

    >>> sel = RegionSelector(image, title="drag over substrate only")
    >>> # ... drag ...
    >>> sid.set_reference_band(*sel.bounds[:2])
    """

    def __init__(
        self,
        image,
        ax=None,
        figsize=(7.0, 7.0),
        title=None,
        initial=None,
        on_select=None,
        cmap="gray",
    ):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector

        self.image = np.asarray(image, float)
        self.bounds = tuple(initial) if initial else None
        self._on_select = on_select

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax, self.fig = ax, ax.figure
        lo, hi = np.nanpercentile(self.image, [0.5, 99.5])
        self.ax.imshow(self.image, cmap=cmap, origin="lower", vmin=lo, vmax=hi)
        self.ax.set_title(title or "drag a rectangle over the reference region", fontsize=9)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._text = self.ax.text(
            0.01,
            0.99,
            "",
            transform=self.ax.transAxes,
            fontsize=7,
            va="top",
            color="#ffb000",
            bbox=dict(fc="k", ec="none", alpha=0.5, pad=1.5),
        )

        self.selector = RectangleSelector(
            self.ax,
            self._callback,
            useblit=False,
            button=[1],
            interactive=True,
            props=dict(facecolor="#ffb000", edgecolor="#ffb000", alpha=0.18, fill=True),
        )
        if self.bounds:
            y0, y1, x0, x1 = self.bounds
            self.selector.extents = (x0, x1, y0, y1)
            self._show()

    def _callback(self, eclick, erelease):
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        ny, nx = self.image.shape
        self.bounds = (
            int(max(0, round(y0))),
            int(min(ny, round(y1))),
            int(max(0, round(x0))),
            int(min(nx, round(x1))),
        )
        self._show()
        if self._on_select:
            self._on_select(self.bounds)

    def set_bounds(self, bounds, notify=False):
        """
        Move the rectangle programmatically, in the same ``(y0, y1, x0, x1)`` order as
        :attr:`bounds`.

        Exists so callers do not have to know that matplotlib's ``RectangleSelector.extents``
        wants ``(xmin, xmax, ymin, ymax)`` -- a different order for the same four numbers, which
        is the kind of detail that silently transposes a selection.  Handy for driving the band
        from an :class:`InterfaceLine`::

            line.on_change = lambda y: sel.set_bounds(band_under(y))
        """
        y0, y1, x0, x1 = (int(round(v)) for v in bounds)
        self.bounds = (y0, y1, x0, x1)
        self.selector.extents = (x0, x1, y0, y1)
        self._show()
        if notify and self._on_select:
            self._on_select(self.bounds)
        return self.bounds

    def _show(self):
        y0, y1, x0, x1 = self.bounds
        self._text.set_text(f"rows {y0}-{y1}  cols {x0}-{x1}\n({y1 - y0} x {x1 - x0} px)")
        self.fig.canvas.draw_idle()


class SiteEditor:
    """
    Show detected sites over an image; left-click adds one, right-click deletes the nearest.

    Automatic detection is good but not perfect, and near an interface the failures are the
    expensive kind: site indices along a row come from rank order, so one spurious or missing
    column shifts every index after it by one and tilts the fitted spacing for that row.  A
    minute of clicking is cheaper than discovering that later.

    ``.sites`` is the current ``(N, 2)`` array of ``(x, y)``.  Added sites are snapped to the
    local intensity centroid by default, so a click only has to be close.

    :meth:`refine` (key ``f``, or the button in ``zbkit.widgets.site_editor_panel``) does the
    stronger version: for each site, a window of ``snap_radius`` px around where it currently is,
    a 2D Gaussian fitted inside that window, and the site moved to the fitted centre -- via
    :func:`snap_sites`, which is purely local and needs no lattice spacing.  Worth doing even
    though clicks are centroid-snapped already, for two reasons.  The click snap is a centroid
    over a fixed 5 px window wherever you clicked, so a click landing between two columns
    returns a centroid between two columns -- a plausible position belonging to no column,
    while the Gaussian fit converges onto one or reports that it could not.  And the fit's
    background is a *plane*, so it does not lean downhill on the intensity gradient that is
    normal near an interface, where a centroid can be biased by a good fraction of a pixel.
    A fit that does not settle within ``snap_radius`` leaves its site alone and is counted, so a
    refine cannot silently invent positions.  It is one undo step (``u``), and the summary line
    under the title says what moved.

    Pass ``interface_y`` and the two phases get different colours, recomputed on every edit --
    so a site added on the wrong side of the interface is visible as the wrong colour rather
    than having to be inferred from its position.  Which side a site belongs to is exactly what
    the downstream measurement splits on, so it is worth being able to see it.

    Keys: ``f`` refine, ``u`` undo, ``r`` reset to the sites passed in, ``h`` toggle the site
    markers.

    >>> ed = SiteEditor(image, sites, interface_y=657)
    >>> # ... click ...
    >>> ed.refine()                       # or press f
    >>> cleaned = ed.sites
    """

    #: substrate (below the interface), film (above).  Chosen to match the blue/orange the
    #: coherency plots use for the same two phases.
    COLOURS = ("#00e5ff", "#ffa040")

    def __init__(
        self,
        image,
        sites,
        ax=None,
        figsize=(8.0, 8.0),
        refine_radius=5.0,
        snap=True,
        ms=4.0,
        title=None,
        cmap="gray",
        zoom=None,
        interface_y=None,
        colours=None,
        snap_radius=8.0,
        fit_radius=None,
        refine_mode="gaussian",
        refine_only_added=True,
        spacing_px=None,
    ):
        import matplotlib.pyplot as plt

        self.image = np.nan_to_num(np.asarray(image, float))
        self._initial = np.asarray(sites, float).reshape(-1, 2).copy()
        self.sites = self._initial.copy()
        self.refine_radius = float(refine_radius)
        self.snap_radius = float(snap_radius)
        self.fit_radius = None if fit_radius is None else float(fit_radius)
        self.refine_mode = str(refine_mode)
        #: default for :meth:`refine`, and the initial state of the panel's checkbox.  True, so a
        #: refine touches what you clicked and leaves the automatic detections alone -- the
        #: detections came from the same data by the same estimators, so re-refining them is at
        #: best a no-op and at worst a second opinion nobody asked for.
        self.refine_only_added = bool(refine_only_added)
        self.snap = bool(snap)
        self.interface_y = None if interface_y is None else float(interface_y)
        self.colours = tuple(colours) if colours else self.COLOURS
        self._history = []
        self._visible = True
        # Which sites were added by hand, carried through edits so ``refine(only_added=True)``
        # can leave the automatic detections alone.
        self._added = np.zeros(len(self.sites), bool)
        # Left to snap_sites, which re-infers it from the current sites plus the image on every
        # refine.  Caching it from the initial set would freeze a bad estimate made when the
        # editor happened to open on only a few sites.
        self.spacing_px = None if spacing_px is None else float(spacing_px)
        self._note = ""
        #: ``info`` dict from the last :meth:`refine`, or ``{}``.
        self.last_refine: dict = {}

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax, self.fig = ax, ax.figure
        lo, hi = np.nanpercentile(self.image, [0.5, 99.5])
        self.ax.imshow(self.image, cmap=cmap, origin="lower", vmin=lo, vmax=hi)

        self._plots = []
        for colour, label in zip(self.colours, ("substrate", "film")):
            (line,) = self.ax.plot(
                [], [], "o", ms=ms, mfc="none", mec=colour, mew=0.8, label=label
            )
            self._plots.append(line)
        if self.interface_y is not None:
            self.ax.axhline(self.interface_y, color="#ffffff", lw=0.6, ls=":", alpha=0.7)
            # A faint white panel: the two marker colours are chosen to read against a dark image
            # and are close to illegible against the light patches of one.  ``frameon=True`` is
            # not redundant -- a house style that sets ``legend.frameon: False`` (this project's
            # does) leaves nothing for facecolor and framealpha to act on, so the box silently
            # never appears.
            self.ax.legend(
                fontsize=7,
                loc="lower right",
                frameon=True,
                facecolor="white",
                edgecolor="0.6",
                framealpha=0.55,
                labelcolor="black",
            )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        # A little slack beyond the data so markers on the edge are drawn whole rather than
        # sliced in half by the axes boundary.
        ny, nx = self.image.shape
        pad = 2.0 * ms
        self.ax.set_xlim(-pad, nx + pad)
        self.ax.set_ylim(-pad, ny + pad)
        self._title = title or (
            "left-click add   right-click remove   f refine   u undo   r reset   h hide"
        )
        self._status = self.ax.set_title(f"{self._title}\n{len(self.sites)} sites", fontsize=8)
        if zoom:
            y0, y1, x0, x1 = zoom
            self.ax.set_xlim(x0 - pad, x1 + pad)
            self.ax.set_ylim(y0 - pad, y1 + pad)
        self._cids = [
            self.fig.canvas.mpl_connect("button_press_event", self._on_click),
            self.fig.canvas.mpl_connect("key_press_event", self._on_key),
        ]
        self._refresh()

    # -- editing ---------------------------------------------------------
    def _push(self):
        """One undo step.  The added-by-hand flags travel with the positions."""
        self._history.append((self.sites.copy(), self._added.copy()))

    def _add(self, x, y):
        p = np.array([[x, y]], float)
        if self.snap:
            p = refine_peaks(self.image, p, radius=self.refine_radius, mode="com")
        self._push()
        self.sites = np.vstack([self.sites, p])
        self._added = np.append(self._added, True)

    def _remove(self, x, y):
        if len(self.sites) == 0:
            return
        d = np.hypot(*(self.sites - np.array([x, y])).T)
        if d.min() > 4 * self.refine_radius:
            return  # ignore clicks nowhere near a site
        self._push()
        i = int(np.argmin(d))
        self.sites = np.delete(self.sites, i, axis=0)
        self._added = np.delete(self._added, i)

    def refine(self, only_added=None, max_move=None, dedupe=True, mode=None):
        """
        Fit each site in a window of ``max_move`` px around it and move it to the fitted centre.

        Local to each site: no lattice spacing, no image-wide peak list, so it works the same on
        three hand-placed sites as on a thousand detected ones, and it iterates internally until
        the positions stop moving, so pressing it twice is a check rather than a second bite.

        ``only_added`` defaults to ``self.refine_only_added`` (True): touch just the hand-placed
        sites and leave the automatic detections alone.  ``mode`` is ``"gaussian"`` or ``"com"``
        and defaults to ``self.refine_mode``.  Returns the :func:`snap_sites` ``info`` dict, also
        kept as ``.last_refine``; one undo step.
        """
        if len(self.sites) == 0:
            return {}
        only_added = self.refine_only_added if only_added is None else bool(only_added)
        idx = None
        if only_added:
            idx = np.nonzero(self._added)[0]
            if len(idx) == 0:
                self._note = "nothing added by hand yet -- refine all, or click first"
                self._refresh()
                return {}
        sites, info = snap_sites(
            self.image,
            self.sites,
            max_move=self.snap_radius if max_move is None else float(max_move),
            fit_radius=self.fit_radius,
            dedupe=dedupe,
            indices=idx,
            mode=self.refine_mode if mode is None else str(mode),
        )
        self._push()
        self.sites = sites
        self._added = self._added[info["keep"]]
        self.last_refine = info
        self._note = info["text"]
        self._refresh()
        return info

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        # Ignore clicks while a pan/zoom tool is armed, or the first zoom drag of a session
        # deletes whatever site it started on.
        if getattr(self.fig.canvas, "toolbar", None) is not None:
            if getattr(self.fig.canvas.toolbar, "mode", ""):
                return
        if event.button == 1:
            self._add(event.xdata, event.ydata)
        elif event.button == 3:
            self._remove(event.xdata, event.ydata)
        else:
            return
        self._refresh()

    def _on_key(self, event):
        if event.key == "f":
            self.refine()
            return
        if event.key == "u" and self._history:
            self.sites, self._added = self._history.pop()
            self._note = ""
        elif event.key == "r":
            self._push()
            self.sites = self._initial.copy()
            self._added = np.zeros(len(self.sites), bool)
            self._note = ""
        elif event.key == "h":
            self._visible = not self._visible
            for line in self._plots:
                line.set_visible(self._visible)
        else:
            return
        self._refresh()

    def _split(self):
        """``(substrate, film)`` index masks, recomputed so edits are coloured correctly."""
        if self.interface_y is None or len(self.sites) == 0:
            return np.ones(len(self.sites), bool), np.zeros(len(self.sites), bool)
        below = self.sites[:, 1] < self.interface_y
        return below, ~below

    def _refresh(self):
        below, above = self._split()
        for line, mask in zip(self._plots, (below, above)):
            line.set_data(self.sites[mask, 0], self.sites[mask, 1])
        counts = (
            f"{int(below.sum())} substrate + {int(above.sum())} film"
            if self.interface_y is not None
            else f"{len(self.sites)}"
        )
        note = f"\n{self._note}" if self._note else ""
        self._status.set_text(f"{self._title}\n{counts} sites{note}")
        self.fig.canvas.draw_idle()

    def disconnect(self):
        for cid in self._cids:
            self.fig.canvas.mpl_disconnect(cid)
