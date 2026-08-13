"""
Single-image scan-drift correction using a region of known crystal as an internal ruler.

``DriftCorrection`` in :mod:`quantem.imaging.drift` corrects drift the well-posed way:
from two or more images with orthogonal scan directions, following Ophus, Ciston &
Nelson (2016).  This module handles the case where you only have **one** image -- no
orthogonal pair, no rotation series, no stack -- by substituting a different piece of
prior knowledge: part of the field of view is a crystal whose projected geometry is
known exactly from symmetry.

Target case: a cross-sectional HAADF image on an uncorrected instrument, epitaxial film
on a relaxed zincblende substrate viewed along <110> (the dumbbell projection), crystal
axes roughly aligned to the pixel grid.

The distortion model
--------------------
The probe rasters with the fast axis along image ``x`` (columns) and the slow axis along
``y`` (rows).  For a sample drifting at constant velocity during the frame, the point
probed at pixel ``(i, j)`` is ``p = (i D + vx tau j, j D + vy tau j)``, with ``D`` the
pixel pitch and ``tau`` the line time.  (Within-line drift is smaller by ~1/n_cols and
is neglected.)  Inverting, a true sample vector ``t`` appears in the image as

.. math::  m = M t, \\qquad M = \\begin{pmatrix} 1 & s \\\\ 0 & k \\end{pmatrix},
           \\qquad s = \\frac{-v_x \\tau}{D + v_y \\tau}, \\qquad k = \\frac{D}{D + v_y \\tau}

Three facts carry the method:

======================  ===========================================================
``M[1, 0] = 0``         the fast-scan axis is undistorted -- it is the ruler
``M[0, 0] = 1``         drift produces no ``x`` scale error
only ``s``, ``k`` free  two unknowns: a shear and a rescaling of the slow axis
======================  ===========================================================

Two degeneracies, both reported rather than hidden
--------------------------------------------------
1. ``k`` is a pure scale change along ``y`` with no angular signature, so it is
   mathematically indistinguishable from scan-coil gain anisotropy (1--2% on an
   uncorrected instrument is normal).  The correction still fixes the image; what you
   cannot do is quote ``vy`` as a calibrated drift velocity without an independent gain
   calibration.

2. Less obvious, and worth being blunt about: **the pure-drift model is not falsifiable
   from a single lattice.**  A measured lattice gives one 2x2 matrix, i.e. four numbers.
   The model has four free parameters -- ``s``, ``k``, plus the nuisance pair (global
   rotation ``theta``, isotropic scale ``sigma``) which are not distortions and must be
   projected out.  Four into four: every 2x2 matrix with *positive determinant* decomposes
   into ``M(s, k) sigma R(theta)``, uniquely -- and image formation is always orientation
   preserving, so that covers every matrix this can encounter.  A "check that the
   ``[1, 0]`` element of an unconstrained affine fit is zero" is therefore vacuous: the
   fit will always find a decomposition.  What is genuinely testable, and is what this module tests, is
   (a) whether the site positions are affine at all (the fit residual),
   (b) whether ``s`` is constant over the frame (:meth:`test_nonlinearity`), and
   (c) whether the fitted ``theta`` is consistent with the scan rotation you set.

Usage
-----
>>> from quantem.imaging.drift_single_image import SingleImageDrift
>>> sid = SingleImageDrift.from_data(image, pixel_size=0.18, flyback_rows=12)
>>> sid.set_reference_band(y0=40, y1=740)          # substrate only
>>> sid.find_sites()                               # dumbbell centroids
>>> sid.fit()                                      # index + solve for (s, k)
>>> print(sid.report_text())
>>> corrected = sid.apply_to_image()               # or sid.apply_to_coordinates(xy)

References
----------
Ophus, Ciston & Nelson, Ultramicroscopy 162, 1 (2016).
Barcena-Gonzalez et al., Microsc. Microanal. 26, 913 (2020) -- CDrift, single image.
Jones & Nellist, Microsc. Microanal. 19, 1050 (2013) -- flyback, scan noise, row-wise.
Nord et al., Adv. Struct. Chem. Imaging 3, 9 (2017) -- Atomap.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, minimum_filter

try:  # optional, only for I/O + viz
    from quantem.core.datastructures.dataset2d import Dataset2d
    from quantem.core.io.serialize import AutoSerialize
except Exception:  # pragma: no cover - standalone use
    Dataset2d = None  # type: ignore[assignment]

    class AutoSerialize:  # type: ignore[no-redef]
        pass


__all__ = [
    "ReferenceGeometry",
    "ZINCBLENDE_110",
    "ROCKSALT_110",
    "SingleImageDrift",
    "decompose_affine",
    "compose_affine",
]


# ===========================================================================
# structures.py -- reference geometry
# ===========================================================================


@dataclass(frozen=True)
class ReferenceGeometry:
    """
    Projected geometry of the reference crystal: a 2-D cell plus a site basis.

    Everything here is fixed by symmetry given one lattice constant, which is what makes
    the crystal usable as a ruler.  ``ax`` runs along the fast scan axis by convention
    (in the target case ``[1-10]``) and ``ay`` along the slow axis (``[001]``), with
    ``axial_angle_deg`` between them (exactly 90 for zincblende <110>).

    **Only the shape matters, not the size.**  The fit needs the *angle* between the two cell
    vectors and the *ratio* of their lengths; the absolute lengths cancel, because an overall
    isotropic scale is one of the nuisance parameters (``sigma``) that gets projected out
    along with the rotation.  So a user supplying their own reference only has to know the
    angle and the ratio -- see :meth:`from_angle_and_ratio`.  Supplying real lengths as well
    buys exactly one extra thing: ``sigma`` then becomes a check on the pixel-size
    calibration instead of an arbitrary number.

    ``site_basis`` holds fractional offsets within the cell for each *detectable site*,
    which is not the same as the atom basis.  On an uncorrected instrument a <110>
    dumbbell is unresolved and the thing a peak finder reports is the dumbbell
    *centroid*, so ``site_basis`` for ``site_mode="centroid"`` has two entries, not
    four.  ``site_basis_columns`` holds the per-column version for the resolved case.

    ``directions`` lists lattice directions with exactly known angles from ``[001]``,
    used for the per-direction angle residuals and the held-out-direction test.  They
    are not extra information for the fit -- the cell already determines them -- but
    they are the most legible way to report what the fit residual means in degrees.
    """

    name: str
    a: float  # cubic lattice constant, Angstrom
    ax_over_a: float  # in-plane period / a
    ay_over_a: float  # out-of-plane period / a
    axial_angle_deg: float = 90.0  # angle between the a and c cell vectors
    site_basis: tuple = ((0.0, 0.0), (0.5, 0.5))
    site_basis_columns: tuple = ((0.0, 0.0), (0.5, 0.5))
    d_dumbbell_over_a: float = 0.0
    directions: tuple = ()  # ((m, n, label), ...)
    zone: str = ""

    # -- derived ------------------------------------------------------------
    @property
    def ax(self) -> float:
        return self.a * self.ax_over_a

    @property
    def ay(self) -> float:
        return self.a * self.ay_over_a

    @property
    def axial_ratio(self) -> float:
        """``|ay| / |ax|`` -- exactly sqrt(2) for zincblende <110>."""
        return self.ay_over_a / self.ax_over_a

    @property
    def d_dumbbell(self) -> float:
        return self.a * self.d_dumbbell_over_a

    @property
    def cell(self) -> NDArray:
        """
        Rows are the ideal cell vectors in Angstrom: ``a`` along ``+x``, ``c`` at
        ``axial_angle_deg`` from it.  Reduces to ``[[ax, 0], [0, ay]]`` at 90 degrees.
        """
        g = np.radians(self.axial_angle_deg)
        return np.array([[self.ax, 0.0], [self.ay * np.cos(g), self.ay * np.sin(g)]])

    def with_lattice_constant(self, a: float) -> "ReferenceGeometry":
        return replace(self, a=float(a))

    @classmethod
    def from_angle_and_ratio(
        cls,
        axial_angle_deg,
        axial_ratio,
        name="user",
        zone="",
        a=1.0,
        site_basis=((0.0, 0.0),),
        site_basis_columns=None,
        directions=(),
        d_dumbbell_over_a=0.0,
    ):
        """
        Build a reference from just the two numbers the fit actually uses.

        ``axial_angle_deg`` is the angle between the two cell vectors and ``axial_ratio`` is
        ``|c| / |a|``.  Lengths are not required: ``a`` defaults to 1, which makes the fitted
        ``sigma`` come out as "pixels per cell edge" rather than as a magnification check.
        Pass the real ``a`` (in Angstrom) if you want the pixel-size cross-check as well.

        ``site_basis`` defaults to one site per cell.  Set it if the reference lattice is
        centred or has several detectable sites per cell -- getting it wrong shows up as a
        large fit residual rather than as a wrong answer, but it does show up.

        >>> # a square reference lattice, one site per cell
        >>> ReferenceGeometry.from_angle_and_ratio(90.0, 1.0)
        >>> # zincblende <110> dumbbell centroids, without knowing the lattice constant
        >>> ReferenceGeometry.from_angle_and_ratio(90.0, np.sqrt(2),
        ...                                        site_basis=((0, 0), (0.5, 0.5)))
        """
        return cls(
            name=name,
            zone=zone,
            a=float(a),
            ax_over_a=1.0,
            ay_over_a=float(axial_ratio),
            axial_angle_deg=float(axial_angle_deg),
            site_basis=tuple(site_basis),
            site_basis_columns=tuple(site_basis_columns or site_basis),
            d_dumbbell_over_a=d_dumbbell_over_a,
            directions=tuple(directions),
        )

    def ideal_vector(self, m: int, n: int) -> NDArray:
        """The ideal (undistorted) vector for lattice indices ``(m, n)``, in Angstrom."""
        return np.asarray([m, n], float) @ self.cell

    def ideal_angle(self, m: int, n: int) -> float:
        """Angle of ``(m, n)`` from ``[001]`` (the ``y`` axis), in degrees, in [0, 180)."""
        v = self.ideal_vector(m, n)
        return float(np.degrees(np.arctan2(abs(v[0]), v[1])) % 180.0)

    def basis(self, site_mode: str = "centroid") -> NDArray:
        """Fractional site offsets for the chosen site mode, shape ``(n_basis, 2)``."""
        b = self.site_basis if site_mode == "centroid" else self.site_basis_columns
        return np.asarray(b, float)

    def site_spacing(self, site_mode: str = "centroid") -> float:
        """
        Shortest site-to-site distance, in Angstrom.

        Sets the peak-finder's minimum separation, and for a centered-rectangular site
        lattice it is the *centering* vector rather than either cell edge -- which is why
        this is computed rather than assumed to be ``min(ax, ay)``.
        """
        b = self.basis(site_mode) @ self.cell
        offs = []
        for m in (-1, 0, 1):
            for n in (-1, 0, 1):
                shift = self.ideal_vector(m, n)
                for i in range(len(b)):
                    for j in range(len(b)):
                        v = b[j] + shift - b[i]
                        if np.hypot(*v) > 1e-9:
                            offs.append(np.hypot(*v))
        return float(min(offs))


#: Zincblende viewed along <110>: the dumbbell projection.  Projected cell is
#: rectangular by cubic symmetry -- ``a`` along ``[001]``, ``a/sqrt(2)`` along
#: ``[1-10]``, exactly 90 deg apart, axial ratio exactly sqrt(2).  One projected
#: sublattice is centered rectangular.  Default ``a`` is GaAs.
ZINCBLENDE_110 = ReferenceGeometry(
    name="zincblende<110>",
    zone="[110]",
    a=5.653,
    ax_over_a=1.0 / np.sqrt(2.0),
    ay_over_a=1.0,
    # dumbbell centroids: two per cell, related by the (1/2, 1/2) centering
    site_basis=((0.0, 0.0), (0.5, 0.5)),
    # resolved columns: the dumbbell split is a/4 along [001]
    site_basis_columns=((0.0, 0.0), (0.0, 0.25), (0.5, 0.5), (0.5, 0.75)),
    d_dumbbell_over_a=0.25,
    directions=((0, 1, "[001]"), (1, 0, "[1-10]"), (-1, 1, "[1-11]-ish"), (-2, 1, "<111>")),
)

#: Rock salt along <110>: a checkerboard, no dumbbells.  Same rectangular cell, so it
#: works as a reference band too if the *film* is the known phase.
ROCKSALT_110 = ReferenceGeometry(
    name="rocksalt<110>",
    zone="[110]",
    a=5.653,
    ax_over_a=1.0 / np.sqrt(2.0),
    ay_over_a=1.0,
    site_basis=((0.0, 0.0), (0.5, 0.5), (0.5, 0.0), (0.0, 0.5)),
    site_basis_columns=((0.0, 0.0), (0.5, 0.5), (0.5, 0.0), (0.0, 0.5)),
    directions=((0, 1, "[001]"), (1, 0, "[1-10]"), (-1, 1, "[1-11]-ish"), (-2, 1, "<111>")),
)


# ===========================================================================
# solve.py -- the affine decomposition
# ===========================================================================


def decompose_affine(A: NDArray) -> dict:
    """
    Decompose a 2x2 image-formation matrix as ``A = M(s, k) sigma R(theta)``.

    ``sigma R(theta)`` takes the ideal crystal into the scan frame (a rotation and an
    isotropic magnification -- neither is a distortion), and ``M`` is the drift shear.
    The order matters and is set by the physics: drift acts *after* the crystal is
    presented to the scan, so ``M`` is on the left.

    Closed form, no iteration.  With ``c = sigma cos(theta)``, ``d = sigma sin(theta)``:
    ``A = [[c + s d, -d + s c], [k d, k c]]``, hence

    * ``theta = atan2(A[1,0], A[1,1])``
    * ``sigma = A[0,0] cos(theta) - A[0,1] sin(theta)``
    * ``s = (A[0,0] sin(theta) + A[0,1] cos(theta)) / sigma``
    * ``k = hypot(A[1,0], A[1,1]) / sigma``

    Exists and is unique for every ``A`` with ``det(A) > 0``, which is why the pure-drift
    model cannot be falsified from a single lattice; see the module docstring.  A negative
    determinant would require ``k < 0``, i.e. a mirrored image, which no imaging geometry
    produces -- so it is rejected rather than accommodated.
    """
    A = np.asarray(A, float)
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        raise ValueError("singular affine matrix -- cannot decompose")
    if det < 0:
        raise ValueError(
            f"affine matrix has negative determinant ({det:.3g}): the image would be "
            "mirrored, which drift cannot do.  Check the coordinate convention of the "
            "site positions -- passing them as (row, col) rather than (x, y) does this."
        )
    theta = np.arctan2(A[1, 0], A[1, 1])
    ct, st = np.cos(theta), np.sin(theta)
    sigma = A[0, 0] * ct - A[0, 1] * st
    if sigma < 0:  # flip branch so sigma > 0
        theta += np.pi
        ct, st = np.cos(theta), np.sin(theta)
        sigma = A[0, 0] * ct - A[0, 1] * st
    s = (A[0, 0] * st + A[0, 1] * ct) / sigma
    k = np.hypot(A[1, 0], A[1, 1]) / sigma
    return dict(
        s=float(s),
        k=float(k),
        theta=float(theta),
        theta_deg=float(np.degrees(theta)),
        sigma=float(sigma),
    )


def compose_affine(s: float, k: float, theta: float = 0.0, sigma: float = 1.0) -> NDArray:
    """Inverse of :func:`decompose_affine`."""
    M = np.array([[1.0, s], [0.0, k]])
    R = sigma * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return M @ R


def drift_matrix(s: float, k: float) -> NDArray:
    """``M(s, k)``: the distortion alone, with rotation and magnification removed."""
    return np.array([[1.0, s], [0.0, k]])


def velocities_from_sk(s: float, k: float, pixel_size: float, line_time: float):
    """
    Implied drift velocity ``(vx, vy)`` in the same length units as ``pixel_size``, per
    second.

    Inverting the model: ``vy = D (1/k - 1) / tau`` and ``vx = -s (D + vy tau) / tau``.
    ``vy`` inherits the ``k`` / gain-anisotropy degeneracy in full; treat it as an upper
    bound on the drift rate, not a measurement.
    """
    vy = pixel_size * (1.0 / k - 1.0) / line_time
    vx = -s * (pixel_size + vy * line_time) / line_time
    return float(vx), float(vy)


# ===========================================================================
# columns.py -- site detection
# ===========================================================================


def estimate_site_spacing(image: NDArray, min_px: float = 3.0, max_frac: float = 0.25) -> float:
    """
    Shortest lattice repeat in the image, in pixels, from the autocorrelation.

    Used to size the peak finder when no pixel size is supplied.  The autocorrelation's
    strongest off-centre peak is the shortest lattice vector; a central exclusion of
    ``min_px`` keeps the self-peak out, and ``max_frac`` of the image size bounds the
    search so a slow illumination ripple cannot win.
    """
    im = np.asarray(image, float)
    im = im - im.mean()
    win = np.hanning(im.shape[0])[:, None] * np.hanning(im.shape[1])[None, :]
    F = np.fft.fft2(im * win)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    cy, cx = np.array(ac.shape) // 2
    yy = (np.arange(ac.shape[0]) - cy)[:, None]
    xx = (np.arange(ac.shape[1]) - cx)[None, :]
    r = np.hypot(xx, yy)
    rmax = max_frac * min(ac.shape)
    mask = (r > min_px) & (r < rmax)
    ac = np.where(mask, ac, -np.inf)
    # only accept true local maxima, so a broad shoulder cannot masquerade as a peak
    loc = ac == maximum_filter(ac, size=3)
    cand = np.argwhere(loc & np.isfinite(ac))
    if len(cand) == 0:
        raise RuntimeError("could not estimate site spacing from the autocorrelation")
    vals = ac[cand[:, 0], cand[:, 1]]
    best = cand[np.argmax(vals)]
    return float(np.hypot(best[1] - cx, best[0] - cy))


def find_peaks(
    image: NDArray,
    min_distance: float,
    threshold_rel: float = 0.15,
    smooth: float = 0.0,
    edge_margin: float = 0.0,
    threshold_mode: str = "prominence",
    valid_mask: NDArray | None = None,
) -> NDArray:
    """
    Local maxima, returned as ``(N, 2)`` array of ``(x, y)`` in pixels.

    Deliberately scale-free: a local-maximum test with a *local* threshold, so a slow
    illumination or thickness gradient across the frame does not delete the dim half of
    the sites the way a global intensity threshold would.

    ``threshold_mode`` decides what "local" means, and at an interface the two differ a lot:

    ``"prominence"`` (default)
        Height above the local *minimum* over ~1.2 lattice spacings -- how far the peak stands
        out of the valley floor immediately around it.  A minimum filter is edge-preserving, so
        this does not care what the phase 40 px away is doing.
    ``"background"``
        The original: height above a Gaussian-smoothed background of sigma ``1.5 * min_distance``.
        A Gaussian bleeds a step across its full width, so where two phases of different
        brightness meet, the brighter one inflates the background on the dimmer side and the
        dimmer side's sites fall under the threshold.  Measured on a synthetic interface with the
        film at 40% of the substrate's brightness: **76% of sites within 40 px of the interface
        detected, against 100% away from it** -- and the same 76% with the film 2.5x *brighter*,
        since the loss is always on the dimmer side.  ``"prominence"`` finds 100% in both.
        Kept for reproducing older results, not recommended.

    Both modes miss a site whose nearest neighbour is closer than ``0.7 * min_distance``, because
    that is the exclusion radius of the local-maximum test.  Rows straddling an interface are
    often closer together than rows within either phase, so ``min_distance`` has to be the
    smallest site-to-site distance *anywhere* in the image, including across the interface --
    with a gap of 0.65 of the passed spacing, both modes lose sites there.

    ``valid_mask`` marks pixels that came from real data, and nothing within ``edge_margin`` of a
    false pixel is returned.  Pass ``apply_to_image``'s ``info["valid"]`` here: a drift-corrected
    image has a *slanted* wedge of fill along two sides, reaching ``|s| * n_rows`` pixels in, and
    a rectangular ``edge_margin`` cannot exclude it.  The half-columns along the wedge boundary
    are otherwise found and refined exactly like real sites -- and with ``threshold_mode
    ="prominence"`` they are *favoured*, since a flat fill makes a generous valley floor.
    """
    im = np.asarray(image, float)
    work = gaussian_filter(im, smooth) if smooth > 0 else im
    size = max(int(round(min_distance * 0.7)), 3)
    peaks = work == maximum_filter(work, size=size, mode="nearest")

    if threshold_mode == "prominence":
        floor = minimum_filter(work, size=max(int(round(min_distance * 1.2)), 3), mode="nearest")
        contrast = work - floor
    elif threshold_mode == "background":
        contrast = work - gaussian_filter(im, max(min_distance * 1.5, 2.0))
    else:
        raise ValueError(f"unknown threshold_mode {threshold_mode!r} (prominence, background)")
    scale = np.percentile(contrast, 99.5) - np.percentile(contrast, 50.0)
    peaks &= contrast > threshold_rel * scale

    ny, nx = im.shape
    m = int(np.ceil(edge_margin))
    if m > 0:
        peaks[:m, :] = peaks[-m:, :] = False
        peaks[:, :m] = peaks[:, -m:] = False
    if valid_mask is not None and edge_margin > 0:
        # Distance to the nearest invalid pixel, so the exclusion follows the shape of the missing
        # data instead of assuming it is rectangular.
        from scipy.ndimage import distance_transform_edt

        peaks &= distance_transform_edt(np.asarray(valid_mask, bool)) >= edge_margin
    yx = np.argwhere(peaks)
    return yx[:, ::-1].astype(float)  # -> (x, y)


def refine_peaks(
    image: NDArray, xy: NDArray, radius: float, mode: str = "com", iterations: int = 3
) -> NDArray:
    """
    Sub-pixel refinement of peak positions.

    ``mode="com"`` (default) is an iterated Gaussian-weighted centre of mass over a
    window of ``radius`` pixels, with a local background removed.  For a marginally
    resolved dumbbell this is the right estimator: the blob is symmetric about the
    dumbbell midpoint, so the centroid *is* the centroid, whereas fitting two
    overlapping Gaussians into an unresolved pair is biased and the bias depends on
    defocus.  ``mode="parabolic"`` is a cheap 3-point vertex fit, useful as a
    cross-check.  ``mode="gaussian"`` delegates to :func:`fit_peaks_gaussian` and discards
    its diagnostics; call that directly if you want them, which for hand-placed sites you do.

    Any residual per-site shape distortion (residual astigmatism, say) is identical at
    every symmetry-equivalent site, so it shifts all sites the same way and does not
    bias the fitted lattice vectors.  It is not worth correcting.
    """
    im = np.asarray(image, float)
    ny, nx = im.shape
    xy = np.asarray(xy, float).copy()

    if mode == "gaussian":
        return fit_peaks_gaussian(im, xy, radius=radius, iterations=max(iterations, 15))[0]

    if mode == "parabolic":
        out = xy.copy()
        ix = np.clip(np.rint(xy[:, 0]).astype(int), 1, nx - 2)
        iy = np.clip(np.rint(xy[:, 1]).astype(int), 1, ny - 2)
        for axis, col in ((1, 0), (0, 1)):
            if axis == 1:
                a, b, c = im[iy, ix - 1], im[iy, ix], im[iy, ix + 1]
                base = ix
            else:
                a, b, c = im[iy - 1, ix], im[iy, ix], im[iy + 1, ix]
                base = iy
            denom = a - 2 * b + c
            shift = np.where(
                np.abs(denom) > 1e-12,
                0.5 * (a - c) / np.where(np.abs(denom) > 1e-12, denom, 1.0),
                0.0,
            )
            out[:, col] = base + np.clip(shift, -0.5, 0.5)
        return out

    if mode != "com":
        raise ValueError(f"unknown refine mode {mode!r} (com, parabolic, gaussian)")

    r = int(np.ceil(radius))
    win = np.arange(-r, r + 1)
    WX, WY = np.meshgrid(win, win, indexing="xy")
    weight = np.exp(-0.5 * (WX**2 + WY**2) / (0.6 * radius) ** 2)
    weight[WX**2 + WY**2 > radius**2] = 0.0

    for _ in range(iterations):
        cx = np.rint(xy[:, 0]).astype(int)
        cy = np.rint(xy[:, 1]).astype(int)
        keep = (cx >= r) & (cx < nx - r) & (cy >= r) & (cy < ny - r)
        idx = np.nonzero(keep)[0]
        if len(idx) == 0:
            break
        # gather all windows at once: (n_sites, 2r+1, 2r+1)
        rows = cy[idx][:, None, None] + WY[None]
        cols = cx[idx][:, None, None] + WX[None]
        patch = im[rows, cols]
        patch = patch - patch.min(axis=(1, 2), keepdims=True)
        w = patch * weight[None]
        tot = w.sum(axis=(1, 2))
        good = tot > 0
        dx = np.where(
            good,
            (w * (cols - cx[idx][:, None, None])).sum(axis=(1, 2)) / np.where(good, tot, 1.0),
            0.0,
        )
        dy = np.where(
            good,
            (w * (rows - cy[idx][:, None, None])).sum(axis=(1, 2)) / np.where(good, tot, 1.0),
            0.0,
        )
        xy[idx, 0] = cx[idx] + dx
        xy[idx, 1] = cy[idx] + dy
    return xy


def fit_peaks_gaussian(
    image: NDArray,
    xy: NDArray,
    radius: float,
    max_move: float | None = None,
    iterations: int = 15,
    seed_com: bool = True,
) -> tuple[NDArray, dict]:
    """
    Fit an isotropic 2D Gaussian on a *planar* background at every site at once.

    ``refine_peaks(mode="com")`` is the right estimator for the lattice fit, where the target
    is an unresolved dumbbell and the centroid is the quantity of interest.  This is for the
    other job -- pinning down a *single* column, one site at a time, from a start that may be
    several pixels off.  Two things it buys, both measured on a synthetic lattice against exact
    truth (``tests/test_coherency.py``):

    * **It converges from a bad start.**  From 3.5 px off, the centroid lands 0.17 px from
      truth and this lands 0.01 px away.  That is the whole point for a hand-placed site: a
      windowed centroid inherits the asymmetry of whatever else is in its window, and a click
      is exactly the case where the window is not centred on the column.
    * **The plane makes it insensitive to a local intensity gradient**, which a centroid is
      not, at all.  Under a background ramp of 10 counts/px against a peak amplitude of 100,
      the centroid is biased 1.35 px downhill -- half a column -- and this stays within 0.01
      px.  Near an interface that gradient is the normal condition, not a pathology.  A *flat*
      background term is not enough; it is worse than the centroid here (0.5 px), because the
      unmodelled ramp is absorbed by moving the centre.

    On a genuinely flat background the centroid is marginally the better of the two (0.004 vs
    0.007 px, both noise-limited), which is one reason the lattice fit keeps it as its default.

    All sites are fitted simultaneously by a batched Levenberg-Marquardt iteration -- seven
    parameters per site ``(x0, y0, amplitude, sigma, background, dbg/dx, dbg/dy)``, the normal
    equations solved as a stack of 7x7 systems -- so a few thousand sites cost about as much as
    one.  The centre step is capped at 1 px per iteration, which keeps a poor start from
    tunnelling into a neighbouring column instead of converging on its own.

    ``max_move`` bounds the distance from the *input* position (default: ``radius``).  A site
    whose fit ends up outside it is **reverted to its input position and flagged in**
    ``info["too_far"]``, not clamped to the bound.  Clamping is the trap: the fit converges,
    returns a plausible number, and nothing tells you the position is really the bound rather
    than the data -- and because the clamped fraction grows with how distorted the image is,
    it flatters the worst data of a set.

    ``sigma`` is bounded above by ``0.5 * radius`` and sites that end on that bound are **not**
    ``ok``.  This is not a cosmetic bound: a Gaussian as wide as its own window is nearly flat
    across it, which makes it degenerate with the background plane -- amplitude and plane trade
    off freely and the centre stops being identifiable, so it wanders to wherever the
    surrounding structure happens to favour.  Measured on a lattice of overlapping blobs, 24-48%
    of sites ended on the old bound of ``radius``, displaced bimodally by a quarter of the
    lattice spacing towards one neighbour or the other.  A wide fitted sigma is therefore a
    statement that this window does not contain one isolated peak, and the honest response is to
    decline the site rather than report the position.

    Returns ``(xy_fitted, info)``, where ``info`` carries per-site ``ok``, ``moved``, ``sigma``,
    ``amplitude``, ``background``, ``rms``, ``too_far``, ``on_sigma_bound`` and ``edge`` arrays.
    Sites that are not ``ok`` come back at their input position, unchanged.
    """
    im = np.asarray(image, float)
    ny, nx = im.shape
    xy_in = np.asarray(xy, float).reshape(-1, 2)
    n = len(xy_in)
    out = xy_in.copy()
    info: dict = {
        "ok": np.zeros(n, bool),
        "moved": np.zeros(n),
        "sigma": np.full(n, np.nan),
        "amplitude": np.full(n, np.nan),
        "background": np.full(n, np.nan),
        "rms": np.full(n, np.nan),
        "too_far": np.zeros(n, bool),
        "on_sigma_bound": np.zeros(n, bool),
        "edge": np.zeros(n, bool),
        "n_ok": 0,
    }
    if n == 0:
        return out, info

    radius = float(radius)
    max_move = radius if max_move is None else float(max_move)
    sigma_lo, sigma_hi = 0.6, max(0.8 * radius, 0.7)
    start = refine_peaks(im, xy_in, radius=radius, mode="com") if seed_com else xy_in.copy()

    r = int(np.ceil(radius))
    win = np.arange(-r, r + 1)
    WX, WY = np.meshgrid(win, win, indexing="xy")
    mask = (WX**2 + WY**2) <= radius**2
    mx = WX[mask].astype(float)
    my = WY[mask].astype(float)

    cx = np.rint(start[:, 0]).astype(int)
    cy = np.rint(start[:, 1]).astype(int)
    inside = (cx >= r) & (cx < nx - r) & (cy >= r) & (cy < ny - r)
    info["edge"] = ~inside
    idx = np.nonzero(inside)[0]
    if len(idx) == 0:
        return out, info

    rows = cy[idx][:, None] + WY[mask][None, :]
    cols = cx[idx][:, None] + WX[mask][None, :]
    data = im[rows, cols]  # (k, m)
    k, m = data.shape

    def model(p):
        """``(prediction, g, dx, dy, sigma^2)`` -- the pieces the Jacobian also needs."""
        dx = mx[None, :] - p[:, 0:1]
        dy = my[None, :] - p[:, 1:2]
        s2 = np.maximum(p[:, 3:4], 1e-3) ** 2
        g = np.exp(-0.5 * (dx**2 + dy**2) / s2)
        plane = p[:, 4:5] + p[:, 5:6] * mx[None, :] + p[:, 6:7] * my[None, :]
        return p[:, 2:3] * g + plane, g, dx, dy, s2

    bg0 = np.median(data, axis=1)
    amp0 = np.maximum(data.max(axis=1) - bg0, 1e-9)
    p = np.stack(
        [
            start[idx, 0] - cx[idx],
            start[idx, 1] - cy[idx],
            amp0,
            np.full(k, np.clip(0.35 * radius, sigma_lo, sigma_hi)),
            bg0,
            np.zeros(k),
            np.zeros(k),
        ],
        axis=1,
    )
    n_par = p.shape[1]
    eye = np.eye(n_par)[None]
    lam = np.full(k, 1e-3)
    pred, g, dx, dy, s2 = model(p)
    res = pred - data
    cost = (res**2).sum(axis=1)

    for _ in range(int(iterations)):
        amp = p[:, 2:3]
        sig = np.maximum(p[:, 3:4], 1e-3)
        ones = np.ones_like(g)
        J = np.stack(
            [
                amp * g * dx / s2,
                amp * g * dy / s2,
                g,
                amp * g * (dx**2 + dy**2) / sig**3,
                ones,
                ones * mx[None, :],
                ones * my[None, :],
            ],
            axis=2,
        )  # (k, m, n_par)
        JtJ = np.einsum("kmi,kmj->kij", J, J)
        Jtr = np.einsum("kmi,km->ki", J, res)
        diag = np.maximum(np.einsum("kii->ki", JtJ), 1e-12)
        lhs = JtJ + lam[:, None, None] * eye * diag[:, None, :]
        try:
            step = -np.linalg.solve(lhs, Jtr[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:  # pragma: no cover - ridge makes this very unlikely
            step = -(np.linalg.pinv(lhs) @ Jtr[:, :, None])[:, :, 0]
        step = np.nan_to_num(step)
        # One pixel per iteration: a start that is off by more than half a lattice spacing
        # should fail to converge, not walk to the neighbouring column and succeed there.
        step[:, :2] = np.clip(step[:, :2], -1.0, 1.0)
        cand = p + step
        cand[:, :2] = np.clip(cand[:, :2], -radius, radius)
        cand[:, 2] = np.maximum(cand[:, 2], 0.0)
        cand[:, 3] = np.clip(cand[:, 3], sigma_lo, sigma_hi)
        pred_c, _, _, _, _ = model(cand)
        cost_c = ((pred_c - data) ** 2).sum(axis=1)
        better = cost_c < cost
        p = np.where(better[:, None], cand, p)
        cost = np.where(better, cost_c, cost)
        lam = np.where(better, np.maximum(lam * 0.3, 1e-8), np.minimum(lam * 4.0, 1e4))
        pred, g, dx, dy, s2 = model(p)
        res = pred - data

    fitted = np.stack([cx[idx] + p[:, 0], cy[idx] + p[:, 1]], axis=1)
    moved = np.hypot(*(fitted - xy_in[idx]).T)
    too_far = moved > max_move
    on_bound = (p[:, 3] <= sigma_lo + 1e-6) | (p[:, 3] >= sigma_hi - 1e-6)
    ok = np.isfinite(fitted).all(axis=1) & (p[:, 2] > 0) & ~too_far & ~on_bound
    out[idx[ok]] = fitted[ok]
    info["ok"][idx] = ok
    info["too_far"][idx] = too_far
    info["on_sigma_bound"][idx] = on_bound
    info["moved"][idx] = np.where(ok, moved, 0.0)
    info["sigma"][idx] = p[:, 3]
    info["amplitude"][idx] = p[:, 2]
    info["background"][idx] = p[:, 4]
    info["rms"][idx] = np.sqrt(cost / m)
    info["n_ok"] = int(ok.sum())
    return out, info


# ===========================================================================
# indexing.py -- lattice vector seeding, integer indexing, robust fit
# ===========================================================================


def seed_lattice_vectors(
    xy: NDArray,
    geometry: ReferenceGeometry,
    site_mode: str = "centroid",
    n_neighbors: int = 12,
    tolerance: float = 0.18,
) -> tuple:
    """
    Approximate cell vectors ``(a_vec, c_vec)`` in pixels, from neighbour statistics.

    Histogramming near-neighbour displacement vectors is more robust than fitting FFT
    peaks: it needs no peak model, degrades gracefully when a few sites are missing, and
    is unaffected by the window function.  The two shortest independent clusters give a
    *primitive* basis; small integer combinations of those are then searched for the pair
    that matches the reference cell's known axial ratio and near-axis orientation.

    No pixel size required -- the axial ratio (exactly sqrt(2) for zincblende <110>)
    identifies the two axes on its own, and the overall scale comes out of the fit as the
    nuisance parameter ``sigma``.  ``tolerance`` is the fractional slack allowed on that
    ratio, and must be comfortably larger than the distortion being measured.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, float)
    if len(xy) < 20:
        raise RuntimeError(f"only {len(xy)} sites -- too few to seed a lattice")
    tree = cKDTree(xy)
    k = min(n_neighbors + 1, len(xy))
    dist, idx = tree.query(xy, k=k)
    vecs = xy[idx[:, 1:]] - xy[:, None, :]
    vecs = vecs.reshape(-1, 2)
    vecs = vecs[vecs[:, 1] > -1e-9]  # half plane: v and -v equivalent
    vecs = vecs[np.hypot(*vecs.T) > 1e-9]

    # cluster by greedy merge in order of increasing length
    order = np.argsort(np.hypot(*vecs.T))
    vecs = vecs[order]
    clusters: list[list[NDArray]] = []
    tol_px = 0.25 * np.median(dist[:, 1])
    for v in vecs:
        for c in clusters:
            if np.hypot(*(v - c[0])) < tol_px:
                c.append(v)
                break
        else:
            clusters.append([v])
    clusters = [c for c in clusters if len(c) >= max(5, 0.05 * len(xy))]
    if len(clusters) < 2:
        raise RuntimeError("could not find two neighbour-vector clusters")
    centres = np.array([np.mean(c, axis=0) for c in clusters])
    counts = np.array([len(c) for c in clusters])

    # primitive pair: shortest, then shortest non-collinear
    lengths = np.hypot(*centres.T)
    o = np.argsort(lengths)
    v1 = centres[o[0]]
    v2 = None
    for i in o[1:]:
        cross = abs(v1[0] * centres[i, 1] - v1[1] * centres[i, 0])
        if cross > 0.25 * lengths[o[0]] ** 2:
            v2 = centres[i]
            break
    if v2 is None:
        raise RuntimeError("neighbour vectors are collinear -- not a 2-D lattice")

    a_vec, c_vec, ija, ijc = conventional_from_primitive(
        v1,
        v2,
        geometry.axial_ratio,
        tolerance,
        context="neighbour vectors of the detected sites",
        want_angle=geometry.axial_angle_deg,
    )
    return (
        a_vec,
        c_vec,
        dict(primitive=(v1, v2), combination=(ija, ijc), cluster_counts=counts.tolist()),
    )


def conventional_from_primitive(
    v1: NDArray,
    v2: NDArray,
    want_ratio: float,
    tolerance: float = 0.18,
    max_index: int = 3,
    context: str = "",
    want_angle: float = 90.0,
    angle_tolerance: float = 12.0,
) -> tuple:
    """
    Find the *conventional* cell as an integer combination of a primitive basis.

    Needed because the shortest two vectors of a centered-rectangular lattice are not the
    cell edges -- for zincblende <110> dumbbell centroids the shortest vector is the
    ``(1/2, 1/2)`` centering vector, and the two shortest independent vectors span a
    primitive oblique cell whose angle is nowhere near 90 deg.  Reporting the angle
    between *those* as evidence about orthogonality would be measuring the wrong thing.

    Selection uses only what symmetry fixes: the axial ratio, the angle between the two
    vectors, that ``a_vec`` lies nearer the ``x`` axis and ``c_vec`` nearer ``y`` (the "axes
    roughly aligned to the pixel grid" assumption, stated rather than hidden), and a
    preference for the shortest such pair.  ``tolerance`` and ``angle_tolerance`` must both
    exceed the distortion being measured.
    """
    B = np.stack([np.asarray(v1, float), np.asarray(v2, float)])
    span = range(-max_index, max_index + 1)
    cands = [(np.array([p, q], float) @ B, (p, q)) for p in span for q in span if (p, q) != (0, 0)]
    # Any integer combination is at least as long as the shorter primitive vector, so
    # this is a tight lower bound and not a tuned threshold.  It rejects the near
    # cancellations that arise when the two primitive vectors are nearly parallel.
    lmin = 0.9 * min(np.hypot(*B[0]), np.hypot(*B[1]))
    best = None
    for va, ija in cands:
        la = np.hypot(*va)
        if la < lmin or abs(va[0]) < abs(va[1]) or va[0] < 0:
            continue
        for vc, ijc in cands:
            lc = np.hypot(*vc)
            if lc < lmin or abs(vc[1]) < abs(vc[0]) or vc[1] < 0:
                continue
            ratio_err = abs((lc / la) / want_ratio - 1.0)
            if ratio_err > tolerance:
                continue
            if abs(va[0] * vc[1] - va[1] * vc[0]) < 0.3 * la * lc:
                continue
            ang = np.degrees(np.arccos(np.clip(np.dot(va, vc) / (la * lc), -1.0, 1.0)))
            if abs(ang - want_angle) > angle_tolerance:
                continue
            # Shortest valid pair wins.  This ordering is essential, not cosmetic: a
            # doubled cell (2a, 2c) has *exactly* the right axial ratio, so a cost
            # dominated by the ratio error is free to pick it, and then sigma comes out
            # at 2 and s and k are nonsense.  Ratio error only breaks ties.
            cost = (la + lc) * (1.0 + 0.1 * ratio_err)
            if best is None or cost < best[0]:
                best = (cost, va, vc, ija, ijc)
    if best is None:
        raise RuntimeError(
            f"no integer combination of the {context or 'primitive basis'} matches the "
            f"reference axial ratio {want_ratio:.4f} to within {tolerance:.0%}.  Either "
            "the zone axis is not the expected one, the reference region is not "
            "single-phase, or the crystal axes are far from the pixel grid."
        )
    return best[1], best[2], best[3], best[4]


def index_sites(
    xy: NDArray, origin: NDArray, a_vec: NDArray, c_vec: NDArray, basis: NDArray
) -> tuple:
    """
    Assign ``(m, n, b)`` to every site by rounding, choosing the basis entry ``b`` that
    gives the smallest residual.  Returns ``(mn, b, residual_frac)``.
    """
    A = np.stack([a_vec, c_vec])  # rows are cell vectors
    uv = np.linalg.solve(A.T, (xy - origin).T).T  # fractional cell coordinates
    best_r = np.full(len(xy), np.inf)
    best_mn = np.zeros((len(xy), 2))
    best_b = np.zeros(len(xy), int)
    for ib, off in enumerate(basis):
        d = uv - off
        mn = np.rint(d)
        r = np.hypot(*(d - mn).T)
        better = r < best_r
        best_r[better] = r[better]
        best_mn[better] = mn[better]
        best_b[better] = ib
    return best_mn.astype(int), best_b, best_r


def fit_lattice(xy: NDArray, mn: NDArray, b: NDArray, basis: NDArray) -> dict:
    """
    Least-squares fit of ``r = r0 + (m + bx) a_vec + (n + by) c_vec`` over all sites.

    Linear in ``(r0, a_vec, c_vec)``, so one ``lstsq`` and done.  The basis offsets are
    *fixed* by the reference geometry rather than fitted: a wrong basis choice then shows
    up as a large residual instead of being quietly absorbed.
    """
    coords = mn.astype(float) + basis[b]
    D = np.column_stack([np.ones(len(xy)), coords[:, 0], coords[:, 1]])
    sol, *_ = np.linalg.lstsq(D, xy, rcond=None)
    pred = D @ sol
    res = xy - pred
    return dict(
        origin=sol[0],
        a_vec=sol[1],
        c_vec=sol[2],
        predicted=pred,
        residual=res,
        rms=float(np.sqrt(np.mean(np.sum(res**2, axis=1)))),
    )


def robust_lattice_fit(
    xy: NDArray,
    geometry: ReferenceGeometry,
    a_vec: NDArray,
    c_vec: NDArray,
    basis: NDArray,
    origin: NDArray | None = None,
    max_index_residual: float = 0.3,
    sigma_clip: float = 4.0,
    clip_floor: float = 0.0,
    iterations: int = 6,
) -> dict:
    """
    Alternate indexing and fitting, clipping outliers, until the site set is stable.

    Sigma clipping rather than RANSAC: the inlier fraction here is high (a good reference
    band is mostly good sites) so the expensive part of RANSAC buys nothing, and clipping
    is deterministic, which matters because the result feeds a bootstrap.

    Rejects on two criteria -- a site whose fractional coordinates are further than
    ``max_index_residual`` from any lattice point is misindexed (defect, adatom, spurious
    peak), and a site whose fit residual exceeds the clip threshold is an outlier.

    ``clip_floor`` (in pixels) is a lower bound on that threshold, and it earns its keep
    whenever the distortion is not purely affine.  With a systematic component present the
    residual distribution is *broad by construction*, and a pure ``median + n*MAD`` rule
    then shaves off its tails -- which are exactly the rows at the top and bottom of the
    band, the ones carrying the vertical lever arm.  Measured on synthetic data with 1 px
    of smooth nonlinear wander, that mechanism alone rejected 21% of sites and biased ``s``
    by 13%.  Setting the floor to ~0.12 of the site spacing keeps genuine misidentifications
    out (they are off by half a spacing or more) while not discarding real displacements.
    """
    xy = np.asarray(xy, float)
    origin = (
        xy[np.argmin(np.sum((xy - xy.mean(0)) ** 2, axis=1))]
        if origin is None
        else np.asarray(origin, float)
    )
    keep = np.ones(len(xy), bool)
    fit = None
    history = []
    for it in range(iterations):
        mn, b, rfrac = index_sites(xy, origin, a_vec, c_vec, basis)
        ok = keep & (rfrac < max_index_residual)
        if ok.sum() < 12:
            raise RuntimeError(
                f"only {ok.sum()} sites survive indexing -- "
                "check the reference band and the site mode"
            )
        fit = fit_lattice(xy[ok], mn[ok], b[ok], basis)
        rr = np.hypot(*fit["residual"].T)
        mad = np.median(np.abs(rr - np.median(rr))) * 1.4826
        thresh = max(np.median(rr) + sigma_clip * max(mad, 1e-9), float(clip_floor))
        newkeep = np.zeros(len(xy), bool)
        newkeep[np.nonzero(ok)[0][rr <= thresh]] = True
        history.append(
            dict(iteration=it, n_used=int(ok.sum()), rms=fit["rms"], threshold=float(thresh))
        )
        origin, a_vec, c_vec = fit["origin"], fit["a_vec"], fit["c_vec"]
        if newkeep.sum() == keep.sum() and it > 0:
            keep = newkeep
            break
        keep = newkeep

    mn, b, rfrac = index_sites(xy, origin, a_vec, c_vec, basis)
    used = keep & (rfrac < max_index_residual)
    fit = fit_lattice(xy[used], mn[used], b[used], basis)
    fit.update(
        used=used,
        mn=mn,
        basis_index=b,
        index_residual=rfrac,
        history=history,
        n_used=int(used.sum()),
        n_total=len(xy),
        n_rejected=int(len(xy) - used.sum()),
    )
    return fit


def affine_from_cell(
    a_vec: NDArray, c_vec: NDArray, geometry: ReferenceGeometry, pixel_size: float | None = None
) -> NDArray:
    """
    The 2x2 matrix taking ideal crystal vectors to measured image vectors.

    Columns of ``A`` are where the reference cell's ``x`` and ``y`` unit directions land.
    Working in units of the reference cell (rather than Angstrom or pixels) keeps the
    nuisance scale ``sigma`` dimensionless and equal to 1 when the pixel size is right.
    """
    A_meas = np.stack([a_vec, c_vec], axis=1)  # columns: measured vectors
    A_ideal = geometry.cell.T  # columns: ideal vectors (general, not necessarily diagonal)
    if pixel_size is not None:
        A_meas = A_meas * float(pixel_size)  # -> Angstrom
    return A_meas @ np.linalg.inv(A_ideal)


# ===========================================================================
# the pipeline
# ===========================================================================


@dataclass
class SingleImageDrift(AutoSerialize):
    """
    Single-image drift/shear correction against a known-crystal reference band.

    Attributes are filled in as the pipeline runs; each stage records what it did into
    :attr:`warnings_raised` and :attr:`report`, so the JSON output is a faithful log
    rather than a summary written after the fact.
    """

    image: NDArray
    pixel_size: float | None = None  # Angstrom / pixel
    line_time: float | None = None  # s, for the implied drift velocity
    geometry: ReferenceGeometry = ZINCBLENDE_110
    site_mode: str = "centroid"  # {"centroid", "per-column"}
    flyback_rows: int = 0
    name: str = "single-image drift"

    # -- filled in by the pipeline -----------------------------------------
    band: tuple | None = None
    sites: NDArray | None = None
    lattice_fit: dict | None = None
    solution: dict | None = None
    nonlinear: dict | None = None
    warnings_raised: list = field(default_factory=list)
    report: dict = field(default_factory=dict)
    _clip_floor: float = 0.0

    # ---------------------------------------------------------------- setup
    @classmethod
    def from_data(
        cls,
        image,
        pixel_size=None,
        line_time=None,
        geometry: ReferenceGeometry = ZINCBLENDE_110,
        lattice_constant: float | None = None,
        site_mode: str = "centroid",
        flyback_rows: int = 0,
        **kw,
    ):
        """
        Build from a 2-D array or a :class:`~quantem.core.datastructures.dataset2d.Dataset2d`.

        Step 0 of the algorithm -- the flyback crop -- happens here, because everything
        downstream (row indices, ``d(t)``, the reference band) should be expressed in
        rows of the *kept* image.  ``flyback_rows`` are dropped from the start of the
        scan and the number dropped is recorded in the report.
        """
        arr, px = _as_array(image, pixel_size)
        if lattice_constant is not None:
            geometry = geometry.with_lattice_constant(lattice_constant)
        arr = np.asarray(arr, float)
        obj = cls(
            image=arr,
            pixel_size=px,
            line_time=line_time,
            geometry=geometry,
            site_mode=site_mode,
            flyback_rows=int(flyback_rows),
            **kw,
        )
        if obj.flyback_rows > 0:
            obj.image = arr[obj.flyback_rows :]
        obj.report["setup"] = dict(
            shape=list(obj.image.shape),
            pixel_size=px,
            line_time=line_time,
            geometry=geometry.name,
            zone=geometry.zone,
            a=geometry.a,
            site_mode=site_mode,
            flyback_rows_dropped=obj.flyback_rows,
            axial_ratio_exact=geometry.axial_ratio,
        )
        if obj.flyback_rows == 0:
            obj._warn(
                "no flyback crop requested; the first 10-20 rows of a real frame "
                "carry the post-flyback settling transient and will bias the fit"
            )
        return obj

    def _warn(self, msg: str):
        self.warnings_raised.append(msg)
        warnings.warn(f"[{self.name}] {msg}", stacklevel=3)

    # --------------------------------------------------------- reference band
    def set_reference_band(
        self,
        y0: int,
        y1: int,
        x0: int = 0,
        x1: int | None = None,
        min_periods: float = 10.0,
        hard_min_periods: float = 3.5,
    ):
        """
        Choose the reference band: substrate only, well clear of the interface.

        Full image width and **as tall as possible** -- the uncertainty on ``s`` scales
        inversely with the vertical lever arm, so a short band is the single most common way
        to get a useless answer.

        Two thresholds rather than one, because the failure mode changes character.  Between
        ``hard_min_periods`` and ``min_periods`` the answer degrades smoothly and honestly:
        measured on synthetic data, 4 periods gave ``s`` to 1.5e-3 and 24 periods to 5e-5,
        with the reported uncertainty tracking that the whole way.  Below that it stops
        degrading and starts being *wrong* -- at 3 periods the lattice seeding latches onto
        the wrong cell and returns ``s = -1.8`` with a small error bar -- so the default
        ``hard_min_periods`` sits between the last band height measured to work and the
        first measured to fail, and that case raises rather than warns.

        ``y0``/``y1`` are rows of the *flyback-cropped* image.
        """
        ny, nx = self.image.shape
        x1 = nx if x1 is None else x1
        y0, y1 = int(max(0, y0)), int(min(ny, y1))
        if y1 - y0 < 8:
            raise ValueError("reference band is empty or inverted")
        self.band = (y0, y1, int(x0), int(x1))

        spacing_px = self._expected_spacing_px()
        periods = (
            (y1 - y0) / (self.geometry.ay / self.pixel_size)
            if self.pixel_size
            else (y1 - y0) / (spacing_px * self.geometry.axial_ratio)
        )
        self.report["band"] = dict(
            y0=y0,
            y1=y1,
            x0=int(x0),
            x1=int(x1),
            height_rows=int(y1 - y0),
            height_periods=float(periods),
        )
        if periods < hard_min_periods:
            self.band = None
            raise ValueError(
                f"reference band spans only {periods:.1f} lattice periods vertically.  "
                f"Below ~{hard_min_periods:.0f} the lattice seeding picks the wrong cell "
                "and the result is wrong rather than merely imprecise, so this refuses to "
                "run.  Use a taller band; if you genuinely have no more substrate, this "
                "method cannot help with this image.  (Override with hard_min_periods.)"
            )
        if periods < min_periods:
            self._warn(
                f"reference band spans only {periods:.1f} lattice periods "
                f"vertically (want >= {min_periods:.0f}); s will be poorly "
                "conditioned and its uncertainty large"
            )
        if y1 > 0.9 * ny:
            self._warn(
                "reference band reaches the top of the frame; if the film is "
                "there, real strain will be absorbed into (s, k)"
            )
        return self

    @property
    def band_image(self) -> NDArray:
        if self.band is None:
            raise RuntimeError("call set_reference_band() first")
        y0, y1, x0, x1 = self.band
        return self.image[y0:y1, x0:x1]

    def _expected_spacing_px(self) -> float:
        """Shortest site spacing in pixels: from the pixel size if known, else measured."""
        if self.pixel_size:
            return self.geometry.site_spacing(self.site_mode) / self.pixel_size
        return estimate_site_spacing(self.band_image if self.band else self.image)

    # ------------------------------------------------------------ find sites
    def find_sites(
        self,
        threshold_rel: float = 0.15,
        smooth: float | None = None,
        refine: str = "com",
        refine_radius: float | None = None,
        edge_margin: float | None = None,
    ):
        """
        Detect and refine reference-band sites (Step 2).

        Defaults target the marginally resolved case: a light smoothing at ~1/4 of the
        site spacing, local maxima, then an iterated centre of mass over ~0.45 of the
        site spacing.  Sites within ``edge_margin`` of the band edge are dropped, because
        a truncated neighbourhood makes the centroid biased in a way that is *not* the
        same at every site -- unlike the interior bias, which cancels out of the lattice
        vectors.
        """
        spacing = self._expected_spacing_px()
        smooth = 0.25 * spacing if smooth is None else smooth
        refine_radius = 0.45 * spacing if refine_radius is None else refine_radius
        edge_margin = 0.6 * spacing if edge_margin is None else edge_margin

        band = self.band_image
        xy = find_peaks(
            band,
            min_distance=spacing,
            threshold_rel=threshold_rel,
            smooth=smooth,
            edge_margin=edge_margin,
        )
        if len(xy) < 30:
            raise RuntimeError(
                f"only {len(xy)} peaks found in the reference band; "
                "check threshold_rel and the site spacing"
            )
        xy = refine_peaks(band, xy, radius=refine_radius, mode=refine)
        y0, y1, x0, x1 = self.band
        self.sites = xy + np.array([x0, y0])  # -> full-image pixel coordinates
        self.report["sites"] = dict(
            n_found=int(len(xy)),
            site_spacing_px=float(spacing),
            smooth_px=float(smooth),
            refine=refine,
            refine_radius_px=float(refine_radius),
            edge_margin_px=float(edge_margin),
            site_mode=self.site_mode,
            expected_sites_per_cell=len(self.geometry.basis(self.site_mode)),
        )
        if self.site_mode == "per-column" and self.pixel_size:
            split_px = self.geometry.d_dumbbell / self.pixel_size
            if split_px < 2.5:
                self._warn(
                    f"site_mode='per-column' with a dumbbell split of only "
                    f"{split_px:.1f} px; on an uncorrected probe the two columns "
                    "are unresolved and per-column fitting is biased -- prefer "
                    "site_mode='centroid'"
                )
        return self

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        tolerance: float = 0.18,
        sigma_clip: float = 4.0,
        clip_floor_fraction: float = 0.12,
        n_bootstrap: int = 300,
        seed: int = 0,
    ):
        """
        Index the sites and solve for the distortion (Steps 3 and 4).

        Fits the full 4-parameter affine by least squares, then decomposes it exactly
        into ``(s, k)`` plus the nuisance pair ``(theta, sigma)``.  Fitting the general
        affine and decomposing is equivalent to fitting the constrained model directly --
        see :func:`decompose_affine` -- and it has the advantage of producing a residual
        that tests something real: whether the site positions are affine *at all*.

        Uncertainties are bootstrapped over sites, which is the honest thing here because
        the errors are dominated by site-to-site scatter rather than by anything with a
        known distribution.
        """
        if self.sites is None:
            raise RuntimeError("call find_sites() first")
        basis = self.geometry.basis(self.site_mode)
        a_vec, c_vec, seed_info = self._seed_lattice(tolerance)
        self._clip_floor = clip_floor_fraction * self._expected_spacing_px()
        fit = robust_lattice_fit(
            self.sites,
            self.geometry,
            a_vec,
            c_vec,
            basis,
            sigma_clip=sigma_clip,
            clip_floor=self._clip_floor,
        )
        self.lattice_fit = fit

        A = affine_from_cell(fit["a_vec"], fit["c_vec"], self.geometry, self.pixel_size)
        dec = decompose_affine(A)

        # --- bootstrap over sites
        rng = np.random.default_rng(seed)
        used = np.nonzero(fit["used"])[0]
        boot = []
        for _ in range(n_bootstrap):
            pick = rng.integers(0, len(used), len(used))
            sub = used[pick]
            f = fit_lattice(self.sites[sub], fit["mn"][sub], fit["basis_index"][sub], basis)
            try:
                d = decompose_affine(
                    affine_from_cell(f["a_vec"], f["c_vec"], self.geometry, self.pixel_size)
                )
            except ValueError:
                continue
            boot.append([d["s"], d["k"], d["theta_deg"], d["sigma"]])
        boot = np.array(boot)
        err = dict(
            zip(
                ["s", "k", "theta_deg", "sigma"],
                boot.std(axis=0, ddof=1) if len(boot) > 2 else [np.nan] * 4,
            )
        )

        rms_px = fit["rms"]
        rms_pm = rms_px * self.pixel_size * 100.0 if self.pixel_size else None
        self.solution = dict(
            **dec,
            A=A,
            M=drift_matrix(dec["s"], dec["k"]),
            uncertainty=err,
            n_bootstrap=int(len(boot)),
            a_vec_px=fit["a_vec"].tolist(),
            c_vec_px=fit["c_vec"].tolist(),
            rms_residual_px=rms_px,
            rms_residual_pm=rms_pm,
            n_sites_used=fit["n_used"],
            n_sites_rejected=fit["n_rejected"],
            seed_info={
                k: (np.asarray(v).tolist() if isinstance(v, (np.ndarray, tuple)) else v)
                for k, v in seed_info.items()
            },
        )
        self._angle_residuals()
        self._pixel_size_check()
        self._report_fit()
        return self

    def _seed_lattice(self, tolerance: float = 0.18):
        """
        Approximate cell vectors to start the indexing from, in pixels.

        The FFT is tried first and the site-neighbour statistics are the fallback, in that
        order for a specific reason: the FFT averages the whole reference band, so it is
        essentially immune to the spurious peaks a low-dose image hands to the peak finder,
        whereas neighbour-vector clustering is not.  At ~10 counts/pixel a few hundred
        noise maxima are enough to create a short spurious cluster, and seeding from that
        sends the indexing to a completely wrong cell -- the failure is silent and total,
        not a graceful loss of precision.  The site positions are still what set the final
        precision; the FFT only has to get the integer indexing right.
        """
        fft = self.fft_check(tolerance=max(tolerance, 0.25))
        if fft.get("ok"):
            a_vec = np.asarray(fft["a_vec_px"], float)
            c_vec = np.asarray(fft["c_vec_px"], float)
            info = dict(source="fft", g1=fft["g1"], g2=fft["g2"], combination=fft["combination"])
            try:
                index_sites(
                    self.sites, self.sites[0], a_vec, c_vec, self.geometry.basis(self.site_mode)
                )
                return a_vec, c_vec, info
            except np.linalg.LinAlgError:
                pass
        try:
            a_vec, c_vec, info = seed_lattice_vectors(
                self.sites, self.geometry, self.site_mode, tolerance=tolerance
            )
            info["source"] = "site neighbour vectors"
            if fft.get("reason"):
                info["fft_fallback_reason"] = fft["reason"]
            return a_vec, c_vec, info
        except RuntimeError as exc:
            raise RuntimeError(
                f"could not seed the lattice.  FFT route: {fft.get('reason', 'failed')}.  "
                f"Neighbour-vector route: {exc}"
            ) from exc

    @staticmethod
    def _angle_of(v) -> float:
        """Angle from ``[001]`` (the ``y`` axis), degrees, folded into [0, 180)."""
        return float(np.degrees(np.arctan2(abs(v[0]), v[1])) % 180.0)

    def _angle_residuals(self):
        """
        Per-direction angle, measured and after correction (Step 4 reporting).

        Read the ``measured`` column, not the ``corrected`` one.  ``measured - ideal``
        says how badly each direction was distorted, which is real information.
        ``corrected - ideal``, on the other hand, is *forced* to equal the nuisance
        rotation ``theta`` for every direction: once ``M`` is divided out, what remains is
        ``sigma R(theta)``, which is exactly rectangular, so all four residuals are equal
        to ``-theta`` by construction and none of them tests anything.  That is why the
        genuinely informative per-direction number is the cross-validated one from
        :meth:`held_out_direction_test`, which is what the figure plots.
        """
        A = self.solution["A"]
        Minv = np.linalg.inv(self.solution["M"])
        rows = []
        for m, n, label in self.geometry.directions:
            ideal = self.geometry.ideal_vector(m, n)
            meas, corr = A @ ideal, Minv @ A @ ideal
            ideal_deg = self.geometry.ideal_angle(m, n)
            rows.append(
                dict(
                    mn=[m, n],
                    label=label,
                    ideal_deg=ideal_deg,
                    measured_deg=self._angle_of(meas),
                    distortion_deg=self._angle_of(meas) - ideal_deg,
                    corrected_deg=self._angle_of(corr),
                    residual_deg=self._angle_of(corr) - ideal_deg,
                    residual_is_theta_by_construction=True,
                )
            )
        self.solution["angles"] = rows

    def _pixel_size_check(self):
        """
        Cross-check the metadata pixel size against the one the crystal implies.

        ``sigma`` is the isotropic scale the fit had to apply to the *stated* pixel size
        to make the reference crystal come out the right size.  If the pixel size is
        right, ``sigma == 1``.  A ``sigma`` of 1.03 does not mean the crystal is 3% large;
        it means the magnification calibration is 3% off, and any lattice parameter
        quoted from this image inherits that error.
        """
        sol = self.solution
        if self.pixel_size:
            sol["pixel_size_stated"] = self.pixel_size
            sol["pixel_size_implied"] = self.pixel_size / sol["sigma"]
            if abs(sol["sigma"] - 1.0) > 0.02:
                self._warn(
                    f"the reference crystal implies a pixel size of "
                    f"{sol['pixel_size_implied']:.4f} A, not the stated "
                    f"{self.pixel_size:.4f} A (sigma = {sol['sigma']:.4f}); the "
                    "magnification calibration is off by that factor"
                )
        else:
            sol["pixel_size_implied"] = self.geometry.ax / np.hypot(*self.lattice_fit["a_vec"])
            sol["pixel_size_stated"] = None

    def _report_fit(self):
        sol = self.solution
        u = sol["uncertainty"]
        rep = dict(
            s=sol["s"],
            s_err=u["s"],
            k=sol["k"],
            k_err=u["k"],
            theta_deg=sol["theta_deg"],
            theta_deg_err=u["theta_deg"],
            sigma=sol["sigma"],
            sigma_err=u["sigma"],
            shear_angle_deg=float(np.degrees(np.arctan(sol["s"]))),
            n_sites_used=sol["n_sites_used"],
            n_sites_rejected=sol["n_sites_rejected"],
            rms_residual_px=sol["rms_residual_px"],
            rms_residual_pm=sol["rms_residual_pm"],
            pixel_size_stated=sol["pixel_size_stated"],
            pixel_size_implied=sol["pixel_size_implied"],
            angles=sol["angles"],
            A=np.asarray(sol["A"]).tolist(),
            M=np.asarray(sol["M"]).tolist(),
        )
        if self.pixel_size and self.line_time:
            vx, vy = velocities_from_sk(sol["s"], sol["k"], self.pixel_size, self.line_time)
            rep["drift_vx_pm_per_s"] = vx * 100.0
            rep["drift_vy_pm_per_s"] = vy * 100.0
            self._warn(
                "vy is reported for completeness only: k is a pure y-scale change "
                "and is degenerate with scan-coil gain anisotropy, so vy is not a "
                "calibrated drift velocity without an independent gain calibration"
            )
        self.report["fit"] = rep

        # Rejection pattern.  A high rejection rate is not itself alarming -- defects and
        # spurious peaks get thrown out, which is the point -- but rejections *clustered in
        # rows* mean the integer indexing is failing wherever the nonlinear excursion is
        # largest, and then (s, k) is fit only on the well-behaved middle of the band and
        # comes out biased with a small, confident-looking error bar.
        n_tot = sol["n_sites_used"] + sol["n_sites_rejected"]
        frac = sol["n_sites_rejected"] / max(n_tot, 1)
        self.report["fit"]["rejected_fraction"] = float(frac)
        if frac > 0.10 and self.lattice_fit is not None:
            y = self.sites[:, 1]
            used = self.lattice_fit["used"]
            nb = max(6, int((self.band[1] - self.band[0]) / 60))
            edges = np.linspace(y.min(), y.max() + 1e-6, nb + 1)
            b = np.clip(np.searchsorted(edges, y) - 1, 0, nb - 1)
            tot = np.bincount(b, minlength=nb).astype(float)
            rej = np.bincount(b[~used], minlength=nb).astype(float)
            rate = np.divide(rej, tot, out=np.zeros(nb), where=tot > 5)
            clustered = float(rate.max()) > 0.5 and float(np.median(rate)) < 0.2
            self.report["fit"]["rejection_rate_by_row_band"] = rate.tolist()
            msg = f"{frac:.0%} of sites were rejected"
            if clustered:
                self._warn(
                    msg + ", and they are clustered in specific row bands rather "
                    "than scattered.  That is the signature of the integer indexing "
                    "failing where the drift excursion is largest: (s, k) is then "
                    "fit only on the well-behaved rows and is biased.  Run "
                    "row_wise_correction(), which iterates the indexing"
                )
            else:
                self._warn(
                    msg + " (scattered across the band, so most likely defects or "
                    "spurious peaks rather than an indexing failure)"
                )

        # Plausibility.  Drift this large would have ruined the image beyond the point of
        # measuring anything, so a big |s| or a wild k means the lattice fit latched onto
        # the wrong cell -- and it will say so with a small error bar, because it fits its
        # wrong cell perfectly well.  Cheap to check, and it converts a silent wrong answer
        # into a loud one.
        implausible = abs(sol["s"]) > 0.3 or not (0.5 < sol["k"] < 2.0)
        self.report["fit"]["implausible"] = bool(implausible)
        if implausible:
            self._warn(
                f"IMPLAUSIBLE SOLUTION: s = {sol['s']:+.3f}, k = {sol['k']:.3f}.  "
                f"Drift of this size would have destroyed the image, so the lattice "
                "fit has almost certainly latched onto the wrong cell.  Check the "
                "reference band height, the site mode, and the zone axis.  Do not "
                "trust the quoted uncertainty -- a wrong cell fits itself well"
            )

        # significance checks
        if u["s"] and abs(sol["s"]) < 2 * u["s"]:
            self._warn(
                f"s = {sol['s']:.2e} +/- {u['s']:.2e} is not significant; there "
                "may be no measurable shear"
            )
        if abs(sol["theta_deg"]) > 5.0:
            self._warn(
                f"fitted global rotation is {sol['theta_deg']:.2f} deg.  That is "
                "absorbed as a nuisance parameter, but if the scan rotation was "
                "set to zero it means the crystal is not aligned to the scan and "
                "the shear/rotation split is only as good as that assumption"
            )

    # ------------------------------------------------- Step 5: nonlinearity
    def test_nonlinearity(
        self,
        n_splits: int = 2,
        sigma_threshold: float = 3.0,
        n_bootstrap: int = 200,
        seed: int = 1,
    ):
        """
        Refit ``s`` in horizontal sub-bands and test whether it is constant (Step 5).

        This is the model check with teeth.  A global affine can only be right if the
        drift velocity was constant; if ``s`` differs between the top and bottom of the
        reference band by more than the site scatter allows, the drift was not linear and
        no single affine will fix the image.

        The comparison threshold is bootstrapped **per split**, not scaled from the global
        uncertainty by a fudge factor.  It has to be: each split has a shorter vertical
        lever arm than the whole band, so its ``s`` is intrinsically noisier, and by a
        factor that depends on how the sites happen to be distributed.  Two splits differ
        significantly when their separation exceeds ``sigma_threshold`` times the
        quadrature sum of their own bootstrap errors.

        Returns, and stores in ``self.nonlinear``, per-split values and a pass/fail flag.
        """
        if self.solution is None:
            raise RuntimeError("call fit() first")
        y0, y1, x0, x1 = self.band
        basis = self.geometry.basis(self.site_mode)
        rng = np.random.default_rng(seed)
        edges = np.linspace(y0, y1, n_splits + 1)
        parts = []
        for i in range(n_splits):
            lo, hi = edges[i], edges[i + 1]
            sel = self.lattice_fit["used"] & (self.sites[:, 1] >= lo) & (self.sites[:, 1] < hi)
            idx = np.nonzero(sel)[0]
            if len(idx) < 20:
                parts.append(
                    dict(
                        y0=float(lo),
                        y1=float(hi),
                        n=int(len(idx)),
                        s=np.nan,
                        s_err=np.nan,
                        k=np.nan,
                    )
                )
                continue
            f = fit_lattice(
                self.sites[idx],
                self.lattice_fit["mn"][idx],
                self.lattice_fit["basis_index"][idx],
                basis,
            )
            d = decompose_affine(
                affine_from_cell(f["a_vec"], f["c_vec"], self.geometry, self.pixel_size)
            )
            boot = []
            for _ in range(n_bootstrap):
                sub = idx[rng.integers(0, len(idx), len(idx))]
                fb = fit_lattice(
                    self.sites[sub],
                    self.lattice_fit["mn"][sub],
                    self.lattice_fit["basis_index"][sub],
                    basis,
                )
                try:
                    boot.append(
                        decompose_affine(
                            affine_from_cell(
                                fb["a_vec"], fb["c_vec"], self.geometry, self.pixel_size
                            )
                        )["s"]
                    )
                except ValueError:
                    continue
            parts.append(
                dict(
                    y0=float(lo),
                    y1=float(hi),
                    n=int(len(idx)),
                    s=d["s"],
                    s_err=float(np.std(boot, ddof=1)) if len(boot) > 2 else np.nan,
                    k=d["k"],
                    theta_deg=d["theta_deg"],
                    rms_px=f["rms"],
                )
            )

        svals = np.array([p["s"] for p in parts], float)
        serrs = np.array([p.get("s_err", np.nan) for p in parts], float)
        good = np.isfinite(svals) & np.isfinite(serrs)
        if good.sum() < 2:
            spread, allowed, passed = np.nan, np.nan, False
        else:
            hi_i, lo_i = (
                np.argmax(np.where(good, svals, -np.inf)),
                np.argmin(np.where(good, svals, np.inf)),
            )
            spread = float(svals[hi_i] - svals[lo_i])
            allowed = float(sigma_threshold * np.hypot(serrs[hi_i], serrs[lo_i]))
            passed = bool(abs(spread) <= allowed)
        self.nonlinear = dict(
            splits=parts,
            s_spread=spread,
            s_allowed=allowed,
            passed=passed,
            n_splits=n_splits,
            sigma_threshold=sigma_threshold,
        )
        self.report["nonlinearity"] = self.nonlinear
        if not passed:
            self._warn(
                f"NONLINEAR DRIFT: s varies by {spread:.2e} across the band, more "
                f"than the {allowed:.2e} the site scatter allows.  A global affine "
                "will not fix this image -- use row_wise_correction(), and treat "
                "the global (s, k) as an average only"
            )
        return self.nonlinear

    # ------------------------------------------- Step 6: row-wise correction
    def _row_trajectory(self, used, predicted, poly_order, min_sites_per_row):
        """One pass of the ``d(t)`` extraction; see :meth:`row_wise_correction`."""
        res = self.sites[used] - predicted
        basis = self.geometry.basis(self.site_mode)
        n_idx = (
            self.lattice_fit["mn"][used][:, 1] + basis[self.lattice_fit["basis_index"][used]][:, 1]
        )
        rows_y = self.sites[used][:, 1]

        # group by atomic row: the n index plus the basis y offset labels the row
        labels, inv = np.unique(np.round(n_idx * 4).astype(int), return_inverse=True)
        counts = np.bincount(inv, minlength=len(labels))
        keep = counts >= min_sites_per_row
        if keep.sum() < poly_order + 2:
            raise RuntimeError(
                f"only {int(keep.sum())} atomic rows have >= {min_sites_per_row} usable "
                f"sites; cannot fit an order-{poly_order} trajectory"
            )

        def agg(f):
            return np.array([f(i) for i in range(len(labels))])[keep]

        t = agg(lambda i: rows_y[inv == i].mean())
        dx = agg(lambda i: res[inv == i, 0].mean())
        dy = agg(lambda i: res[inv == i, 1].mean())
        sem = agg(lambda i: res[inv == i, 0].std(ddof=1) / max(np.sqrt((inv == i).sum()), 1.0))
        order = np.argsort(t)
        return (t[order], dx[order], dy[order], sem[order], int(len(labels)), int(keep.sum()))

    def row_wise_correction(self, poly_order: int = 2, min_sites_per_row: int = 6):
        """
        Extract the drift trajectory ``d(t)`` row by row and fit a low-order polynomial.

        Row index is time, so the mean ``x`` offset of each row of atoms from the fitted
        ideal lattice *is* the drift trajectory.  Fitting order 2--3 is deliberate: this is
        a few tens to a few hundred noisy points, and a high-order polynomial will happily
        absorb real lattice information.

        A single pass, deliberately
        ---------------------------
        An earlier version iterated -- undo the fitted trajectory, re-index, re-measure --
        because whole atomic rows were dropping out of the fit wherever the excursion was
        large, leaving ``d(t)`` measured only across the middle of the band.  That turned
        out to be a symptom of the sigma clip shaving the tails of a systematic trend, which
        is now fixed at source by ``clip_floor`` in :func:`robust_lattice_fit`.  With the
        cause removed, a single pass covers the whole band, and iterating actively hurt:
        each pass re-fits the lattice, which re-absorbs part of the trajectory already
        removed, so the accumulated polynomial double-counts.  On synthetic data with 1 px
        of wander, one pass recovered the injected curve to 0.02 px RMS and two passes to
        1.1 px.  The workaround is gone rather than kept alongside the fix.

        Two limitations, both reported rather than papered over.  ``d(t)`` is only measurable
        where the known substrate is, so extending it into the film assumes the trajectory
        stays smooth -- see ``extrapolated_fraction``.  And when the excursion grows to a
        sizeable fraction of the site spacing, the global affine has already absorbed part of
        it, so the extracted amplitude is an *underestimate*; a warning fires in that regime.
        """
        if self.lattice_fit is None:
            raise RuntimeError("call fit() first")
        ny = self.image.shape[0]
        used = self.lattice_fit["used"]
        t, dx, dy, sem, n_rows_all, n_rows_used = self._row_trajectory(
            used, self.lattice_fit["predicted"], poly_order, min_sites_per_row
        )
        poly_x = np.polyfit(t, dx, poly_order)
        poly_y = np.polyfit(t, dy, poly_order)

        traj_rms = float(np.std(np.polyval(poly_x, t)))
        spacing = self._expected_spacing_px()
        # Is the row-to-row structure real, or is it just the site scatter reshuffled?
        # std(dx) against the per-row standard error answers that, and it does so without
        # reference to the fitted amplitude -- which matters, because the amplitude is the
        # quantity suspected of being too small.  Keying the warning on the amplitude
        # itself would be circular: an underestimated curve would fail to warn that it is
        # underestimated, which is exactly the failure seen while developing this.
        significance = float(np.std(dx) / max(np.mean(sem), 1e-12))
        self.nonlinear = dict(self.nonlinear or {})
        self.nonlinear.update(
            row_trajectory=dict(
                t=t.tolist(),
                dx=dx.tolist(),
                dy=dy.tolist(),
                sem=sem.tolist(),
                poly_x=poly_x.tolist(),
                poly_y=poly_y.tolist(),
                poly_order=poly_order,
                n_rows=int(len(t)),
                n_rows_total=n_rows_all,
                measured_span=[float(t.min()), float(t.max())],
                extrapolated_fraction=float(1.0 - (t.max() - t.min()) / ny),
                residual_rms_px=float(np.std(dx - np.polyval(poly_x, t))),
                trajectory_rms_px=traj_rms,
                trajectory_over_spacing=float(traj_rms / max(spacing, 1e-9)),
                significance=significance,
            )
        )
        self.report["nonlinearity"] = self.nonlinear
        frac = self.nonlinear["row_trajectory"]["extrapolated_fraction"]
        if frac > 0.25:
            self._warn(
                f"{frac:.0%} of the image lies outside the rows where d(t) could "
                "be measured; the row-wise correction is extrapolated there and "
                "assumes the drift trajectory stays smooth"
            )
        if significance < 3.0:
            self._warn(
                f"the row-to-row structure is only {significance:.1f}x the per-row "
                "standard error, so d(t) is not clearly distinguishable from site "
                "scatter; applying it will add noise rather than remove distortion"
            )
        else:
            self._warn(
                f"d(t) is significant ({significance:.0f}x the per-row standard "
                f"error, {traj_rms:.2f} px RMS), which means the global affine "
                "absorbed part of the same curve: treat this amplitude as a LOWER "
                "BOUND and the global (s, k) as biased.  The linear and nonlinear "
                "parts of a drift trajectory are not cleanly separable from a "
                "single image"
            )
        return self.nonlinear["row_trajectory"]

    # -------------------------------------------------------- Step 7: apply
    def apply_to_coordinates(self, xy: NDArray, row_wise: bool = False) -> NDArray:
        """
        Apply ``M^-1`` to coordinates (the preferred mode).

        No interpolation and no resolution loss, so this is what any strain or
        lattice-parameter measurement should use.  Note that only ``M`` is undone: the
        nuisance rotation and scale are left alone, because they are not distortions and
        removing them would silently recalibrate the image.
        """
        if self.solution is None:
            raise RuntimeError("call fit() first")
        xy = np.asarray(xy, float)
        if row_wise:
            xy = self._undo_row_offsets(xy)
        return xy @ np.linalg.inv(self.solution["M"]).T

    def _undo_row_offsets(self, xy: NDArray) -> NDArray:
        traj = (self.nonlinear or {}).get("row_trajectory")
        if traj is None:
            raise RuntimeError("call row_wise_correction() first")
        out = np.asarray(xy, float).copy()
        out[:, 0] -= np.polyval(traj["poly_x"], out[:, 1])
        out[:, 1] -= np.polyval(traj["poly_y"], out[:, 1])
        return out

    def apply_to_image(
        self,
        order: int = 3,
        row_wise: bool = False,
        output_shape: tuple | None = None,
        cval: float = np.nan,
    ):
        """
        Resample the whole image with ``M^-1`` for display (Step 7, image mode).

        Applied to the *entire* image, not just the reference band, because the
        distortion is global -- the band is only where it could be measured.  This
        interpolates, and a cubic spline resample of an atomic-resolution image visibly
        softens the peaks; use :meth:`apply_to_coordinates` for measurement and this for
        figures.  Returns ``(corrected, extent_info)``.
        """
        if self.solution is None:
            raise RuntimeError("call fit() first")
        im = self.image
        M = self.solution["M"]
        Minv = np.linalg.inv(M)
        ny, nx = im.shape

        corners = np.array([[0, 0], [nx, 0], [0, ny], [nx, ny]], float)
        out_corners = corners @ Minv.T
        lo = out_corners.min(axis=0)
        hi = out_corners.max(axis=0)
        if output_shape is None:
            output_shape = (int(np.ceil(hi[1] - lo[1])), int(np.ceil(hi[0] - lo[0])))
        oy, ox = output_shape
        gx, gy = np.meshgrid(np.arange(ox) + lo[0], np.arange(oy) + lo[1])
        src = np.stack([gx.ravel(), gy.ravel()], axis=1) @ M.T
        if row_wise:
            traj = (self.nonlinear or {}).get("row_trajectory")
            if traj is None:
                raise RuntimeError("call row_wise_correction() first")
            src[:, 0] += np.polyval(traj["poly_x"], src[:, 1])
            src[:, 1] += np.polyval(traj["poly_y"], src[:, 1])
        out = map_coordinates(
            im, [src[:, 1], src[:, 0]], order=order, mode="constant", cval=cval
        ).reshape(oy, ox)
        # Which output pixels came from real data.  The shear leaves a *slanted* wedge of
        # out-of-bounds fill along two sides, reaching |s| * n_rows pixels into the frame, and a
        # rectangular edge margin cannot exclude it.  Callers that go looking for peaks need this:
        # a wedge filled with a constant is flat, but the boundary between it and the data is a
        # step, and the half-columns along that step are found and refined like any other site.
        valid = np.isfinite(out)
        info = dict(
            order=order,
            row_wise=bool(row_wise),
            origin_offset=lo.tolist(),
            output_shape=[int(oy), int(ox)],
            valid_fraction=float(valid.mean()),
            valid=valid,
            note=("cubic-spline resample: this blurs; use apply_to_coordinates for measurements"),
        )
        self.report["apply_image"] = info
        return out, info

    # -------------------------------------------------------- diagnostics
    def fft_check(
        self,
        image: NDArray | None = None,
        band: tuple | None = None,
        n_peaks: int = 24,
        exclude_radius: float = 8.0,
        rel_threshold: float = 0.02,
        tolerance: float = 0.25,
    ) -> dict:
        """
        Measure the substrate spot lattice from the FFT: axial angle and axial ratio.

        Shares none of the site-finding or least-squares code, which is the point --
        agreement between this and :meth:`fit` is meaningful precisely because the two
        paths are independent.  Run it before and after correction; afterwards the axes
        should be orthogonal with the reference crystal's exact axial ratio.

        Two traps this avoids.  First, the strongest low-frequency peaks come from
        illumination ripple and the interface, not the lattice, so everything inside
        ``exclude_radius`` pixels of DC is masked.  Second -- and this is the one that
        gives silently wrong answers -- for a centered-rectangular lattice the two
        shortest *reciprocal* vectors do not correspond to the cell edges (the ``(1,0)``
        and ``(0,1)`` conventional spots are extinguished by the centering).  So the
        primitive reciprocal cell is inverted to a primitive real-space cell and the
        conventional cell recovered from that, via
        :func:`conventional_from_primitive`.
        """
        im = (
            self.band_image
            if image is None
            else (image if band is None else image[band[0] : band[1], band[2] : band[3]])
        )
        im = np.asarray(im, float)
        im = np.nan_to_num(im - np.nanmean(im))
        ny, nx = im.shape
        win = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
        F = np.abs(np.fft.fftshift(np.fft.fft2(im * win)))
        cy, cx = ny // 2, nx // 2
        yy = (np.arange(ny) - cy)[:, None]
        xx = (np.arange(nx) - cx)[None, :]
        r = np.hypot(xx, yy)
        Fm = np.where(r > exclude_radius, F, 0.0)
        loc = Fm == maximum_filter(Fm, size=5)
        cand = np.argwhere(loc & (Fm > rel_threshold * Fm.max()))
        if len(cand) < 2:
            return dict(ok=False, reason="fewer than two FFT peaks above threshold")
        vals = Fm[cand[:, 0], cand[:, 1]]
        strongest = cand[np.argsort(vals)[::-1][:n_peaks]]
        pk = np.stack([strongest[:, 1], strongest[:, 0]], axis=1).astype(float)
        # Sub-pixel refinement matters more here than anywhere else in the module.  A spot
        # at radius ~50 px pinned to the integer grid carries a ~0.5/50 = 1% direction
        # error, which is ~0.6 deg -- larger than the distortions being measured, so an
        # unrefined FFT check would report a "residual shear" that is pure quantization.
        # Refine on |F|^2, not |F| and definitely not log|F|: the centroid of a windowed
        # spot sitting on the DC skirt is pulled inward by the skirt, and squaring
        # suppresses it.  Measured on a synthetic image with a known answer, |F|^2 at
        # radius 4 leaves ~0.09 deg of angle error against ~0.6 deg for log|F|.
        pk = refine_peaks(F**2, pk, radius=4.0, mode="com", iterations=2)
        pk = pk - np.array([cx, cy])
        pk = pk[pk[:, 1] >= -1e-9]  # Friedel: keep a half plane
        pk = pk[np.hypot(*pk.T) > exclude_radius]
        pk = pk[np.argsort(np.hypot(*pk.T))]
        if len(pk) < 2:
            return dict(ok=False, reason="fewer than two independent FFT peaks")

        # Choose the pair that *generates* the observed spot pattern, rather than simply
        # taking the two shortest peaks.  Necessary because a single noise maximum that
        # happens to be shorter than the real spots -- and at 10-20 counts/pixel there
        # usually is one -- would otherwise be adopted as g1, and the resulting cell is
        # not slightly wrong but entirely wrong.  Scoring by "how many of the other
        # peaks are integer combinations of this pair" makes a lone spurious peak
        # unable to win, because nothing else in the pattern is consistent with it.
        g1, g2, best_score = None, None, -1.0
        n_cand = min(len(pk), 8)
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                p, q = pk[i], pk[j]
                B = np.stack([p, q])
                if abs(np.linalg.det(B)) < 0.3 * np.hypot(*p) * np.hypot(*q):
                    continue
                hk = np.linalg.solve(B.T, pk.T).T
                resid = np.abs(hk - np.rint(hk)).max(axis=1)
                explained = float(np.mean(resid < 0.15))
                # prefer the shortest generating pair; the length term only breaks ties
                score = explained - 1e-3 * (np.hypot(*p) + np.hypot(*q)) / max(
                    np.hypot(*pk[-1]), 1.0
                )
                if score > best_score:
                    g1, g2, best_score = p, q, score
        if g1 is None:
            return dict(ok=False, reason="FFT peaks are collinear")

        # reciprocal (in cycles per image) -> primitive real-space cell in pixels
        G = np.stack([g1 / np.array([nx, ny]), g2 / np.array([nx, ny])])
        try:
            Breal = np.linalg.inv(G).T  # rows = primitive real vectors
            a_vec, c_vec, ija, ijc = conventional_from_primitive(
                Breal[0],
                Breal[1],
                self.geometry.axial_ratio,
                tolerance,
                context="FFT reciprocal cell",
                want_angle=self.geometry.axial_angle_deg,
            )
        except (np.linalg.LinAlgError, RuntimeError) as exc:
            return dict(ok=False, reason=str(exc), g1=g1.tolist(), g2=g2.tolist())

        la, lc = np.hypot(*a_vec), np.hypot(*c_vec)
        # No abs() on the dot product: folding an obtuse angle back below 90 deg would
        # throw away the sign of the shear, which is the one thing this check is for.
        ang = float(np.degrees(np.arccos(np.clip(np.dot(a_vec, c_vec) / (la * lc), -1.0, 1.0))))
        ratio = float(lc / la)
        return dict(
            ok=True,
            g1=g1.tolist(),
            g2=g2.tolist(),
            a_vec_px=a_vec.tolist(),
            c_vec_px=c_vec.tolist(),
            combination=[list(ija), list(ijc)],
            angle_deg=ang,
            ratio=ratio,
            ideal_angle_deg=float(self.geometry.axial_angle_deg),
            ideal_ratio=float(self.geometry.axial_ratio),
            angle_error_deg=ang - float(self.geometry.axial_angle_deg),
            ratio_error=float(ratio - self.geometry.axial_ratio),
        )

    def held_out_direction_test(self, mn=(-2, 1), fraction: float = 0.5, seed: int = 0):
        """
        Cross-validation on a non-orthogonal direction with an exactly known angle.

        Fit the lattice on a random half of the sites, correct the *other* half with the
        resulting ``M``, and measure the ``(m, n)`` direction's angle in the held-out
        coordinates.  For zincblende <110>, ``(-2, 1)`` is a ``<111>`` direction at
        exactly 54.7356 deg from ``[001]``.

        Note what this does and does not test.  It is a *precision* check -- it asks
        whether the noise on the site positions is small enough to place a known angle
        correctly.  It is not a model check: fitting an affine to two exactly-determined
        directions reproduces all the others by construction, so any version of this test
        that reuses the fitting sites is vacuous.
        """
        if self.lattice_fit is None:
            raise RuntimeError("call fit() first")
        rng = np.random.default_rng(seed)
        basis = self.geometry.basis(self.site_mode)
        used = np.nonzero(self.lattice_fit["used"])[0]
        rng.shuffle(used)
        cut = int(fraction * len(used))
        tr, te = used[:cut], used[cut:]
        if min(len(tr), len(te)) < 20:
            raise RuntimeError("not enough sites to split for cross-validation")

        f = fit_lattice(
            self.sites[tr], self.lattice_fit["mn"][tr], self.lattice_fit["basis_index"][tr], basis
        )
        d = decompose_affine(
            affine_from_cell(f["a_vec"], f["c_vec"], self.geometry, self.pixel_size)
        )
        Minv = np.linalg.inv(drift_matrix(d["s"], d["k"]))

        # measure the direction in the held-out set: fit its own cell, correct, take angle
        g = fit_lattice(
            self.sites[te], self.lattice_fit["mn"][te], self.lattice_fit["basis_index"][te], basis
        )
        rows = []
        directions = (
            [tuple(mn)] if mn is not None else [(m, n) for m, n, _ in self.geometry.directions]
        )
        labels = {(m, n): lab for m, n, lab in self.geometry.directions}
        for m, n in directions:
            v = m * (Minv @ g["a_vec"]) + n * (Minv @ g["c_vec"])
            ideal = self.geometry.ideal_angle(m, n)
            rows.append(
                dict(
                    mn=[m, n],
                    label=labels.get((m, n), f"({m},{n})"),
                    ideal_deg=ideal,
                    measured_deg=self._angle_of(v),
                    error_deg=self._angle_of(v) - ideal,
                )
            )
        out = dict(
            directions=rows,
            n_train=int(len(tr)),
            n_test=int(len(te)),
            s_train=d["s"],
            k_train=d["k"],
            fraction=fraction,
            seed=seed,
        )
        if len(rows) == 1:
            out.update(rows[0])
        self.report["held_out_direction"] = out
        return out

    def held_out_all_directions(self, n_repeats: int = 20, fraction: float = 0.5, seed: int = 0):
        """
        :meth:`held_out_direction_test` over every reference direction and several random
        splits, reporting the mean and spread of each direction's angle error.

        Repeating the split matters: a single 50/50 draw gives one number with no error
        bar, and the spread across draws is the only honest statement of how well this
        image places a known angle.
        """
        rows = {}
        for i in range(n_repeats):
            out = self.held_out_direction_test(mn=None, fraction=fraction, seed=seed + i)
            for r in out["directions"]:
                rows.setdefault(
                    tuple(r["mn"]), dict(label=r["label"], ideal_deg=r["ideal_deg"], errors=[])
                )["errors"].append(r["error_deg"])
        summary = []
        for mn, r in rows.items():
            e = np.array(r["errors"])
            summary.append(
                dict(
                    mn=list(mn),
                    label=r["label"],
                    ideal_deg=r["ideal_deg"],
                    mean_error_deg=float(e.mean()),
                    std_error_deg=float(e.std(ddof=1)) if len(e) > 1 else np.nan,
                    max_abs_error_deg=float(np.abs(e).max()),
                    n_repeats=int(len(e)),
                )
            )
        out = dict(directions=summary, n_repeats=n_repeats, fraction=fraction)
        self.report["held_out_directions"] = out
        return out

    # ------------------------------------------------------------- reporting
    def to_json(self, path: str | None = None) -> str:
        """Serialise the whole report, including every warning raised, in order."""
        rep = dict(self.report)
        rep["warnings"] = list(self.warnings_raised)
        rep["degeneracies"] = [
            "k is a pure y-scale change and is indistinguishable from scan-coil gain "
            "anisotropy; do not quote vy as a calibrated drift velocity without an "
            "independent gain calibration.",
            "The pure-drift model is not falsifiable from a single lattice: every "
            "positive-determinant 2x2 matrix decomposes uniquely into M(s,k) sigma "
            "R(theta), and image formation is always orientation preserving. The "
            "meaningful checks are the fit residual, the nonlinearity split test, and "
            "whether theta matches the scan rotation you set.",
        ]
        txt = json.dumps(_jsonable(rep), indent=2)
        if path:
            with open(path, "w") as fh:
                fh.write(txt)
        return txt

    def report_text(self) -> str:
        """Human-readable summary -- the thing to paste into a lab notebook."""
        if self.solution is None:
            return "no fit yet"
        s, u = self.solution, self.solution["uncertainty"]
        L = [
            f"=== {self.name} : {self.geometry.name} {self.geometry.zone}, "
            f"a = {self.geometry.a:.4f} A ===",
            f"image {self.image.shape[0]} x {self.image.shape[1]} px, "
            f"{self.flyback_rows} flyback rows dropped",
            f"reference band rows {self.band[0]}-{self.band[1]} "
            f"({self.report['band']['height_periods']:.1f} lattice periods tall)",
            f"sites: {s['n_sites_used']} used, {s['n_sites_rejected']} rejected, "
            f"RMS residual {s['rms_residual_px']:.3f} px"
            + (f" = {s['rms_residual_pm']:.1f} pm" if s["rms_residual_pm"] else ""),
            "",
            "distortion  M = [[1, s], [0, k]]",
            f"  s      = {s['s']:+.5f} +/- {u['s']:.5f}   "
            f"(shear of the slow axis, {np.degrees(np.arctan(s['s'])):+.3f} deg)",
            f"  k      = {s['k']:.5f} +/- {u['k']:.5f}   "
            f"(slow-axis scale; DEGENERATE with scan gain anisotropy)",
            "nuisance parameters, projected out (not distortions)",
            f"  theta  = {s['theta_deg']:+.3f} +/- {u['theta_deg']:.3f} deg",
            f"  sigma  = {s['sigma']:.5f} +/- {u['sigma']:.5f}",
        ]
        if s["pixel_size_stated"]:
            L.append(
                f"  pixel size: stated {s['pixel_size_stated']:.4f} A, "
                f"crystal implies {s['pixel_size_implied']:.4f} A"
            )
        if "drift_vx_pm_per_s" in self.report.get("fit", {}):
            f_ = self.report["fit"]
            L += [
                "",
                f"implied drift: vx = {f_['drift_vx_pm_per_s']:+.1f} pm/s, "
                f"vy = {f_['drift_vy_pm_per_s']:+.1f} pm/s (vy uncalibrated)",
            ]
        L += [
            "",
            "per-direction angles (deg).  'distortion' = how far the raw image put each",
            "direction; 'residual' after correction equals -theta for every direction by",
            "construction and tests nothing -- see held-out below.",
        ]
        for a in s["angles"]:
            L.append(
                f"  {a['label']:>12s} ({a['mn'][0]:+d},{a['mn'][1]:+d})  "
                f"ideal {a['ideal_deg']:7.3f}  measured {a['measured_deg']:7.3f}  "
                f"distortion {a['distortion_deg']:+7.3f}  "
                f"residual {a['residual_deg']:+.2e}"
            )
        hod = self.report.get("held_out_directions")
        if hod:
            L += ["", "held-out direction test (fit on half the sites, measure the other half)"]
            for d in hod["directions"]:
                L.append(
                    f"  {d['label']:>12s} ({d['mn'][0]:+d},{d['mn'][1]:+d})  "
                    f"ideal {d['ideal_deg']:7.3f}  "
                    f"error {d['mean_error_deg']:+.4f} +/- {d['std_error_deg']:.4f}  "
                    f"(worst {d['max_abs_error_deg']:.4f})"
                )
        if self.nonlinear and "passed" in self.nonlinear:
            n = self.nonlinear
            L += [
                "",
                f"nonlinearity split test: {'PASS' if n['passed'] else 'FAIL'} "
                f"(s spread {n['s_spread']:.2e}, allowed {n['s_allowed']:.2e})",
            ]
            for p in n["splits"]:
                L.append(
                    f"  rows {p['y0']:7.1f}-{p['y1']:7.1f}  n={p['n']:5d}  "
                    f"s = {p['s']:+.5f} +/- {p.get('s_err', float('nan')):.5f}"
                )
        if self.warnings_raised:
            L += ["", "warnings"]
            L += [f"  ! {w}" for w in self.warnings_raised]
        return "\n".join(L)

    # ------------------------------------------------------------- figures
    def plot(self, corrected_image=None, figsize=(8.0, 8.0)):
        """
        A 2x2: the raw image and its FFT above, the corrected image and its FFT below.

        Both FFTs are of the **same rows** -- the reference band, raw against corrected -- which is
        the only way the pair says anything: the spots should go from a skewed net to an orthogonal
        one with the reference axial ratio.  Comparing the raw band against the whole corrected
        frame, which this used to do, puts a single crystal beside two lattices and an interface,
        and the panels then differ for reasons that have nothing to do with the correction.

        A fifth panel spanning the bottom appears once :meth:`row_wise_correction` has run, showing
        ``d(t)``; without it there is nothing to put there.

        Returns ``(fig, axes)``.  Uses plain matplotlib so it works standalone.
        """
        import matplotlib.pyplot as plt

        n = self.nonlinear or {}
        fifth = bool(n.get("row_trajectory") or n.get("splits"))
        h = figsize[1] * (1.35 if fifth else 1.0)
        fig = plt.figure(figsize=(figsize[0], h), constrained_layout=True)
        gs = fig.add_gridspec(3 if fifth else 2, 2, height_ratios=(1, 1, 0.55) if fifth else None)
        ax = [fig.add_subplot(gs[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
        if fifth:
            ax.append(fig.add_subplot(gs[2, :]))
        if self.solution:
            rms = f", fit RMS {self.lattice_fit['rms']:.3f} px" if self.lattice_fit else ""
            fig.suptitle(
                f"s = {self.solution['s']:+.5f}   k = {self.solution['k']:.5f}{rms}", fontsize=9
            )

        y0, y1, x0, x1 = self.band
        lo, hi = np.nanpercentile(self.image, [0.5, 99.5])
        ax[0].imshow(self.image, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        ax[0].axhline(y0, color="tab:orange", lw=0.8)
        ax[0].axhline(y1, color="tab:orange", lw=0.8)
        if self.lattice_fit is not None:
            u = self.lattice_fit["used"]
            # A few thousand accepted sites at full opacity paint the reference band solid
            # and hide the very image the panel exists to show, so show a sample of them.
            # Rejections are the interesting ones and are always drawn in full.
            used_xy = self.sites[u]
            step = max(1, len(used_xy) // 600)
            ax[0].plot(
                *used_xy[::step].T,
                ".",
                ms=0.8,
                alpha=0.5,
                color="tab:cyan",
                label=f"used ({u.sum()}, showing 1 in {step})"
                if step > 1
                else f"used ({u.sum()})",
            )
            if (~u).any():
                ax[0].plot(
                    *self.sites[~u].T,
                    "x",
                    ms=2.5,
                    mew=0.6,
                    color="tab:red",
                    label=f"rejected ({(~u).sum()})",
                )
            ax[0].legend(fontsize=5, loc="lower center", ncol=2, framealpha=0.6)
        ax[0].set_title("raw image + reference band", fontsize=8)

        # The corrected image, and the band mapped into it: k rescales y, so the band's rows land at
        # [y0/k, y1/k].  Drawn on the same axes scale as the raw panel above so the shear is visible
        # as a change in shape rather than only as a change in the FFT.
        after_band = None
        if corrected_image is not None:
            k = float(self.solution["k"]) if self.solution else 1.0
            ci = np.asarray(corrected_image, float)
            r0, r1 = (int(round(v / k)) for v in (y0, y1))
            r0, r1 = max(0, r0), min(ci.shape[0], r1)
            if r1 - r0 > 8:
                after_band = ci[r0:r1, x0:x1]
            clo, chi = np.nanpercentile(ci, [0.5, 99.5])
            ax[2].imshow(ci, cmap="gray", origin="lower", vmin=clo, vmax=chi)
            ax[2].axhline(r0, color="tab:orange", lw=0.8)
            ax[2].axhline(r1, color="tab:orange", lw=0.8)
            ax[2].set_title("corrected image", fontsize=8)
        else:
            ax[2].axis("off")

        for a, im_, t in (
            (ax[1], self.band_image, "FFT: reference band, raw"),
            (ax[3], after_band, "FFT: same band, corrected"),
        ):
            if im_ is None:
                a.axis("off")
                continue
            d = np.asarray(im_, float)
            d = np.nan_to_num(d - np.nanmean(d))
            w = np.hanning(d.shape[0])[:, None] * np.hanning(d.shape[1])[None, :]
            F = np.abs(np.fft.fftshift(np.fft.fft2(d * w)))
            a.imshow(np.log1p(F / F.max() * 1e3), cmap="inferno", origin="lower")
            c = np.array(F.shape) // 2
            h = min(F.shape) // 6
            a.set_xlim(c[1] - h, c[1] + h)
            a.set_ylim(c[0] - h, c[0] + h)
            a.set_title(t, fontsize=8)
            a.set_xticks([])
            a.set_yticks([])

        traj = n.get("row_trajectory")
        if not fifth:
            for a in (ax[0], ax[2]):
                a.set_xticks([])
                a.set_yticks([])
            return fig, ax
        if traj:
            t = np.array(traj["t"])
            ax[4].errorbar(
                t, traj["dx"], yerr=traj["sem"], fmt=".", ms=2.5, lw=0.5, label="d(t) measured"
            )
            tt = np.linspace(0, self.image.shape[0], 200)
            ax[4].plot(
                tt,
                np.polyval(traj["poly_x"], tt),
                "-",
                lw=1.0,
                label=f"order {traj['poly_order']} fit",
            )
            ax[4].axvspan(
                t.max(), self.image.shape[0], color="tab:red", alpha=0.12, label="extrapolated"
            )
            ax[4].set_xlabel("row (= time)", fontsize=7)
            ax[4].set_ylabel("dx (px)", fontsize=7)
            ax[4].legend(fontsize=6)
            ax[4].set_title("row-wise drift trajectory", fontsize=8)
        else:
            sv = [p["s"] for p in n["splits"]]
            yc = [0.5 * (p["y0"] + p["y1"]) for p in n["splits"]]
            se = [p.get("s_err", np.nan) for p in n["splits"]]
            ax[4].errorbar(yc, sv, yerr=se, fmt="o-", ms=3, lw=0.8, capsize=2)
            ax[4].axhline(self.solution["s"], ls="--", lw=0.7, color="k", label="global s")
            ax[4].set_xlabel("band centre row", fontsize=7)
            ax[4].set_ylabel("s", fontsize=7)
            ax[4].legend(fontsize=6)
            ax[4].set_title(f"nonlinearity: {'PASS' if n['passed'] else 'FAIL'}", fontsize=8)
        for a in (ax[0], ax[2]):
            a.set_xticks([])
            a.set_yticks([])
        return fig, ax


# ===========================================================================
# helpers
# ===========================================================================


def _as_array(image, pixel_size):
    """Accept a Dataset2d, a hyperspy signal, or a bare array; return (array, pixel_size)."""
    if Dataset2d is not None and isinstance(image, Dataset2d):
        px = pixel_size
        if px is None:
            samp = np.atleast_1d(np.asarray(image.sampling, float))
            px = float(samp[-1])
        return np.asarray(image.array, float), px
    if hasattr(image, "data") and hasattr(image, "axes_manager"):  # hyperspy
        px = pixel_size
        if px is None:
            try:
                px = float(image.axes_manager[-1].scale)
            except Exception:
                px = None
        return np.asarray(image.data, float), px
    return np.asarray(image, float), pixel_size


def _jsonable(obj):
    """Recursively convert numpy types so json.dumps does not choke."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)
