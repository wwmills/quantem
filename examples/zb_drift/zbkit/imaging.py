"""
Uncorrected-STEM image formation with drift applied *in scan coordinates*.

Why not just warp a finished image
----------------------------------
The obvious way to make a test case is to render a clean image and then
``map_coordinates`` it with a known affine.  Don't.  That does two wrong things at
once: it interpolates (so the ground truth is contaminated by spline error exactly at
the atom peaks you are about to fit), and it warps the *noise and the probe* along
with the object, which is backwards -- a real scan lays a fixed-shape probe down on a
moving sample.

So here the probe never moves and never shears.  For every scan pixel ``(ix, iy)`` we
work out where on the sample the probe actually landed,

.. math::  p_x = \\Delta\\,i_x + v_x \\tau\\, i_y, \\qquad p_y = (\\Delta + v_y \\tau)\\, i_y

and evaluate the incoherent image ``I = P (x) O`` at that point.  Because ``P (x) O``
is a sum of Gaussians (see :class:`GaussianMixtureKernel`) it can be evaluated
*analytically* at arbitrary positions, so there is no interpolation anywhere in the
forward model and the recovered ``(s, k)`` can be compared against the input to
machine precision.

Inverting the map above gives the distortion the correction code sees,

.. math::  M = \\begin{pmatrix} 1 & s \\\\ 0 & k \\end{pmatrix}, \\quad
           s = \\frac{-v_x \\tau}{\\Delta + v_y \\tau}, \\quad k = \\frac{\\Delta}{\\Delta + v_y \\tau}

with the two structural facts the whole single-image method rests on: the fast axis is
undistorted (``M[1, 0] = 0``) and there is no ``x`` scale error (``M[0, 0] = 1``).

Conventions
-----------
Images are ``[iy, ix]``: ``iy`` is the row = slow scan axis = ``y`` = growth direction,
``ix`` is the column = fast scan axis = ``x``.  ``k`` is spatial frequency in 1/A.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

_HC = 12398.4198        # eV A
_MC2 = 510998.95        # eV


def wavelength(energy):
    """Relativistic electron wavelength in Angstrom for ``energy`` in eV."""
    return _HC / np.sqrt(energy * (2 * _MC2 + energy))


@dataclass
class Probe:
    """
    Round-aperture STEM probe.  Defaults describe an **uncorrected** 200 kV
    instrument: ``Cs = 1.2 mm`` and the aperture at its Scherzer optimum, which lands
    the probe at roughly 1.6 A -- coarser than the 1.41 A GaAs dumbbell split, so the
    dumbbells come out marginally unresolved.  That is the regime this whole project
    is about, and it is why the correction code defaults to a dumbbell *centroid*
    site rather than fitting two overlapping Gaussians.
    """

    energy: float = 200e3            # eV
    Cs: float = 1.2e7                # A  (1 mm = 1e7 A); positive = uncorrected
    defocus: float = None            # A, positive = underfocus; None -> Scherzer
    semiangle: float = None          # rad; None -> Scherzer optimum aperture
    C5: float = 0.0                  # A
    astig_a: float = 0.0             # A, twofold astigmatism magnitude
    astig_phi: float = 0.0           # rad
    focal_spread: float = 60.0       # A, chromatic defocus spread
    source_size: float = 0.5         # A, HWHM-ish Gaussian source image
    aperture_smooth: float = 0.02    # soft aperture edge, fraction of k_max

    @property
    def wavelength(self):
        return wavelength(self.energy)

    @property
    def scherzer_semiangle(self):
        """``(4 lambda / Cs)^(1/4)`` -- the aperture that balances Cs against diffraction."""
        return (4.0 * self.wavelength / abs(self.Cs)) ** 0.25

    @property
    def scherzer_defocus(self):
        """``0.87 sqrt(Cs lambda)`` underfocus, the probe-forming optimum."""
        return 0.87 * np.sqrt(abs(self.Cs) * self.wavelength)

    @property
    def alpha(self):
        return self.scherzer_semiangle if self.semiangle is None else self.semiangle

    @property
    def df(self):
        return self.scherzer_defocus if self.defocus is None else self.defocus

    @property
    def scherzer_probe_size(self):
        """``0.43 (Cs lambda^3)^(1/4)``, the textbook uncorrected probe diameter."""
        return 0.43 * (abs(self.Cs) * self.wavelength ** 3) ** 0.25

    def chi(self, kx, ky, defocus=None):
        """Aberration phase, ``k`` in 1/A."""
        lam, k2 = self.wavelength, kx ** 2 + ky ** 2
        df = self.df if defocus is None else defocus
        c = np.pi * lam * k2 * (0.5 * self.Cs * lam ** 2 * k2 - df)
        if self.C5:
            c += np.pi / 3.0 * self.C5 * lam ** 5 * k2 ** 3
        if self.astig_a:
            c += np.pi * lam * k2 * self.astig_a * np.cos(
                2 * (np.arctan2(ky, kx) - self.astig_phi))
        return c

    def intensity(self, shape, pixel_size, n_focal=5):
        """
        Probe intensity on an ``[iy, ix]`` grid, centred, unit sum.

        Focal spread is done the honest way for an intensity -- an incoherent
        weighted average of ``|psi|^2`` over defocus by Gauss--Hermite quadrature --
        rather than as an envelope on the amplitude, which would wrongly preserve
        coherence.  Finite source size is a Gaussian blur of the intensity.
        """
        ky, kx = _kgrid(shape, pixel_size)
        k = np.hypot(kx, ky)
        kmax = self.alpha / self.wavelength
        A = 0.5 * (1.0 - np.tanh((k - kmax) / max(self.aperture_smooth * kmax, 1e-9)))

        if self.focal_spread <= 0:
            nodes = [(1.0, 0.0)]
        else:
            x, w = np.polynomial.hermite_e.hermegauss(n_focal)
            nodes = list(zip(w / w.sum(), x * self.focal_spread))

        intensity = np.zeros(shape, float)
        for weight, ddf in nodes:
            psi = np.fft.ifft2(A * np.exp(-1j * self.chi(kx, ky, self.df + ddf)))
            intensity += weight * np.abs(psi) ** 2
        if self.source_size > 0:
            g = np.exp(-2.0 * np.pi ** 2 * self.source_size ** 2 * (kx ** 2 + ky ** 2))
            intensity = np.real(np.fft.ifft2(np.fft.fft2(intensity) * g))
        intensity = np.fft.fftshift(np.clip(intensity, 0.0, None))
        return intensity / intensity.sum()

    def radial_profile(self, r, shape=(512, 512), pixel_size=0.05):
        """Azimuthally averaged probe intensity, interpolated onto radii ``r`` (A)."""
        img = self.intensity(shape, pixel_size)
        ny, nx = shape
        yy = (np.arange(ny) - ny // 2)[:, None] * pixel_size
        xx = (np.arange(nx) - nx // 2)[None, :] * pixel_size
        rr = np.hypot(xx, yy).ravel()
        order = np.argsort(rr)
        rr, vv = rr[order], img.ravel()[order] / pixel_size ** 2   # -> per A^2
        # bin to a fine radial grid, then interpolate
        edges = np.linspace(0.0, rr.max(), 4 * max(shape) )
        idx = np.clip(np.searchsorted(edges, rr) - 1, 0, len(edges) - 2)
        n = np.bincount(idx, minlength=len(edges) - 1)
        s = np.bincount(idx, weights=vv, minlength=len(edges) - 1)
        good = n > 0
        centres = 0.5 * (edges[:-1] + edges[1:])
        return np.interp(r, centres[good], (s[good] / n[good]))

    def measured_fwhm(self, pixel_size=0.02, extent=12.0):
        """FWHM of the radial intensity profile, in Angstrom."""
        r = np.arange(0.0, extent, pixel_size)
        p = self.radial_profile(r, shape=(1024, 1024), pixel_size=0.04)
        half = 0.5 * p[0]
        below = np.nonzero(p < half)[0]
        if len(below) == 0:
            return np.nan
        i = below[0]
        r0, r1, p0, p1 = r[i - 1], r[i], p[i - 1], p[i]
        return 2.0 * (r0 + (p0 - half) * (r1 - r0) / (p0 - p1))


def _kgrid(shape, pixel_size):
    """fft-ordered ``(ky, kx)`` in 1/A for an ``[iy, ix]`` array."""
    sy, sx = (pixel_size, pixel_size) if np.isscalar(pixel_size) else pixel_size
    ky = np.fft.fftfreq(shape[0], sy)[:, None]
    kx = np.fft.fftfreq(shape[1], sx)[None, :]
    return ky, kx


# ---------------------------------------------------------------------------
# the object kernel: probe (x) column, as a sum of Gaussians
# ---------------------------------------------------------------------------


@dataclass
class GaussianMixtureKernel:
    """
    ``P (x) G_column`` approximated as ``sum_n c_n Normal(0, sigma_n^2)`` in 2-D.

    The point of this representation is that the image can then be evaluated
    *analytically* at the arbitrary, non-grid positions the drifting scan visits.
    The alternative -- render on a grid and interpolate -- would put spline error into
    the ground truth.

    ``amps`` are integrated weights (they sum to 1); ``sigmas`` are in Angstrom.
    ``rel_error`` is the relative L2 error of the fit to the true radial profile,
    area-weighted, and is worth printing: it bounds how much of any residual in the
    validation is the kernel's fault rather than the algorithm's.
    """

    amps: np.ndarray
    sigmas: np.ndarray
    rel_error: float = np.nan
    cutoff_sigma: float = 3.5
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_probe(cls, probe, column_sigma=0.45, n_sigma=24, sigma_max=None,
                   shape=(1024, 1024), pixel_size=0.04, cutoff_sigma=3.5,
                   amp_floor=1e-3):
        """
        Fit the mixture to the true probe profile, convolved with a Gaussian column of
        width ``column_sigma`` (thermal spread + finite column extent).

        Solved as non-negative least squares on a fixed log-spaced ``sigma`` grid.
        That makes it linear in the only free parameters, so there is no initial guess
        to get wrong and no chance of settling into a local minimum -- and NNLS
        naturally returns a sparse subset of the grid.
        """
        from scipy.optimize import nnls

        sigma_max = sigma_max or 8.0
        r = np.linspace(0.0, cutoff_sigma * sigma_max, 900)
        target = probe.radial_profile(r, shape=shape, pixel_size=pixel_size)
        # convolve the (radially symmetric) probe with the column Gaussian in 2-D:
        # easiest exactly on the grid, so do it in Fourier space on the profile's own
        # 2-D grid instead of approximating -- see _blur_radial below.
        target = _blur_radial(r, target, column_sigma)

        sig = np.geomspace(max(0.25 * column_sigma, 0.05), sigma_max, n_sigma)
        # area weighting: the fit should care about integrated intensity, not about
        # the single point at r = 0.
        w = np.sqrt(np.maximum(r, 0.5 * (r[1] - r[0])))
        A = (np.exp(-0.5 * (r[:, None] / sig[None, :]) ** 2)
             / (2 * np.pi * sig[None, :] ** 2)) * w[:, None]
        c, _ = nnls(A, target * w)

        keep = c > amp_floor * c.sum()
        c, sig = c[keep], sig[keep]
        model = (np.exp(-0.5 * (r[:, None] / sig[None, :]) ** 2)
                 / (2 * np.pi * sig[None, :] ** 2)) @ c
        rel = float(np.linalg.norm((model - target) * w) / np.linalg.norm(target * w))
        c = c / c.sum()
        return cls(c, sig, rel, cutoff_sigma,
                   meta=dict(column_sigma=column_sigma,
                             probe_fwhm=float(probe.measured_fwhm()),
                             energy=probe.energy, Cs=probe.Cs, alpha=probe.alpha,
                             defocus=probe.df))

    @classmethod
    def gaussian(cls, sigma, cutoff_sigma=4.0):
        """A single Gaussian -- the cheap kernel for parameter sweeps and unit tests."""
        return cls(np.array([1.0]), np.array([float(sigma)]), 0.0, cutoff_sigma)

    @property
    def cutoff(self):
        return self.cutoff_sigma * float(self.sigmas.max())

    def __call__(self, r2):
        """Evaluate at squared radii ``r2`` (A^2)."""
        out = np.zeros(np.shape(r2), float)
        for c, s in zip(self.amps, self.sigmas):
            out += c / (2 * np.pi * s ** 2) * np.exp(-0.5 * r2 / s ** 2)
        return out


def _blur_radial(r, profile, sigma):
    """
    Convolve a radially symmetric 2-D function (given as a 1-D profile) with a 2-D
    Gaussian of width ``sigma``, via the Hankel-transform identity: the 2-D Fourier
    transform of a radial function is its Hankel transform, and a Gaussian stays a
    Gaussian, so this is a multiply in Hankel space.  Done by direct quadrature,
    which at 900 samples is instant and avoids pulling in a Hankel library.
    """
    from scipy.special import j0

    if sigma <= 0:
        return profile
    q = np.linspace(0, 6.0 / max(sigma, 1e-6), 1200)
    # forward Hankel: F(q) = 2 pi int f(r) J0(2 pi q r) r dr
    F = 2 * np.pi * np.trapezoid(profile[None, :] * j0(2 * np.pi * q[:, None] * r[None, :])
                                 * r[None, :], r, axis=1)
    F = F * np.exp(-2 * np.pi ** 2 * sigma ** 2 * q ** 2)
    out = 2 * np.pi * np.trapezoid(F[None, :] * j0(2 * np.pi * q[None, :] * r[:, None])
                                   * q[None, :], q, axis=1)
    return np.clip(out, 0.0, None)


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------


@dataclass
class ScanDrift:
    """
    Where the probe lands, row by row.

    Parameterised two equivalent ways.  Either give the *distortion* directly
    (``s``, ``k``) -- convenient because that is what the correction reports -- or
    give a physical drift velocity (``vx``, ``vy`` in A/s) together with the line time
    ``tau``, and let :meth:`from_velocity` convert.  They are related by
    ``s = -vx tau / (Delta + vy tau)`` and ``k = Delta / (Delta + vy tau)``.

    Everything else here is a *row offset*: a per-row displacement of the sample that
    is not part of the linear model.

    ``nonlinear_amplitude``
        RMS of a smooth (Gaussian-correlated) random walk added to the sample
        position, in Angstrom.  **Linearly detrended against row index**, so that
        turning it on does not secretly change the true ``(s, k)`` and make the
        validation unfalsifiable.  This is what Step 5's upper/lower-half test is
        supposed to catch.
    ``jitter_amplitude``
        Uncorrelated per-row ``x`` displacement -- scan noise rather than drift.
        Also detrended.
    ``flyback_amplitude`` / ``flyback_rows``
        Exponentially decaying ``x`` offset over the first rows of the frame, the
        settling transient after each flyback that makes the top of a real frame
        untrustworthy.  Applied to the first rows so that cropping them is a
        meaningful thing for the pipeline to do.
    """

    s: float = 0.0
    k: float = 1.0
    nonlinear_amplitude: float = 0.0
    nonlinear_correlation: float = 0.25    # as a fraction of the number of rows
    jitter_amplitude: float = 0.0
    flyback_amplitude: float = 0.0         # A
    flyback_rows: float = 6.0              # decay constant, rows
    seed: int = 0

    @classmethod
    def from_velocity(cls, vx, vy, line_time, pixel_size, **kw):
        """Build from a physical drift velocity (A/s) and line time (s)."""
        denom = pixel_size + vy * line_time
        return cls(s=-vx * line_time / denom, k=pixel_size / denom, **kw)

    def velocity(self, pixel_size, line_time):
        """Recover ``(vx, vy)`` in A/s from ``(s, k)``.  See the degeneracy note."""
        vy = pixel_size * (1.0 / self.k - 1.0) / line_time
        vx = -self.s * (pixel_size + vy * line_time) / line_time
        return vx, vy

    @property
    def matrix(self):
        """``M``: maps a true sample vector (in pixel units) to its image vector."""
        return np.array([[1.0, self.s], [0.0, self.k]])

    @property
    def inverse_matrix(self):
        return np.linalg.inv(self.matrix)

    def row_offsets(self, n_rows, pixel_size):
        """
        ``(xoff, yoff)``: sample-frame offset of each row, in Angstrom, *excluding*
        the linear part (which lives in ``M``).
        """
        rng = np.random.default_rng(self.seed)
        j = np.arange(n_rows, dtype=float)
        xoff = np.zeros(n_rows)
        yoff = np.zeros(n_rows)

        if self.nonlinear_amplitude > 0:
            from scipy.ndimage import gaussian_filter1d
            corr = max(self.nonlinear_correlation * n_rows, 1.0)
            for out in (xoff, yoff):
                walk = np.cumsum(rng.standard_normal(n_rows))
                walk = gaussian_filter1d(walk, corr, mode="nearest")
                out += walk
            for out in (xoff, yoff):
                out -= np.polyval(np.polyfit(j, out, 1), j)     # detrend: purely nonlinear
                out *= self.nonlinear_amplitude / max(out.std(), 1e-12)

        if self.jitter_amplitude > 0:
            jit = rng.standard_normal(n_rows)
            jit -= np.polyval(np.polyfit(j, jit, 1), j)
            xoff = xoff + self.jitter_amplitude * jit / max(jit.std(), 1e-12)

        if self.flyback_amplitude > 0:
            fly = self.flyback_amplitude * np.exp(-j / max(self.flyback_rows, 1e-6))
            fly = fly - np.polyval(np.polyfit(j, fly, 1), j)
            xoff = xoff + fly

        return xoff, yoff

    def sample_positions(self, shape, pixel_size, origin=(0.0, 0.0)):
        """
        Sample-frame ``(x, y)`` the probe visits, as separable pieces.

        Returns ``(x_of_col, xoff_of_row, y_of_row)`` such that
        ``p_x(iy, ix) = x_of_col[ix] + xoff_of_row[iy]`` and ``p_y(iy, ix) = y_of_row[iy]``.
        Separability is what makes the row-wise renderer fast; it holds because drift
        within a single line is smaller than the drift between lines by ~1/n_cols and
        is neglected, which is the same approximation the correction's model makes.
        """
        ny, nx = shape
        Minv = self.inverse_matrix
        j = np.arange(ny, dtype=float)
        xoff_nl, yoff_nl = self.row_offsets(ny, pixel_size)
        x_of_col = np.arange(nx, dtype=float) * pixel_size + origin[0]
        xoff_of_row = Minv[0, 1] * j * pixel_size + xoff_nl
        y_of_row = Minv[1, 1] * j * pixel_size + yoff_nl + origin[1]
        return x_of_col, xoff_of_row, y_of_row


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------


def render(columns, shape, pixel_size, kernel, scan=None, origin=(0.0, 0.0),
           thickness=1.0, progress=False):
    """
    Incoherent HAADF image of ``columns``, sampled at the positions the drifting scan
    actually visits.

    Row-wise: for each row, take the columns within the kernel cutoff in ``y``, and
    accumulate their (analytic) Gaussian contributions into the row's pixels using a
    per-component window.  Cost scales as ``n_pixels`` times column density times the
    kernel area, and is dominated by the broadest mixture component.

    Returns the image in ``[iy, ix]``, in arbitrary but linear units (proportional to
    ``occ * Z**gamma * thickness``).
    """
    scan = ScanDrift() if scan is None else scan
    ny, nx = shape
    x_of_col, xoff_of_row, y_of_row = scan.sample_positions(shape, pixel_size, origin)

    tx, ty = columns.xy[:, 0], columns.xy[:, 1]
    w = columns.weights * float(thickness)
    order = np.argsort(ty)
    tx, ty, w = tx[order], ty[order], w[order]

    image = np.zeros(shape, float)
    rows = range(ny)
    if progress:
        from tqdm.auto import tqdm
        rows = tqdm(rows, desc="scanning", leave=False)

    for iy in rows:
        y0 = y_of_row[iy]
        xshift = xoff_of_row[iy]
        acc = np.zeros(nx)
        for c, sig in zip(kernel.amps, kernel.sigmas):
            cut = kernel.cutoff_sigma * sig
            lo, hi = np.searchsorted(ty, [y0 - cut, y0 + cut])
            if hi <= lo:
                continue
            dy2 = (ty[lo:hi] - y0) ** 2
            # pixel index nearest each column, and a shared window around it
            centre = (tx[lo:hi] - xshift - x_of_col[0]) / pixel_size
            half = int(np.ceil(cut / pixel_size))
            win = np.arange(-half, half + 1)
            idx = np.rint(centre)[:, None] + win[None, :]
            dx = (idx * pixel_size + x_of_col[0] + xshift) - tx[lo:hi][:, None]
            val = (w[lo:hi][:, None] * c / (2 * np.pi * sig ** 2)
                   * np.exp(-0.5 * (dx ** 2 + dy2[:, None]) / sig ** 2))
            idx = idx.astype(np.int64).ravel()
            val = val.ravel()
            inside = (idx >= 0) & (idx < nx)
            acc += np.bincount(idx[inside], weights=val[inside], minlength=nx)
        image[iy] = acc
    return image


def ground_truth_positions(columns, shape, pixel_size, scan, origin=(0.0, 0.0),
                           label=None):
    """
    Where each column *appears* in the distorted image, in pixels ``(ix, iy)``.

    Exact for the linear part of the scan model; the row offsets are inverted by one
    Newton step, which converges immediately because they vary slowly with row.
    Use this to score fitted site positions without re-deriving the map by hand.
    """
    scan = ScanDrift() if scan is None else scan
    cols = columns if label is None else columns.select(columns.label == label)
    t = (cols.xy - np.asarray(origin)) / pixel_size
    m = t @ scan.matrix.T                                  # linear part
    ny = shape[0]
    xoff, yoff = scan.row_offsets(ny, pixel_size)
    for _ in range(3):
        j = np.clip(np.rint(m[:, 1]).astype(int), 0, ny - 1)
        m[:, 0] = (t[:, 0] - xoff[j] / pixel_size) + scan.s * m[:, 1]
        m[:, 1] = scan.k * (t[:, 1] - yoff[j] / pixel_size)
    return m


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------


def apply_mtf(image, mtf_width=0.55):
    """Gaussian detector/amplifier MTF; ``mtf_width`` is the 1/e point as a fraction
    of Nyquist."""
    ky, kx = _kgrid(image.shape, 1.0)         # in cycles/pixel
    k2 = kx ** 2 + ky ** 2
    return np.real(np.fft.ifft2(np.fft.fft2(image) * np.exp(-k2 / (0.5 * mtf_width) ** 2)))


def add_poisson_noise(image, mean_counts=40.0, rng=None):
    """
    Shot noise at ``mean_counts`` electrons per pixel on average.  Returns an image
    with the same mean as the input, so downstream display limits do not shift.
    """
    rng = np.random.default_rng() if rng is None else rng
    scale = mean_counts / max(image.mean(), 1e-12)
    return rng.poisson(np.clip(image * scale, 0.0, None)).astype(float) / scale


def add_illumination_ripple(image, amplitude=0.0, n_periods=(1.5, 2.5), seed=0):
    """
    Slow multiplicative gain variation across the frame.

    Included because it breaks any peak finder that uses a global intensity
    threshold, and a real frame always has some.  Local-maximum detection is
    scale-free and should shrug it off -- which is the point of testing with it on.
    """
    if amplitude <= 0:
        return image
    rng = np.random.default_rng(seed)
    ny, nx = image.shape
    y = np.linspace(0, 2 * np.pi, ny)[:, None]
    x = np.linspace(0, 2 * np.pi, nx)[None, :]
    g = (np.sin(n_periods[0] * y + rng.uniform(0, 6.28))
         + np.sin(n_periods[1] * x + rng.uniform(0, 6.28)))
    return image * (1.0 + amplitude * g / 2.0)


def normalize(image, percentile=(0.1, 99.9)):
    """Scale to [0, 1] using robust limits."""
    lo, hi = np.nanpercentile(image, percentile)
    return np.clip((image - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


# ---------------------------------------------------------------------------
# one-call simulation
# ---------------------------------------------------------------------------


def simulate(columns, shape=(1024, 1024), pixel_size=0.18, probe=None, kernel=None,
             scan=None, mean_counts=40.0, mtf_width=0.55, ripple=0.0, seed=0,
             column_sigma=0.45, progress=False):
    """
    Full forward model: probe -> kernel -> drifting scan -> MTF -> shot noise.

    Returns ``(image, info)`` where ``info`` carries the kernel, the scan, and the
    ground-truth ``(s, k)`` and drift matrix, so the correction can be scored later
    without the caller having to keep track.
    """
    probe = Probe() if probe is None else probe
    kernel = (GaussianMixtureKernel.from_probe(probe, column_sigma=column_sigma)
              if kernel is None else kernel)
    scan = ScanDrift() if scan is None else scan
    rng = np.random.default_rng(seed)

    image = render(columns, shape, pixel_size, kernel, scan, progress=progress)
    if mtf_width:
        image = apply_mtf(image, mtf_width)
    image = add_illumination_ripple(image, ripple, seed=seed)
    if mean_counts:
        image = add_poisson_noise(image, mean_counts, rng=rng)

    info = dict(
        kernel=kernel, scan=scan, probe=probe, pixel_size=pixel_size, shape=shape,
        s_true=scan.s, k_true=scan.k, M_true=scan.matrix,
        mean_counts=mean_counts, mtf_width=mtf_width, ripple=ripple,
        probe_fwhm=kernel.meta.get("probe_fwhm", np.nan),
        kernel_rel_error=kernel.rel_error,
        ground_truth=columns.meta.get("ground_truth", {}),
    )
    return image, info
