"""
Projected structures: a zincblende <110> substrate with an epitaxial cubic film on top.

Coordinates are Angstrom, ``x`` in-plane (the fast scan axis, along [1-10]) and ``y``
the growth direction (the slow scan axis, along [001]).  Structures are 2-D projected
atomic columns; the imaging model in :mod:`zbkit.imaging` convolves them with a probe.

The substrate is the ruler
--------------------------
Zincblende viewed along <110> gives the familiar dumbbell projection, and its
projected cell is fixed by cubic symmetry alone:

* ``[001]`` repeat = ``a``
* ``[1-10]`` repeat = ``a/sqrt(2)``
* the angle between them is **exactly 90 deg** and the length ratio **exactly sqrt(2)**
* dumbbell splitting = ``a/4`` along ``[001]``

Those constants are what the single-image drift correction fits against, so this
module and :mod:`quantem.imaging.drift_single_image` must agree on them; the reference
geometry lives in the quantem module and is imported here for the assertion in
:func:`check_reference_consistency`.

The film is *pseudomorphic*: in-plane locked to the substrate, out-of-plane relaxed by
the tetragonal Poisson response.  That means the film's ``y`` period differs from the
substrate's by a known amount, which is the thing a bad drift correction would
corrupt -- ``k`` is a pure ``y`` rescaling and so masquerades exactly as an
out-of-plane strain.  Getting ``k`` right is the difference between measuring the
film's strain and measuring the microscope's drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------


@dataclass
class Columns:
    """Projected atomic columns over a rectangular field of view."""

    xy: np.ndarray                     # (N, 2) positions in Angstrom, (x, y)
    Z: np.ndarray                      # (N,) atomic number
    occ: np.ndarray                    # (N,) atoms per column per projected slice
    fov: tuple                         # (Lx, Ly) in Angstrom
    label: np.ndarray | None = None    # (N,) int region tag: 0 = substrate, 1 = film
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.Z)

    def __add__(self, other):
        """Stack two column sets, keeping the larger field of view."""
        fov = (max(self.fov[0], other.fov[0]), max(self.fov[1], other.fov[1]))
        meta = dict(self.meta)
        meta.update(other.meta)
        lab_a = np.zeros(len(self), int) if self.label is None else self.label
        lab_b = np.zeros(len(other), int) if other.label is None else other.label
        return Columns(
            np.concatenate([self.xy, other.xy]),
            np.concatenate([self.Z, other.Z]),
            np.concatenate([self.occ, other.occ]),
            fov,
            np.concatenate([lab_a, lab_b]),
            meta,
        )

    def select(self, mask):
        lab = None if self.label is None else self.label[mask]
        return Columns(self.xy[mask], self.Z[mask], self.occ[mask], self.fov, lab,
                       dict(self.meta))

    def crop(self, margin=0.0, ymin=None, ymax=None):
        """Drop columns outside the field of view (plus an optional ``y`` band)."""
        Lx, Ly = self.fov
        x, y = self.xy[:, 0], self.xy[:, 1]
        m = (x >= -margin) & (x < Lx + margin) & (y >= -margin) & (y < Ly + margin)
        if ymin is not None:
            m &= y >= ymin
        if ymax is not None:
            m &= y < ymax
        return self.select(m)

    def displaced(self, u_fn):
        """Copy with ``u_fn(x, y) -> (ux, uy)`` added to every position."""
        ux, uy = u_fn(self.xy[:, 0], self.xy[:, 1])
        out = self.select(slice(None))
        out.xy = self.xy + np.stack([ux, uy], axis=1)
        return out

    @property
    def weights(self):
        """HAADF column weight, ``occ * Z**gamma``; ``gamma`` from ``meta``."""
        gamma = self.meta.get("Z_exponent", 1.7)
        return self.occ * np.asarray(self.Z, float) ** gamma


# ---------------------------------------------------------------------------
# projected lattices
# ---------------------------------------------------------------------------


def commensurate(target, period):
    """Round ``target`` down-or-up to a whole number of ``period``, at least one."""
    return max(1, round(target / period)) * period


def _tile(fov, ax, ay, basis, margin, ymin=None, ymax=None, y0=0.0, x0=0.0, label=0,
          Z_exponent=1.7):
    """Tile a rectangular cell ``ax x ay`` with ``basis`` = [(fx, fy, Z, occ), ...]."""
    Lx, Ly = fov
    lo_y = (-margin if ymin is None else ymin - margin) - y0
    hi_y = (Ly + margin if ymax is None else ymax + margin) - y0
    nx = np.arange(int(np.floor((-margin - x0) / ax)) - 1,
                   int(np.ceil((Lx + margin - x0) / ax)) + 2)
    ny = np.arange(int(np.floor(lo_y / ay)) - 1, int(np.ceil(hi_y / ay)) + 2)
    NX, NY = np.meshgrid(nx, ny, indexing="ij")
    cells = np.stack([NX.ravel(), NY.ravel()], axis=1).astype(float)

    xy, Zs, occs = [], [], []
    for fx, fy, Z, occ in basis:
        pos = (cells + np.array([fx, fy])) * np.array([ax, ay]) + np.array([x0, y0])
        xy.append(pos)
        Zs.append(np.full(len(pos), Z))
        occs.append(np.full(len(pos), float(occ)))
    cols = Columns(np.concatenate(xy), np.concatenate(Zs), np.concatenate(occs),
                   (Lx, Ly), None, meta=dict(Z_exponent=Z_exponent))
    cols.label = np.full(len(cols), label, int)
    return cols.crop(margin=margin, ymin=ymin, ymax=ymax)


def zincblende_110(fov, a=5.653, Z_cation=31, Z_anion=33, margin=10.0,
                   ymin=None, ymax=None, y0=0.0, label=0):
    """
    Zincblende projected along <110>: the dumbbell substrate.  Default is GaAs
    (``a = 5.653 A``, Ga/As).

    The rectangular projected cell is ``a/sqrt(2)`` wide by ``a`` tall and holds four
    columns as two dumbbells split by ``a/4 = 1.413 A``.  On an uncorrected
    instrument that split sits right at the resolution limit, which is the entire
    point of the ``site_mode="centroid"`` default in the correction code -- fitting
    two overlapping Gaussians into an unresolved pair is a biased way to find a
    lattice.

    One projected sublattice is *centered rectangular*: the second dumbbell sits at
    ``(0.5 a/sqrt(2), 0.5 a)``.  The correction code fits that basis offset rather
    than assuming it, so a wrong choice shows up as a residual instead of a bias.
    """
    ax, ay = a / np.sqrt(2.0), a
    basis = [
        (0.00, 0.000, Z_cation, 1.0), (0.00, 0.250, Z_anion, 1.0),
        (0.50, 0.500, Z_cation, 1.0), (0.50, 0.750, Z_anion, 1.0),
    ]
    cols = _tile(fov, ax, ay, basis, margin, ymin, ymax, y0, label=label)
    cols.meta.update(substrate=dict(
        name="zincblende<110>", a=a, ax=ax, ay=ay, d_dumbbell=a / 4.0,
        Z=(Z_cation, Z_anion), slice_thickness=ax,
    ))
    return cols


def rocksalt_110(fov, a=5.653, a_perp=None, Z_cation=38, Z_anion=16, margin=10.0,
                 ymin=None, ymax=None, y0=0.0, label=1):
    """
    Rock-salt projected along <110>: the cubic film.

    In this projection the cell is ``a/sqrt(2)`` by ``a_perp`` with four columns on a
    checkerboard -- cations at ``(0, 0)`` and ``(1/2, 1/2)``, anions at ``(1/2, 0)``
    and ``(0, 1/2)``.  No dumbbells, so the film is immediately distinguishable from
    the substrate by eye, which is what you want in a figure whose message is
    "the correction was derived from the bottom of the image and applied to all of it".

    The default ``Z`` is Sr/S-like.  That is a deliberate choice of *scattering
    power*, not a claim about a real SrS/GaAs interface: at ``Z^1.7`` it puts the mean
    film column within ~15% of the mean GaAs column, so both halves of the image are
    legible on one grey scale.  A film of genuinely light elements (MgO, say) is
    perfectly simulable here but comes out ~5x dimmer than the substrate, which
    buries the film's atoms in shot noise and makes the figure about display limits
    instead of about drift.

    ``a_perp`` overrides the out-of-plane period for a tetragonally distorted
    pseudomorphic film; ``a`` then sets only the in-plane period.
    """
    ax = a / np.sqrt(2.0)
    ay = a if a_perp is None else a_perp
    basis = [
        (0.00, 0.00, Z_cation, 1.0), (0.50, 0.50, Z_cation, 1.0),
        (0.50, 0.00, Z_anion, 1.0), (0.00, 0.50, Z_anion, 1.0),
    ]
    cols = _tile(fov, ax, ay, basis, margin, ymin, ymax, y0, label=label)
    cols.meta.update(film=dict(
        name="rocksalt<110>", a=a, ax=ax, ay=ay, Z=(Z_cation, Z_anion),
        slice_thickness=ax,
    ))
    return cols


def cubic_film_100(fov, d=4.15, d_perp=None, Z=40, margin=10.0, ymin=None, ymax=None,
                   y0=0.0, x0=0.0, label=1, rumple=None):
    """
    Cubic film viewed down a <100> zone axis: a **square net** of projected period ``d``,
    with the growth direction along the film's ``[001]``.

    This is the film orientation to use when the film is rotated 45 deg in-plane relative
    to a <110>-viewed zincblende substrate -- the common epitaxial relationship in which
    the film's ``[100]`` lies along the substrate's ``[1-10]``, so the same electron beam
    that sees dumbbells in the substrate sees a <100> projection in the film.

    Why a plain square net.  Take rock salt down ``[100]``: the cation fcc sublattice
    projects to ``(0,0), (1/2,1/2), (0,1/2), (1/2,0)`` in ``(y, z)`` fractional
    coordinates, and the anion sublattice -- cations displaced by ``(1/2, 0, 0)``, i.e.
    *along the beam* -- projects onto exactly the same four points.  Every column is
    therefore mixed cation/anion, all columns are equivalent, and the projection is a
    single square net of period ``a/2``.  There is no chemical contrast to model and no
    checkerboard, which is what makes this projection so easy to tell apart from the
    substrate's dumbbells by eye.

    ``d`` is the *projected* period, which is the only thing the image knows about, so the
    parameter is the projection rather than a cubic constant plus a rule for halving it.
    The default 4.15 A is a synthetic choice: comfortably resolved by a 1.8 A probe,
    visibly distinct from the substrate's 4.00 A in-plane period, and a 3.8% misfit
    against it -- big enough to matter, small enough that a coherent interface is credible.
    ``d_perp`` overrides the out-of-plane period for a tetragonally distorted film.

    ``rumple`` makes successive layers differ, which a single tiled cell cannot express.  It takes
    a dict, any key optional:

    ``x`` (A)
        alternate layers shifted along the row by ``+x/2`` and ``-x/2`` -- a registry zig-zag, the
        thing that shows up in the disregistry panel as film rows alternating about a common line.
    ``y`` (fraction)
        alternate *gaps* scaled by ``1 +/- y`` -- a buckled stack, which alters the out-of-plane
        row pitch layer by layer while leaving the mean unchanged.
    ``d`` (fraction)
        alternate layers' in-plane period scaled by ``1 +/- d`` -- layers with genuinely different
        in-plane spacing, so ``rows/side`` changes the answer.
    ``decay`` (A)
        amplitudes fall off as ``exp(-height / decay)`` from the interface, since a real rumpling
        relaxes into the film rather than persisting to the surface.  Omit for uniform.

    A rumpled film is the case that breaks a measurement quietly, because a *single* film row still
    looks perfectly periodic; only the comparison between rows shows it.
    """
    dy = d if d_perp is None else d_perp
    if not rumple:
        cols = _tile(fov, d, dy, [(0.0, 0.0, Z, 1.0)], margin, ymin, ymax, y0, x0=x0,
                     label=label)
        cols.meta.update(film=dict(
            name="cubic<100>", d=d, ax=d, ay=dy, Z=(Z,), zone="[100]",
            slice_thickness=d,
        ))
        return cols

    # Layer by layer, because the modulation is per-layer by definition.  Each layer is one row of
    # a 1-D tiling, so the same _tile machinery still places the columns within the row.
    amp_x = float(rumple.get("x", 0.0))
    amp_y = float(rumple.get("y", 0.0))
    amp_d = float(rumple.get("d", 0.0))
    decay = rumple.get("decay", None)
    Ly = fov[1] if ymax is None else ymax
    layers, y = [], float(y0)
    k = 0
    while y <= Ly + margin:
        fade = 1.0 if decay is None else float(np.exp(-(y - y0) / float(decay)))
        sign = 1.0 if k % 2 == 0 else -1.0
        d_k = d * (1.0 + sign * amp_d * fade)
        layer = _tile((fov[0], fov[1]), d_k, dy, [(0.0, 0.0, Z, 1.0)], margin,
                      ymin=y - 1e-6, ymax=y + 1e-6, y0=y,
                      x0=x0 + sign * 0.5 * amp_x * fade, label=label)
        if len(layer):
            layers.append(layer)
        y += dy * (1.0 + sign * amp_y * fade)
        k += 1
    cols = layers[0]
    for extra in layers[1:]:
        cols = cols + extra
    cols.meta.update(film=dict(
        name="cubic<100> rumpled", d=d, ax=d, ay=dy, Z=(Z,), zone="[100]",
        slice_thickness=d, rumple=dict(x=amp_x, y=amp_y, d=amp_d, decay=decay),
    ))
    return cols


# ---------------------------------------------------------------------------
# epitaxy
# ---------------------------------------------------------------------------


def tetragonal_a_perp(a_relaxed, a_substrate, poisson=0.30):
    """
    Out-of-plane lattice parameter of a fully strained (pseudomorphic) cubic film.

    In-plane misfit strain ``eps_par = (a_sub - a_rel)/a_rel`` is accommodated
    elastically; biaxial strain with a free surface gives
    ``eps_perp = -2 nu/(1 - nu) eps_par``.
    """
    eps_par = (a_substrate - a_relaxed) / a_relaxed
    eps_perp = -2.0 * poisson / (1.0 - poisson) * eps_par
    return a_relaxed * (1.0 + eps_perp), eps_par, eps_perp


def relaxing_dislocation_spacing(d_relaxed, d_substrate):
    """
    Dislocation spacing that *fully* relaxes a given misfit: ``S = b / |eps|``.

    An array of edge dislocations with Burgers vector ``b`` spaced ``S`` apart relieves an
    in-plane strain of ``b / S``, so relaxing a misfit ``eps`` needs ``S = b / |eps|``.  Worth
    computing rather than guessing: at the 3.8% misfit used here that is ~105 Angstrom, and
    picking a "reasonable-looking" 38 Angstrom instead over-relaxes by nearly 3x, which turns
    a demonstration of a dislocated interface into a demonstration of a badly strained one.
    """
    eps = abs(d_relaxed - d_substrate) / d_substrate
    if eps < 1e-9:
        return np.inf
    return d_substrate / eps


def misfit_staircase(spacing, burgers, width=None, x_offset=0.0, sign=+1.0):
    """
    In-plane displacement field of a periodic array of misfit dislocations.

    Returns ``u(x) -> ux``: a smooth staircase that gains ``sign * burgers`` every ``spacing``
    Angstrom, each step spread over ``width`` (default ``0.2 * spacing``).  That is what a
    relaxed epitaxial film does -- it stays registered to the substrate across one domain,
    slips by a lattice vector at a dislocation, then re-registers.

    Worth having as a case distinct from uniform relaxation, because the two look nothing
    alike in the measurement this project exists to make: uniform relaxation tilts the
    ``x``-versus-index line to a different slope, while a dislocation array keeps the
    substrate slope and inserts *steps*.  A tool that can only see the first would call a
    dislocated interface coherent.

    ``sign`` matters and is easy to get backwards.  A film whose natural period exceeds the
    substrate's is *compressed* when coherent, so relaxing it must make its average spacing
    *larger* -- ``sign = +1``.  Getting this wrong produces a film that is more strained than
    the coherent one, in the opposite direction, which reads as a plausible measurement.

    ``width`` should be a decent fraction of ``spacing``.  Squeezing a full Burgers vector
    into a few Angstrom gives a local strain of tens of percent, close enough to collapse
    neighbouring columns into one another that the peak finder merges them.
    """
    width = 0.2 * spacing if width is None else width

    def u(x):
        x = np.asarray(x, float)
        n = int(np.ceil((np.nanmax(x) - np.nanmin(x)) / spacing)) + 3
        start = np.floor(np.nanmin(x) / spacing) - 1
        out = np.zeros_like(x)
        for j in range(n):
            xc = (start + j) * spacing + x_offset
            out += sign * burgers * 0.5 * (1.0 + np.tanh((x - xc) / max(width, 1e-6)))
        return out

    return u


def epitaxial_stack(
    fov=(120.0, 140.0),
    a_substrate=5.653,
    film_zone="100",
    d_film_relaxed=4.15,
    a_film_relaxed=5.870,
    poisson=0.30,
    film_fraction=0.45,
    relaxation=0.0,
    misfit_dislocation_spacing=None,
    misfit_step_width=None,
    interface_gap=None,
    Z_substrate=(31, 33),
    Z_film=(40,),
    interface_roughness=0.0,
    rumple=None,
    rng=None,
):
    """
    Zincblende <110> dumbbell substrate with a cubic film grown on top.

    ``film_zone``
        ``"100"`` (default) puts the film on a ``<100>`` zone axis -- a square net,
        growth along the film's ``[001]``, corresponding to a 45 deg in-plane rotation
        relative to the ``<110>``-viewed substrate.  ``"110"`` keeps the older rock-salt
        ``<110>`` checkerboard.

    How coherent the interface is, is a *parameter*, because it is the quantity the
    downstream measurement exists to determine.  Three regimes, and the point of the
    ``x``-versus-site-index plot is that they are distinguishable:

    ``relaxation=0``, no dislocations
        Coherent.  The film's in-plane period is locked to the substrate's, so film and
        substrate rows share one slope.
    ``relaxation=1``
        Uniformly relaxed to its own bulk period: film rows take a *different slope* and
        drift steadily away from the substrate's.
    ``misfit_dislocation_spacing=S``
        Relaxed by a dislocation array: film rows keep the substrate slope but *step* by one
        lattice vector every ``S`` Angstrom.  Pass ``"full"`` for the spacing that exactly
        relieves the misfit (:func:`relaxing_dislocation_spacing`), which is what a real film
        tends toward -- ~105 Angstrom at the default 3.8% misfit.

    Intermediate ``relaxation`` interpolates the in-plane period; the out-of-plane period
    always follows from the Poisson response to whatever in-plane strain is left, so the
    ground truth stays exact in every case.

    The field of view is snapped to a whole number of substrate cells in both directions so
    the substrate FFT has no wrap seam -- worth doing because the before/after FFT is one of
    the diagnostics, and an incommensurate FOV puts streaks through it that look like the
    very distortion being measured.  The default is ~12 nm, sized for a real
    atomic-resolution frame rather than for a figure.

    Returns ``Columns`` with ``meta['ground_truth']`` recording every parameter the
    correction and the coherency measurement get scored against.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    ax = a_substrate / np.sqrt(2.0)
    Lx = commensurate(fov[0], ax)
    Ly = commensurate(fov[1], a_substrate)
    y_interface = commensurate((1.0 - film_fraction) * Ly, a_substrate)

    if film_zone == "100":
        d_relaxed = d_film_relaxed
        # in-plane period: interpolate between locked-to-substrate and fully relaxed
        d_par = (1.0 - relaxation) * ax + relaxation * d_relaxed
    elif film_zone == "110":
        d_relaxed = a_film_relaxed / np.sqrt(2.0)
        d_par = (1.0 - relaxation) * ax + relaxation * d_relaxed
    else:
        raise ValueError(f"film_zone must be '100' or '110', got {film_zone!r}")

    eps_par = (d_par - d_relaxed) / d_relaxed
    eps_perp = -2.0 * poisson / (1.0 - poisson) * eps_par
    d_perp = d_relaxed * (1.0 + eps_perp)
    if interface_gap is None:
        interface_gap = 0.25 * (a_substrate + d_perp) / 2.0

    sub = zincblende_110((Lx, Ly), a=a_substrate, Z_cation=Z_substrate[0],
                         Z_anion=Z_substrate[1], ymax=y_interface, label=0)
    if film_zone == "100":
        film = cubic_film_100((Lx, Ly), d=d_par, d_perp=d_perp, Z=Z_film[0],
                              ymin=y_interface, y0=y_interface + interface_gap, label=1,
                              rumple=rumple)
    else:
        film = rocksalt_110((Lx, Ly), a=d_par * np.sqrt(2.0), a_perp=d_perp,
                            Z_cation=Z_film[0], Z_anion=Z_film[-1], ymin=y_interface,
                            y0=y_interface + interface_gap, label=1)

    if misfit_dislocation_spacing:
        # Relax by slipping rather than by straining: the film keeps the substrate's
        # in-plane period and loses one period at each dislocation.  Applied to film
        # columns only, as the far-field limit of an interfacial dislocation array.
        #
        # Two details that are not cosmetic.  The staircase is re-centred to zero mean over
        # the field of view, and the film is tiled with a margin wide enough to cover the
        # full drop -- otherwise the columns march off one side and leave a blank strip
        # where the film should be, which reads as a real feature rather than as the tiling
        # artefact it is.
        if misfit_dislocation_spacing == "full":
            misfit_dislocation_spacing = relaxing_dislocation_spacing(d_relaxed, ax)
        misfit_dislocation_spacing = float(misfit_dislocation_spacing)
        n_disl = Lx / misfit_dislocation_spacing
        pad = d_par * (n_disl + 2) + 12.0
        film = (cubic_film_100((Lx, Ly), d=d_par, d_perp=d_perp, Z=Z_film[0],
                               ymin=y_interface, y0=y_interface + interface_gap,
                               margin=pad, label=1)
                if film_zone == "100" else
                rocksalt_110((Lx, Ly), a=d_par * np.sqrt(2.0), a_perp=d_perp,
                             Z_cation=Z_film[0], Z_anion=Z_film[-1], ymin=y_interface,
                             y0=y_interface + interface_gap, margin=pad, label=1))
        # Sign from the physics, not from taste: a film whose natural period exceeds the
        # substrate's is compressed when coherent, so relaxing must widen its mean spacing.
        u = misfit_staircase(misfit_dislocation_spacing, d_par, misfit_step_width,
                             x_offset=0.35 * misfit_dislocation_spacing,
                             sign=np.sign(d_relaxed - ax) or 1.0)
        shift = u(film.xy[:, 0])
        inside = (film.xy[:, 0] >= 0) & (film.xy[:, 0] <= Lx)
        film.xy[:, 0] += shift - (shift[inside].mean() if inside.any() else 0.0)
        film = film.crop(margin=2.0)

    if interface_roughness > 0:
        # Push columns within ~2 cells of the interface up or down along a smooth lateral
        # wiggle.  Cosmetic, but a dead-straight interface makes choosing the reference
        # region look easier than it is on real data.
        kx = 2 * np.pi / Lx * rng.integers(2, 5)
        phase = rng.uniform(0, 2 * np.pi)
        for cols, sign in ((sub, +1.0), (film, -1.0)):
            d = np.abs(cols.xy[:, 1] - y_interface)
            w = np.exp(-0.5 * (d / (1.5 * a_substrate)) ** 2)
            cols.xy[:, 1] += sign * interface_roughness * w * np.cos(kx * cols.xy[:, 0] + phase)

    stack = sub + film
    stack.meta["Z_exponent"] = 1.7
    coherent = bool(relaxation == 0.0 and not misfit_dislocation_spacing)
    stack.meta["ground_truth"] = dict(
        a_substrate=a_substrate, ax_substrate=ax, ay_substrate=a_substrate,
        d_dumbbell=a_substrate / 4.0,
        film_zone=film_zone, d_film_relaxed=d_relaxed,
        rumple=rumple,
        d_film_parallel=d_par, d_film_perp=d_perp,
        eps_parallel=eps_par, eps_perpendicular=eps_perp, poisson=poisson,
        relaxation=relaxation, coherent=coherent,
        misfit_dislocation_spacing=misfit_dislocation_spacing,
        misfit_strain=(d_relaxed - ax) / ax,
        y_interface=y_interface, interface_gap=interface_gap,
        fov=(Lx, Ly), axial_ratio=np.sqrt(2.0), axial_angle_deg=90.0,
    )
    return stack


# ---------------------------------------------------------------------------
# consistency check against the correction code's reference geometry
# ---------------------------------------------------------------------------


def check_reference_consistency(a=5.653):
    """
    Assert that this simulator and the reference geometry the correction fits against
    describe the same crystal.

    A validation sweep that recovers ``(s, k)`` perfectly because both sides share the
    *same wrong* constants has validated nothing, so this is worth an explicit call.
    Returns a dict of the compared quantities.
    """
    from quantem.imaging.drift_single_image import ZINCBLENDE_110

    ref = ZINCBLENDE_110.with_lattice_constant(a)
    sim_ax, sim_ay = a / np.sqrt(2.0), a
    out = dict(
        ax=(sim_ax, ref.ax), ay=(sim_ay, ref.ay),
        axial_ratio=(sim_ay / sim_ax, ref.axial_ratio),
        d_dumbbell=(a / 4.0, ref.d_dumbbell),
    )
    for name, (mine, theirs) in out.items():
        if not np.isclose(mine, theirs, rtol=1e-12, atol=1e-12):
            raise AssertionError(f"reference geometry mismatch in {name}: "
                                 f"zbkit={mine!r} vs quantem={theirs!r}")
    return out
