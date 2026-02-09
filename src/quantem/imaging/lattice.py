import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import least_squares

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization import show_2d


class Lattice(AutoSerialize):
    """
    Atomic lattice fitting in 2D.
    """

    _token = object()

    def __init__(
        self,
        image: Dataset2d,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use Lattice.from_data() to instantiate this class.")
        self._image: Dataset2d = image

    # --- Constructors ---
    @classmethod
    def from_data(
        cls,
        image: Dataset2d | NDArray,
        normalize_min: bool = True,
        normalize_max: bool = True,
    ) -> "Lattice":
        """
        Create a Lattice instance from a 2D image-like input.

        Parameters:
        - image: A 2D numpy array or a Dataset2d instance representing the image.
        - normalize_min: If True, shift the image so its minimum becomes 0.
        - normalize_max: If True, scale the image by its maximum after min-shift
          so values are in [0, 1]. If the maximum is 0 or non-finite (NaN/Inf),
          scaling is skipped to avoid invalid operations.

        Notes:
        - Non-2D inputs and empty arrays raise a ValueError.
        - Inputs with boolean dtype are safely converted to float before normalization.
        - NaN values are ignored when computing min/max (using nanmin/nanmax). If the
          data is all-NaN, normalization is skipped.
        """
        if isinstance(image, Dataset2d):
            ds2d = image
            # Ensure numeric operations are valid (e.g., for bool dtype)
            ds2d.array = np.asarray(ds2d.array, dtype=float)
            # Validate shape
            if ds2d.array.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if ds2d.array.size == 0:
                raise ValueError("Input image array must not be empty.")
        else:
            # Validate dimensionality and emptiness before any processing
            arr = np.asarray(image)
            if arr.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if arr.size == 0:
                raise ValueError("Input image array must not be empty.")
            # Convert to float for safe arithmetic (handles bool arrays)
            arr = arr.astype(float, copy=False)
            if hasattr(Dataset2d, "from_array") and callable(getattr(Dataset2d, "from_array")):
                ds2d = Dataset2d.from_array(arr)  # type: ignore[attr-defined]
            else:
                ds2d = Dataset2d(arr)  # type: ignore[call-arg]

        # Normalization (robust to constant, NaN, and bool inputs)
        if normalize_min:
            # Use nanmin to ignore NaNs; if all-NaN, skip
            try:
                min_val = np.nanmin(ds2d.array)
                if np.isfinite(min_val):
                    ds2d.array = ds2d.array - min_val
            except ValueError:
                # Raised when all values are NaN; skip
                pass

        if normalize_max:
            # Use nanmax to ignore NaNs; skip division if max <= 0 or not finite
            try:
                max_val = np.nanmax(ds2d.array)
                if np.isfinite(max_val) and max_val > 0.0:
                    ds2d.array = ds2d.array / max_val
            except ValueError:
                # Raised when all values are NaN; skip
                pass

        return cls(image=ds2d, _token=cls._token)

    # --- Properties ---
    @property
    def image(self) -> Dataset2d:
        return self._image

    @image.setter
    def image(self, value: Dataset2d | NDArray):
        if isinstance(value, Dataset2d):
            # Ensure numeric dtype to avoid boolean arithmetic issues downstream
            value.array = np.asarray(value.array, dtype=float)
            # Validate shape
            if value.array.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if value.array.size == 0:
                raise ValueError("Input image array must not be empty.")
            self._image = value
        else:
            arr = np.asarray(value)
            if arr.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if arr.size == 0:
                raise ValueError("Input image array must not be empty.")
            arr = arr.astype(float, copy=False)
            if hasattr(Dataset2d, "from_array") and callable(getattr(Dataset2d, "from_array")):
                self._image = Dataset2d.from_array(arr)  # type: ignore[attr-defined]
            else:
                self._image = Dataset2d(arr)  # type: ignore[call-arg]

    # --- Functions ---
    def define_lattice(
        self,
        origin,
        u,
        v,
        refine_lattice: bool = True,
        block_size: int = -1,
        plot_lattice: bool = True,
        bound_num_vectors: int | None = None,
        input_mask=None,
        refine_maxiter: int = 200,
        **kwargs,
    ) -> "Lattice":
        """
        Define the lattice for the image using the origin and the u and v vectors starting from the origin.
        The lattice is defined as r = r0 + nu + mv.

        Parameters
        ----------
        origin : NDArray[2] | Sequence[float]
            Start point (r0) to define the lattice.
            Enter as (row, col) as a numpy array, list, or tuple.
            Ideally a lattice point.
        u : NDArray[2] | Sequence[float]
            Basis vector u to define the lattice.
            Enter as (row, col) as a numpy array, list, or tuple.
        v : NDArray[2] | Sequence[float]
            Basis vector v to define the lattice.
            Enter as (row, col) as a numpy array, list, or tuple.
        refine_lattice : bool, default=True
            If True, refines the values of r0, u, and v by maximizing the bilinear intensity sum.
        block_size : int, default=-1
            Fit the lattice points in steps of block_size * lattice_vectors(u, v).
            For example, if block_size = 5, then the lattice points will be fit in steps of
            (-5, 5)u * (-5, 5)v -> (-10, 10)u * (-10, 10)v -> ...
            block_size = -1 means the entire image will be fit at once.
        plot_lattice : bool, default=True
            If True, the lattice vectors and lines will be plotted overlaid on the image.
        bound_num_vectors : int | None, default=None
            The maximum number of lattice vectors to plot in each direction.
            For example, if bound_num_vectors = 5, lattice lines between (-5, 5)u * (-5, 5)v will be plotted.
            If None, the plotting bounds are set to the image edges.
        refine_maxiter : int, default=200
            Maximum number of iterations for the lattice refinement optimizer (Powell method).
        **kwargs
            Additional keyword arguments forwarded to the plotting function (show_2d), e.g., cmap, title, etc.

        Returns
        -------
        self : Lattice
            Returns the same object, modified in-place.
            The final values of r0, u, v are stored in self._lat.
        """
        # Lattice
        self._lat = np.vstack(
            (
                np.array(origin),
                np.array(u),
                np.array(v),
            )
        )
        if not self._lat.shape == (3, 2):
            raise ValueError("origin, u, v must be in (row, col) format only.")

        # Refine lattice coordinates
        # Note that we currently assume corners are local maxima
        if refine_lattice:
            from scipy.optimize import minimize

            H, W = self._image.shape  # rows (x), cols (y)
            im = np.asarray(self._image.array, dtype=float)
            r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)  # (x, y)

            corners = np.array(
                [
                    [0.0, 0.0],
                    [float(H), 0.0],
                    [0.0, float(W)],
                    [float(H), float(W)],
                ],
                dtype=float,
            )

            # a,b from corners; A = [u v] in columns (2x2), rhs = (corner - r0)
            A = np.column_stack((u, v))  # (2,2)
            ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]  # (2,4)

            a_min, a_max = int(np.floor(ab[0].min())), int(np.ceil(ab[0].max()))
            b_min, b_max = int(np.floor(ab[1].min())), int(np.ceil(ab[1].max()))

            max_ind = max(abs(a_min), a_max, abs(b_min), b_max)
            steps = (
                [*np.arange(0, max_ind + 1, block_size)[1:], max_ind] if max_ind > 0 else [max_ind]
            )

            PENALTY = 1e10
            H_CLIP = H - 2
            W_CLIP = W - 2

            a_range = np.arange(max(a_min, -max_ind), min(a_max, max_ind) + 1, dtype=np.int32)
            b_range = np.arange(max(b_min, -max_ind), min(b_max, max_ind) + 1, dtype=np.int32)
            aa, bb = np.meshgrid(a_range, b_range, indexing="ij")

            # Pre-compute all masks and bases
            all_masks = {}
            all_bases = {}
            for curr_block_size in steps:
                a_min_blk = max(a_min, -curr_block_size)
                a_max_blk = min(a_max, curr_block_size)
                b_min_blk = max(b_min, -curr_block_size)
                b_max_blk = min(b_max, curr_block_size)

                mask = (
                    (aa >= a_min_blk) & (aa <= a_max_blk) & (bb >= b_min_blk) & (bb <= b_max_blk)
                )

                aa_masked = aa[mask]
                bb_masked = bb[mask]

                all_masks[curr_block_size] = mask
                all_bases[curr_block_size] = np.column_stack(
                    [np.ones(aa_masked.size), aa_masked.ravel(), bb_masked.ravel()]
                )

            # Pre-allocate cache
            max_points = max(basis.shape[0] for basis in all_bases.values())
            x0_cache = np.empty(max_points, dtype=np.int32)
            y0_cache = np.empty(max_points, dtype=np.int32)
            dx_cache = np.empty(max_points, dtype=np.float64)
            dy_cache = np.empty(max_points, dtype=np.float64)

            def bilinear_sum(im_: np.ndarray, xy: np.ndarray) -> float:
                """Sum of bilinearly interpolated intensities at (x,y) points."""

                n_points = xy.shape[0]
                if n_points == 0:
                    return 0.0

                x, y = xy[:, 0], xy[:, 1]

                # Filter points that are within valid bounds for bilinear interpolation
                # Need x in [0, H-2] and y in [0, W-2] so that x+1 and y+1 are valid
                valid_mask = (
                    (x >= 0)
                    & (x <= H_CLIP)
                    & (y >= 0)
                    & (y <= W_CLIP)
                    & np.isfinite(x)
                    & np.isfinite(y)
                )

                if input_mask is not None:
                    pixel_buffer = 1028
                    input_mask_padded = np.zeros(
                        [
                            input_mask.shape[0] + 2 * pixel_buffer,
                            input_mask.shape[1] + 2 * pixel_buffer,
                        ]
                    ).astype(bool)
                    input_mask_padded[pixel_buffer:-pixel_buffer, pixel_buffer:-pixel_buffer] = (
                        input_mask
                    )
                    x_round = np.round(x).astype(np.int32) + pixel_buffer
                    y_round = np.round(y).astype(np.int32) + pixel_buffer
                    valid_mask &= (
                        input_mask_padded[x_round, y_round]
                        & input_mask_padded[x_round + 1, y_round]
                        & input_mask_padded[x_round, y_round + 1]
                        & input_mask_padded[x_round + 1, y_round + 1]
                    )

                n_valid = np.sum(valid_mask)
                if n_valid == 0:
                    return -PENALTY

                x_valid = x[valid_mask]
                y_valid = y[valid_mask]

                x0, y0 = x0_cache[:n_valid], y0_cache[:n_valid]
                dx, dy = dx_cache[:n_valid], dy_cache[:n_valid]

                np.floor(x_valid, out=dx)
                x0[:] = dx.astype(np.int32)
                np.floor(y_valid, out=dy)
                y0[:] = dy.astype(np.int32)

                np.subtract(x_valid, x0, out=dx)
                np.subtract(y_valid, y0, out=dy)

                Ia = im_[x0, y0]
                Ib = im_[x0 + 1, y0]
                Ic = im_[x0, y0 + 1]
                Id = im_[x0 + 1, y0 + 1]

                return np.sum(
                    Ia * (1 - dx) * (1 - dy)
                    + Ib * dx * (1 - dy)
                    + Ic * (1 - dx) * dy
                    + Id * dx * dy
                )

            current_basis = None

            def objective(theta: np.ndarray) -> float:
                # theta is 6-vector -> (3,2) matrix [[r0],[u],[v]]
                lat = theta.reshape(3, 2)
                xy = current_basis @ lat  # (N,2) with columns (x,y)
                # Negative: maximize intensity sum by minimizing its negative
                return -bilinear_sum(im, xy)

            minimize_options = {
                "maxiter": int(refine_maxiter),
                "xtol": 1e-3,
                "ftol": 1e-3,
                "disp": False,
            }

            lat_flat = self._lat.astype(np.float32).reshape(-1)

            for curr_block_size in steps:
                current_basis = all_bases[curr_block_size]

                res = minimize(
                    objective,
                    lat_flat,
                    method="Powell",
                    options=minimize_options,
                )

                # Update for next iteration
                lat_flat = res.x
                self._lat = res.x.reshape(3, 2)

        # plotting
        if plot_lattice:
            fig, ax = show_2d(
                self._image.array,
                returnfig=True,
                **kwargs,
            )

            # Put the image at lowest zorder so overlays sit on top
            if ax.images:
                ax.images[-1].set_zorder(0)

            H, W = self._image.shape  # rows (x), cols (y)
            r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)  # each (x, y) == (row, col)

            # print(r0)
            # print(u)
            # print(v)

            # -------------------------------
            # Origin marker (TOP of stack)
            # -------------------------------
            ax.scatter(
                r0[1],
                r0[0],  # (y, x)
                s=60,
                edgecolor=(0, 0, 0),
                facecolor=(0, 0.5, 0),
                marker="s",
                zorder=30,
            )

            # -------------------------------
            # Lattice vectors as arrows
            # -------------------------------
            n_vec = int(bound_num_vectors) if bound_num_vectors is not None else 1

            # draw n_vec arrows for u (red)
            for k in range(1, n_vec + 1):
                tip = r0 + k * u
                ax.arrow(
                    r0[1],
                    r0[0],  # base (y, x)
                    (tip - r0)[1],
                    (tip - r0)[0],  # delta (y, x)
                    length_includes_head=True,
                    head_width=4.0,
                    head_length=6.0,
                    linewidth=2.0,
                    color="red",
                    zorder=20,
                )

            # draw n_vec arrows for v (cyan)
            for k in range(1, n_vec + 1):
                tip = r0 + k * v
                ax.arrow(
                    r0[1],
                    r0[0],
                    (tip - r0)[1],
                    (tip - r0)[0],
                    length_includes_head=True,
                    head_width=4.0,
                    head_length=6.0,
                    linewidth=2.0,
                    color=(0.0, 0.7, 1.0),
                    zorder=20,
                )

            # -----------------------------------------
            # Solve for a,b at plot corners (bounds)
            # -----------------------------------------
            if bound_num_vectors is None:
                corners = np.array(
                    [
                        [0.0, 0.0],
                        [float(H), 0.0],
                        [0.0, float(W)],
                        [float(H), float(W)],
                    ]
                )
            else:
                n = float(bound_num_vectors)
                corners = np.array(
                    [
                        r0 - n * u,
                        r0 - n * v,
                        r0 + n * u,
                        r0 + n * v,
                    ],
                    dtype=float,
                )

            # a,b from corners; A = [u v] in columns (2x2), rhs = (corner - r0)
            A = np.column_stack((u, v))  # shape (2,2)
            ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]  # (2,4)

            a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
            b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

            # -----------------------------------------
            # Clipping rectangle (image or custom)
            # -----------------------------------------
            if bound_num_vectors is None:
                x_lo, x_hi = 0.0, float(H)  # rows
                y_lo, y_hi = 0.0, float(W)  # cols
            else:
                # Bounds are the min/max over the provided corners
                x_lo, x_hi = float(np.min(corners[:, 0])), float(np.max(corners[:, 0]))
                y_lo, y_hi = float(np.min(corners[:, 1])), float(np.max(corners[:, 1]))

            def clipped_segment(base: np.ndarray, direction: np.ndarray):
                """Clip base + t*direction to rectangle [x_lo,x_hi] x [y_lo,y_hi]."""
                x0, y0 = base
                dx, dy = direction
                t0, t1 = -np.inf, np.inf
                eps = 1e-12

                # x in [x_lo, x_hi]
                if abs(dx) < eps:
                    if not (x_lo <= x0 <= x_hi):
                        return None
                else:
                    tx0 = (x_lo - x0) / dx
                    tx1 = (x_hi - x0) / dx
                    t_enter, t_exit = (tx0, tx1) if tx0 <= tx1 else (tx1, tx0)
                    t0, t1 = max(t0, t_enter), min(t1, t_exit)

                # y in [y_lo, y_hi]
                if abs(dy) < eps:
                    if not (y_lo <= y0 <= y_hi):
                        return None
                else:
                    ty0 = (y_lo - y0) / dy
                    ty1 = (y_hi - y0) / dy
                    t_enter, t_exit = (ty0, ty1) if ty0 <= ty1 else (ty1, ty0)
                    t0, t1 = max(t0, t_enter), min(t1, t_exit)

                if t0 > t1:
                    return None

                p1 = base + t0 * direction  # (x, y)
                p2 = base + t1 * direction
                return p1, p2

            # -----------------------------------------
            # Lattice lines (zorder above image)
            # Using x=rows, y=cols: plot(y, x)
            # -----------------------------------------

            # Lines parallel to v (vary a)
            for a in range(a_min, a_max + 1):
                base = r0 + a * u
                seg = clipped_segment(base, v)
                if seg is None:
                    continue
                (x1, y1), (x2, y2) = seg
                ax.plot([y1, y2], [x1, x2], color=(0.0, 0.7, 1.0), lw=1, clip_on=True, zorder=10)

            # Lines parallel to u (vary b)
            # print(b_min)
            # print(b_max)
            for b in range(b_min, b_max + 1):
                base = r0 + b * v
                seg = clipped_segment(base, u)
                if seg is None:
                    continue
                (x1, y1), (x2, y2) = seg
                ax.plot([y1, y2], [x1, x2], color="red", lw=1, clip_on=True, zorder=10)

            # Axes limits (x=rows vertical; y=cols horizontal)
            ax.set_xlim(y_lo, y_hi)
            ax.set_ylim(x_hi, x_lo)

        return self

    def add_atoms(
        self,
        positions_frac,
        numbers=None,
        intensity_min=None,
        intensity_radius=None,
        plot_atoms=True,
        *,
        edge_min_dist_px=None,
        mask=None,
        contrast_min=None,
        annulus_radii=None,
        **kwargs,
    ) -> "Lattice":
        """
        Add atoms for each lattice site by sampling all integer lattice translations that fall inside
        the image, measuring local intensity, and filtering candidates by bounds, edge distance,
        mask, and optional intensity/contrast thresholds. Optionally plots the detected atoms.

        Parameters
        ----------
        positions_frac : array-like, shape (S, 2)
            Fractional positions (a, b) of S lattice sites within the unit cell. These are offsets
            relative to the lattice origin r0 and basis vectors (u, v), and are used to tile the
            image with candidate atom centers at all visible integer translations.
        numbers : array-like of int, shape (S,), optional
            Identifier per site (e.g., species or label). If None, uses 1..S. Used only for plotting
            color coding; not used in detection logic.
        intensity_min : float, optional
            Minimum mean intensity inside the detection disk required to keep a candidate atom.
            If None, no intensity thresholding is applied.
        intensity_radius : float, optional
            Radius (in pixels) of the detection disk used to compute the mean intensity at each
            candidate center. If None, an automatic radius is estimated as half of the nearest-neighbor
            spacing in pixels (see Notes).
        plot_atoms : bool, default True
            If True, displays the image and overlays the detected atoms for each site.
        edge_min_dist_px : float, optional
            Minimum distance (in pixels) that candidate centers must maintain from the image borders.
            If a mask is provided and a distance transform can be computed, this same threshold is also
            used to enforce a minimum distance from masked boundaries.
        mask : array-like of bool, shape (H, W), optional
            Binary mask defining valid regions. If provided:
            - When a distance transform is available, candidates must be at least edge_min_dist_px away
            from masked boundaries.
            - Otherwise, candidates are kept only if the nearest integer-pixel location is True in the mask.
        contrast_min : float, optional
            Minimum contrast required to keep a candidate, defined as (disk mean) - (annulus mean).
            If None, no contrast thresholding is applied.
        annulus_radii : tuple of float, optional
            Inner and outer radii (in pixels) of the background annulus used for contrast estimation.
            If None, defaults to (1.5 * intensity_radius, 3.0 * intensity_radius).
        **kwargs
            Additional keyword arguments forwarded to the plotting helper (show_2d) when plot_atoms is True.

        Returns
        -------
        self
            The current object, with the following side effects:
            - self._positions_frac set from positions_frac
            - self._num_sites set to S
            - self._numbers set from numbers or default sequence
            - self.atoms populated with detected atom data per site

        Raises
        ------
        ValueError
            If a provided mask does not match the image shape (H, W).

        Side Effects
        ------------
        self.atoms : Vector
            shape=(S,), fields=("x", "y", "a", "b", "int_peak"), units=("px", "px", "ind", "ind", "counts").
            For each site index s, self.atoms[s] holds a table with one row per detected atom:
            - x, y: pixel coordinates of the atom center (x is row, y is column; origin at top-left)
            - a, b: fractional lattice indices for that atom (including the site's fractional offset plus integer translations)
            - int_peak: mean intensity inside the detection disk at (x, y)

        Notes
        -----
        Lattice and image geometry
            - The image array is of shape (H, W), where x indexes rows and y indexes columns.
            - Lattice parameters are taken from self._lat = [r0, u, v], with r0 the origin (in pixels)
            and u, v the lattice basis vectors (in pixels). Candidate centers are generated by tiling
            each site's fractional offset across all integer translations that map into the image bounds.
            - The visible range of integer translations (a, b) is determined by projecting the image corners
            through the inverse lattice transform.

        Automatic detection radius (when intensity_radius is None)
            - If there are at least two sites, the nearest-neighbor spacing is computed from fractional
            differences between site positions, accounting for periodic wrapping, and converted to pixels
            via the lattice matrix [u v]. The radius is set to half of this spacing.
            - If there is only one site, the spacing fallback is min(||u||, ||v||, ||u+v||, ||u-v||), and the
            radius is half of this value.
            - If the estimate is invalid or non-positive, a robust fallback of 0.5 * (0.5 * (||u|| + ||v||)) is used.

        Filtering
            - Candidates must lie fully within image bounds and satisfy the edge_min_dist_px constraint.
            - If mask is provided and a distance transform can be computed, candidates must also be at least
            edge_min_dist_px inside the masked region; otherwise, the mask must be True at the nearest integer pixel.
            - intensity_min filters by the disk mean; contrast_min filters by the difference between the disk mean
            and the annulus mean, where the annulus default is (1.5 * r, 3.0 * r).

        Plotting
            - When plot_atoms is True, the image is shown and detected atoms are rendered as semi-transparent
            colored markers per site. Colors are determined by site numbers. Axes are set to match image
            coordinates (x increasing downward).
        """
        if not hasattr(self, "_lat") or self._lat is None:
            raise ValueError(
                "Lattice vectors have not been fitted. Please call define_lattice() first."
            )
        # Handle empty positions early without creating a Vector of length 0
        positions_frac_arr = np.asarray(positions_frac, dtype=float)
        if positions_frac_arr.size == 0:
            # Bookkeeping for consistency
            self._positions_frac = np.empty((0, 2), dtype=float)
            self._num_sites = 0
            self._numbers = (
                np.array([], dtype=int)
                if numbers is None
                else np.atleast_1d(np.array(numbers, dtype=int))
            )
            # Do not construct an empty Vector with zero shape (causes error). Just return.
            return self

        self._positions_frac = np.atleast_2d(np.array(positions_frac, dtype=float))
        self._num_sites = self._positions_frac.shape[0]
        self._numbers = (
            np.arange(1, self._num_sites + 1, dtype=int)
            if numbers is None
            else np.atleast_1d(np.array(numbers, dtype=int))
        )

        im = np.asarray(self._image.array, dtype=float)
        H, W = self._image.shape  # x=rows, y=cols
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        A = np.column_stack((u, v))

        corners = np.array(
            [[0.0, 0.0], [float(H), 0.0], [0.0, float(W)], [float(H), float(W)]], dtype=float
        )
        ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]
        a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
        b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

        def _auto_radius_px() -> float:
            S = self._positions_frac
            if S.shape[0] >= 2:
                d = S[:, None, :] - S[None, :, :]
                d = d - np.round(d)
                same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
                dpix = d @ A.T
                dist = np.linalg.norm(dpix, axis=2)
                dist[same] = np.inf
                nn = float(np.min(dist))
            else:
                nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
            if not np.isfinite(nn) or nn <= 0:
                nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
            return 0.5 * nn

        r_px = float(intensity_radius) if intensity_radius is not None else _auto_radius_px()
        rin, rout = (1.5 * r_px, 3.0 * r_px) if annulus_radii is None else annulus_radii
        R_disk = int(np.ceil(r_px))
        R_ring = int(np.ceil(rout))
        edge_thresh = float(edge_min_dist_px) if edge_min_dist_px is not None else 0.0

        DT = None
        if mask is not None:
            m = np.asarray(mask).astype(bool)
            if m.shape != (H, W):
                raise ValueError(f"mask shape {m.shape} must match image shape {(H, W)}")
            try:
                from scipy.ndimage import distance_transform_edt

                DT = distance_transform_edt(m)
            except Exception:
                DT = None

        def mean_disk(x: float, y: float) -> float:
            ix0, iy0 = int(np.floor(x)), int(np.floor(y))
            i0, i1 = max(0, ix0 - R_disk), min(H - 1, ix0 + R_disk)
            j0, j1 = max(0, iy0 - R_disk), min(W - 1, iy0 + R_disk)
            ii = np.arange(i0, i1 + 1)[:, None]
            jj = np.arange(j0, j1 + 1)[None, :]
            dx, dy = ii - x, jj - y
            mask_circle = (dx * dx + dy * dy) <= (r_px * r_px)
            vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_circle]
            if vals.size == 0:
                return float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
            return float(vals.mean())

        def mean_std_annulus(x: float, y: float) -> tuple[float, float]:
            ix0, iy0 = int(np.floor(x)), int(np.floor(y))
            i0, i1 = max(0, ix0 - R_ring), min(H - 1, ix0 + R_ring)
            j0, j1 = max(0, iy0 - R_ring), min(W - 1, iy0 + R_ring)
            ii = np.arange(i0, i1 + 1)[:, None]
            jj = np.arange(j0, j1 + 1)[None, :]
            dx, dy = ii - x, jj - y
            r2 = dx * dx + dy * dy
            mask_ring = (r2 >= rin * rin) & (r2 <= rout * rout)
            vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_ring]
            if vals.size == 0:
                val = float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
                return val, 0.0
            return float(vals.mean()), float(vals.std(ddof=0))

        self.atoms = Vector.from_shape(
            shape=(self._num_sites,),
            fields=("x", "y", "a", "b", "int_peak"),
            units=("px", "px", "ind", "ind", "counts"),
        )

        for a0 in range(self._num_sites):
            da, db = self._positions_frac[a0, 0], self._positions_frac[a0, 1]
            aa, bb = np.meshgrid(
                np.arange(a_min - 1 + da, a_max + 1 + da),
                np.arange(b_min - 1 + db, b_max + 1 + db),
                indexing="ij",
            )
            basis = np.vstack((np.ones(aa.size), aa.ravel(), bb.ravel())).T
            xy = basis @ self._lat  # (N,2) in (x,y)

            x, y = xy[:, 0], xy[:, 1]
            in_bounds = (x >= 0.0) & (x <= H - 1) & (y >= 0.0) & (y <= W - 1)
            border_ok = (
                (x - edge_thresh >= 0.0)
                & (x + edge_thresh <= H - 1)
                & (y - edge_thresh >= 0.0)
                & (y + edge_thresh <= W - 1)
            )

            if mask is not None:
                if DT is not None:
                    ii = np.clip(np.round(x).astype(int), 0, H - 1)
                    jj = np.clip(np.round(y).astype(int), 0, W - 1)
                    mask_ok = DT[ii, jj] >= edge_thresh
                else:
                    m = np.asarray(mask).astype(bool)
                    mask_ok = m[
                        np.clip(np.round(x).astype(int), 0, H - 1),
                        np.clip(np.round(y).astype(int), 0, W - 1),
                    ]
            else:
                mask_ok = np.ones_like(in_bounds, dtype=bool)

            int_center = np.empty(xy.shape[0], dtype=float)
            for i in range(xy.shape[0]):
                int_center[i] = mean_disk(x[i], y[i])

            keep = in_bounds & border_ok & mask_ok
            if intensity_min is not None:
                keep &= int_center >= float(intensity_min)
            if contrast_min is not None:
                bg_mean = np.empty(xy.shape[0], dtype=float)
                for i in range(xy.shape[0]):
                    bg_mean[i], _ = mean_std_annulus(x[i], y[i])
                keep &= (int_center - bg_mean) >= float(contrast_min)

            if np.any(keep):
                arr = np.vstack(
                    (x[keep], y[keep], basis[keep, 1], basis[keep, 2], int_center[keep])
                ).T
            else:
                arr = np.zeros((0, 5), dtype=float)

            # --- Correct API usage ---
            self.atoms.set_data(arr, a0)

        if plot_atoms:
            fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)
            for a0 in range(self._num_sites):
                cell = self.atoms.get_data(a0)
                if isinstance(cell, list) or cell is None or cell.size == 0:
                    continue
                x = self.atoms[a0]["x"]
                y = self.atoms[a0]["y"]
                rgb = site_colors(int(self._numbers[a0]))
                ax.scatter(
                    y,
                    x,
                    s=18,
                    facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
                    edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                    linewidths=0.75,
                    marker="o",
                    zorder=18,
                )
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        return self

    def refine_atoms(
        self,
        fit_radius=None,
        max_nfev: int = 200,
        max_move_px: float | None = None,
        plot_atoms: bool = False,
        **kwargs,
    ) -> "Lattice":
        """
        Refine atom centers by local 2D Gaussian fitting around each previously detected atom.
        Updates atom positions and peak intensity and adds per-atom sigma and background fields.
        Optionally plots the refined atoms.
        Parameters
        ----------
        fit_radius : float, optional
            Radius (in pixels) of the circular fitting region around each atom's current center.
            If None, an automatic radius is estimated as half of the nearest-neighbor spacing
            between lattice sites in pixels. When there is only one site, the spacing fallback
            is min(||u||, ||v||, ||u+v||, ||u-v||) where u and v are lattice vectors. If this
            estimate is invalid or non-positive, a robust fallback is used.
        max_nfev : int, default 200
            Maximum number of function evaluations for the non-linear least-squares solver.
        max_move_px : float, optional
            Maximum allowed movement (in pixels) of the refined center from its initial position.
            If None, defaults to the fitting radius. Bounds also enforce staying within image limits.
        plot_atoms : bool, default False
            If True, displays the image and overlays the refined atom positions.
        **kwargs
            Additional keyword arguments forwarded to the plotting helper when plot_atoms is True.

        Returns
        -------
        self
            The current object, with self.atoms updated per site to refined values.

        Raises
        ------
        ValueError
            If no atoms are present to refine (call add_atoms() first).

        Side Effects
        ------------
        self.atoms : Vector
            For each site index s, the per-atom rows are updated:
            - x, y: pixel coordinates refined by local Gaussian fitting (x is row, y is column).
            - int_peak: updated to the fitted Gaussian amplitude at the center.
            - sigma: added or updated; the fitted Gaussian width (pixels).
            - int_bg: added or updated; the fitted local constant background level.
            If "sigma" and "int_bg" fields do not exist, they are added automatically.

        Notes
        -----
        Model and fitting
            - A circular patch of radius fit_radius is extracted around each atom's current center.
            - Within that patch, a 2D isotropic Gaussian plus constant background is fit:
            I(x, y) = amp * exp(-0.5 * r^2 / sigma^2) + bg, where r^2 is the squared distance
            to the fitted center (x_c, y_c).
            - Initial guesses:
            - Center starts at the current atom position.
            - amp starts from the central pixel value minus the local median background.
            - sigma starts at max(0.5 * fit_radius, 0.5).
            - bg starts at the median of the patch outside the circular mask (or full patch median).
            - Parameter bounds:
            - Center (x_c, y_c) limited to within max_move_px of the initial center and within
                image bounds.
            - amp in [0, max(pmax - pmin, 4 * amp0)], using local patch extrema and initial amp0.
            - sigma in [0.25, max(2 x fit_radius, 1.0)].
            - bg in [pmin * (pmax - pmin), pmax + (pmax - pmin)].
            - Optimization uses scipy.optimize.least_squares with "trf" method and "soft_l1" loss.

        Automatic fitting radius (when fit_radius is None)
            - If there are at least two sites, the nearest-neighbor spacing is computed from fractional
            differences between site positions (wrapped to [-0.5, 0.5]) and converted to pixels using
            the lattice matrix [u v]; the radius is set to half of this spacing.
            - If there is only one site, the spacing fallback is min(||u||, ||v||, ||u+v||, ||u-v||),
            and the radius is half of this value.
            - If the estimate is invalid or non-positive, a robust fallback is used based on the lattice
            vector norms to ensure a reasonable, non-zero radius.

        Plotting
            - When plot_atoms is True, the image is shown and refined atom centers are rendered as
            semi-transparent colored markers per site. Colors are determined by site numbers.
            - Axes are set to match image coordinates (x increasing downward).
        """

        if not hasattr(self, "atoms"):
            raise ValueError("No atoms to refine. Call add_atoms() first.")

        im = np.asarray(self._image.array, dtype=float)
        H, W = self._image.shape
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        A = np.column_stack((u, v))

        def _auto_radius_px() -> float:
            S = np.asarray(getattr(self, "_positions_frac", [[0.0, 0.0]]), dtype=float)
            if S.shape[0] >= 2:
                d = S[:, None, :] - S[None, :, :]
                d = d - np.round(d)
                same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
                dpix = d @ A.T
                dist = np.linalg.norm(dpix, axis=2)
                dist[same] = np.inf
                nn = float(np.min(dist))
            else:
                nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
            if not np.isfinite(nn) or nn <= 0:
                nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
            return 0.5 * nn

        r_fit = float(fit_radius) if fit_radius is not None else _auto_radius_px()
        R = int(np.ceil(r_fit))
        max_move = float(max_move_px) if max_move_px is not None else r_fit

        # Ensure extra fields exist
        needed = [f for f in ("sigma", "int_bg") if f not in self.atoms.fields]
        if needed:
            self.atoms.add_fields(needed)

        # Single lookup of column indices for writing
        idx_x = self.atoms.fields.index("x")
        idx_y = self.atoms.fields.index("y")
        idx_amp = self.atoms.fields.index("int_peak")
        idx_sigma = self.atoms.fields.index("sigma")
        idx_bg = self.atoms.fields.index("int_bg")
        # i_arr = np.arange(205,209)
        # i_arr = np.arange(3380,3385)

        for s in range(self._num_sites):
            row = self.atoms.get_data(s)
            if isinstance(row, list) or row is None or row.size == 0:
                continue

            # Intuitive reads: per-cell field arrays
            x_arr = self.atoms[s]["x"]
            y_arr = self.atoms[s]["y"]

            updated = row.copy()
            for i in range(row.shape[0]):
                x0, y0 = float(x_arr[i]), float(y_arr[i])

                ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
                i0, i1 = max(0, ix0 - R), min(H - 1, ix0 + R)
                j0, j1 = max(0, iy0 - R), min(W - 1, iy0 + R)
                if i1 <= i0 or j1 <= j0:  # this doesn't do anything
                    continue

                patch = im[i0 : i1 + 1, j0 : j1 + 1]

                # broadcast coordinate grids to patch shape
                ii = np.arange(i0, i1 + 1)[:, None]
                jj = np.arange(j0, j1 + 1)[None, :]
                II = np.broadcast_to(ii, patch.shape)
                JJ = np.broadcast_to(jj, patch.shape)

                r2 = (II - x0) ** 2 + (JJ - y0) ** 2
                mask = r2 <= (
                    r_fit * r_fit
                )  # why not just square this with **? Or square root instead of r2
                if not np.any(mask):
                    continue

                vals = patch[mask].astype(float).ravel()
                pmin, pmax = float(vals.min()), float(vals.max())
                bg0 = float(np.median(patch[~mask])) if np.any(~mask) else float(np.median(patch))
                amp0 = max(float(im[np.clip(ix0, 0, H - 1), np.clip(iy0, 0, W - 1)] - bg0), 1e-6)
                sig0 = max(r_fit * 0.5, 0.5)

                x_coords = II[mask].astype(float).ravel()
                y_coords = JJ[mask].astype(float).ravel()

                def residual(theta):
                    x_c, y_c, amp, sig, bg = theta
                    sig2 = max(sig, 1e-6) ** 2
                    rr = (x_coords - x_c) ** 2 + (y_coords - y_c) ** 2
                    model = amp * np.exp(-0.5 * rr / sig2) + bg
                    return model - vals

                # movement-limited bounds + image bounds
                x_lb = max(x0 - max_move, 0.0)
                x_ub = min(x0 + max_move, H - 1.0)
                y_lb = max(y0 - max_move, 0.0)
                y_ub = min(y0 + max_move, W - 1.0)

                lb = [x_lb, y_lb, 0.0, 0.25, pmin - (pmax - pmin)]
                ub = [
                    x_ub,
                    y_ub,
                    max(pmax - pmin, amp0 * 4.0),
                    max(2.0 * r_fit, 1.0),
                    pmax + (pmax - pmin),
                ]
                theta0 = [x0, y0, amp0, sig0, bg0]

                res = least_squares(
                    residual,
                    theta0,
                    bounds=(lb, ub),
                    method="trf",
                    loss="soft_l1",
                    max_nfev=int(max_nfev),
                    xtol=1e-6,
                    ftol=1e-6,
                    gtol=1e-6,
                )

                x_c, y_c, amp, sig, bg = res.x
                updated[i, idx_x] = x_c
                updated[i, idx_y] = y_c
                updated[i, idx_amp] = amp
                updated[i, idx_sigma] = sig
                updated[i, idx_bg] = bg
                # if i in i_arr:
                #     # print(r_fit)
                #     plt.figure(figsize = (10,3))
                #     # plt.subplot(121)
                #     plt.imshow(patch)
                #     plt.colorbar()
                #     plt.title('V site')
                #     # plt.title('W site')
                #     # plt.title('amp: '+ str(np.round(amp,2)) + ', max: '+str(np.round(np.max(patch),2)))
                #     plt.axis('off')
                #     plt.tight_layout()
                #     print('Fit amplitude:', np.round(amp, 2))
                #     print('Raw max amplitude:', np.round(np.max(patch), 2))
                #     patch[mask]=0
                #     print('Background median:', np.round(np.median(patch[patch>0]), 2))
                #     # plt.subplot(122)
                #     # plt.imshow(patch)
                #     # # plt.title('bg med: '+str(np.round(np.median(patch[patch>0]),2)))
                #     # plt.title('amp: '+ str(np.round(amp,2)) + ', max: '+str(np.round(np.max(patch),2)))
                #     # plt.subplot(133)
                #     # plt.imshow(mask)

            self.atoms.set_data(updated, s)

        if hasattr(self, "check_for_dislocations"):
            if self.check_for_dislocations is True:
                # Ensure extra fields exist
                needed = [f for f in ("sigma", "int_bg") if f not in self.atoms_dislocation.fields]
                if needed:
                    self.atoms_dislocation.add_fields(needed)

                # Single lookup of column indices for writing
                idx_x = self.atoms_dislocation.fields.index("x")
                idx_y = self.atoms_dislocation.fields.index("y")
                idx_amp = self.atoms_dislocation.fields.index("int_peak")
                idx_sigma = self.atoms_dislocation.fields.index("sigma")
                idx_bg = self.atoms_dislocation.fields.index("int_bg")

                for s in range(self._num_sites):
                    row = self.atoms_dislocation.get_data(s)
                    if isinstance(row, list) or row is None or row.size == 0:
                        continue

                    # Intuitive reads: per-cell field arrays
                    x_arr = self.atoms_dislocation[s]["x"]
                    y_arr = self.atoms_dislocation[s]["y"]

                    updated = row.copy()
                    for i in range(row.shape[0]):
                        x0, y0 = float(x_arr[i]), float(y_arr[i])

                        ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
                        i0, i1 = max(0, ix0 - R), min(H - 1, ix0 + R)
                        j0, j1 = max(0, iy0 - R), min(W - 1, iy0 + R)
                        if i1 <= i0 or j1 <= j0:
                            continue

                        patch = im[i0 : i1 + 1, j0 : j1 + 1]

                        # broadcast coordinate grids to patch shape
                        ii = np.arange(i0, i1 + 1)[:, None]
                        jj = np.arange(j0, j1 + 1)[None, :]
                        II = np.broadcast_to(ii, patch.shape)
                        JJ = np.broadcast_to(jj, patch.shape)

                        r2 = (II - x0) ** 2 + (JJ - y0) ** 2
                        mask = r2 <= (r_fit * r_fit)
                        if not np.any(mask):
                            continue

                        vals = patch[mask].astype(float).ravel()
                        pmin, pmax = float(vals.min()), float(vals.max())
                        bg0 = (
                            float(np.median(patch[~mask]))
                            if np.any(~mask)
                            else float(np.median(patch))
                        )
                        amp0 = max(
                            float(im[np.clip(ix0, 0, H - 1), np.clip(iy0, 0, W - 1)] - bg0), 1e-6
                        )
                        sig0 = max(r_fit * 0.5, 0.5)

                        x_coords = II[mask].astype(float).ravel()
                        y_coords = JJ[mask].astype(float).ravel()

                        def residual(theta):
                            x_c, y_c, amp, sig, bg = theta
                            sig2 = max(sig, 1e-6) ** 2
                            rr = (x_coords - x_c) ** 2 + (y_coords - y_c) ** 2
                            model = amp * np.exp(-0.5 * rr / sig2) + bg
                            return model - vals

                        # movement-limited bounds + image bounds
                        x_lb = max(x0 - max_move, 0.0)
                        x_ub = min(x0 + max_move, H - 1.0)
                        y_lb = max(y0 - max_move, 0.0)
                        y_ub = min(y0 + max_move, W - 1.0)

                        lb = [x_lb, y_lb, 0.0, 0.25, pmin - (pmax - pmin)]
                        ub = [
                            x_ub,
                            y_ub,
                            max(pmax - pmin, amp0 * 4.0),
                            max(2.0 * r_fit, 1.0),
                            pmax + (pmax - pmin),
                        ]
                        theta0 = [x0, y0, amp0, sig0, bg0]

                        res = least_squares(
                            residual,
                            theta0,
                            bounds=(lb, ub),
                            method="trf",
                            loss="soft_l1",
                            max_nfev=int(max_nfev),
                            xtol=1e-6,
                            ftol=1e-6,
                            gtol=1e-6,
                        )

                        x_c, y_c, amp, sig, bg = res.x
                        updated[i, idx_x] = x_c
                        updated[i, idx_y] = y_c
                        updated[i, idx_amp] = amp
                        updated[i, idx_sigma] = sig
                        updated[i, idx_bg] = bg

                    self.atoms_dislocation.set_data(updated, s)

        if plot_atoms:
            fig, ax = show_2d(self._image.array, figsize=(10, 10), returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)
            for s in range(self._num_sites):
                cell = self.atoms.get_data(s)
                if isinstance(cell, list) or cell is None or cell.size == 0:
                    continue
                xs = self.atoms[s]["x"]
                ys = self.atoms[s]["y"]
                rgb = site_colors(int(self._numbers[s]))
                ax.scatter(
                    ys,
                    xs,
                    s=18,
                    facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
                    edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                    linewidths=0.75,
                    marker="o",
                    zorder=25,
                )
                if hasattr(self, "check_for_dislocations"):
                    if self.check_for_dislocations is True:
                        # print("trying to plot dislocations")
                        cell = self.atoms_dislocation.get_data(s)
                        if isinstance(cell, list) or cell is None or cell.size == 0:
                            continue
                        xs = self.atoms_dislocation[s]["x"]
                        ys = self.atoms_dislocation[s]["y"]
                        # print(xs)
                        rgb = site_colors(int(self._numbers[s] + 1))
                        ax.scatter(
                            ys,
                            xs,
                            s=18,
                            facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
                            edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                            linewidths=0.75,
                            marker="o",
                            zorder=25,
                        )

            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        return self

    def atoms_first(
        self,
        origin=None,
        u=None,
        v=None,
        positions_frac=None,
        tolerance_uv: float = 1.1,
        numbers=None,
        edge_min_dist_px=None,
        subpixel: str = "poly",
        upsample_factor: int = 16,
        sigma: float = 0,
        minAbsoluteIntensity: float = 0,
        minRelativeIntensity: float = 0,
        relativeToPeak: float = 0,
        minSpacing: float = 0,
        edgeBoundary: int = 1,
        maxNumPeaks: int = 5000,
        plot_atoms=True,
        input_mask=None,
        refine_lattice=True,
        refine_maxiter: int = 200,
        intensity_radius=None,
        intensity_min: float | None = None,
        contrast_min=None,
        annulus_radii=None,
        check_uv_duplication=True,
        check_for_dislocations=False,
        merge_dislocation=False,
        **kwargs,
    ):
        self.check_for_dislocations = check_for_dislocations and check_uv_duplication
        # find all candidates above threshold
        maxima_candidates = self.get_maxima_2D(
            self.image.array,
            subpixel=subpixel,
            upsample_factor=upsample_factor,
            sigma=sigma,
            minAbsoluteIntensity=minAbsoluteIntensity,
            minRelativeIntensity=minRelativeIntensity,
            relativeToPeak=relativeToPeak,
            minSpacing=minSpacing,
            edgeBoundary=edgeBoundary,
            maxNumPeaks=maxNumPeaks,
        )
        H, W = self._image.shape  # x=rows, y=cols

        if origin is None:
            max_intensity_index = np.argmax(maxima_candidates[:]["intensity"])
            origin_x = maxima_candidates[max_intensity_index]["x"]
            origin_y = maxima_candidates[max_intensity_index]["y"]
            origin = np.array([origin_x, origin_y])

        if u is None or v is None:
            num_peaks_search = 20
            num_peaks_use = 2
            center_ignore_buffer = 15
            minSpacingPeaks = 5
            uv_result_inv = self.auto_peak_finder(
                num_peaks_search=num_peaks_search,
                num_peaks_use=num_peaks_use,
                center_ignore_buffer=center_ignore_buffer,
                minSpacingPeaks=minSpacingPeaks,
            )

            g_vector_1_c = np.array([uv_result_inv[0]["x"], uv_result_inv[0]["y"]])
            g_vector_2_c = np.array([uv_result_inv[1]["x"], uv_result_inv[1]["y"]])
            g_vec1 = np.zeros(2)
            g_vec1[0] = (g_vector_1_c[0] - (0.5 * H)) / H
            g_vec1[1] = (g_vector_1_c[1] - (0.5 * W)) / W
            g_vec2 = np.zeros(2)
            g_vec2[0] = (g_vector_2_c[0] - (0.5 * H)) / H
            g_vec2[1] = (g_vector_2_c[1] - (0.5 * W)) / W
            g_matrix = np.array([g_vec1, g_vec2])
            a_matrix = np.linalg.inv(g_matrix)
            a_transpose = a_matrix.T
            u = np.array([a_transpose[0, 0], a_transpose[0, 1]])
            v = np.array([a_transpose[1, 0], a_transpose[1, 1]])
            self.u = u
            self.v = v

        if positions_frac is None:
            positions_frac = np.atleast_2d(np.array((0, 0)))

        self._positions_frac = np.atleast_2d(np.array(positions_frac, dtype=float))
        self._num_sites = self._positions_frac.shape[0]
        self._numbers = (
            np.arange(1, self._num_sites + 1, dtype=int)
            if numbers is None
            else np.atleast_1d(np.array(numbers, dtype=int))
        )

        self._lat = np.vstack(
            (
                np.array(origin),
                np.array(u),
                np.array(v),
            )
        )

        im = np.asarray(self._image.array, dtype=float)
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        A = np.column_stack((u, v))

        def _auto_radius_px() -> float:
            S = self._positions_frac
            if S.shape[0] >= 2:
                d = S[:, None, :] - S[None, :, :]
                d = d - np.round(d)
                same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
                dpix = d @ A.T
                dist = np.linalg.norm(dpix, axis=2)
                dist[same] = np.inf
                nn = float(np.min(dist))
            else:
                nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
            if not np.isfinite(nn) or nn <= 0:
                nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
            return 0.5 * nn

        r_px = float(intensity_radius) if intensity_radius is not None else _auto_radius_px()
        rin, rout = (1.5 * r_px, 3.0 * r_px) if annulus_radii is None else annulus_radii
        R_disk = int(np.ceil(r_px))
        R_ring = int(np.ceil(rout))

        def mean_disk(x: float, y: float) -> float:
            ix0, iy0 = int(np.floor(x)), int(np.floor(y))
            i0, i1 = max(0, ix0 - R_disk), min(H - 1, ix0 + R_disk)
            j0, j1 = max(0, iy0 - R_disk), min(W - 1, iy0 + R_disk)
            ii = np.arange(i0, i1 + 1)[:, None]
            jj = np.arange(j0, j1 + 1)[None, :]
            dx, dy = ii - x, jj - y
            mask_circle = (dx * dx + dy * dy) <= (r_px * r_px)
            vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_circle]
            if vals.size == 0:
                return float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
            return float(vals.mean())

        def mean_std_annulus(x: float, y: float) -> tuple[float, float]:
            ix0, iy0 = int(np.floor(x)), int(np.floor(y))
            i0, i1 = max(0, ix0 - R_ring), min(H - 1, ix0 + R_ring)
            j0, j1 = max(0, iy0 - R_ring), min(W - 1, iy0 + R_ring)
            ii = np.arange(i0, i1 + 1)[:, None]
            jj = np.arange(j0, j1 + 1)[None, :]
            dx, dy = ii - x, jj - y
            r2 = dx * dx + dy * dy
            mask_ring = (r2 >= rin * rin) & (r2 <= rout * rout)
            vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_ring]
            if vals.size == 0:
                val = float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
                return val, 0.0
            return float(vals.mean()), float(vals.std(ddof=0))

        # mask of where in real space maxima can occur
        H, W = self._image.shape  # x=rows, y=cols
        edge_thresh = float(edge_min_dist_px) if edge_min_dist_px is not None else 0.0

        DT = None
        if input_mask is not None:
            m = np.asarray(input_mask).astype(bool)
            if m.shape != (H, W):
                raise ValueError(f"mask shape {m.shape} must match image shape {(H, W)}")
            try:
                from scipy.ndimage import distance_transform_edt

                DT = distance_transform_edt(m)
            except Exception:
                DT = None

        # find the maxima closest to the origin:
        maxima_candidates_x = maxima_candidates[:]["x"]
        maxima_candidates_y = maxima_candidates[:]["y"]

        pm_arr = np.array([-1, 1])  # np.array([-1,0,1])
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        uv_arr = np.array([np.asarray(u), np.asarray(v)])
        uv_norm = 0.5 * (u_norm + v_norm)
        self.uv_norm = uv_norm
        self.uv_arr = uv_arr
        self.tolerance_uv = tolerance_uv
        x = maxima_candidates_x
        y = maxima_candidates_y

        in_bounds = (x >= 0.0) & (x <= H - 1) & (y >= 0.0) & (y <= W - 1)
        border_ok = (
            (x - edge_thresh >= 0.0)
            & (x + edge_thresh <= H - 1)
            & (y - edge_thresh >= 0.0)
            & (y + edge_thresh <= W - 1)
        )
        if input_mask is not None:
            if DT is not None:
                ii = np.clip(np.round(x).astype(int), 0, H - 1)
                jj = np.clip(np.round(y).astype(int), 0, W - 1)
                mask_ok = DT[ii, jj] >= edge_thresh
            else:
                m = np.asarray(input_mask).astype(bool)
                mask_ok = m[
                    np.clip(np.round(x).astype(int), 0, H - 1),
                    np.clip(np.round(y).astype(int), 0, W - 1),
                ]
        else:
            mask_ok = np.ones_like(in_bounds, dtype=bool)

        int_center = np.empty(x.shape[0], dtype=float)
        for i in range(x.shape[0]):
            int_center[i] = mean_disk(x[i], y[i])

        keep = in_bounds & border_ok & mask_ok
        if intensity_min is not None:
            keep &= int_center >= float(intensity_min)
        if contrast_min is not None:
            bg_mean = np.empty(x.shape[0], dtype=float)
            for i in range(x.shape[0]):
                bg_mean[i], _ = mean_std_annulus(x[i], y[i])
            keep &= (int_center - bg_mean) >= float(contrast_min)

        if np.any(keep):
            maxima_candidates = maxima_candidates[keep]
        else:
            raise ValueError("Zero maxima candidates kept")

        # find the maxima closest to the origin:
        maxima_candidates_x = maxima_candidates[:]["x"]
        maxima_candidates_y = maxima_candidates[:]["y"]
        maxima_candidates_intensity = maxima_candidates[:]["intensity"]

        # the unique ids array is an array of the original index, candidacy, a (of a * u), and b (of b * v)
        unique_ids = np.zeros([6, len(maxima_candidates)])
        unique_ids[0, :] = np.arange(0, len(maxima_candidates))
        unique_ids[4, :] = -1 * np.arange(1, 1 + len(maxima_candidates))
        unique_ids[5, :] -= 1

        # Show the candidates that were found (tuning the find peaks functionality)
        if plot_atoms:
            fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)
            xs = maxima_candidates_x
            ys = maxima_candidates_y
            rgb = site_colors(int(self._numbers[0]))
            ax.scatter(
                ys,
                xs,
                s=18,
                facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
                edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                linewidths=0.75,
                marker="o",
                zorder=25,
            )
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        radial_dist = (
            (maxima_candidates_x - origin[0]) ** 2 + (maxima_candidates_y - origin[1]) ** 2
        ) ** (0.5)
        origin_candidate_index = np.argmin(
            radial_dist
        )  # use the first minima, if there are multiple
        unique_ids[1, origin_candidate_index] = 1
        unique_ids[4, origin_candidate_index] = 0

        atoms_found_this_iteration = np.zeros(len(maxima_candidates))
        atoms_found_prev_iteration = np.zeros(len(maxima_candidates))
        atoms_found_previous_iterations = np.zeros(len(maxima_candidates), dtype=bool)
        atoms_found_prev_iteration[origin_candidate_index] = 1
        found_atoms_in_prev_iteration = True
        iteration_while = 0

        def check_dislocations():
            if check_for_dislocations:
                for atom_index in range(len(maxima_candidates)):
                    if unique_ids[1, atom_index] == 1:
                        for pm in pm_arr:
                            for uv_index, lat_vec in enumerate(uv_arr):
                                position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
                                position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
                                radial_dist = (
                                    (maxima_candidates_x - position_x) ** 2
                                    + (maxima_candidates_y - position_y) ** 2
                                ) ** (0.5)
                                radial_dist[atom_index] = (
                                    uv_norm * (tolerance_uv - 1) * 2
                                )  # make sure that self is outside of range
                                if (radial_dist < (uv_norm * (tolerance_uv - 1))).any():
                                    successful_candidate_index = np.argmin(radial_dist)
                                    if unique_ids[1, successful_candidate_index] == 2:
                                        atoms_found_this_iteration[successful_candidate_index] += 1
                                        unique_ids[1, successful_candidate_index] = (
                                            3  # for being found in dislocation search
                                        )
                                        unique_ids[2, successful_candidate_index] = unique_ids[
                                            2, atom_index
                                        ] + pm * int(uv_index == 0)
                                        unique_ids[3, successful_candidate_index] = unique_ids[
                                            3, atom_index
                                        ] + pm * int(uv_index == 1)
                                        unique_ids[4, successful_candidate_index] = 0
                maxima_dislocation_x = maxima_candidates_x[unique_ids[1, :] == 3]
                maxima_dislocation_y = maxima_candidates_y[unique_ids[1, :] == 3]
                maxima_dislocation_u = unique_ids[2, unique_ids[1, :] == 3]
                maxima_dislocation_v = unique_ids[3, unique_ids[1, :] == 3]
                maxima_dislocation_intensity = maxima_candidates_intensity[unique_ids[1, :] == 3]
                arr = np.vstack(
                    (
                        maxima_dislocation_x,
                        maxima_dislocation_y,
                        maxima_dislocation_u,
                        maxima_dislocation_v,
                        maxima_dislocation_intensity,
                    )
                ).T
                return arr

        # first, a loop that finds all of the A sites

        a0 = 0  # here we are just doing a0
        while found_atoms_in_prev_iteration is True:
            for atom_index in range(len(maxima_candidates)):
                if atoms_found_prev_iteration[atom_index] > 0:
                    for pm in pm_arr:
                        for uv_index, lat_vec in enumerate(uv_arr):
                            position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
                            position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
                            radial_dist = (
                                (maxima_candidates_x - position_x) ** 2
                                + (maxima_candidates_y - position_y) ** 2
                            ) ** (0.5)
                            radial_dist[atom_index] = (
                                uv_norm * (tolerance_uv - 1) * 2
                            )  # make sure that self is outside of range
                            if (radial_dist < (uv_norm * (tolerance_uv - 1))).any():
                                successful_candidate_index = np.argmin(radial_dist)
                                if unique_ids[1, successful_candidate_index] == 0:
                                    atoms_found_this_iteration[successful_candidate_index] += 1
                                    unique_ids[1, successful_candidate_index] = 1
                                    unique_ids[2, successful_candidate_index] = unique_ids[
                                        2, atom_index
                                    ] + pm * int(uv_index == 0)
                                    unique_ids[3, successful_candidate_index] = unique_ids[
                                        3, atom_index
                                    ] + pm * int(uv_index == 1)
                                    unique_ids[4, successful_candidate_index] = 0
                                    unique_ids[5, successful_candidate_index] = 0
            # check if any atom was somehow still found twice:
            assert np.max(atoms_found_this_iteration) < 2
            # check if any found atoms have the same uv index
            if check_uv_duplication:
                uv_pairs = unique_ids[1:6, :].T
                unique_pairs, inverse, counts = np.unique(
                    uv_pairs, axis=0, return_inverse=True, return_counts=True
                )
                duplicate_groups = [
                    np.where(inverse == k)[0] for k, c in enumerate(counts) if c > 1
                ]
                mask_atoms_found = atoms_found_this_iteration.astype(bool)
                if len(duplicate_groups) != 0:
                    for duplicate_group in duplicate_groups:
                        duplicate_group = np.asarray(duplicate_group)
                        duplicate_atoms_index_found_previous_iterations = duplicate_group[
                            atoms_found_previous_iterations[duplicate_group]
                        ]
                        if duplicate_atoms_index_found_previous_iterations.size > 1:
                            if origin_candidate_index not in duplicate_group:
                                raise ValueError(
                                    "The duplicate atoms finding code is somehow bugged"
                                )
                            else:
                                kept_index = origin_candidate_index
                        elif duplicate_atoms_index_found_previous_iterations.size == 1:
                            kept_index = duplicate_atoms_index_found_previous_iterations
                        else:
                            kept_index = duplicate_group[mask_atoms_found[duplicate_group]][0]
                        wipe_indicies = duplicate_group[duplicate_group != kept_index]
                        unique_ids[1, wipe_indicies] = (
                            2  # signals to not accept for this maxima anymore
                        )
                        unique_ids[2:4, wipe_indicies] = 0
                        unique_ids[5, wipe_indicies] = -1
                        unique_ids[4, wipe_indicies] = -1 * (wipe_indicies + 1)
                        atoms_found_previous_iterations[wipe_indicies] = False
                        mask_atoms_found[wipe_indicies] = False
                        atoms_found_this_iteration[wipe_indicies] = 0
            if np.sum(atoms_found_this_iteration) == 0:
                found_atoms_in_prev_iteration = False
                print("stopping search")

            atoms_found_previous_iterations |= atoms_found_this_iteration.astype(bool)

            atoms_found_prev_iteration = atoms_found_this_iteration.copy()
            atoms_found_this_iteration = np.zeros(len(maxima_candidates))
            iteration_while += 1

        maxima_accepted_x = maxima_candidates_x[unique_ids[1, :] == 1]
        maxima_accepted_y = maxima_candidates_y[unique_ids[1, :] == 1]

        maxima_accepted_u = unique_ids[2, unique_ids[1, :] == 1]
        maxima_accepted_v = unique_ids[3, unique_ids[1, :] == 1]

        maxima_accepted_intensity = maxima_candidates_intensity[unique_ids[1, :] == 1]

        self.atoms = Vector.from_shape(
            shape=(self._num_sites),
            fields=("x", "y", "a", "b", "int_peak"),
            units=("px", "px", "ind", "ind", "counts"),
        )
        if not merge_dislocation or not check_for_dislocations:
            arr = np.vstack(
                (
                    maxima_accepted_x,
                    maxima_accepted_y,
                    maxima_accepted_u,
                    maxima_accepted_v,
                    maxima_accepted_intensity,
                )
            ).T
            self.atoms.set_data(arr, 0)

        if merge_dislocation and check_for_dislocations:
            maxima_merge_x = maxima_candidates_x[np.isin(unique_ids[1, :], [1, 3])]
            maxima_merge_y = maxima_candidates_y[np.isin(unique_ids[1, :], [1, 3])]

            maxima_merge_u = unique_ids[2, np.isin(unique_ids[1, :], [1, 3])]
            maxima_merge_v = unique_ids[3, np.isin(unique_ids[1, :], [1, 3])]

            maxima_merge_intensity = maxima_candidates_intensity[np.isin(unique_ids[1, :], [1, 3])]
            arr = np.vstack(
                (
                    maxima_merge_x,
                    maxima_merge_y,
                    maxima_merge_u,
                    maxima_merge_v,
                    maxima_merge_intensity,
                )
            ).T
            self.atoms.set_data(arr, 0)

        # second, a loop that uses these A sites to find all other sites
        found_atoms_in_prev_iteration = True
        while found_atoms_in_prev_iteration is True:
            for atom_index in range(len(maxima_candidates)):
                if (
                    not unique_ids[5, atom_index] == 0
                ):  # skip this 'for' iteration if the atom is not an A site
                    continue
                # print(unique_ids[5, atom_index])
                for a0 in range(self._num_sites - 1):
                    a0 += 1  # we don't need to go over the 0 index again
                    positions_around_A_site = self.get_xy_shifts(a0)
                    positions_around_A_site_norm = np.linalg.norm(positions_around_A_site, axis=1)
                    for pos_index, pos_vec in enumerate(positions_around_A_site):
                        position_x = pos_vec[0] + maxima_candidates_x[atom_index]
                        position_y = pos_vec[1] + maxima_candidates_y[atom_index]
                        radial_dist = (
                            (maxima_candidates_x - position_x) ** 2
                            + (maxima_candidates_y - position_y) ** 2
                        ) ** (0.5)
                        radial_dist[atom_index] = (
                            positions_around_A_site_norm[pos_index] * (tolerance_uv - 1) * 2
                        )  # make sure that self is outside of range
                        if (
                            radial_dist
                            < (positions_around_A_site_norm[pos_index] * (tolerance_uv - 1))
                        ).any():
                            successful_candidate_index = np.argmin(radial_dist)
                            if unique_ids[1, successful_candidate_index] == 0:
                                atoms_found_this_iteration[successful_candidate_index] += 1
                                unique_ids[1, successful_candidate_index] = 1
                                unique_ids[2, successful_candidate_index] = unique_ids[
                                    2, atom_index
                                ]
                                unique_ids[3, successful_candidate_index] = unique_ids[
                                    3, atom_index
                                ]
                                unique_ids[4, successful_candidate_index] = 0
                                unique_ids[5, successful_candidate_index] = a0
                # check if any atom was somehow still found twice:
                assert np.max(atoms_found_this_iteration) < 2
                # uv duplication check won't work as is. since our uv coordinates will have to move to a floating point for extra sites, we will need to use a threshold
            if np.sum(atoms_found_this_iteration) == 0:
                found_atoms_in_prev_iteration = False
                print("stopping search")

            atoms_found_previous_iterations |= atoms_found_this_iteration.astype(bool)
            atoms_found_prev_iteration = atoms_found_this_iteration.copy()
            atoms_found_this_iteration = np.zeros(len(maxima_candidates))
            iteration_while += 1

        if plot_atoms:
            fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)
            xs = maxima_accepted_x
            ys = maxima_accepted_y
            rgb = site_colors(int(self._numbers[0]))
            ax.scatter(
                ys,
                xs,
                s=18,
                facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
                edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                linewidths=0.75,
                marker="o",
                zorder=25,
            )
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        for a0 in range(self._num_sites - 1):
            a0 += 1
            mask_1 = unique_ids[1, :] == 1
            mask_2 = unique_ids[5, :] == a0
            mask = mask_1.astype(bool) & mask_2.astype(bool)
            maxima_accepted_x = maxima_candidates_x[mask]
            maxima_accepted_y = maxima_candidates_y[mask]
            maxima_accepted_u = unique_ids[2, mask]
            maxima_accepted_v = unique_ids[3, mask]
            maxima_accepted_intensity = maxima_candidates_intensity[mask]
            arr = np.vstack(
                (
                    maxima_accepted_x,
                    maxima_accepted_y,
                    maxima_accepted_u,
                    maxima_accepted_v,
                    maxima_accepted_intensity,
                )
            ).T
            self.atoms.set_data(arr, a0)

        if plot_atoms:
            fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)
            xs = maxima_accepted_x
            ys = maxima_accepted_y
            rgb = site_colors(int(self._numbers[0]))
            ax.scatter(
                ys,
                xs,
                s=18,
                facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
                edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                linewidths=0.75,
                marker="o",
                zorder=25,
            )
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        return self

    def atoms_first_uvw(
        self,
        origin=None,
        u=None,
        v=None,
        positions_frac=None,
        tolerance_uvw: float = 1.1,
        w=None,
        numbers=None,
        edge_min_dist_px=None,
        subpixel: str = "poly",
        upsample_factor: int = 16,
        sigma: float = 0,
        minAbsoluteIntensity: float = 0,
        minRelativeIntensity: float = 0,
        relativeToPeak: float = 0,
        minSpacing: float = 0,
        edgeBoundary: int = 1,
        maxNumPeaks: int = 5000,
        plot_atoms=True,
        input_mask=None,
        refine_lattice=True,
        refine_maxiter: int = 200,
        intensity_radius=None,
        intensity_min: float | None = None,
        contrast_min=None,
        annulus_radii=None,
        check_uv_duplication=True,
        check_for_dislocations=False,
        merge_dislocation=False,
        num_peaks_search=20,
        num_peaks_use=2,
        center_ignore_buffer=15,
        minSpacingPeaks=5,
        use_found_peaks_directly=False,
        tolerance_b=None,
        **kwargs,
    ):
        self.check_for_dislocations = check_for_dislocations and check_uv_duplication
        # find all candidates above threshold
        maxima_candidates = self.get_maxima_2D(
            self.image.array,
            subpixel=subpixel,
            upsample_factor=upsample_factor,
            sigma=sigma,
            minAbsoluteIntensity=minAbsoluteIntensity,
            minRelativeIntensity=minRelativeIntensity,
            relativeToPeak=relativeToPeak,
            minSpacing=minSpacing,
            edgeBoundary=edgeBoundary,
            maxNumPeaks=maxNumPeaks,
        )

        H, W = self._image.shape  # x=rows, y=cols

        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        show_2d(
            self._image.array,
            figax=(fig, ax),
            # scalebar = {
            #     'sampling':data_1['pixelSize'][1] * 2,
            #     'units':"nm",
            #     'length':4,
            #     'loc':3,
            #     # 'font_size': scalebar_fontsize,
            #     'width_px':20
            # },
            # lower_quantile = 0.23,
            cbar=True,
        )

        if origin is None:
            max_intensity_index = np.argmax(maxima_candidates[:]["intensity"])
            origin_x = maxima_candidates[max_intensity_index]["x"]
            origin_y = maxima_candidates[max_intensity_index]["y"]
            origin = np.array([origin_x, origin_y])

        if u is None or v is None:
            uv_result_inv = self.auto_peak_finder(
                num_peaks_search=num_peaks_search,
                num_peaks_use=num_peaks_use,
                center_ignore_buffer=center_ignore_buffer,
                minSpacingPeaks=minSpacingPeaks,
            )

            g_vector_1_c = np.array([uv_result_inv[0]["x"], uv_result_inv[0]["y"]])
            g_vector_2_c = np.array([uv_result_inv[1]["x"], uv_result_inv[1]["y"]])
            g_vec1 = np.zeros(2)
            g_vec1[0] = (g_vector_1_c[0] - (0.5 * H)) / H
            g_vec1[1] = (g_vector_1_c[1] - (0.5 * W)) / W
            g_vec2 = np.zeros(2)
            g_vec2[0] = (g_vector_2_c[0] - (0.5 * H)) / H
            g_vec2[1] = (g_vector_2_c[1] - (0.5 * W)) / W
            g_matrix = np.array([g_vec1, g_vec2])
            a_matrix = np.linalg.inv(g_matrix)
            a_transpose = a_matrix.T
            u = np.array([a_transpose[0, 0], a_transpose[0, 1]])
            v = np.array([a_transpose[1, 0], a_transpose[1, 1]])
            self.u = u
            self.v = v

        if positions_frac is None:
            positions_frac = np.atleast_2d(np.array((0, 0)))  # 1, 1
        # if (positions_frac[0] == np.array([0,0])).all():
        # positions_frac[0] = np.atleast_2d(np.array((1,1)))

        self._positions_frac = np.atleast_2d(np.array(positions_frac, dtype=float))
        self._num_sites = self._positions_frac.shape[0]
        if numbers is None:
            self._numbers = np.arange(1, self._num_sites + 1, dtype=int)
        else:
            self._numbers = np.atleast_1d(np.array(numbers, dtype=int))

        # print("numbers",self._numbers)
        # print("num sites",self._num_sites)

        if w is None:
            if np.abs(
                np.rad2deg(np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))
            ) > np.deg2rad(90):
                w = np.asarray(u) + np.asarray(v)
                w_sign = 1
            else:
                w = np.asarray(u) - np.asarray(v)
                w_sign = -1
        else:
            w_sign = 1

        self._lat = np.vstack(
            (
                np.array(origin),
                np.array(u),
                np.array(v),
            )
        )

        if tolerance_b is None:
            tolerance_b = tolerance_uvw

        im = np.asarray(self._image.array, dtype=float)
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        A = np.column_stack((u, v))

        def _auto_radius_px() -> float:
            S = self._positions_frac
            if S.shape[0] >= 2:
                d = S[:, None, :] - S[None, :, :]
                d = d - np.round(d)
                same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
                dpix = d @ A.T
                dist = np.linalg.norm(dpix, axis=2)
                dist[same] = np.inf
                nn = float(np.min(dist))
            else:
                nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
            if not np.isfinite(nn) or nn <= 0:
                nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
            return 0.5 * nn

        r_px = float(intensity_radius) if intensity_radius is not None else _auto_radius_px()
        rin, rout = (1.5 * r_px, 3.0 * r_px) if annulus_radii is None else annulus_radii
        R_disk = int(np.ceil(r_px))
        R_ring = int(np.ceil(rout))

        def mean_disk(x: float, y: float) -> float:
            ix0, iy0 = int(np.floor(x)), int(np.floor(y))
            i0, i1 = max(0, ix0 - R_disk), min(H - 1, ix0 + R_disk)
            j0, j1 = max(0, iy0 - R_disk), min(W - 1, iy0 + R_disk)
            ii = np.arange(i0, i1 + 1)[:, None]
            jj = np.arange(j0, j1 + 1)[None, :]
            dx, dy = ii - x, jj - y
            mask_circle = (dx * dx + dy * dy) <= (r_px * r_px)
            vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_circle]
            if vals.size == 0:
                return float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
            return float(vals.mean())

        def mean_std_annulus(x: float, y: float) -> tuple[float, float]:
            ix0, iy0 = int(np.floor(x)), int(np.floor(y))
            i0, i1 = max(0, ix0 - R_ring), min(H - 1, ix0 + R_ring)
            j0, j1 = max(0, iy0 - R_ring), min(W - 1, iy0 + R_ring)
            ii = np.arange(i0, i1 + 1)[:, None]
            jj = np.arange(j0, j1 + 1)[None, :]
            dx, dy = ii - x, jj - y
            r2 = dx * dx + dy * dy
            mask_ring = (r2 >= rin * rin) & (r2 <= rout * rout)
            vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_ring]
            if vals.size == 0:
                val = float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
                return val, 0.0
            return float(vals.mean()), float(vals.std(ddof=0))

        # mask of where in real space maxima can occur
        H, W = self._image.shape  # x=rows, y=cols
        edge_thresh = float(edge_min_dist_px) if edge_min_dist_px is not None else 0.0

        DT = None
        if input_mask is not None:
            m = np.asarray(input_mask).astype(bool)
            if m.shape != (H, W):
                raise ValueError(f"mask shape {m.shape} must match image shape {(H, W)}")
            try:
                from scipy.ndimage import distance_transform_edt

                DT = distance_transform_edt(m)
            except Exception:
                DT = None

        # find the maxima closest to the origin:
        maxima_candidates_x = maxima_candidates[:]["x"]
        maxima_candidates_y = maxima_candidates[:]["y"]

        pm_arr = np.array([-1, 1])
        # pm_arr = np.array([-1,0,1])
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        w_norm = np.linalg.norm(w)
        uvw_arr = np.array([np.asarray(u), np.asarray(v), np.asarray(w)])
        uvw_norm = 0.5 * (u_norm + v_norm + w_norm)
        self.uv_norm = uvw_norm
        self.uv_arr = uvw_arr
        self.tolerance_uv = tolerance_uvw
        x = maxima_candidates_x
        y = maxima_candidates_y

        in_bounds = (x >= 0.0) & (x <= H - 1) & (y >= 0.0) & (y <= W - 1)
        border_ok = (
            (x - edge_thresh >= 0.0)
            & (x + edge_thresh <= H - 1)
            & (y - edge_thresh >= 0.0)
            & (y + edge_thresh <= W - 1)
        )
        if input_mask is not None:
            if DT is not None:
                ii = np.clip(np.round(x).astype(int), 0, H - 1)
                jj = np.clip(np.round(y).astype(int), 0, W - 1)
                mask_ok = DT[ii, jj] >= edge_thresh
            else:
                m = np.asarray(input_mask).astype(bool)
                mask_ok = m[
                    np.clip(np.round(x).astype(int), 0, H - 1),
                    np.clip(np.round(y).astype(int), 0, W - 1),
                ]
        else:
            mask_ok = np.ones_like(in_bounds, dtype=bool)

        int_center = np.empty(x.shape[0], dtype=float)
        for i in range(x.shape[0]):
            int_center[i] = mean_disk(x[i], y[i])

        keep = in_bounds & border_ok & mask_ok
        if intensity_min is not None:
            keep &= int_center >= float(intensity_min)
        if contrast_min is not None:
            bg_mean = np.empty(x.shape[0], dtype=float)
            for i in range(x.shape[0]):
                bg_mean[i], _ = mean_std_annulus(x[i], y[i])
            keep &= (int_center - bg_mean) >= float(contrast_min)

        if np.any(keep):
            maxima_candidates = maxima_candidates[keep]
        else:
            raise ValueError("Zero maxima candidates kept")

        # find the maxima closest to the origin:
        maxima_candidates_x = maxima_candidates[:]["x"]
        maxima_candidates_y = maxima_candidates[:]["y"]
        maxima_candidates_intensity = maxima_candidates[:]["intensity"]

        # the unique ids array is an array of the original index, candidacy, a (of a * u), and b (of b * v), and c (of c * w)
        unique_ids = np.zeros([6, len(maxima_candidates)])
        unique_ids[0, :] = np.arange(0, len(maxima_candidates))
        unique_ids[4, :] = -1 * np.arange(1, 1 + len(maxima_candidates))
        unique_ids[5, :] -= 1

        if plot_atoms:
            fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)
            xs = maxima_candidates_x
            ys = maxima_candidates_y
            rgb = site_colors(int(self._numbers[0]))
            ax.scatter(
                ys,
                xs,
                s=18,
                facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
                edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                linewidths=0.75,
                marker="o",
                zorder=25,
            )
            ax.scatter(origin[1], origin[0], c="red", marker="x", s=80)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        if use_found_peaks_directly:
            self.atoms = Vector.from_shape(
                shape=(self._num_sites),
                fields=("x", "y", "a", "b", "int_peak"),
                units=("px", "px", "ind", "ind", "counts"),
            )
            maxima_accepted_x = maxima_candidates_x
            maxima_accepted_y = maxima_candidates_y
            maxima_accepted_u = np.zeros_like(maxima_accepted_x)
            maxima_accepted_v = np.zeros_like(maxima_accepted_x)
            maxima_accepted_intensity = maxima_candidates_intensity
            arr = np.vstack(
                (
                    maxima_accepted_x,
                    maxima_accepted_y,
                    maxima_accepted_u,
                    maxima_accepted_v,
                    maxima_accepted_intensity,
                )
            ).T
            self.atoms.set_data(arr, 0)
            return self

        radial_dist = (
            (maxima_candidates_x - origin[0]) ** 2 + (maxima_candidates_y - origin[1]) ** 2
        ) ** (0.5)
        origin_candidate_index = np.argmin(
            radial_dist
        )  # use the first minima, if there are multiple
        unique_ids[1, origin_candidate_index] = 1
        unique_ids[4, origin_candidate_index] = 0

        atoms_found_this_iteration = np.zeros(len(maxima_candidates))
        atoms_found_prev_iteration = np.zeros(len(maxima_candidates))
        atoms_found_previous_iterations = np.zeros(len(maxima_candidates), dtype=bool)
        atoms_found_prev_iteration[origin_candidate_index] = 1
        found_atoms_in_prev_iteration = True
        iteration_while = 0

        def check_dislocations():
            if check_for_dislocations:
                for atom_index in range(len(maxima_candidates)):
                    if unique_ids[1, atom_index] == 1:
                        for pm in pm_arr:
                            for uvw_index, lat_vec in enumerate(uvw_arr):
                                position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
                                position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
                                radial_dist = (
                                    (maxima_candidates_x - position_x) ** 2
                                    + (maxima_candidates_y - position_y) ** 2
                                ) ** (0.5)
                                radial_dist[atom_index] = (
                                    uvw_norm * (tolerance_uvw - 1) * 2
                                )  # make sure that self is outside of range
                                if (radial_dist < (uvw_norm * (tolerance_uvw - 1))).any():
                                    successful_candidate_index = np.argmin(radial_dist)
                                    if unique_ids[1, successful_candidate_index] == 2:
                                        atoms_found_this_iteration[successful_candidate_index] += 1
                                        unique_ids[1, successful_candidate_index] = (
                                            3  # for being found in dislocation search
                                        )
                                        unique_ids[2, successful_candidate_index] = (
                                            unique_ids[2, atom_index]
                                            + pm * int(uvw_index == 0)
                                            + pm * int(uvw_index == 2)
                                        )
                                        unique_ids[3, successful_candidate_index] = (
                                            unique_ids[3, atom_index]
                                            + pm * int(uvw_index == 1)
                                            - w_sign * pm * int(uvw_index == 2)
                                        )
                                        unique_ids[4, successful_candidate_index] = 0
                maxima_dislocation_x = maxima_candidates_x[unique_ids[1, :] == 3]
                maxima_dislocation_y = maxima_candidates_y[unique_ids[1, :] == 3]
                maxima_dislocation_u = unique_ids[2, unique_ids[1, :] == 3]
                maxima_dislocation_v = unique_ids[3, unique_ids[1, :] == 3]
                maxima_dislocation_intensity = maxima_candidates_intensity[unique_ids[1, :] == 3]
                arr = np.vstack(
                    (
                        maxima_dislocation_x,
                        maxima_dislocation_y,
                        maxima_dislocation_u,
                        maxima_dislocation_v,
                        maxima_dislocation_intensity,
                    )
                ).T
                return arr

        # first, a loop that finds all of the A sites
        a0 = 0  # here we are just doing a0
        while found_atoms_in_prev_iteration is True:
            for atom_index in range(len(maxima_candidates)):
                if atoms_found_prev_iteration[atom_index] > 0:
                    for pm in pm_arr:
                        for uvw_index, lat_vec in enumerate(uvw_arr):
                            position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
                            position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
                            radial_dist = (
                                (maxima_candidates_x - position_x) ** 2
                                + (maxima_candidates_y - position_y) ** 2
                            ) ** (0.5)
                            radial_dist[atom_index] = (
                                np.inf
                            )  # uvw_norm * (tolerance_uvw - 1) * 2 # make sure that self is outside of range
                            if (radial_dist < (uvw_norm * (tolerance_uvw - 1))).any():
                                successful_candidate_index = np.argmin(radial_dist)
                                # print('threshold',(uvw_norm * (tolerance_uvw - 1)))
                                # print('uvw_norm',(uvw_norm))
                                # print('tolerance_uvw',(tolerance_uvw))
                                # print('radial distance of successful candiate',radial_dist[successful_candidate_index])
                                if unique_ids[1, successful_candidate_index] == 0:
                                    atoms_found_this_iteration[successful_candidate_index] += 1
                                    unique_ids[1, successful_candidate_index] = 1
                                    unique_ids[2, successful_candidate_index] = (
                                        unique_ids[2, atom_index]
                                        + pm * int(uvw_index == 0)
                                        + pm * int(uvw_index == 2)
                                    )
                                    unique_ids[3, successful_candidate_index] = (
                                        unique_ids[3, atom_index]
                                        + pm * int(uvw_index == 1)
                                        + w_sign * pm * int(uvw_index == 2)
                                    )
                                    unique_ids[4, successful_candidate_index] = 0
                                    unique_ids[5, successful_candidate_index] = a0
            # check if any atom was somehow still found twice:
            assert np.max(atoms_found_this_iteration) < 2
            # check if any found atoms have the same uv index
            if check_uv_duplication:
                uv_pairs = unique_ids[1:6, :].T
                unique_pairs, inverse, counts = np.unique(
                    uv_pairs, axis=0, return_inverse=True, return_counts=True
                )
                duplicate_groups = [
                    np.where(inverse == k)[0] for k, c in enumerate(counts) if c > 1
                ]
                mask_atoms_found = atoms_found_this_iteration.astype(bool)
                if len(duplicate_groups) != 0:
                    for duplicate_group in duplicate_groups:
                        duplicate_group = np.asarray(duplicate_group)
                        duplicate_atoms_index_found_previous_iterations = duplicate_group[
                            atoms_found_previous_iterations[duplicate_group]
                        ]
                        if duplicate_atoms_index_found_previous_iterations.size > 1:
                            if origin_candidate_index not in duplicate_group:
                                raise ValueError(
                                    "The duplicate atoms finding code is somehow bugged"
                                )
                            else:
                                kept_index = origin_candidate_index
                        elif duplicate_atoms_index_found_previous_iterations.size == 1:
                            kept_index = duplicate_atoms_index_found_previous_iterations
                        else:
                            kept_index = duplicate_group[mask_atoms_found[duplicate_group]][0]
                        wipe_indicies = duplicate_group[duplicate_group != kept_index]
                        unique_ids[1, wipe_indicies] = (
                            2  # this signals to not accept for this maxima anymore (and flags this as a dulpicate)
                        )
                        unique_ids[2:4, wipe_indicies] = 0
                        unique_ids[5, wipe_indicies] = -1
                        unique_ids[4, wipe_indicies] = -1 * (wipe_indicies + 1)
                        atoms_found_previous_iterations[wipe_indicies] = False
                        mask_atoms_found[wipe_indicies] = False
                        atoms_found_this_iteration[wipe_indicies] = 0
            if np.sum(atoms_found_this_iteration) == 0:
                found_atoms_in_prev_iteration = False
                # print('stopping search')

            atoms_found_previous_iterations |= atoms_found_this_iteration.astype(bool)

            atoms_found_prev_iteration = atoms_found_this_iteration.copy()
            atoms_found_this_iteration = np.zeros(len(maxima_candidates))
            iteration_while += 1
        if check_for_dislocations:
            atom_arr = check_dislocations()
            self.atoms_dislocation = Vector.from_shape(
                shape=(self._num_sites),
                fields=("x", "y", "a", "b", "int_peak"),
                units=("px", "px", "ind", "ind", "counts"),
            )
            self.atoms_dislocation.set_data(atom_arr, 0)

        # add interactive bit here

        maxima_accepted_x = maxima_candidates_x[unique_ids[1, :] == 1]
        maxima_accepted_y = maxima_candidates_y[unique_ids[1, :] == 1]

        maxima_accepted_u = unique_ids[2, unique_ids[1, :] == 1]
        maxima_accepted_v = unique_ids[3, unique_ids[1, :] == 1]

        maxima_accepted_intensity = maxima_candidates_intensity[unique_ids[1, :] == 1]

        self.atoms = Vector.from_shape(
            shape=(self._num_sites),
            fields=("x", "y", "a", "b", "int_peak"),
            units=("px", "px", "ind", "ind", "counts"),
        )

        if not merge_dislocation or not check_for_dislocations:
            arr = np.vstack(
                (
                    maxima_accepted_x,
                    maxima_accepted_y,
                    maxima_accepted_u,
                    maxima_accepted_v,
                    maxima_accepted_intensity,
                )
            ).T
            self.atoms.set_data(arr, 0)

        if merge_dislocation and check_for_dislocations:
            maxima_merge_x = maxima_candidates_x[np.isin(unique_ids[1, :], [1, 3])]
            maxima_merge_y = maxima_candidates_y[np.isin(unique_ids[1, :], [1, 3])]

            maxima_merge_u = unique_ids[2, np.isin(unique_ids[1, :], [1, 3])]
            maxima_merge_v = unique_ids[3, np.isin(unique_ids[1, :], [1, 3])]

            maxima_merge_intensity = maxima_candidates_intensity[np.isin(unique_ids[1, :], [1, 3])]
            arr = np.vstack(
                (
                    maxima_merge_x,
                    maxima_merge_y,
                    maxima_merge_u,
                    maxima_merge_v,
                    maxima_merge_intensity,
                )
            ).T
            self.atoms.set_data(arr, 0)

        # second, a loop that uses these A sites to find all other sites
        found_atoms_in_prev_iteration = True
        while found_atoms_in_prev_iteration is True:
            for atom_index in range(len(maxima_candidates)):
                if (
                    not unique_ids[5, atom_index] == 0
                ):  # skip this 'for' iteration if the atom is not an A site
                    continue
                # print(unique_ids[5, atom_index])
                for a0 in range(self._num_sites - 1):
                    a0 += 1  # we don't need to go over the 0 index again
                    positions_around_A_site = self.get_xy_shifts(a0)
                    positions_around_A_site_norm = np.linalg.norm(positions_around_A_site, axis=1)
                    for pos_index, pos_vec in enumerate(positions_around_A_site):
                        position_x = pos_vec[0] + maxima_candidates_x[atom_index]
                        position_y = pos_vec[1] + maxima_candidates_y[atom_index]
                        radial_dist = (
                            (maxima_candidates_x - position_x) ** 2
                            + (maxima_candidates_y - position_y) ** 2
                        ) ** (0.5)
                        radial_dist[atom_index] = (
                            positions_around_A_site_norm[pos_index] * (tolerance_b - 1) * 2
                        )  # make sure that self is outside of range
                        if (
                            radial_dist
                            < (positions_around_A_site_norm[pos_index] * (tolerance_b - 1))
                        ).any():
                            successful_candidate_index = np.argmin(radial_dist)
                            if unique_ids[1, successful_candidate_index] == 0:
                                atoms_found_this_iteration[successful_candidate_index] += 1
                                unique_ids[1, successful_candidate_index] = 1
                                unique_ids[2, successful_candidate_index] = unique_ids[
                                    2, atom_index
                                ]
                                unique_ids[3, successful_candidate_index] = unique_ids[
                                    3, atom_index
                                ]
                                unique_ids[4, successful_candidate_index] = 0
                                unique_ids[5, successful_candidate_index] = a0
                # check if any atom was somehow still found twice:
                assert np.max(atoms_found_this_iteration) < 2
                # uv duplication check won't work as is. since our uv coordinates will have to move to a floating point for extra sites, we will need to use a threshold
            if np.sum(atoms_found_this_iteration) == 0:
                found_atoms_in_prev_iteration = False
                # print('stopping search')

            atoms_found_previous_iterations |= atoms_found_this_iteration.astype(bool)

            atoms_found_prev_iteration = atoms_found_this_iteration.copy()
            atoms_found_this_iteration = np.zeros(len(maxima_candidates))
            iteration_while += 1

        # if plot_atoms:
        #     fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
        #     if ax.images:
        #         ax.images[-1].set_zorder(0)
        #     xs = maxima_accepted_x
        #     ys = maxima_accepted_y
        #     rgb = site_colors(int(self._numbers[0]))
        #     ax.scatter(
        #         ys,
        #         xs,
        #         s=18,
        #         facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
        #         edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
        #         linewidths=0.75,
        #         marker="o",
        #         zorder=25,
        #     )
        #     ax.set_xlim(0, W)
        #     ax.set_ylim(H, 0)

        for a0 in range(self._num_sites - 1):
            a0 += 1
            mask_1 = unique_ids[1, :] == 1
            mask_2 = unique_ids[5, :] == a0
            mask = mask_1.astype(bool) & mask_2.astype(bool)
            maxima_accepted_x = maxima_candidates_x[mask]
            maxima_accepted_y = maxima_candidates_y[mask]

            maxima_accepted_u = unique_ids[2, mask]
            maxima_accepted_v = unique_ids[3, mask]

            maxima_accepted_intensity = maxima_candidates_intensity[mask]

            arr = np.vstack(
                (
                    maxima_accepted_x,
                    maxima_accepted_y,
                    maxima_accepted_u,
                    maxima_accepted_v,
                    maxima_accepted_intensity,
                )
            ).T
            self.atoms.set_data(arr, a0)

        if plot_atoms:
            fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)
            for a0 in range(self._num_sites):
                atoms_arr = self.atoms.get_data(a0)
                xs = atoms_arr[:, 0]
                ys = atoms_arr[:, 1]
                # xs = maxima_accepted_x
                # ys = maxima_accepted_y
                rgb = site_colors(int(self._numbers[0] + 2 * a0))
                ax.scatter(
                    ys,
                    xs,
                    s=200 * (a0 + 1),
                    facecolor=(rgb[0], rgb[1], rgb[2], 0.85),
                    edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                    linewidths=0.75,
                    marker="o",
                    zorder=25,
                )
            ax.scatter(origin[1], origin[0], c="red", marker="x", s=80)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        return self

    def find_correct_b(
        self,
    ):
        return 0

    def auto_find_b_frac_orientation(
        self,
        num_b_per_uc=1,
        order=1,
        interpolate_intensity=True,
        avg_inside_radius=False,
        max_inside_radius=False,
        fit_guassian=False,
        radius=4,
        max_shift_gauss=3,
        dedup_cutoff_px=5,
        plot_atoms=True,
        print_message=True,
    ):
        def generate_frac_variants(frac, tol=1e-6):
            import itertools

            frac = np.asarray(frac, dtype=float)
            variants = []
            for perm in set(itertools.permutations(frac)):
                for signs in itertools.product([1, -1], repeat=2):
                    v = np.array(perm) * np.array(signs)
                    variants.append(v)
            unique = []
            for v in variants:
                if not any(np.allclose(v, u, atol=tol) for u in unique):
                    unique.append(v)
            return np.array(unique)

        candidates = generate_frac_variants(self._positions_frac[1])

        intensities = np.zeros([candidates.shape[0]])
        frac_ind = 0
        for frac in candidates:
            self._positions_frac[1] = frac
            self.measure_b_intensity_near_a(
                num_b_per_uc=num_b_per_uc,
                order=order,
                interpolate_intensity=interpolate_intensity,
                avg_inside_radius=avg_inside_radius,
                max_inside_radius=max_inside_radius,
                fit_guassian=fit_guassian,
                radius=radius,
                max_shift_gauss=max_shift_gauss,
                dedup_cutoff_px=dedup_cutoff_px,
                plot_atoms=plot_atoms,
            )

            intensities[frac_ind] = np.sum(self.bsites_assume[:, 0])
            frac_ind += 1

        best_index = np.argmax(intensities)

        if print_message:
            print(
                "Found orientation of b site. Setting position fraction to", candidates[best_index]
            )
        self._positions_frac[1] = candidates[best_index]

        return self

    def measure_b_intensity_near_a(
        self,
        num_b_per_uc=1,
        order=1,
        interpolate_intensity=True,
        avg_inside_radius=False,
        max_inside_radius=False,
        fit_guassian=False,
        radius=4,
        max_shift_gauss=3,
        dedup_cutoff_px=5,
        plot_atoms=True,
        title="",
        **kwargs,
    ):
        # going to assume that there is a b site near the a sites.
        # i am just going to measure the intensity of the b sites
        # it is important to avoid double counting the sites
        # in the case that i am writing this for, the basis consists of two atoms
        # this means that i only have to check for the b site in one location for every a site
        # nominally all a sites are present, but they could also not be present.
        # in case the a sites are not present, i could try doing it from multiple directions and then keeping sites only once where at least one b site turned up.
        from scipy.ndimage import map_coordinates
        from scipy.optimize import least_squares
        from scipy.spatial import cKDTree

        def extract_circular_roi(image, x0, y0, radius):
            x_min = int(np.floor(x0 - radius))
            x_max = int(np.floor(x0 + radius + 1))
            y_min = int(np.floor(y0 - radius))
            y_max = int(np.floor(y0 + radius + 1))

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)

            roi = image[x_min:x_max, y_min:y_max]

            x = np.arange(x_min, x_max)
            y = np.arange(y_min, y_max)

            xx, yy = np.meshgrid(x, y, indexing="ij")

            rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

            mask = rr <= radius

            return roi, mask, xx, yy

        def gaussian_2d(params, x, y):
            amp, x0, y0, sigma, offset = params
            return amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)) + offset

        def fit_gaussian_local(image, x_init, y_init, radius, max_shift_px):
            roi, mask, xx, yy = extract_circular_roi(image, x_init, y_init, radius)

            z = roi[mask]
            x = xx[mask]
            y = yy[mask]
            amp0 = z.max() - z.min()
            offset0 = z.min()
            sigma0 = radius / 2

            p0 = [amp0, x_init, y_init, sigma0, offset0]

            bounds = (
                [0, x_init - max_shift_px, y_init - max_shift_px, 0.5, -np.inf],
                [np.inf, x_init + max_shift_px, y_init + max_shift_px, radius, np.inf],
            )

            def residuals(p):
                return gaussian_2d(p, x, y) - z

            res = least_squares(residuals, p0, bounds=bounds)

            return res.x, res.cost

        atoms_arr = self.atoms.get_data(0)
        a_x = atoms_arr[:, 0]
        a_y = atoms_arr[:, 1]
        positions_around_A_site = self.get_xy_shifts(
            1
        )  # this function is just for B sites right now, so hard coding this 1 (zero indexed)
        positions_per_uc = positions_around_A_site[:num_b_per_uc]

        n_A = a_x.shape[0]
        n_B = num_b_per_uc

        bsite_data = np.full((n_B, n_A, 3), np.nan, dtype=float)
        H, W = self._image.shape  # x=rows, y=cols

        for atom_index in range(a_x.shape[0]):
            for pos_index, pos_vec in enumerate(positions_per_uc):
                position_x = pos_vec[0] + a_x[atom_index]
                position_y = pos_vec[1] + a_y[atom_index]

                if not (0 <= np.round(position_x) < H and 0 <= np.round(position_y) < W):
                    continue
                if interpolate_intensity:
                    intensity = map_coordinates(
                        self._image.array,
                        [[position_x], [position_y]],
                        order=order,
                        mode="nearest",
                    )[0]
                elif avg_inside_radius:
                    roi, mask, _, _ = extract_circular_roi(
                        self._image.array, position_x, position_y, radius
                    )
                    intensity = roi[mask].mean()
                elif max_inside_radius:
                    roi, mask, xx, yy = extract_circular_roi(
                        self._image.array, position_x, position_y, radius
                    )
                    idx = np.argmax(roi[mask])
                    intensity_a = roi[mask]
                    xs_a = xx[mask]
                    ys_a = yy[mask]
                    intensity = intensity_a[idx]
                    xs = xs_a[idx]
                    ys = ys_a[idx]
                elif fit_guassian:
                    gauss_params, fit_cost = fit_gaussian_local(
                        self._image.array,
                        position_x,
                        position_y,
                        radius=radius,
                        max_shift_px=max_shift_gauss,
                    )
                    intensity_less_offset, position_x, position_y, sigma, offset = gauss_params
                    intensity = intensity_less_offset + offset
                else:
                    intensity = self._image.array[
                        np.round(position_x).astype(int), np.round(position_y).astype(int)
                    ]
                bsite_data[pos_index, atom_index, :] = intensity, position_x, position_y

        def deduplicate_positions(positions, cutoff_px):
            tree = cKDTree(positions)
            pairs = tree.query_pairs(cutoff_px)

            parent = np.arange(len(positions))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            def union(i, j):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

            for i, j in pairs:
                union(i, j)

            clusters = {}
            for i in range(len(positions)):
                r = find(i)
                clusters.setdefault(r, []).append(i)
            return np.array([positions[idxs].mean(axis=0) for idxs in clusters.values()])

        bsite_flat = bsite_data.reshape(-1, 3)
        mask = np.isfinite(bsite_flat[:, 0])
        bsite_flat_valid = bsite_flat[mask]

        positions = bsite_flat_valid[:, 1:3]  # (N, 2)
        intensities = bsite_flat_valid[:, 0]

        dedup_positions = deduplicate_positions(positions, cutoff_px=dedup_cutoff_px)

        tree = cKDTree(positions)

        unique_data = np.zeros((len(dedup_positions), 3))

        for i, pos in enumerate(dedup_positions):
            idxs = tree.query_ball_point(pos, dedup_cutoff_px)
            best = idxs[np.argmax(intensities[idxs])]

            unique_data[i, 0] = intensities[best]
            unique_data[i, 1:] = positions[best]

        # save the result
        self.bsites_assume = unique_data

        if plot_atoms:
            fig, ax = show_2d(self._image.array, figsize=(10, 10), returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)

            atoms_arr = self.atoms.get_data(0)
            xs = atoms_arr[:, 0]
            ys = atoms_arr[:, 1]
            rgb = site_colors(int(self._numbers[0]))
            ax.scatter(
                ys,
                xs,
                s=200,
                facecolor=(rgb[0], rgb[1], rgb[2], 0.85),
                edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                linewidths=0.75,
                marker="o",
                zorder=25,
            )

            rgb = site_colors(int(self._numbers[0] + 2))
            ax.scatter(
                unique_data[:, 2],
                unique_data[:, 1],
                s=200,
                facecolor=(rgb[0], rgb[1], rgb[2], 0.05),
                edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                linewidths=0.75,
                marker="o",
                zorder=25,
            )
            ax.set_title(title)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        return self

    def measure_intensity_input_image(
        self,
        input_image,
        order=1,
        interpolate_intensity=False,
        avg_inside_radius=False,
        max_inside_radius=False,
        fit_guassian=True,
        radius=5,
        max_shift_gauss=2,
        plot_atoms=True,
        title="",
        **kwargs,
    ):
        from scipy.ndimage import map_coordinates
        from scipy.optimize import least_squares

        self.input_image = input_image

        def extract_circular_roi(image, x0, y0, radius):
            x_min = int(np.floor(x0 - radius))
            x_max = int(np.floor(x0 + radius + 1))
            y_min = int(np.floor(y0 - radius))
            y_max = int(np.floor(y0 + radius + 1))

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)

            roi = image[x_min:x_max, y_min:y_max]

            x = np.arange(x_min, x_max)
            y = np.arange(y_min, y_max)

            xx, yy = np.meshgrid(x, y, indexing="ij")

            rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

            mask = rr <= radius

            return roi, mask, xx, yy

        def gaussian_2d(params, x, y):
            amp, x0, y0, sigma, offset = params
            return amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)) + offset

        def fit_gaussian_local(image, x_init, y_init, radius, max_shift_px):
            roi, mask, xx, yy = extract_circular_roi(image, x_init, y_init, radius)

            z = roi[mask]
            x = xx[mask]
            y = yy[mask]
            amp0 = z.max() - z.min()
            offset0 = z.min()
            sigma0 = radius / 2

            p0 = [amp0, x_init, y_init, sigma0, offset0]

            bounds = (
                [0, x_init - max_shift_px, y_init - max_shift_px, 0.5, -np.inf],
                [np.inf, x_init + max_shift_px, y_init + max_shift_px, radius, np.inf],
            )

            def residuals(p):
                return gaussian_2d(p, x, y) - z

            res = least_squares(residuals, p0, bounds=bounds)

            return res.x, res.cost

        atoms_arr = self.atoms.get_data(0)
        a_x = atoms_arr[:, 0]
        a_y = atoms_arr[:, 1]

        b_x = self.bsites_assume[:, 1]
        b_y = self.bsites_assume[:, 2]
        # positions_around_A_site = self.get_xy_shifts(1) # this function is just for B sites right now, so hard coding this 1 (zero indexed)
        # positions_per_uc = positions_around_A_site[:num_b_per_uc]
        # positions_around_A_site_norm = np.linalg.norm(positions_around_A_site, axis = 1)

        n_A = a_x.shape[0]
        n_B = b_x.shape[0]

        H, W = self._image.shape  # x=rows, y=cols
        a_int_input = np.zeros(n_A)
        b_int_input = np.zeros(n_B)

        for atom_index in range(a_x.shape[0]):
            position_x = a_x[atom_index]
            position_y = a_y[atom_index]

            if not (0 <= np.round(position_x) < H and 0 <= np.round(position_y) < W):
                continue
            if interpolate_intensity:
                intensity = map_coordinates(
                    input_image, [[position_x], [position_y]], order=order, mode="nearest"
                )[0]
            elif avg_inside_radius:
                roi, mask, _, _ = extract_circular_roi(input_image, position_x, position_y, radius)
                intensity = roi[mask].mean()
            elif max_inside_radius:
                roi, mask, xx, yy = extract_circular_roi(
                    input_image, position_x, position_y, radius
                )
                idx = np.argmax(roi[mask])
                intensity_a = roi[mask]
                intensity = intensity_a[idx]
            elif fit_guassian:
                gauss_params, fit_cost = fit_gaussian_local(
                    self._image.array,
                    position_x,
                    position_y,
                    radius=radius,
                    max_shift_px=max_shift_gauss,
                )
                intensity_less_offset, position_x, position_y, sigma, offset = gauss_params
                intensity = intensity_less_offset + offset
            else:
                intensity = self._image.array[
                    np.round(position_x).astype(int), np.round(position_y).astype(int)
                ]
            a_int_input[atom_index] = intensity

        for atom_index in range(b_x.shape[0]):
            position_x = b_x[atom_index]
            position_y = b_y[atom_index]

            if not (0 <= np.round(position_x) < H and 0 <= np.round(position_y) < W):
                continue
            if interpolate_intensity:
                intensity = map_coordinates(
                    input_image, [[position_x], [position_y]], order=order, mode="nearest"
                )[0]
            elif avg_inside_radius:
                roi, mask, _, _ = extract_circular_roi(input_image, position_x, position_y, radius)
                intensity = roi[mask].mean()
            elif max_inside_radius:
                roi, mask, xx, yy = extract_circular_roi(
                    input_image, position_x, position_y, radius
                )
                idx = np.argmax(roi[mask])
                intensity_a = roi[mask]
                intensity = intensity_a[idx]
            elif fit_guassian:
                gauss_params, fit_cost = fit_gaussian_local(
                    self._image.array,
                    position_x,
                    position_y,
                    radius=radius,
                    max_shift_px=max_shift_gauss,
                )
                intensity_less_offset, position_x, position_y, sigma, offset = gauss_params
                intensity = intensity_less_offset + offset
            else:
                intensity = self._image.array[
                    np.round(position_x).astype(int), np.round(position_y).astype(int)
                ]
            b_int_input[atom_index] = intensity

        if plot_atoms:
            fig, ax = show_2d(input_image, figsize=(10, 10), returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)

            rgb = site_colors(int(self._numbers[0]))
            ax.scatter(
                a_y,
                a_x,
                s=200,
                facecolor=(rgb[0], rgb[1], rgb[2], 0.85),
                edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                linewidths=0.75,
                marker="o",
                zorder=25,
            )

            rgb = site_colors(int(self._numbers[0] + 2))
            ax.scatter(
                b_y,
                b_x,
                s=300,
                facecolor=(rgb[0], rgb[1], rgb[2], 0.01),
                edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                linewidths=0.75,
                marker="o",
                zorder=25,
            )
            ax.set_title(title)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        self.input_image_a_int = a_int_input
        self.input_image_b_int = b_int_input

        return self

    def get_ab_positions(
        self,
    ):
        positions_b = self.bsites_assume[:, 1:]
        a_x = self.atoms[0]["x"]
        a_y = self.atoms[0]["y"]
        positions_a = np.column_stack((a_x, a_y))
        return positions_a, positions_b

    def bsites_assume_preliminary(
        self,
    ):
        # get the A sites
        a_x = self.atoms[0]["x"]
        a_y = self.atoms[0]["y"]
        a_int = self.atoms[0]["int_peak"]

        # bsite_data[pos_index, atom_index, :] = intensity, position_x, position_y
        # self.bsites_assume = unique_data
        b_int = self.bsites_assume[:, 0]
        b_x = self.bsites_assume[:, 1]
        b_y = self.bsites_assume[:, 2]

        plt.figure(figsize=(5, 5), dpi=300)
        plt.plot(a_int, c="#4281f5", label="A intensities")
        plt.plot(np.sort(b_int), c="#ef42f5", label="B intensities")
        plt.legend()

        a_int_normalized = a_int.copy() / np.max(a_int)
        b_int_normalized = b_int.copy() / np.max(a_int)  # normalizing by A

        plt.figure(figsize=(5, 5), dpi=300)
        plt.plot(a_int_normalized, c="#4281f5", label="A intensities")
        plt.plot(np.sort(b_int_normalized), c="#ef42f5", label="B intensities")
        plt.legend()

        n_bins = 50
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins = (bin_edges[:-1] + bin_edges[1:]) / 2

        # plt.figure(figsize = (5,5), dpi = 300)
        # plt.hist(a_hist, bins)#, color = '#4281f5')
        # plt.title('A Site Histogram')
        # # plt.legend()
        # plt.figure(figsize = (5,5), dpi = 300)
        # plt.hist(b_hist, bins)#, color = '#ef42f5')
        # plt.title('B Site Histogram')
        # # plt.legend()
        plt.figure(figsize=(5, 5), dpi=300)
        plt.hist(a_int_normalized, bins)  # , color = '#4281f5')
        plt.title("A Site Histogram")
        # plt.legend()
        plt.figure(figsize=(5, 5), dpi=300)
        plt.hist(b_int_normalized, bins)  # , color = '#ef42f5')
        plt.title("B Site Histogram")
        plt.yscale("log")
        # plt.legend()

        plt.figure(figsize=(5, 5), dpi=300)
        plt.scatter(a_y, a_x, c=a_int_normalized, cmap="viridis", label="A sites")
        plt.scatter(b_y, b_x, c=b_int_normalized, cmap="magma", label="B sites")
        plt.colorbar()
        plt.ylim([np.max(a_y) + 20, -20])

        return self

    # also I think it is necessary to have a function that compares the intensity of a B site to that of its neighboring A sites.
    # ohh new idea as well. Now that I have the positions of the atoms, I can go back and measure the intensity over the original image.

    def delta_intensities_assume(
        self,
        uc_val=2,
        delta_input_cutoff=None,
        plot_atoms=False,
    ):
        from scipy.spatial import cKDTree

        neighbor_cutoff_pix = uc_val * self.uv_norm
        #
        positions_b = self.bsites_assume[:, 1:]
        b_int = self.bsites_assume[:, 0]
        a_x = self.atoms[0]["x"]
        a_y = self.atoms[0]["y"]
        a_int = self.atoms[0]["int_peak"]
        b_int /= np.max(a_int)
        a_int /= np.max(a_int)
        positions_a = np.column_stack((a_x, a_y))

        positions_all = np.concatenate([positions_a, positions_b])
        num_a = positions_a.shape[0]
        num_b = positions_b.shape[0]

        tree = cKDTree(positions_all)

        delta_data = np.zeros((num_b, 5))

        for i, pos in enumerate(positions_b):
            idxs = np.asarray(tree.query_ball_point(pos, neighbor_cutoff_pix))
            idxs_a = idxs[idxs < num_a]
            idxs_b = idxs[idxs > num_a]

            a_neighbor_intensities = a_int[idxs_a]
            median_a_intensity = np.median(a_neighbor_intensities)
            b_neighbor_intensities = b_int[idxs_b - num_a]
            median_b_intensity = np.median(b_neighbor_intensities)
            all_neighbor_intensities = np.concatenate(
                [a_neighbor_intensities, b_neighbor_intensities]
            )
            median_all_intensity = np.median(all_neighbor_intensities)

            delta_data[i, 0] = b_int[i] - median_a_intensity  # the delta intensity
            delta_data[i, 1] = b_int[i] - median_b_intensity  # the delta intensity
            delta_data[i, 2] = b_int[i] - median_all_intensity  # the delta intensity
            delta_data[i, 3] = idxs_a.shape[0]  # number of neighbors used
            delta_data[i, 4] = idxs_b.shape[0]  # number of neighbors used

        self.delta_assume = delta_data

        # delta_intensities, num_neighbors = lattice.intensity_neighborhood(neighborhood_units = neighborhood_units, return_delta = True)

        plt.figure(figsize=(24, 7), dpi=300)
        # plt.rcParams['font.family'] = 'serif'
        # params = {'mathtext.default': 'regular' }
        # plt.rcParams.update(params)
        plt.subplot(141)
        plt.scatter(self.delta_assume[:, 0], self.delta_assume[:, 3], alpha=0.2)
        plt.title("Number of A neighbors for each site")
        plt.xlabel('$I_{site}-median(I_{A neighbors})$ "(∆I)"')
        plt.ylabel("Number of neighbors in px range " + str(np.round(neighbor_cutoff_pix)))

        num_bins = 100
        range_bins = [-0.12, 0.1]
        hist_bins = np.linspace(range_bins[0], range_bins[1], num_bins)
        plt.subplot(142)
        plt.hist(self.delta_assume[:, 0], bins=hist_bins)
        plt.xlabel('$I_{site}-median(I_{A neighbors})$ "(∆I)"')
        plt.ylabel("A site count")
        plt.title("Histogram of ∆Intensity")
        plt.grid("on")

        delta_histogram, bin_edges = np.histogram(self.delta_assume[:, 0], num_bins, range_bins)

        def double_gaussian(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
            return amp1 * np.exp(-((x - mean1) ** 2) / (2 * sigma1**2)) + amp2 * np.exp(
                -((x - mean2) ** 2) / (2 * sigma2**2)
            )

        p0 = [100, -0.025, 0.01, 10, -0.07, 0.01]
        p0 = np.array(p0)

        def double_gaussian_penalized(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
            model = amp1 * np.exp(-((x - mean1) ** 2) / (2 * sigma1**2)) + amp2 * np.exp(
                -((x - mean2) ** 2) / (2 * sigma2**2)
            )

            penalty_strength = 1e6
            penalty = penalty_strength * np.sum(
                (np.array([amp1, mean1, sigma1, amp2, mean2, sigma2]) - p0) ** 2
            )

            return model + penalty / len(x)

        x = np.linspace(range_bins[0], range_bins[1], num_bins)
        y = delta_histogram
        from scipy.optimize import curve_fit

        try:
            popt, _ = curve_fit(double_gaussian_penalized, x, y, p0=p0, maxfev=10000)
        except (RuntimeError, ValueError):
            popt = np.asarray(p0)

        amp1, mean1, sigma1, amp2, mean2, sigma2 = popt

        plt.subplot(143)
        plt.plot(x, y, "k.", label="Data")
        plt.plot(x, double_gaussian(x, *popt), "r-", label="Total fit")
        plt.plot(x, amp1 * np.exp(-((x - mean1) ** 2) / (2 * sigma1**2)), "b--")
        plt.plot(x, amp2 * np.exp(-((x - mean2) ** 2) / (2 * sigma2**2)), "g--")
        plt.vlines(np.array([mean1, mean2]), 0.8, 250, label="Gaussian Means")
        plt.legend()
        plt.ylim([0.8, 250])
        plt.xlabel('$I_{site}-median(I_{neighbors})$ "(∆I)"')
        plt.ylabel("Atomic site count")
        plt.title("Gaussian fit of ∆Intensity")
        plt.grid("on")
        plt.subplot(144)
        plt.plot(x, y, "k.", label="Data")
        plt.plot(x, double_gaussian(x, *popt), "r-", label="Total fit")
        plt.plot(x, amp1 * np.exp(-((x - mean1) ** 2) / (2 * sigma1**2)), "b--")
        plt.plot(x, amp2 * np.exp(-((x - mean2) ** 2) / (2 * sigma2**2)), "g--")
        plt.vlines(np.array([mean1, mean2]), 0.8, 250, label="Gaussian Means")
        plt.legend()
        plt.yscale("log")
        plt.ylim([0.8, 250])
        plt.xlabel('$I_{site}-median(I_{neighbors})$ "(∆I)"')
        plt.ylabel("Atomic site count [log scale]")
        plt.title("Gaussian fit of ∆Intensity")
        plt.grid("on")

        plt.figure(figsize=(6, 7), dpi=300)
        plt.hist(self.delta_assume[:, 0], bins=hist_bins)
        plt.xlabel(
            r"$I_{site}-\mathrm{median}(I_{A\,\mathrm{neighbors}})$ $(\Delta I)$", fontsize=15
        )
        plt.ylabel("A-site count", fontsize=15)
        # plt.title("Histogram of ∆Intensity")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid("on")

        # use the estimates to retrieve the sites
        # positions_b = self.bsites_assume[:,1:]
        # b_int = self.bsites_assume[:,0]
        # a_x = self.atoms[0]["x"]
        # a_y = self.atoms[0]["y"]
        # a_int = self.atoms[0]["int_peak"]
        # b_int /= np.max(a_int)
        # a_int /= np.max(a_int)

        if delta_input_cutoff is None:
            positions_b_defect = positions_b[self.delta_assume[:, 0] < (mean2 + sigma2 * 1.2), :]
        else:
            positions_b_defect = positions_b[self.delta_assume[:, 0] < delta_input_cutoff, :]

        if plot_atoms:
            fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

            import matplotlib.patches as patches

            show_2d(
                self._image.array,
                # lower_quantile = 0.23,
                figax=(fig, ax),
            )

            for i in range(positions_b_defect.shape[0]):
                circle = patches.Circle(
                    (positions_b_defect[i, 1], positions_b_defect[i, 0]),
                    10,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
                ax.add_patch(circle)

            fig.text(
                0.5,
                -0.05,
                "Number of A sites counted: "
                + str(num_a)
                + ", number of B sites counted: "
                + str(num_b)
                + ", number of defects counted: "
                + str(positions_b_defect.shape[0]),
                ha="center",
                va="top",
            )

        return self

    def circle_defects(self, delta_threshold, out, show_histogram=False, figax=None):
        import matplotlib.patches as patches

        out.clear_output(wait=True)
        with out:
            if figax is None:
                fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
            else:
                fig, ax = figax

            ax.clear()

            positions_b = self.bsites_assume[:, 1:]
            positions_b_defect = positions_b[self.delta_assume[:, 0] < delta_threshold, :]

            show_2d(
                self._image.array,
                figax=(fig, ax),
            )

            for i in range(positions_b_defect.shape[0]):
                circle = patches.Circle(
                    (positions_b_defect[i, 1], positions_b_defect[i, 0]),
                    10,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
                ax.add_patch(circle)

            if show_histogram:
                num_bins = 100
                range_bins = [
                    np.min(self.delta_assume[:, 0]) - 0.01,
                    np.max(self.delta_assume[:, 0]) + 0.01,
                ]
                hist_bins = np.linspace(range_bins[0], range_bins[1], num_bins)
                # plt.subplot(142)
                plt.figure()
                plt.hist(self.delta_assume[:, 0], bins=hist_bins)
                plt.xlabel('$I_{site}-median(I_{A neighbors})$ "(∆I)"')
                plt.ylabel("A site count")
                plt.title("Histogram of ∆Intensity")
                plt.grid("on")

        return self

    # def interactive_circle_defects(self):

    #     import ipywidgets as widgets
    #     from IPython.display import display
    #     import matplotlib.pyplot as plt

    #     delta_vals = self.delta_assume[:,0]
    #     delta_min = float(delta_vals.min())
    #     delta_max = float(delta_vals.max())
    #     delta0 = float(np.median(delta_vals))

    #     slider = widgets.FloatSlider(
    #         value=delta0,
    #         min=delta_min,
    #         max=delta_max,
    #         step=(delta_max - delta_min) / 500,
    #         description=r'$\Delta I$',
    #         continuous_update=True,
    #         readout_format='.4f'
    #     )

    #     out = widgets.Output()

    #     def _update(delta_threshold):
    #         out.clear_output(wait=True)
    #         with out:
    #             fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    #             self.circle_defects(
    #                 delta_threshold=delta_threshold,
    #                 figax=(fig, ax)
    #             )
    #             plt.show()

    #     ui = widgets.VBox([slider])
    #     out_plot = widgets.interactive_output(_update, {'delta_threshold': slider})

    #     display(ui, out_plot)

    def interactive_circle_defects(
        self,
        scalebar=None,
        ind=None,
        init_threshold_low=None,
        init_threshold_high=None,
    ):
        import ipywidgets as widgets
        import matplotlib.patches as patches
        from ipywidgets import interactive_output

        if init_threshold_low is None:
            init_threshold_low = np.min(self.delta_assume[:, 0]) + 0.01
        if init_threshold_high is None:
            init_threshold_high = np.median(self.delta_assume[:, 0])
        thresh_slider_l = widgets.FloatSlider(
            value=init_threshold_low,
            min=np.min(self.delta_assume[:, 0]) - 0.01,
            max=np.max(self.delta_assume[:, 0]) + 0.01,
            description="threshold",
            step=0.001,
            continuous_update=True,
            readout_format=".4f",
        )
        thresh_slider_h = widgets.FloatSlider(
            value=init_threshold_high,
            min=np.min(self.delta_assume[:, 0]) - 0.01,
            max=np.max(self.delta_assume[:, 0]) + 0.01,
            description="threshold",
            step=0.001,
            continuous_update=True,
            readout_format=".4f",
        )
        out = widgets.Output()

        def circle_defects(delta_high, delta_low, show_histogram=False, figax=None):
            out.clear_output(wait=True)
            with out:
                if figax is None:
                    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
                else:
                    fig, ax = figax

                ax.clear()

                positions_b = self.bsites_assume[:, 1:]

                delta = self.delta_assume[:, 0]
                mask_mono = (delta >= delta_low) & (delta < delta_high)
                mask_di = delta < delta_low
                positions_b_mono = positions_b[mask_mono, :]
                positions_b_di = positions_b[mask_di, :]

                # positions_b_defect = positions_b[lattice.delta_assume[:,0] < delta_threshold,:]
                a_x = self.atoms[0]["x"]
                a_y = self.atoms[0]["y"]

                num_a = a_x.shape[0]
                num_b = positions_b.shape[0]

                # a rough area estimate using the site positions:
                min_x = min((np.min(a_x), np.min(positions_b[:, 0])))
                max_x = max((np.max(a_x), np.max(positions_b[:, 0])))

                min_y = min((np.min(a_y), np.min(positions_b[:, 1])))
                max_y = max((np.max(a_y), np.max(positions_b[:, 1])))

                x_length_pix = max_x - min_x
                y_length_pix = max_y - min_y

                pixel_size = scalebar["sampling"]
                pixel_units = scalebar["units"]
                if pixel_units == "nm":
                    cm_multiplier = 1e7

                area_analyzed = (
                    pixel_size**2 * y_length_pix * x_length_pix / (cm_multiplier**2)
                )  # area in cm

                monovacancy_density = positions_b_mono.shape[0] / area_analyzed
                divacancy_density = positions_b_di.shape[0] / area_analyzed

                show_2d(
                    self._image.array,
                    figax=(fig, ax),
                    scalebar=scalebar,
                )

                for i in range(positions_b_mono.shape[0]):
                    circle = patches.Circle(
                        (positions_b_mono[i, 1], positions_b_mono[i, 0]),
                        10,
                        fill=False,
                        edgecolor="red",
                        linewidth=1,
                    )
                    ax.add_patch(circle)

                for i in range(positions_b_di.shape[0]):
                    circle = patches.Circle(
                        (positions_b_di[i, 1], positions_b_di[i, 0]),
                        10,
                        fill=False,
                        edgecolor="blue",
                        linewidth=1,
                    )
                    ax.add_patch(circle)

                rect = patches.Rectangle(
                    (min_y - 5, min_x - 5),
                    max_y - min_y + 10,
                    max_x - min_x + 10,
                    linewidth=0.5,
                    edgecolor="#695147",
                    facecolor="none",
                )

                ax.add_patch(rect)

                fig.text(
                    0.5,
                    -0.01,
                    "Number of A sites: "
                    + str(num_a)
                    + ", Number of B sites: "
                    + str(num_b)
                    + "\nNumber of mono Se vacancies: "
                    + str(positions_b_mono.shape[0])
                    + ", Number of di Se vacancies: "
                    + str(positions_b_di.shape[0])
                    + f"\nMonovacancy density: {monovacancy_density:.1e} cm$^{{-2}}$"
                    + f", Divacancy density: {divacancy_density:.1e} cm$^{{-2}}$"
                    + f"\nArea analyzed: {area_analyzed:.1e} cm$^{{2}}$",
                    ha="center",
                    va="top",
                )

                if ind is not None:
                    ax.set_title("Pair index: " + str(ind))

                if show_histogram:
                    num_bins = 100
                    range_bins = [
                        np.min(self.delta_assume[:, 0]) - 0.01,
                        np.max(self.delta_assume[:, 0]) + 0.01,
                    ]
                    hist_bins = np.linspace(range_bins[0], range_bins[1], num_bins)
                    # plt.subplot(142)
                    plt.figure()
                    plt.hist(self.delta_assume[:, 0], bins=hist_bins)
                    plt.xlabel('$I_{site}-median(I_{A neighbors})$ "(∆I)"')
                    plt.ylabel("A site count")
                    plt.title("Histogram of ∆Intensity")
                    plt.grid("on")

        interactive_output(
            circle_defects, {"delta_low": thresh_slider_l, "delta_high": thresh_slider_h}
        )

        # display(ui, out_plot)
        return self

    def delta_intensities_input(
        self,
        uc_val=2,
    ):
        from scipy.spatial import cKDTree

        neighbor_cutoff_pix = uc_val * self.uv_norm

        # self.input_image_a_int = a_int_input
        # self.input_image_b_int = b_int_input
        positions_b = self.bsites_assume[:, 1:]
        b_int = self.input_image_b_int
        a_x = self.atoms[0]["x"]
        a_y = self.atoms[0]["y"]
        a_int = self.input_image_a_int
        b_int /= np.max(a_int)
        a_int /= np.max(a_int)
        positions_a = np.column_stack((a_x, a_y))

        positions_all = np.concatenate([positions_a, positions_b])
        num_a = positions_a.shape[0]
        num_b = positions_b.shape[0]

        tree = cKDTree(positions_all)

        delta_data = np.zeros((num_b, 5))

        for i, pos in enumerate(positions_b):
            idxs = np.asarray(tree.query_ball_point(pos, neighbor_cutoff_pix))
            idxs_a = idxs[idxs < num_a]
            idxs_b = idxs[idxs > num_a]

            a_neighbor_intensities = a_int[idxs_a]
            median_a_intensity = np.median(a_neighbor_intensities)
            b_neighbor_intensities = b_int[idxs_b - num_a]
            median_b_intensity = np.median(b_neighbor_intensities)
            all_neighbor_intensities = np.concatenate(
                [a_neighbor_intensities, b_neighbor_intensities]
            )
            median_all_intensity = np.median(all_neighbor_intensities)

            delta_data[i, 0] = b_int[i] - median_a_intensity  # the delta intensity
            delta_data[i, 1] = b_int[i] - median_b_intensity  # the delta intensity
            delta_data[i, 2] = b_int[i] - median_all_intensity  # the delta intensity
            delta_data[i, 3] = idxs_a.shape[0]  # number of neighbors used
            delta_data[i, 4] = idxs_b.shape[0]  # number of neighbors used

        self.delta_input = delta_data

        # delta_intensities, num_neighbors = lattice.intensity_neighborhood(neighborhood_units = neighborhood_units, return_delta = True)

        plt.figure(figsize=(24, 7), dpi=300)
        # plt.rcParams['font.family'] = 'serif'
        # params = {'mathtext.default': 'regular' }
        # plt.rcParams.update(params)
        plt.subplot(141)
        plt.scatter(self.delta_input[:, 0], self.delta_input[:, 3], alpha=0.2)
        plt.title("Number of A neighbors for each site")
        plt.xlabel('$I_{site}-median(I_{A neighbors})$ "(∆I)"')
        plt.ylabel("Number of neighbors in px range " + str(np.round(neighbor_cutoff_pix)))

        num_bins = 100
        range_bins = [-0.6, 0.1]
        hist_bins = np.linspace(range_bins[0], range_bins[1], num_bins)
        plt.subplot(142)
        plt.hist(self.delta_input[:, 0], bins=hist_bins)
        plt.xlabel('$I_{site}-median(I_{A neighbors})$ "(∆I)"')
        plt.ylabel("A site count")
        plt.title("Histogram of ∆Intensity")
        plt.grid("on")

        delta_histogram, bin_edges = np.histogram(self.delta_input[:, 0], num_bins, range_bins)

        def double_gaussian(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
            return amp1 * np.exp(-((x - mean1) ** 2) / (2 * sigma1**2)) + amp2 * np.exp(
                -((x - mean2) ** 2) / (2 * sigma2**2)
            )

        p0 = [30, -0.2, 0.15, 5, -0.5, 0.1]
        p0 = np.array(p0)

        def double_gaussian_penalized(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
            model = amp1 * np.exp(-((x - mean1) ** 2) / (2 * sigma1**2)) + amp2 * np.exp(
                -((x - mean2) ** 2) / (2 * sigma2**2)
            )

            penalty_strength = 1e6
            penalty = penalty_strength * np.sum(
                (np.array([amp1, mean1, sigma1, amp2, mean2, sigma2]) - p0) ** 2
            )

            return model + penalty / len(x)

        x = np.linspace(range_bins[0], range_bins[1], num_bins)
        y = delta_histogram
        from scipy.optimize import curve_fit

        try:
            popt, _ = curve_fit(double_gaussian_penalized, x, y, p0=p0, maxfev=10000)
        except (RuntimeError, ValueError):
            popt = np.asarray(p0)

        amp1, mean1, sigma1, amp2, mean2, sigma2 = popt

        plt.subplot(143)
        plt.plot(x, y, "k.", label="Data")
        plt.plot(x, double_gaussian(x, *popt), "r-", label="Total fit")
        plt.plot(x, amp1 * np.exp(-((x - mean1) ** 2) / (2 * sigma1**2)), "b--")
        plt.plot(x, amp2 * np.exp(-((x - mean2) ** 2) / (2 * sigma2**2)), "g--")
        plt.vlines(np.array([mean1, mean2]), 0.8, 250, label="Gaussian Means")
        plt.legend()
        plt.ylim([0.8, 250])
        plt.xlabel('$I_{site}-median(I_{neighbors})$ "(∆I)"')
        plt.ylabel("Atomic site count")
        plt.title("Gaussian fit of ∆Intensity")
        plt.grid("on")
        plt.subplot(144)
        plt.plot(x, y, "k.", label="Data")
        plt.plot(x, double_gaussian(x, *popt), "r-", label="Total fit")
        plt.plot(x, amp1 * np.exp(-((x - mean1) ** 2) / (2 * sigma1**2)), "b--")
        plt.plot(x, amp2 * np.exp(-((x - mean2) ** 2) / (2 * sigma2**2)), "g--")
        plt.vlines(np.array([mean1, mean2]), 0.8, 250, label="Gaussian Means")
        plt.legend()
        plt.yscale("log")
        plt.ylim([0.8, 250])
        plt.xlabel('$I_{site}-median(I_{neighbors})$ "(∆I)"')
        plt.ylabel("Atomic site count [log scale]")
        plt.title("Gaussian fit of ∆Intensity")
        plt.grid("on")

        # use the estimates to retrieve the sites
        # positions_b = self.bsites_assume[:,1:]
        # b_int = self.bsites_assume[:,0]
        # a_x = self.atoms[0]["x"]
        # a_y = self.atoms[0]["y"]
        # a_int = self.atoms[0]["int_peak"]
        # b_int /= np.max(a_int)
        # a_int /= np.max(a_int)

        positions_b_defect = positions_b[self.delta_input[:, 0] < (mean2 + sigma2 * 1.2), :]

        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

        import matplotlib.patches as patches

        show_2d(
            self.input_image,
            # lower_quantile = 0.23,
            figax=(fig, ax),
        )

        for i in range(positions_b_defect.shape[0]):
            circle = patches.Circle(
                (positions_b_defect[i, 1], positions_b_defect[i, 0]),
                10,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(circle)

        return self

    def line_profile(self, origin, direction, num_samples=None):
        nx, ny = self._image.array.shape
        # print(nx, ny)
        print(direction)
        # print(origin)
        x0, y0 = origin
        v = np.array(direction, dtype=float)
        v /= np.linalg.norm(v)

        if num_samples is None:
            num_samples = int(np.hypot(nx, ny))

        corners = np.array([[0, 0], [nx, 0], [0, ny], [nx, ny]])
        t_values = []
        for corner in corners:
            dx, dy = corner - origin
            t_values.append(np.dot([dx, dy], v))  # projection along direction
        t_min, t_max = min(t_values), max(t_values)
        # print(t_min, t_max, num_samples)
        t = np.linspace(t_min, t_max, num_samples)

        # Coordinates along the line
        x = x0 + v[0] * t
        y = y0 + v[1] * t

        # Interpolated intensities
        profile = map_coordinates(self.image.array, [x, y], order=1, mode="nearest")

        # Clip valid region (inside image bounds)
        mask = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
        return t[mask], profile[mask], x[mask], y[mask]

    # overloading this handle
    def line_profile_pts(
        image,
        p1=None,
        p2=None,
        origin=None,
        direction=None,
        num_samples=None,
        order=1,
        mode="nearest",
    ):
        nx, ny = image.shape

        if p1 is not None and p2 is not None:
            p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
            v = p2 - p1
            length = np.linalg.norm(v)
            v /= length
            if num_samples is None:
                num_samples = int(length)
            t = np.linspace(0, length, num_samples)
            x = p1[0] + v[0] * t
            y = p1[1] + v[1] * t

        elif origin is not None and direction is not None:
            x0, y0 = origin
            v = np.array(direction, dtype=float)
            v /= np.linalg.norm(v)

            if num_samples is None:
                num_samples = int(np.hypot(nx, ny))

            corners = np.array([[0, 0], [nx, 0], [0, ny], [nx, ny]])
            t_values = [(np.dot(corner - origin, v)) for corner in corners]
            t_min, t_max = min(t_values), max(t_values)
            t = np.linspace(t_min, t_max, num_samples)
            x = x0 + v[0] * t
            y = y0 + v[1] * t
        else:
            raise ValueError("Specify either (p1, p2) or (origin, direction).")

        # Interpolate
        profile = map_coordinates(image, [x, y], order=order, mode=mode)

        # Mask for valid coordinates
        mask = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)

        return t[mask], profile[mask], x[mask], y[mask]

    def find_b_sites(
        self,
        max_perpendicular_distance=5,
        sigma_perp=5,
        sigma_parallel=5,
    ):
        atoms_arr = self.atoms.get_data(0)
        a_x = atoms_arr[:, 0]
        a_y = atoms_arr[:, 1]
        a_intensity = self.atoms[0]["int_peak"]

        for lat_vec in self.uv_arr:
            for atom_index in range(a_x.shape[0]):
                neighbor_x = [
                    a_x[i] for i in self.atom_neighbor_arr[:, atom_index] if i is not None
                ]
                neighbor_y = [
                    a_y[i] for i in self.atom_neighbor_arr[:, atom_index] if i is not None
                ]
                atom_x = a_x[atom_index]
                atom_y = a_y[atom_index]
                position_x = lat_vec[0] + atom_x
                position_y = lat_vec[1] + atom_y
                radial_dist = (
                    (neighbor_x - position_x) ** 2 + (neighbor_y - position_y) ** 2
                ) ** (0.5)
                if (radial_dist < (self.uv_norm * (self.tolerance_uv - 1))).any():
                    successful_candidate_index = np.argmin(radial_dist)
                    successful_candidate_index = self.atom_neighbor_arr[
                        successful_candidate_index, atom_index
                    ]
                    vec_x = a_x[successful_candidate_index] - atom_x
                    vec_y = a_y[successful_candidate_index] - atom_y
                    line_profile_vector = np.array([vec_x, vec_y])
                    line_profile_origin = np.array([atom_x, atom_y])
                    slice_coordinates, slice_y, x_coords_slice, y_coords_slice = self.line_profile(
                        line_profile_origin, line_profile_vector
                    )
                    mask, near_peaks, t_values, distances = self.select_peaks_near_line(
                        atom_index, line_profile_vector, max_perpendicular_distance
                    )

                    t_profile = np.asarray(slice_coordinates)
                    t_values = np.asarray(t_values)
                    distances = np.asarray(distances)

                    A = a_intensity[mask]
                    A_eff = A * np.exp(-0.5 * (distances / sigma_perp) ** 2)
                    gaussian_sum = np.zeros_like(t_profile, dtype=float)
                    for t_i, A_i in zip(t_values, A_eff):
                        gaussian_sum += A_i * np.exp(
                            -0.5 * ((t_profile - t_i) / sigma_parallel) ** 2
                        )
                    if atom_index < 3:
                        plt.figure()
                        plt.plot()
                        plt.figure(figsize=(10, 4))
                        plt.subplot(1, 2, 1)
                        plt.imshow(self.image.array, cmap="gray", origin="upper")
                        plt.plot(y_coords_slice, x_coords_slice, "r-", lw=1)
                        plt.title("Line through image")

                        plt.subplot(1, 2, 2)
                        plt.plot(slice_coordinates, slice_y)
                        plt.plot(slice_coordinates, gaussian_sum)
                        plt.title("Line profile")
                        plt.xlabel("Distance along line (pixels)")
                        plt.ylabel("Intensity")
                        plt.tight_layout()
                        plt.show()
        return self

    def find_parallel_sites(
        self,
        t,
        profile,
        x_coords,
        y_coords,
        direction,
        origin=None,
        max_perpendicular_distance=10,
        gaussian_smooth_data=None,
        return_fit=False,
        plot_profile=True,
    ):
        a_intensity = self.atoms[0]["int_peak"]
        bg_intensity = self.atoms[0]["int_bg"]
        sigma_arr = self.atoms[0]["sigma"]
        if origin is None:
            t_near_zero = np.argmin(np.abs(t))
            x_origin = x_coords[t_near_zero]
            y_origin = y_coords[t_near_zero]
        else:
            x_origin = origin[0]
            y_origin = origin[1]
        mask, near_peaks, t_values_s, distances = self.select_peaks_near_line(
            x_origin,
            y_origin,
            direction=direction,
            max_perpendicular_distance=max_perpendicular_distance,
        )

        t_profile = np.asarray(profile)
        t_values = np.asarray(t)
        distances = np.asarray(distances)

        A = a_intensity[mask]
        B = bg_intensity[mask]
        sigma_perp = sigma_arr[mask]
        sigma_parallel = sigma_arr[mask]
        A_eff = A * np.exp(-0.5 * (distances / sigma_perp) ** 2)

        gaussian_sum = np.zeros_like(t_values, dtype=float)
        s_index = 0
        for t_i, A_i in zip(t_values_s, A_eff):
            gaussian_sum += A_i * np.exp(-0.5 * ((t_values - t_i) / sigma_parallel[s_index]) ** 2)
            s_index += 1

        bg_interp_func = interp1d(
            t_values_s, B, kind="linear", bounds_error=False, fill_value=(B[0], B[-1])
        )

        B_interp = bg_interp_func(t_values)
        gaussian_sum += B_interp

        nx, ny = self.image.array.shape

        if plot_profile is True:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(self.image.array, cmap="gray", origin="upper")
            plt.plot(y_coords, x_coords, "r-", lw=1)
            plt.quiver(
                y_coords[0],
                x_coords[0],
                direction[1],
                direction[0],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="red",
                zorder=10,
            )
            plt.title("Line through image")
            plt.xlim([0, nx - 1])
            plt.ylim([ny - 1, 0])

            plt.subplot(1, 2, 2)
            if gaussian_smooth_data is not None:
                plt.plot(t_values, gaussian_filter(t_profile, gaussian_smooth_data))
            else:
                plt.plot(t_values, t_profile)
            plt.plot(t_values, gaussian_sum, alpha=0.5)
            plt.title("Line profile")
            plt.xlabel("Distance along line (pixels)")
            plt.ylabel("Intensity")
            plt.tight_layout()
            plt.show()
        if return_fit:
            return gaussian_sum

        return self

    def unit_vector(
        self,
        v,
    ):
        v = np.array(v, dtype=float)
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("direction vector cannot be zero")
        return v / norm

    def project_point_to_line_rowmajor(self, point, origin, direction):
        p = np.array(point, dtype=float)
        origin = np.array(origin, dtype=float)
        v = self.unit_vector(direction)
        dp = p - origin
        s = np.dot(dp, v)
        proj = origin + s * v
        perp_vec = dp - s * v
        perp_dist = np.linalg.norm(perp_vec)
        return s, perp_dist, proj

    def select_peaks_near_line(
        self,
        atom_index,
        direction,
        max_perpendicular_distance,
    ):
        atoms_arr = self.atoms.get_data(0)
        a_xy = atoms_arr[:, 0:2]
        v = self.unit_vector(direction)
        center_atom_to_rest = a_xy - a_xy[atom_index]

        t_values_all = np.dot(center_atom_to_rest, v)

        rest_proj = np.outer(t_values_all, v)
        perp = center_atom_to_rest - rest_proj
        distances_all = np.linalg.norm(perp, axis=1)

        # Select peaks within threshold
        mask = distances_all <= max_perpendicular_distance

        near_peaks = a_xy[mask]
        t_values = t_values_all[mask]
        distances = distances_all[mask]

        return mask, near_peaks, t_values, distances

    def select_peaks_near_line_center(
        self,
        x_c,
        y_c,
        direction,
        max_perpendicular_distance,
    ):
        atoms_arr = self.atoms.get_data(0)
        a_xy = atoms_arr[:, 0:2]
        v = self.unit_vector(direction)
        # v = self.unit_vector(direction)

        # v = self.unit_vector(direction[::-1])

        center_atom_to_rest = a_xy - np.array([x_c, y_c])

        t_values_all = np.dot(center_atom_to_rest, v)

        rest_proj = np.outer(t_values_all, v)
        perp = center_atom_to_rest - rest_proj
        distances_all = np.linalg.norm(perp, axis=1)

        # Select peaks within threshold
        mask = distances_all <= max_perpendicular_distance
        # print(mask[mask == True])

        near_peaks = a_xy[mask]
        t_values = t_values_all[mask]
        distances = distances_all[mask]

        return mask, near_peaks, t_values, distances

    def gaussian(self, s, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((s - mu) / sigma) ** 2)

    def get_xy_shifts(
        self,
        a0,
    ):
        position_fraction = self._positions_frac[a0]

        if (position_fraction == np.zeros([2])).all():
            p1 = np.array([1, 0]) @ self.uv_arr[:2]
            p2 = np.array([-1, 0]) @ self.uv_arr[:2]
            p3 = np.array([0, 1]) @ self.uv_arr[:2]
            p4 = np.array([0, -1]) @ self.uv_arr[:2]
        else:
            p1 = (np.array([0, 0]) + position_fraction) @ self.uv_arr[:2]
            p2 = (np.array([-1, 0]) + position_fraction) @ self.uv_arr[:2]
            p3 = (np.array([0, -1]) + position_fraction) @ self.uv_arr[:2]
            p4 = (np.array([-1, -1]) + position_fraction) @ self.uv_arr[:2]

        return np.array([p1, p2, p3, p4])

    def get_a_positions_near_b_site(
        self,
    ):
        b_frac = self._positions_frac[1]
        p1 = (np.array([0, 0]) - b_frac) @ self.uv_arr[:2]
        p2 = (np.array([1, 0]) - b_frac) @ self.uv_arr[:2]
        p3 = (np.array([0, 1]) - b_frac) @ self.uv_arr[:2]
        p4 = (np.array([1, 1]) - b_frac) @ self.uv_arr[:2]
        return np.array([p1, p2, p3, p4])

    def organize_b_neighbors(
        self,
        site_search_radius=2,
        num_bins=128,
        tolerance_uv=None,
        num_sites_use=1,
        # centers = None,
    ):
        a_x_b = self.atoms.get_data(1)[:, 0]
        a_y_b = self.atoms.get_data(1)[:, 1]

        a_x_a = self.atoms.get_data(0)[:, 0]
        a_y_a = self.atoms.get_data(0)[:, 1]
        b_neighbor_arr = np.empty((2, 4, a_x_b.shape[0]), dtype=object)
        pm_arr = np.array([-1, 1])
        a_positions_around_site = self.get_a_positions_near_b_site()
        pm_arr = np.array([-1, 1])
        uvw_arr = self.uv_arr
        uvw_norm = self.uv_norm
        tolerance_uv = self.tolerance_uv
        for atom_index in range(a_x_b.shape[0]):
            # B sites near the B sites (displaced by the lattice vectors, hopefully)
            for pm in pm_arr:
                for uvw_index, lat_vec in enumerate(uvw_arr):
                    position_x = pm * lat_vec[0] + a_x_b[atom_index]
                    position_y = pm * lat_vec[1] + a_y_b[atom_index]
                    radial_dist = ((a_x_b - position_x) ** 2 + (a_y_b - position_y) ** 2) ** (0.5)
                    radial_dist[atom_index] = (
                        uvw_norm * (tolerance_uv - 1) * 2
                    )  # make sure that self is outside of range
                    if (radial_dist < (uvw_norm * (tolerance_uv - 1))).any():
                        successful_candidate_index = np.argmin(radial_dist)
                        if pm == 1 and uvw_index == 0:
                            b_neighbor_arr[1, 0, atom_index] = int(successful_candidate_index)

                        if pm == -1 and uvw_index == 0:
                            b_neighbor_arr[1, 1, atom_index] = int(successful_candidate_index)

                        if pm == 1 and uvw_index == 1:
                            b_neighbor_arr[1, 2, atom_index] = int(successful_candidate_index)

                        if pm == -1 and uvw_index == 1:
                            b_neighbor_arr[1, 3, atom_index] = int(successful_candidate_index)
                        # leaving out the w vector for right now...

            # A sites near the B sites
            # critically, the successful candidate indices that are written in this loop are only valid for the A site variables.
            for pos_index, pos_vec in enumerate(a_positions_around_site):
                position_x = pos_vec[0] + a_x_b[atom_index]
                position_y = pos_vec[1] + a_y_b[atom_index]
                radial_dist = ((a_x_a - position_x) ** 2 + (a_y_a - position_y) ** 2) ** (0.5)
                # this line is not necessary because the coordinate of the b site is not in the a site coordinate list
                # radial_dist[atom_index] = self.uv_norm * (tolerance_uv - 1) * 2 # make sure that self is outside of range
                if (radial_dist < (self.uv_norm * (tolerance_uv - 1))).any():
                    successful_candidate_index = np.argmin(radial_dist)
                    if pos_index == 0:
                        b_neighbor_arr[0, 0, atom_index] = int(successful_candidate_index)

                    if pos_index == 1:
                        b_neighbor_arr[0, 1, atom_index] = int(successful_candidate_index)

                    if pos_index == 2:
                        b_neighbor_arr[0, 2, atom_index] = int(successful_candidate_index)

                    if pos_index == 3:
                        b_neighbor_arr[0, 3, atom_index] = int(successful_candidate_index)

                    if pos_index == 4:
                        b_neighbor_arr[0, 4, atom_index] = int(successful_candidate_index)

                    if pos_index == 5:
                        b_neighbor_arr[0, 5, atom_index] = int(successful_candidate_index)

        self.b_neighbor_arr = b_neighbor_arr

        # if centers is None:
        #     centers = np.arange(0,2)
        # if isinstance(centers, int):
        #     centers = np.arange(0,centers)
        # plt.figure()
        # plt.imshow(self.image.array, cmap = 'gray')
        # a_x_b = self.atoms.get_data(1)[:,0]
        # a_y_b = self.atoms.get_data(1)[:,1]
        # a_x_a = self.atoms.get_data(0)[:,0]
        # a_y_a = self.atoms.get_data(0)[:,1]

        # for atom_b_index in centers:
        #     # plt.scatter(a_y_b[atom_b_index], a_x_b[atom_b_index], color = 'blue', zorder = 10)
        #     # plt.scatter(a_y_b[self.b_neighbor_arr[1,:,atom_b_index].astype(int)], a_x_b[self.b_neighbor_arr[1,:,atom_b_index].astype(int)], color = 'red',  alpha = 0.5)
        #     # plt.scatter(a_y_a[self.b_neighbor_arr[0,:,atom_b_index].astype(int)], a_x_a[self.b_neighbor_arr[0,:,atom_b_index].astype(int)], color = 'green')

        # # for atom_b_index in centers:
        #     plt.scatter(a_y_b[atom_b_index], a_x_b[atom_b_index], color='blue', zorder=10)

        #     # filter valid neighbors
        #     valid_neighbors_1 = [i for i in self.b_neighbor_arr[1, :, atom_b_index] if i is not None]
        #     valid_neighbors_0 = [i for i in self.b_neighbor_arr[0, :, atom_b_index] if i is not None]

        #     plt.scatter(a_y_b[np.array(valid_neighbors_1, int)], a_x_b[np.array(valid_neighbors_1, int)], color='red', alpha=0.5)
        #     plt.scatter(a_y_a[np.array(valid_neighbors_0, int)], a_x_a[np.array(valid_neighbors_0, int)], color='green')

        return self

    # def organize_nearest_neighbors_2(
    #     self,
    #     site_search_radius = 2,
    #     num_bins = 128,
    #     tolerance_uv = None,
    #     num_sites_use = 1,
    #     which_neighbors = None,
    #     which_centers = None,
    #     return_neigbhor_arr = False,
    # ):
    #     if tolerance_uv is None:
    #         tolerance_uv = self.tolerance_uv

    #     if which_neighbors is None:
    #         which_neighbors = np.array([0])
    #     if which_centers is None:
    #         which_centers = np.array([0])

    #     pm_arr = np.array([1,-1])
    #     uv_arr = self.uv_arr
    #     uvw_arr = np.zeros([self._num_sites, 3,2])
    #     for v0 in range(self._num_sites):
    #         unit_shifts = np.array([self._positions_frac[v0, 0], self._positions_frac[v0, 1]])
    #         positive_vector = unit_shifts.copy()
    #         negative_vector = unit_shifts.copy()
    #         negative_vector[1] *= -1
    #         if np.abs(np.rad2deg(np.arccos(np.dot(positive_vector, negative_vector)/(np.linalg.norm(positive_vector) * np.linalg.norm(negative_vector))))) > np.deg2rad(90):
    #             w_ = np.asarray(positive_vector)+np.asarray(negative_vector)
    #             w_sign = 1
    #         else:
    #             w_ = np.asarray(positive_vector)-np.asarray(negative_vector)
    #             w_sign = -1

    #         p_v_c = positive_vector @ uv_arr[:2]
    #         n_v_c = negative_vector @ uv_arr[:2]
    #         w_v_c = w_ @ uv_arr[:2]
    #         uvw_arr[v0,:] = np.array([p_v_c, n_v_c, w_v_c])

    #     a0_iter = 0
    #     for a0 in which_centers:
    #         a_x_c = self.atoms.get_data(a0)[:,0]
    #         a_y_c = self.atoms.get_data(a0)[:,1]
    #         atom_neighbor_arr = np.empty((which_centers.shape[0], which_neighbors.shape[0], 6, a_x_c.shape[0]), dtype=object)
    #         a1_iter = 0
    #         for a1 in which_neighbors:
    #             a_x_n = self.atoms.get_data(a1)[:,0]
    #             a_y_n = self.atoms.get_data(a1)[:,1]
    #             for atom_index in range(a_x_c.shape[0]):
    #                 for pm in pm_arr:
    #                     for uvw_index, lat_vec in enumerate(uvw_arr[a1]):
    #                         position_x = pm * lat_vec[0] + a_x_c[atom_index]
    #                         position_y = pm * lat_vec[1] + a_y_c[atom_index]
    #                         radial_dist = ((a_x_n - position_x)**2 + (a_y_n - position_y)**2)**(0.5)
    #                         radial_dist[atom_index] = self.uv_norm * (tolerance_uv - 1) * 2 # make sure that self is outside of range
    #                         if (radial_dist < (self.uv_norm * (tolerance_uv - 1))).any():
    #                             successful_candidate_index = np.argmin(radial_dist)
    #                             if (pm == 1 and uvw_index == 0):
    #                                 atom_neighbor_arr[a0_iter, a1_iter, 0,atom_index] = int(successful_candidate_index)

    #                             if (pm == -1 and uvw_index == 0):
    #                                 atom_neighbor_arr[a0_iter, a1_iter, 1,atom_index] = int(successful_candidate_index)

    #                             if (pm == 1 and uvw_index == 1):
    #                                 atom_neighbor_arr[a0_iter, a1_iter, 2,atom_index] = int(successful_candidate_index)

    #                             if (pm == -1 and uvw_index == 1):
    #                                 atom_neighbor_arr[a0_iter, a1_iter, 3,atom_index] = int(successful_candidate_index)

    #                             if (pm == 1 and uvw_index == 2):
    #                                 atom_neighbor_arr[a0_iter, a1_iter, 4,atom_index] = int(successful_candidate_index)

    #                             if (pm == -1 and uvw_index == 2):
    #                                 atom_neighbor_arr[a0_iter, a1_iter, 5,atom_index] = int(successful_candidate_index)
    #             a1_iter += 1
    #         a0_iter += 1
    #     self.atom_neighbor_arr = atom_neighbor_arr
    #     self.which_centers = which_centers
    #     self.which_neighbors = which_neighbors

    #     plt.figure()
    #     index_show = 10
    #     print(atom_neighbor_arr[0,0,:,:index_show])
    #     for atom_index in range(index_show):
    #         not_none_indices = np.asarray([i for i in atom_neighbor_arr[0,0,:,atom_index] if i is not None], dtype = int)
    #         plt.scatter(self.atoms.get_data(0)[not_none_indices,1], self.atoms.get_data(0)[not_none_indices,0])
    #     if return_neigbhor_arr:
    #         return atom_neighbor_arr, which_centers, which_neighbors
    #     return self

    def organize_nearest_neighbors(
        self,
        site_search_radius=2,
        num_bins=128,
        tolerance_uv=None,
        num_sites_use=1,
    ):
        if tolerance_uv is None:
            tolerance_uv = self.tolerance_uv
        for a0 in range(num_sites_use):
            atoms_arr = self.atoms.get_data(a0)
            a_x = atoms_arr[:, 0]
            a_y = atoms_arr[:, 1]
            pm_arr = np.array([1, -1])

            atom_neighbor_arr = np.empty((6, a_x.shape[0]), dtype=object)
            has_six_neighbors_arr = np.zeros(a_x.shape[0])
            for atom_index in range(a_x.shape[0]):
                for pm in pm_arr:
                    for uvw_index, lat_vec in enumerate(self.uv_arr):
                        position_x = pm * lat_vec[0] + a_x[atom_index]
                        position_y = pm * lat_vec[1] + a_y[atom_index]
                        radial_dist = ((a_x - position_x) ** 2 + (a_y - position_y) ** 2) ** (0.5)
                        radial_dist[atom_index] = (
                            self.uv_norm * (tolerance_uv - 1) * 2
                        )  # make sure that self is outside of range
                        if (radial_dist < (self.uv_norm * (tolerance_uv - 1))).any():
                            successful_candidate_index = np.argmin(radial_dist)
                            if pm == 1 and uvw_index == 0:
                                atom_neighbor_arr[0, atom_index] = int(successful_candidate_index)
                                has_six_neighbors_arr[atom_index] += 1

                            if pm == -1 and uvw_index == 0:
                                atom_neighbor_arr[1, atom_index] = int(successful_candidate_index)
                                has_six_neighbors_arr[atom_index] += 1

                            if pm == 1 and uvw_index == 1:
                                atom_neighbor_arr[2, atom_index] = int(successful_candidate_index)
                                has_six_neighbors_arr[atom_index] += 1

                            if pm == -1 and uvw_index == 1:
                                atom_neighbor_arr[3, atom_index] = int(successful_candidate_index)
                                has_six_neighbors_arr[atom_index] += 1

                            if pm == 1 and uvw_index == 2:
                                atom_neighbor_arr[4, atom_index] = int(successful_candidate_index)
                                has_six_neighbors_arr[atom_index] += 1

                            if pm == -1 and uvw_index == 2:
                                atom_neighbor_arr[5, atom_index] = int(successful_candidate_index)
                                has_six_neighbors_arr[atom_index] += 1

            self.has_six_neighbors_arr = has_six_neighbors_arr == 6
            self.atom_neighbor_arr = atom_neighbor_arr
        return self

    def get_next_neighborhood_layer(
        self,
        atom_index,
    ):
        neighbor_arr = np.asarray(
            [i for i in self.atom_neighbor_arr[:, atom_index] if i is not None], dtype=int
        )
        neighbors_pass = [i for i in neighbor_arr if not self.added_to_neighbor_list_already[i]]

        self.added_to_neighbor_list_already[neighbors_pass] = 1

        return neighbors_pass

    def get_next_neighborhood_layer_arr(
        self,
        atom_indexes,
    ):
        atom_indexes_less_none = [i for i in atom_indexes if i is not None]
        arr_present = False
        arr = None
        for atom_index in atom_indexes_less_none:
            if arr_present:
                arr = np.concatenate(
                    (arr, np.asarray(self.get_next_neighborhood_layer(atom_index)))
                )
            else:
                arr = np.asarray(self.get_next_neighborhood_layer(atom_index))
                arr_present = True
        if arr is None:
            return None
        else:
            return np.asarray(arr, dtype=int)

    def get_next_neighborhood_layer_b(
        self,
        atom_index,
        a_or_b,
    ):
        neighbor_arr = np.asarray(
            [i for i in self.b_neighbor_arr[a_or_b, :, atom_index] if i is not None], dtype=int
        )
        if a_or_b == 0:
            neighbors_pass = [
                i for i in neighbor_arr if not self.added_to_neighbor_list_already[i]
            ]
            self.added_to_neighbor_list_already[neighbors_pass] = 1
        if a_or_b == 1:
            neighbors_pass = [
                i for i in neighbor_arr if not self.added_to_neighbor_list_already_b[i]
            ]
            self.added_to_neighbor_list_already_b[neighbors_pass] = 1

        return neighbors_pass

    def get_next_neighborhood_layer_arr_b(
        self,
        atom_indexes,
        a_or_b,
    ):
        atom_indexes_less_none = [i for i in atom_indexes if i is not None]
        arr_present = False
        arr = None
        for atom_index in atom_indexes_less_none:
            if arr_present:
                arr = np.concatenate(
                    (arr, np.asarray(self.get_next_neighborhood_layer_b(atom_index, a_or_b)))
                )
            else:
                arr = np.asarray(self.get_next_neighborhood_layer_b(atom_index, a_or_b))
                arr_present = True

        # if a_or_b  == 0:
        #     arr = arr[1:] # get rid of the b index
        if arr is None:
            return None
        else:
            return np.asarray(arr, dtype=int)

    def neighborhood_unfinished(
        self,
        neighborhood_units=2,
    ):
        self.neighborhood_units = neighborhood_units
        for a0 in range(self._num_sites):
            atoms_arr = self.atoms.get_data(a0)
            a_x = atoms_arr[:, 0]
            atom_neighbor_list = []
            self.added_to_neighbor_list_already = np.zeros(a_x.shape[0])
            self.num_neighbors = np.zeros(a_x.shape[0])
            for atom_index in range(a_x.shape[0]):
                neighbors_search = np.asarray([atom_index])
                self.added_to_neighbor_list_already[atom_index] = 1
                neighbors_search_out = np.array([atom_index])
                for neighbor_iteration in range(neighborhood_units):
                    neighbors_search_out = self.get_next_neighborhood_layer_arr(
                        neighbors_search_out
                    )
                    neighbors_search = np.concatenate((neighbors_search, neighbors_search_out))
                    if neighbors_search_out.size == 0:
                        break
                atom_neighbor_list.append(neighbors_search)
                self.num_neighbors[atom_index] = (
                    np.sum(self.added_to_neighbor_list_already) - 1
                )  # minus one because the central atom is not neighbor
                self.added_to_neighbor_list_already = np.zeros(a_x.shape[0])
        self.atom_neighbor_layer_arr = atom_neighbor_list
        return self

    def neighborhood_a(
        self,
        neighborhood_units=2,
    ):
        self.neighborhood_units = neighborhood_units
        for a0 in range(1):
            atoms_arr = self.atoms.get_data(a0)
            a_x = atoms_arr[:, 0]
            atom_neighbor_list = []
            self.added_to_neighbor_list_already = np.zeros(a_x.shape[0])
            self.num_neighbors = np.zeros(a_x.shape[0])
            for atom_index in range(a_x.shape[0]):
                neighbors_search = np.asarray([atom_index])
                self.added_to_neighbor_list_already[atom_index] = 1
                neighbors_search_out = np.array([atom_index])
                for neighbor_iteration in range(neighborhood_units):
                    neighbors_search_out = self.get_next_neighborhood_layer_arr(
                        neighbors_search_out
                    )
                    neighbors_search = np.concatenate((neighbors_search, neighbors_search_out))
                    if neighbors_search_out.size == 0:
                        break
                atom_neighbor_list.append(neighbors_search)
                self.num_neighbors[atom_index] = (
                    np.sum(self.added_to_neighbor_list_already) - 1
                )  # minus one because the central atom is not neighbor
                self.added_to_neighbor_list_already = np.zeros(a_x.shape[0])
        self.atom_neighbor_layer_arr = atom_neighbor_list
        return self

    # neighborhood b collects all of the b and a neighbors in the vicinity of central b atoms
    def neighborhood_b(
        self,
        neighborhood_units=2,
    ):
        self.neighborhood_units = neighborhood_units
        a_x_b = self.atoms.get_data(1)[:, 0]
        a_x_a = self.atoms.get_data(0)[:, 0]
        atom_neighbor_list_b = []
        atom_neighbor_list_a = []
        self.added_to_neighbor_list_already_b = np.zeros(a_x_b.shape[0])
        self.added_to_neighbor_list_already = np.zeros(a_x_a.shape[0])
        self.num_neighbors = np.zeros(a_x_b.shape[0])
        for atom_b_index in range(a_x_b.shape[0]):
            neighbors_search_a = np.asarray([atom_b_index])
            neighbors_search_b = np.asarray([atom_b_index])
            self.added_to_neighbor_list_already_b[atom_b_index] = 1
            neighbors_search_out_b = np.array([atom_b_index])
            neighbors_search_out_a = np.array([atom_b_index])
            for neighbor_iteration in range(neighborhood_units):
                if neighbor_iteration == 0:
                    neighbors_search_out_a = self.get_next_neighborhood_layer_arr_b(
                        neighbors_search_out_a, 0
                    )
                else:
                    neighbors_search_out_a = self.get_next_neighborhood_layer_arr(
                        neighbors_search_out_a
                    )

                neighbors_search_out_b = self.get_next_neighborhood_layer_arr_b(
                    neighbors_search_out_b, 1
                )
                neighbors_search_a = np.concatenate((neighbors_search_a, neighbors_search_out_a))
                if neighbor_iteration == 0:
                    neighbors_search_a = neighbors_search_a[1:]
                #     if atom_b_index < 2:
                #         print(atom_b_index)
                #         print(neighbors_search_a)
                # if atom_b_index < 10:
                #     print(neighbors_search_a)
                neighbors_search_b = np.concatenate((neighbors_search_b, neighbors_search_out_b))
                if neighbors_search_out_a.size == 0:
                    break
            atom_neighbor_list_a.append(neighbors_search_a)
            atom_neighbor_list_b.append(neighbors_search_b)
            self.num_neighbors[atom_b_index] = (
                np.sum(self.added_to_neighbor_list_already_b)
                + np.sum(self.added_to_neighbor_list_already)
                - 2
            )  # minus one because the central atom is not neighbor, and counted twice
            self.added_to_neighbor_list_already_b = np.zeros(a_x_b.shape[0])
            self.added_to_neighbor_list_already = np.zeros(a_x_a.shape[0])
        self.atom_neighbor_layer_arr_a = atom_neighbor_list_a
        self.atom_neighbor_layer_arr_b = atom_neighbor_list_b
        return self

    # for this, the number of A site neighbors will be good probably
    def find_neighbors_in_tolerance(
        self,
        tolerance=None,
    ):
        if not hasattr(self, "uv_norm"):
            self.uv_norm = np.mean(np.linalg.norm(self._lat[1:], axis=1))
        if tolerance is None:
            tolerance = self.uv_norm * 1.1
        num_sites = len(self._positions_frac)
        a_x = self.atoms.get_data(0)[:, 0]
        a_y = self.atoms.get_data(0)[:, 1]
        count_a_neighbors = np.zeros([num_sites, 2 * a_x.shape[0]])
        for site_index in range(num_sites):
            a_x_n = self.atoms.get_data(site_index)[:, 0]
            a_y_n = self.atoms.get_data(site_index)[:, 1]
            for atom_index in range(a_x_n.shape[0]):
                a_x_i = a_x_n[atom_index]
                a_y_i = a_y_n[atom_index]
                a_x_ai = a_x - a_x_i
                a_y_ai = a_y - a_y_i
                radial_dist = np.sqrt(a_x_ai**2 + a_y_ai**2)
                if site_index == 0:
                    radial_dist[atom_index] = (
                        tolerance * 2
                    )  # make sure that self is outside of range
                count_a_neighbors[site_index, atom_index] = np.sum(radial_dist < tolerance)
        self.count_a_neighbors = count_a_neighbors
        return self

    def remove_atoms_with_too_few_neighbors(
        self,
        min_neighbors=None,
        return_removed=False,
    ):
        if min_neighbors is None:
            min_neighbors = 2

        num_sites = len(self._positions_frac)
        removed = []
        for site_index in range(num_sites):
            site_data = self.atoms.get_data(site_index)
            keep_mask = self.count_a_neighbors[site_index, : site_data.shape[0]] >= min_neighbors
            updated = site_data[keep_mask]
            removed.append(site_data[~keep_mask])
            self.atoms.set_data(updated, site_index)
        if return_removed:
            return removed
        return self

    def find_atoms_with_too_few_neighbors(
        self,
        min_neighbors=None,
    ):
        if min_neighbors is None:
            min_neighbors = 2

        num_sites = len(self._positions_frac)
        found = []
        for site_index in range(num_sites):
            site_data = self.atoms.get_data(site_index)
            keep_mask = self.count_a_neighbors[site_index, : site_data.shape[0]] >= min_neighbors
            found.append(~keep_mask)
        return found

    def plot_neighbors(
        self,
        centers=None,
    ):
        if centers is None:
            centers = np.arange(0, 2)
        if isinstance(centers, int):
            centers = np.arange(0, centers)
        plt.figure()
        plt.imshow(self.image.array, cmap="gray")
        a_x_b = self.atoms.get_data(1)[:, 0]
        a_y_b = self.atoms.get_data(1)[:, 1]
        a_x_a = self.atoms.get_data(0)[:, 0]
        a_y_a = self.atoms.get_data(0)[:, 1]

        for atom_b_index in centers:
            plt.scatter(a_y_b[atom_b_index], a_x_b[atom_b_index], color="blue", zorder=10)
            plt.scatter(
                a_y_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)],
                a_x_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)],
                color="red",
                alpha=0.5,
            )
            plt.scatter(
                a_y_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)],
                a_x_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)],
                color="green",
            )

    def intensity_neighborhood(
        self,
        neighborhood_units=2,
        return_delta=False,
    ):
        self.neighborhood_units = neighborhood_units
        self.neighborhood_a(neighborhood_units=neighborhood_units)
        for a0 in range(self._num_sites):
            a_x = self.atoms[0]["x"]
            a_intensity = self.atoms[0]["int_peak"]
            delta_intensity = np.zeros([a_x.shape[0]])
            for atom_index in range(len(self.atom_neighbor_layer_arr)):
                neighbor_intensities = a_intensity[
                    self.atom_neighbor_layer_arr[atom_index][1:]
                ]  # 1: excludes the first one, which is itself
                median_intensity = np.median(neighbor_intensities)
                delta_intensity[atom_index] = a_intensity[atom_index] - median_intensity
        self.delta_intensities = delta_intensity
        if return_delta:
            return delta_intensity, self.num_neighbors
        else:
            return self

    def neighborhood_b_circular_distance(
        self,
        circular_radius_cutoff=None,
    ):
        a_x_b = self.atoms.get_data(1)[:, 0]
        a_y_b = self.atoms.get_data(1)[:, 1]
        a_x_a = self.atoms.get_data(0)[:, 0]
        a_y_a = self.atoms.get_data(0)[:, 1]

        if circular_radius_cutoff is None:
            lattice_spacing = self.uv_norm * np.linalg.norm(self._positions_frac[1]) * 0.9
            circular_radius_cutoff = lattice_spacing * self.neighborhood_units

        for atom_b_index in range(a_x_b.shape[0]):
            b_neighbor_x = a_x_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)]
            b_neighbor_y = a_y_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)]
            a_neighbor_x = a_x_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)]
            a_neighbor_y = a_y_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)]
            b_x = a_x_b[atom_b_index]
            b_y = a_y_b[atom_b_index]
            radial_dist_b = np.sqrt((b_neighbor_x - b_x) ** 2 + (b_neighbor_y - b_y) ** 2)
            radial_dist_a = np.sqrt((a_neighbor_x - b_x) ** 2 + (a_neighbor_y - b_y) ** 2)
            mask = radial_dist_b < circular_radius_cutoff
            self.atom_neighbor_layer_arr_b[atom_b_index] = self.atom_neighbor_layer_arr_b[
                atom_b_index
            ][mask]
            mask = radial_dist_a < circular_radius_cutoff
            self.atom_neighbor_layer_arr_a[atom_b_index] = self.atom_neighbor_layer_arr_a[
                atom_b_index
            ][mask]

        return self

    def gauss_2D_rot(
        self,
        x,
        y,
        xc,
        yc,
        xs,
        ys,
        A,
        B,
        theta,
    ):
        a = np.cos(theta) ** 2 / (2 * xs**2) + np.sin(theta) ** 2 / (2 * ys**2)
        b = -np.cos(theta) * np.sin(theta) / (2 * xs**2) + np.cos(theta) * np.sin(theta) / (
            2 * ys**2
        )
        c = np.sin(theta) ** 2 / (2 * xs**2) + np.cos(theta) ** 2 / (2 * ys**2)

        # arr_pd = np.array([[a, b], [b, c]])
        # check that arr_pd is positive definite
        # condition_1 = a > 0
        # condition_2 = a * c - b**2 > 0
        assert xs > 0
        assert ys > 0

        gaussian_2d = (
            A * np.exp(-(a * (x - xc) ** 2 + 2 * b * (x - xc) * (y - yc) + c * (y - yc) ** 2)) + B
        )
        return gaussian_2d

    def local_fitting_subtraction(
        self,
        fit_radius=None,
        max_nfev: int = 200,
        max_move_px: float | None = None,
        plot_atoms: bool = False,
    ):
        im = np.asarray(self._image.array, dtype=float)
        H, W = self._image.shape
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        A = np.column_stack((u, v))

        def _auto_radius_px() -> float:
            S = np.asarray(getattr(self, "_positions_frac", [[0.0, 0.0]]), dtype=float)
            if S.shape[0] >= 2:
                d = S[:, None, :] - S[None, :, :]
                d = d - np.round(d)
                same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
                dpix = d @ A.T
                dist = np.linalg.norm(dpix, axis=2)
                dist[same] = np.inf
                nn = float(np.min(dist))
            else:
                nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
            if not np.isfinite(nn) or nn <= 0:
                nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
            return 0.5 * nn

        r_fit = float(fit_radius) if fit_radius is not None else _auto_radius_px()
        R = int(np.ceil(r_fit))
        max_move = float(max_move_px) if max_move_px is not None else r_fit

        # Ensure extra fields exist
        needed = [f for f in ("sigma", "int_bg") if f not in self.atoms.fields]
        if needed:
            self.atoms.add_fields(needed)

        # Single lookup of column indices for writing
        idx_x = self.atoms.fields.index("x")
        idx_y = self.atoms.fields.index("y")
        idx_amp = self.atoms.fields.index("int_peak")
        idx_sigma = self.atoms.fields.index("sigma")
        idx_bg = self.atoms.fields.index("int_bg")

        a_x_b = self.atoms.get_data(1)[:, 0]
        a_y_b = self.atoms.get_data(1)[:, 1]
        a_s_b = self.atoms[1]["sigma"]
        a_ip_b = self.atoms[1]["int_peak"]

        a_x_a = self.atoms.get_data(0)[:, 0]
        a_y_a = self.atoms.get_data(0)[:, 1]
        a_s_a = self.atoms[0]["sigma"]
        a_ip_a = self.atoms[0]["int_peak"]

        H, W = self._image.shape
        window_pix = 10

        row = self.atoms.get_data(1)
        updated = row.copy()

        for atom_b_index in range(a_x_b.shape[0]):
            b_neighbor_x = a_x_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)]
            b_neighbor_y = a_y_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)]
            b_neighbor_s = a_s_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)]
            b_neighbor_ip = a_ip_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)]

            a_neighbor_x = a_x_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)]
            a_neighbor_y = a_y_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)]
            a_neighbor_s = a_s_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)]
            a_neighbor_ip = a_ip_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)]

            window_fit_x_max = np.ceil(
                np.max(np.concatenate([a_neighbor_x, b_neighbor_x])) + window_pix
            ).astype(int)
            window_fit_x_min = np.floor(
                np.min(np.concatenate([a_neighbor_x, b_neighbor_x])) - window_pix
            ).astype(int)
            window_fit_y_max = np.ceil(
                np.max(np.concatenate([a_neighbor_y, b_neighbor_y])) + window_pix
            ).astype(int)
            window_fit_y_min = np.floor(
                np.min(np.concatenate([a_neighbor_y, b_neighbor_y])) - window_pix
            ).astype(int)

            window_fit_x_max = min(window_fit_x_max, H)
            window_fit_x_min = max(window_fit_x_min, 0)
            window_fit_y_max = min(window_fit_y_max, W)
            window_fit_y_min = max(window_fit_y_min, 0)

            x = np.arange(window_fit_x_min, window_fit_x_max)
            y = np.arange(window_fit_y_min, window_fit_y_max)
            xx, yy = np.meshgrid(x, y, indexing="ij")
            sub_window = self._image.array[
                window_fit_x_min:window_fit_x_max, window_fit_y_min:window_fit_y_max
            ].copy()
            if plot_atoms:
                sub_window_before = sub_window.copy()

            for a_neighbor_index in range(a_neighbor_x.shape[0]):
                sub_window -= self.gauss_2D_rot(
                    xx,
                    yy,
                    a_neighbor_x[a_neighbor_index],
                    a_neighbor_y[a_neighbor_index],
                    a_neighbor_s[a_neighbor_index],
                    a_neighbor_s[a_neighbor_index],
                    a_neighbor_ip[a_neighbor_index],
                    0,
                    0,
                )
            for b_neighbor_index in range(b_neighbor_x.shape[0]):
                if self.atom_neighbor_layer_arr_b[atom_b_index][b_neighbor_index] != atom_b_index:
                    sub_window -= self.gauss_2D_rot(
                        xx,
                        yy,
                        b_neighbor_x[b_neighbor_index],
                        b_neighbor_y[b_neighbor_index],
                        b_neighbor_s[b_neighbor_index],
                        b_neighbor_s[b_neighbor_index],
                        b_neighbor_ip[b_neighbor_index],
                        0,
                        0,
                    )
                # since refine atoms already exists, going to start this without doing any additional refinement of the A site gaussians.
                # so calculate the gaussians, subtratct them, and decide what to do about background

            x0, y0 = float(a_x_b[atom_b_index]), float(a_y_b[atom_b_index])

            ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
            i0, i1 = max(0, ix0 - R), min(H - 1, ix0 + R)
            j0, j1 = max(0, iy0 - R), min(W - 1, iy0 + R)
            if i1 <= i0 or j1 <= j0:
                continue

            patch = im[i0 : i1 + 1, j0 : j1 + 1]

            # broadcast coordinate grids to patch shape
            ii = np.arange(i0, i1 + 1)[:, None]
            jj = np.arange(j0, j1 + 1)[None, :]
            II = np.broadcast_to(ii, patch.shape)
            JJ = np.broadcast_to(jj, patch.shape)

            r2 = (II - x0) ** 2 + (JJ - y0) ** 2
            mask = r2 <= (
                r_fit * r_fit
            )  # why not just square this with **? Or square root instead of r2
            if not np.any(mask):
                continue

            vals = patch[mask].astype(float).ravel()
            pmin, pmax = float(vals.min()), float(vals.max())
            bg0 = float(np.median(patch[~mask])) if np.any(~mask) else float(np.median(patch))
            amp0 = max(float(im[np.clip(ix0, 0, H - 1), np.clip(iy0, 0, W - 1)] - bg0), 1e-6)
            sig0 = max(r_fit * 0.5, 0.5)

            x_coords = II[mask].astype(float).ravel()
            y_coords = JJ[mask].astype(float).ravel()

            def residual(theta):
                x_c, y_c, amp, sig, bg = theta
                sig2 = max(sig, 1e-6) ** 2
                rr = (x_coords - x_c) ** 2 + (y_coords - y_c) ** 2
                model = amp * np.exp(-0.5 * rr / sig2) + bg
                return model - vals

            # movement-limited bounds + image bounds
            x_lb = max(x0 - max_move, 0.0)
            x_ub = min(x0 + max_move, H - 1.0)
            y_lb = max(y0 - max_move, 0.0)
            y_ub = min(y0 + max_move, W - 1.0)

            lb = [x_lb, y_lb, 0.0, 0.25, pmin - (pmax - pmin)]
            ub = [
                x_ub,
                y_ub,
                max(pmax - pmin, amp0 * 4.0),
                max(2.0 * r_fit, 1.0),
                pmax + (pmax - pmin),
            ]
            theta0 = [x0, y0, amp0, sig0, bg0]

            res = least_squares(
                residual,
                theta0,
                bounds=(lb, ub),
                method="trf",
                loss="soft_l1",
                max_nfev=int(max_nfev),
                xtol=1e-6,
                ftol=1e-6,
                gtol=1e-6,
            )

            x_c, y_c, amp, sig, bg = res.x
            updated[atom_b_index, idx_x] = x_c
            updated[atom_b_index, idx_y] = y_c
            updated[atom_b_index, idx_amp] = amp
            updated[atom_b_index, idx_sigma] = sig
            updated[atom_b_index, idx_bg] = bg
            if plot_atoms:
                if atom_b_index < 2:
                    plt.figure()
                    plt.subplot(121)
                    plt.imshow(sub_window_before)
                    plt.axis("off")
                    plt.subplot(122)
                    plt.imshow(sub_window)
                    plt.axis("off")

        # plot the difference in values
        # intensity, xc, yc, sigmas
        delta_int = a_ip_b - updated[:, idx_amp]
        delta_xc = a_x_b - updated[:, idx_x]
        delta_yc = a_y_b - updated[:, idx_y]
        delta_sig = a_s_b - updated[:, idx_sigma]

        # plt.figure(figsize = (10,10))
        # plt.subplot(221)
        # plt.imshow(self._image.array, cmap = 'gray')
        # plt.scatter(updated[:,idx_y], updated[:,idx_x], c = delta_xc, s = 40, alpha = 0.5, cmap = 'magma_r')
        # plt.title('Delta X Center')
        # plt.axis('off')
        # plt.subplot(222)
        # plt.imshow(self._image.array, cmap = 'gray')
        # plt.scatter(updated[:,idx_y], updated[:,idx_x], c = delta_yc, s = 40, alpha = 0.5, cmap = 'magma_r')
        # plt.title('Delta Y Center')
        # plt.axis('off')
        # plt.subplot(223)
        # plt.imshow(self._image.array, cmap = 'gray')
        # plt.scatter(updated[:,idx_y], updated[:,idx_x], c = delta_int, s = 40, alpha = 0.5, cmap = 'magma_r')
        # plt.title('Delta Intensity')
        # plt.axis('off')
        # plt.subplot(224)
        # plt.imshow(self._image.array, cmap = 'gray')
        # plt.scatter(updated[:,idx_y], updated[:,idx_x], c = delta_sig, s = 40, alpha = 0.5, cmap = 'magma_r')
        # plt.title('Delta Sigma')
        # plt.axis('off')
        # plt.tight_layout()

        s_plot = 40
        alpha_plot = 0.7
        cmap_plot = "magma"

        fig = plt.figure(figsize=(10, 10))

        ax1 = plt.subplot(221)
        ax1.imshow(self._image.array, cmap="gray")
        sc1 = ax1.scatter(
            updated[:, idx_y],
            updated[:, idx_x],
            c=delta_xc,
            s=s_plot,
            alpha=alpha_plot,
            cmap=cmap_plot,
        )
        ax1.set_title("Delta X Center")
        ax1.axis("off")
        fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = plt.subplot(222)
        ax2.imshow(self._image.array, cmap="gray")
        sc2 = ax2.scatter(
            updated[:, idx_y],
            updated[:, idx_x],
            c=delta_yc,
            s=s_plot,
            alpha=alpha_plot,
            cmap=cmap_plot,
        )
        ax2.set_title("Delta Y Center")
        ax2.axis("off")
        fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = plt.subplot(223)
        ax3.imshow(self._image.array, cmap="gray")
        sc3 = ax3.scatter(
            updated[:, idx_y],
            updated[:, idx_x],
            c=delta_int,
            s=s_plot,
            alpha=alpha_plot,
            cmap=cmap_plot,
        )
        ax3.set_title("Delta Intensity")
        ax3.axis("off")
        fig.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)

        ax4 = plt.subplot(224)
        ax4.imshow(self._image.array, cmap="gray")
        sc4 = ax4.scatter(
            updated[:, idx_y],
            updated[:, idx_x],
            c=delta_sig,
            s=s_plot,
            alpha=alpha_plot,
            cmap=cmap_plot,
        )
        ax4.set_title("Delta Sigma")
        ax4.axis("off")
        fig.colorbar(sc4, ax=ax4, fraction=0.046, pad=0.04)

        plt.tight_layout()

        self.atoms.set_data(updated, 1)
        return self

    def local_fitting_subtraction_1(
        self,
        atom_b_index,
        fit_radius=None,
        max_nfev: int = 200,
        max_move_px: float | None = None,
        plot_atoms: bool = False,
    ):
        im = np.asarray(self._image.array, dtype=float)
        H, W = self._image.shape
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        A = np.column_stack((u, v))

        def _auto_radius_px() -> float:
            S = np.asarray(getattr(self, "_positions_frac", [[0.0, 0.0]]), dtype=float)
            if S.shape[0] >= 2:
                d = S[:, None, :] - S[None, :, :]
                d = d - np.round(d)
                same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
                dpix = d @ A.T
                dist = np.linalg.norm(dpix, axis=2)
                dist[same] = np.inf
                nn = float(np.min(dist))
            else:
                nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
            if not np.isfinite(nn) or nn <= 0:
                nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
            return 0.5 * nn

        r_fit = float(fit_radius) if fit_radius is not None else _auto_radius_px()
        R = int(np.ceil(r_fit))
        max_move = float(max_move_px) if max_move_px is not None else r_fit

        # Ensure extra fields exist
        needed = [f for f in ("sigma", "int_bg") if f not in self.atoms.fields]
        if needed:
            self.atoms.add_fields(needed)

        a_x_b = self.atoms.get_data(1)[:, 0]
        a_y_b = self.atoms.get_data(1)[:, 1]
        a_s_b = self.atoms[1]["sigma"]
        a_ip_b = self.atoms[1]["int_peak"]
        a_ib_b = self.atoms[1]["int_bg"]

        a_x_a = self.atoms.get_data(0)[:, 0]
        a_y_a = self.atoms.get_data(0)[:, 1]
        a_s_a = self.atoms[0]["sigma"]
        a_ip_a = self.atoms[0]["int_peak"]

        H, W = self._image.shape
        window_pix = 10

        b_neighbor_x = a_x_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)]
        b_neighbor_y = a_y_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)]
        b_neighbor_s = a_s_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)]
        b_neighbor_ip = a_ip_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)]

        a_neighbor_x = a_x_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)]
        a_neighbor_y = a_y_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)]
        a_neighbor_s = a_s_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)]
        a_neighbor_ip = a_ip_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)]

        window_fit_x_max = np.ceil(
            np.max(np.concatenate([a_neighbor_x, b_neighbor_x])) + window_pix
        ).astype(int)
        window_fit_x_min = np.floor(
            np.min(np.concatenate([a_neighbor_x, b_neighbor_x])) - window_pix
        ).astype(int)
        window_fit_y_max = np.ceil(
            np.max(np.concatenate([a_neighbor_y, b_neighbor_y])) + window_pix
        ).astype(int)
        window_fit_y_min = np.floor(
            np.min(np.concatenate([a_neighbor_y, b_neighbor_y])) - window_pix
        ).astype(int)

        window_fit_x_max = min(window_fit_x_max, H)
        window_fit_x_min = max(window_fit_x_min, 0)
        window_fit_y_max = min(window_fit_y_max, W)
        window_fit_y_min = max(window_fit_y_min, 0)

        x = np.arange(window_fit_x_min, window_fit_x_max)
        y = np.arange(window_fit_y_min, window_fit_y_max)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        sub_window = self._image.array[
            window_fit_x_min:window_fit_x_max, window_fit_y_min:window_fit_y_max
        ].copy()
        if plot_atoms:
            sub_window_before = sub_window.copy()
        for a_neighbor_index in range(a_neighbor_x.shape[0]):
            sub_window -= self.gauss_2D_rot(
                xx,
                yy,
                a_neighbor_x[a_neighbor_index],
                a_neighbor_y[a_neighbor_index],
                a_neighbor_s[a_neighbor_index],
                a_neighbor_s[a_neighbor_index],
                a_neighbor_ip[a_neighbor_index],
                0,
                0,
            )
        for b_neighbor_index in range(b_neighbor_x.shape[0]):
            if self.atom_neighbor_layer_arr_b[atom_b_index][b_neighbor_index] != atom_b_index:
                sub_window -= self.gauss_2D_rot(
                    xx,
                    yy,
                    b_neighbor_x[b_neighbor_index],
                    b_neighbor_y[b_neighbor_index],
                    b_neighbor_s[b_neighbor_index],
                    b_neighbor_s[b_neighbor_index],
                    b_neighbor_ip[b_neighbor_index],
                    0,
                    0,
                )
                # since refine atoms already exists, going to start this without doing any additional refinement of the A site gaussians.
                # so calculate the gaussians, subtratct them, and decide what to do about background

        x0, y0 = float(a_x_b[atom_b_index]), float(a_y_b[atom_b_index])

        ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
        i0, i1 = max(0, ix0 - R), min(H - 1, ix0 + R)
        j0, j1 = max(0, iy0 - R), min(W - 1, iy0 + R)
        if i1 <= i0 or j1 <= j0:  # this doesn't do anything
            return (
                a_x_b[atom_b_index],
                a_y_b[atom_b_index],
                a_ip_b[atom_b_index],
                a_s_b[atom_b_index],
                a_ib_b[atom_b_index],
            )

        patch = im[i0 : i1 + 1, j0 : j1 + 1]

        # broadcast coordinate grids to patch shape
        ii = np.arange(i0, i1 + 1)[:, None]
        jj = np.arange(j0, j1 + 1)[None, :]
        II = np.broadcast_to(ii, patch.shape)
        JJ = np.broadcast_to(jj, patch.shape)

        r2 = (II - x0) ** 2 + (JJ - y0) ** 2
        mask = r2 <= (
            r_fit * r_fit
        )  # why not just square this with **? Or square root instead of r2
        if not np.any(mask):
            return (
                a_x_b[atom_b_index],
                a_y_b[atom_b_index],
                a_ip_b[atom_b_index],
                a_s_b[atom_b_index],
                a_ib_b[atom_b_index],
            )

        vals = patch[mask].astype(float).ravel()
        pmin, pmax = float(vals.min()), float(vals.max())
        bg0 = float(np.median(patch[~mask])) if np.any(~mask) else float(np.median(patch))
        amp0 = max(float(im[np.clip(ix0, 0, H - 1), np.clip(iy0, 0, W - 1)] - bg0), 1e-6)
        sig0 = max(r_fit * 0.5, 0.5)

        x_coords = II[mask].astype(float).ravel()
        y_coords = JJ[mask].astype(float).ravel()

        def residual(theta):
            x_c, y_c, amp, sig, bg = theta
            sig2 = max(sig, 1e-6) ** 2
            rr = (x_coords - x_c) ** 2 + (y_coords - y_c) ** 2
            model = amp * np.exp(-0.5 * rr / sig2) + bg
            return model - vals

        # movement-limited bounds + image bounds
        x_lb = max(x0 - max_move, 0.0)
        x_ub = min(x0 + max_move, H - 1.0)
        y_lb = max(y0 - max_move, 0.0)
        y_ub = min(y0 + max_move, W - 1.0)

        lb = [x_lb, y_lb, 0.0, 0.25, pmin - (pmax - pmin)]
        ub = [
            x_ub,
            y_ub,
            max(pmax - pmin, amp0 * 4.0),
            max(2.0 * r_fit, 1.0),
            pmax + (pmax - pmin),
        ]
        theta0 = [x0, y0, amp0, sig0, bg0]

        res = least_squares(
            residual,
            theta0,
            bounds=(lb, ub),
            method="trf",
            loss="soft_l1",
            max_nfev=int(max_nfev),
            xtol=1e-6,
            ftol=1e-6,
            gtol=1e-6,
        )

        if plot_atoms:
            plt.figure()
            plt.subplot(121)
            plt.imshow(sub_window_before)
            plt.axis("off")
            plt.subplot(122)
            plt.imshow(sub_window)
            plt.axis("off")

        x_c, y_c, amp, sig, bg = res.x
        return x_c, y_c, amp, sig, bg

    def local_fitting_subtraction_loop(
        self,
        fit_radius,
        max_move_px,
        max_nfev,
    ):
        # Single lookup of column indices for writing
        idx_x = self.atoms.fields.index("x")
        idx_y = self.atoms.fields.index("y")
        idx_amp = self.atoms.fields.index("int_peak")
        idx_sigma = self.atoms.fields.index("sigma")
        idx_bg = self.atoms.fields.index("int_bg")

        row = self.atoms.get_data(1)
        updated = row.copy()

        for atom_b_index in range(row.shape[0]):
            x_c, y_c, amp, sig, bg = self.local_fitting_subtraction_1(
                atom_b_index,
                fit_radius,
                max_move_px,
                max_nfev,
            )
            updated[atom_b_index, idx_x] = x_c
            updated[atom_b_index, idx_y] = y_c
            updated[atom_b_index, idx_amp] = amp
            updated[atom_b_index, idx_sigma] = sig
            updated[atom_b_index, idx_bg] = bg

        self.atoms.set_data(updated, 1)

    def auto_peak_finder(
        self,
        num_peaks_search=20,
        num_peaks_use=2,
        center_ignore_buffer=15,
        minSpacingPeaks=5,
    ):
        diffraction_peaks_list = self.locate_diffraction_spots(
            num_peaks_search,
            center_ignore_buffer=center_ignore_buffer,
            minSpacingPeaks=minSpacingPeaks,
        )
        if num_peaks_use == 2:
            peakA, peakB = self.locate_first_order_peaks(diffraction_peaks_list)
            diffraction_peaks_list = np.array([peakA, peakB])
        else:
            diffraction_peaks_list = np.array(
                [[diffraction_peaks_list[i]] for i in range(1, (num_peaks_use + 1))]
            )
        return diffraction_peaks_list

    def locate_first_order_peaks(
        self,
        peakCoordinates: np.dtype([("x", float), ("y", float), ("intensity", float)]),
    ):
        """
        Locate three low-order linearly independent peaks in k-space.

        Parameters
        ----------
        peakCoordinates: (number of peaks) np.ndarrary, np.dtype([("x", float), ("y", float), ("intensity", float)])
            An array of input peaks. This array should contain at least 2 linearly independent Bragg vectors.

        Returns
        -------
        peakA: np.dtype([("x", float), ("y", float), ("intensity", float)])
            The first peak (closest to central peak).
        peakB: np.dtype([("x", float), ("y", float), ("intensity", float)])
            The first peak (second closest to central peak).
        """
        nx, ny = self._image.shape
        midX = nx // 2
        midY = ny // 2
        peakCoordinatesRespCenter = np.zeros(
            len(peakCoordinates),
            dtype=np.dtype([("x", float), ("y", float), ("intensity", float)]),
        )
        peakCoordinatesRespCenter["x"] = peakCoordinates["x"] - midX
        peakCoordinatesRespCenter["y"] = peakCoordinates["y"] - midY
        peakRadialDistCenter = (
            peakCoordinatesRespCenter["x"] ** 2 + peakCoordinatesRespCenter["y"] ** 2
        )

        smallestRadiiIndices = np.argsort(peakRadialDistCenter)
        peakCoordinatesRespCenter = peakCoordinatesRespCenter[smallestRadiiIndices]

        # The closest peak should be the zero order peak - not interested in that.
        if peakRadialDistCenter[0] < 5:
            peakAInd = 1
            peakBInd = None
        else:
            peakAInd = 0
            peakBInd = None

        crossAWithRest = np.zeros(
            [len(peakCoordinates) - 2]
        )  # this 2 comes from the A peak and the central peak that are excluded from consideration for the B and C peaks
        peakA_xy = self.get_xy(peakCoordinatesRespCenter[peakAInd])
        for peakIndex in np.arange(2, len(peakCoordinates)):
            currentPeak = self.get_xy(peakCoordinatesRespCenter[peakIndex])
            crossAWithRest[peakIndex - 2] = np.cross(peakA_xy, currentPeak)
        threshold = 5 * (np.min(np.abs(crossAWithRest)) + 0.1)

        thresholdCondition = np.abs(crossAWithRest) > threshold
        if np.any(thresholdCondition):
            peakBInd = (
                np.argmax(thresholdCondition) + 2
            )  # returning the 2 that was subtracted above
        else:
            print("Lowering threshold B")
            threshold = 2 * (np.min(np.abs(crossAWithRest)) + 0.1)
            thresholdCondition = np.abs(crossAWithRest) > threshold
            peakBInd = np.argmax(thresholdCondition) + 2

        peakA = np.zeros(1, dtype=np.dtype([("x", float), ("y", float), ("intensity", float)]))
        peakB = np.zeros(1, dtype=np.dtype([("x", float), ("y", float), ("intensity", float)]))

        peakA["x"] = peakCoordinates["x"][smallestRadiiIndices[peakAInd]]
        peakA["y"] = peakCoordinates["y"][smallestRadiiIndices[peakAInd]]
        peakA["intensity"] = peakCoordinates["intensity"][smallestRadiiIndices[peakAInd]]
        peakB["x"] = peakCoordinates["x"][smallestRadiiIndices[peakBInd]]
        peakB["y"] = peakCoordinates["y"][smallestRadiiIndices[peakBInd]]
        peakB["intensity"] = peakCoordinates["intensity"][smallestRadiiIndices[peakBInd]]
        return peakA, peakB

    def locate_diffraction_spots(
        self,
        maxNumPeaks_in: int,
        minSpacingPeaks: int = 0,
        center_ignore_buffer: int | None = None,
    ):
        """
        Calls the maxima finder.

        Parameters
        ----------
        maxNumPeaks_in: int
            The number of peaks to return. Noisier data should use a smaller value. For 2D crystals, more than 3 peaks should be sought.
        Returns
        -------
        peakList: (maxNumPeaks_in) np.ndarray, np.dtype([("x", float), ("y", float), ("intensity", float)])
            An array of peak coordinates with a custom datatype.
        """
        nx, ny = self._image.shape
        peakList = self.get_maxima_2D(
            np.abs(np.fft.fftshift(np.fft.fft2(self._image.array))),
            maxNumPeaks=maxNumPeaks_in,
            minSpacing=minSpacingPeaks,
        )
        if center_ignore_buffer is not None:
            x_dist_to_center = peakList["x"] - nx / 2
            y_dist_to_center = peakList["y"] - ny / 2
            rad_dist_to_center = np.sqrt(x_dist_to_center**2 + y_dist_to_center**2)
            peakList = peakList[rad_dist_to_center > center_ignore_buffer]
            zero_peak = np.zeros(1, np.dtype([("x", float), ("y", float), ("intensity", float)]))
            zero_peak["x"] = nx / 2
            zero_peak["y"] = ny / 2
            peakList = np.append(zero_peak, peakList)
            return peakList
        else:
            return peakList

    def get_xy_2(
        self,
        coords_arr: np.dtype([("x", float), ("y", float), ("intensity", float)]),
    ):
        """
        Converts the custom dtype to an np.ndarray.

        Parameters
        ----------
        coords_arr: np.dtype([("x", float), ("y", float), ("intensity", float)])
            A single set of peak coordinates that has not already been indexed.

        Returns
        -------
        xyCoords: (2) np.ndarray
            A simple array with two entries giving the x (row) and y (column) coordinates of the input peak.
        """
        xyCoords = np.array([coords_arr["x"][0], coords_arr["y"][0]])
        return xyCoords

    def get_xy(
        self,
        coords_arr: np.dtype([("x", float), ("y", float), ("intensity", float)]),
    ):
        """
        Converts the custom dtype to an np.ndarray.

        Parameters
        ----------
        coords_arr: np.dtype([("x", float), ("y", float), ("intensity", float)])
            A single set of peak coordinates.

        Returns
        -------
        xyCoords: (2) np.ndarray
            A simple array with three entries giving the x (row) and y (column) coordinates of the input peak.
        """
        xyCoords = np.array([coords_arr["x"], coords_arr["y"]])
        return xyCoords

    def get_maxima_2D(
        self,
        ar: np.ndarray,
        subpixel: str = "poly",
        upsample_factor: int = 16,
        sigma: float = 0,
        minAbsoluteIntensity: float = 0,
        minRelativeIntensity: float = 0,
        relativeToPeak: float = 0,
        minSpacing: float = 0,
        edgeBoundary: int = 1,
        maxNumPeaks: int = 1,
        _ar_FT: np.ndarray | None = None,
    ):
        """
        Finds the maximal points of a 2D array.

        Parameters
        ----------
        ar: (nx, ny) np.ndarray
            The 2D image with peaks.
        subpixel: string
            specifies the subpixel resolution algorithm to use.
            must be in ('pixel','poly','multicorr'), which correspond
            to pixel resolution, subpixel resolution by fitting a
            parabola, and subpixel resultion by Fourier upsampling.
        upsample_factor: int
            the upsampling factor for the 'multicorr' algorithm
        sigma: float
            If > 0, applies a gaussian filter
        maxNumPeaks: int
            The maximum number of maxima to return
        minAbsoluteIntensity, minRelativeIntensity, relativeToPeak,
            minSpacing, edgeBoundary, maxNumPeaks: filtering applied
            after maximum detection and before subpixel refinement.
            Parameter descriptions in filter_2D_maxima.
        _ar_FT: (nx, ny) np.ndarray, complex
            If 'multicorr' is used and this is not None, uses this argument
            as the Fourier transform of `ar`, instead of recomputing it

        Returns
        -------
        maxima: np.ndarray, np.dtype([("x", float), ("y", float), ("intensity", float)])
            A structured array of maxima with fields 'x','y','intensity'
        """

        subpixel_modes = ("pixel", "poly", "multicorr")
        er = f"Unrecognized subpixel option {subpixel}. Must be in {subpixel_modes}"
        assert subpixel in subpixel_modes, er

        # gaussian filtering
        ar = ar if sigma <= 0 else gaussian_filter(ar, sigma)

        # local pixelwise maxima
        maxima_bool = (
            (ar >= np.roll(ar, (-1, 0), axis=(0, 1)))
            & (ar > np.roll(ar, (1, 0), axis=(0, 1)))
            & (ar >= np.roll(ar, (0, -1), axis=(0, 1)))
            & (ar > np.roll(ar, (0, 1), axis=(0, 1)))
            & (ar >= np.roll(ar, (-1, -1), axis=(0, 1)))
            & (ar > np.roll(ar, (-1, 1), axis=(0, 1)))
            & (ar >= np.roll(ar, (1, -1), axis=(0, 1)))
            & (ar > np.roll(ar, (1, 1), axis=(0, 1)))
        )

        # remove edges
        assert isinstance(edgeBoundary, (int, np.integer))
        if edgeBoundary < 1:
            edgeBoundary = 1
        maxima_bool[:edgeBoundary, :] = False
        maxima_bool[-edgeBoundary:, :] = False
        maxima_bool[:, :edgeBoundary] = False
        maxima_bool[:, -edgeBoundary:] = False

        # get indices
        # sort by intensity
        maxima_x, maxima_y = np.nonzero(maxima_bool)
        dtype = np.dtype([("x", float), ("y", float), ("intensity", float)])
        maxima = np.zeros(len(maxima_x), dtype=dtype)
        maxima["x"] = maxima_x
        maxima["y"] = maxima_y
        maxima["intensity"] = ar[maxima_x, maxima_y]
        maxima = np.sort(maxima, order="intensity")[::-1]

        if len(maxima) == 0:
            return maxima

        # filter
        maxima = self.filter_2D_maxima(
            maxima,
            minAbsoluteIntensity=minAbsoluteIntensity,
            minRelativeIntensity=minRelativeIntensity,
            relativeToPeak=relativeToPeak,
            minSpacing=minSpacing,
            edgeBoundary=edgeBoundary,
            maxNumPeaks=maxNumPeaks,
        )

        if subpixel == "pixel":
            return maxima

        # Parabolic subpixel refinement
        for i in range(len(maxima)):
            Ix1_ = ar[int(maxima["x"][i]) - 1, int(maxima["y"][i])].astype(np.float64)
            Ix0 = ar[int(maxima["x"][i]), int(maxima["y"][i])].astype(np.float64)
            Ix1 = ar[int(maxima["x"][i]) + 1, int(maxima["y"][i])].astype(np.float64)
            Iy1_ = ar[int(maxima["x"][i]), int(maxima["y"][i]) - 1].astype(np.float64)
            Iy0 = ar[int(maxima["x"][i]), int(maxima["y"][i])].astype(np.float64)
            Iy1 = ar[int(maxima["x"][i]), int(maxima["y"][i]) + 1].astype(np.float64)
            deltax = (Ix1 - Ix1_) / (4 * Ix0 - 2 * Ix1 - 2 * Ix1_)
            deltay = (Iy1 - Iy1_) / (4 * Iy0 - 2 * Iy1 - 2 * Iy1_)
            maxima["x"][i] += deltax
            maxima["y"][i] += deltay
            maxima["intensity"][i] = self.linear_interpolation_2D(
                ar, maxima["x"][i], maxima["y"][i]
            )

        if subpixel == "poly":
            return maxima

        # Fourier upsampling
        if _ar_FT is None:
            _ar_FT = np.fft.fft2(ar)
        for ipeak in range(len(maxima["x"])):
            xyShift = np.array((maxima["x"][ipeak], maxima["y"][ipeak]))
            # we actually have to lose some precision and go down to half-pixel
            # accuracy for multicorr
            xyShift[0] = np.round(xyShift[0] * 2) / 2
            xyShift[1] = np.round(xyShift[1] * 2) / 2

            subShift = self.upsampled_correlation(_ar_FT, upsample_factor, xyShift)
            maxima["x"][ipeak] = subShift[0]
            maxima["y"][ipeak] = subShift[1]

        maxima = np.sort(maxima, order="intensity")[::-1]
        return maxima

    def filter_2D_maxima(
        self,
        maxima,
        minAbsoluteIntensity=0,
        minRelativeIntensity=0,
        relativeToPeak=0,
        minSpacing=0,
        edgeBoundary=1,
        maxNumPeaks=1,
    ):
        """
        Args:
            maxima : a numpy structured array with fields 'x', 'y', 'intensity'
            minAbsoluteIntensity : delete counts with intensity below this value
            minRelativeIntensity : delete counts with intensity below this value times
                the intensity of the i'th peak, where i is given by `relativeToPeak`
            relativeToPeak : see above
            minSpacing : if two peaks are within this euclidean distance from one
                another, delete the less intense of the two
            edgeBoundary : delete peaks within this distance of the image edge
            maxNumPeaks : an integer. defaults to 1

        Returns:
            a numpy structured array with fields 'x', 'y', 'intensity'
        """

        # Remove maxima which are too dim
        if minAbsoluteIntensity > 0:
            deletemask = maxima["intensity"] < minAbsoluteIntensity
            maxima = maxima[~deletemask]

        # Remove maxima which are too dim, compared to the n-th brightest
        if (minRelativeIntensity > 0) & (len(maxima) > relativeToPeak):
            assert isinstance(relativeToPeak, (int, np.integer))
            deletemask = (
                maxima["intensity"] / maxima["intensity"][relativeToPeak] < minRelativeIntensity
            )
            maxima = maxima[~deletemask]

        # Remove maxima which are too close
        if minSpacing > 0:
            deletemask = np.zeros(len(maxima), dtype=bool)
            for i in range(len(maxima)):
                if deletemask[i] == False:  # noqa: E712
                    tooClose = (
                        (maxima["x"] - maxima["x"][i]) ** 2 + (maxima["y"] - maxima["y"][i]) ** 2
                    ) < minSpacing**2
                    tooClose[: i + 1] = False
                    deletemask[tooClose] = True
            maxima = maxima[~deletemask]

        # Remove maxima in excess of maxNumPeaks
        if maxNumPeaks is not None:
            if len(maxima) > maxNumPeaks:
                maxima = maxima[:maxNumPeaks]

        return maxima

    def linear_interpolation_2D(self, ar, x, y):
        """
        Calculates the 2D linear interpolation of array ar at position x,y using the four
        nearest array elements.
        """
        x0, x1 = int(np.floor(x)), int(np.ceil(x))
        y0, y1 = int(np.floor(y)), int(np.ceil(y))
        dx = x - x0
        dy = y - y0
        return (
            (1 - dx) * (1 - dy) * ar[x0, y0]
            + (1 - dx) * dy * ar[x0, y1]
            + dx * (1 - dy) * ar[x1, y0]
            + dx * dy * ar[x1, y1]
        )

    def upsampled_correlation(self, imageCorr, upsampleFactor, xyShift, device="cpu"):
        """
        Refine the correlation peak of imageCorr around xyShift by DFT upsampling.

        There are two approaches to Fourier upsampling for subpixel refinement: (a) one
        can pad an (appropriately shifted) FFT with zeros and take the inverse transform,
        or (b) one can compute the DFT by matrix multiplication using modified
        transformation matrices. The former approach is straightforward but requires
        performing the FFT algorithm (which is fast) on very large data. The latter method
        trades one speedup for a slowdown elsewhere: the matrix multiply steps are expensive
        but we operate on smaller matrices. Since we are only interested in a very small
        region of the FT around a peak of interest, we use the latter method to get
        a substantial speedup and enormous decrease in memory requirement. This
        "DFT upsampling" approach computes the transformation matrices for the matrix-
        multiply DFT around a small 1.5px wide region in the original `imageCorr`.

        Following the matrix multiply DFT we use parabolic subpixel fitting to
        get even more precision! (below 1/upsampleFactor pixels)

        NOTE: previous versions of multiCorr operated in two steps: using the zero-
        padding upsample method for a first-pass factor-2 upsampling, followed by the
        DFT upsampling (at whatever user-specified factor). I have implemented it
        differently, to better support iterating over multiple peaks. **The DFT is always
        upsampled around xyShift, which MUST be specified to HALF-PIXEL precision
        (no more, no less) to replicate the behavior of the factor-2 step.**
        (It is possible to refactor this so that peak detection is done on a Fourier
        upsampled image rather than using the parabolic subpixel and rounding as now...
        I like keeping it this way because all of the parameters and logic will be identical
        to the other subpixel methods.)


        Args:
            imageCorr (complex valued ndarray):
                Complex product of the FFTs of the two images to be registered
                i.e. m = np.fft.fft2(DP) * probe_kernel_FT;
                imageCorr = np.abs(m)**(corrPower) * np.exp(1j*np.angle(m))
            upsampleFactor (int):
                Upsampling factor. Must be greater than 2. (To do upsampling
                with factor 2, use upsampleFFT, which is faster.)
            xyShift:
                Location in original image coordinates around which to upsample the
                FT. This should be given to exactly half-pixel precision to
                replicate the initial FFT step that this implementation skips

        Returns:
            (2-element np array): Refined location of the peak in image coordinates.
        """

        if device == "cpu":
            xp = np

        assert upsampleFactor > 2

        xyShift[0] = xp.round(xyShift[0] * upsampleFactor) / upsampleFactor
        xyShift[1] = xp.round(xyShift[1] * upsampleFactor) / upsampleFactor

        globalShift = xp.fix(xp.ceil(upsampleFactor * 1.5) / 2)

        upsampleCenter = xp.asarray(globalShift - upsampleFactor * xyShift)

        imageCorrUpsample = xp.conj(
            self.dftUpsample(xp.conj(imageCorr), upsampleFactor, upsampleCenter, device=device)
        )

        xySubShift = xp.asarray(
            xp.unravel_index(imageCorrUpsample.argmax(), imageCorrUpsample.shape)
        )

        # add a subpixel shift via parabolic fitting
        try:
            icc = xp.real(
                imageCorrUpsample[
                    xySubShift[0] - 1 : xySubShift[0] + 2,
                    xySubShift[1] - 1 : xySubShift[1] + 2,
                ]
            )
            dx = (icc[2, 1] - icc[0, 1]) / (4 * icc[1, 1] - 2 * icc[2, 1] - 2 * icc[0, 1])
            dy = (icc[1, 2] - icc[1, 0]) / (4 * icc[1, 1] - 2 * icc[1, 2] - 2 * icc[1, 0])
        except (IndexError, TypeError):
            dx, dy = (
                0,
                0,
            )  # this is the case when the peak is near the edge and one of the above values does not exist

        xySubShift = xySubShift - globalShift

        xyShift = xyShift + (xySubShift + xp.array([dx, dy])) / upsampleFactor

        return xyShift

    def dftUpsample(self, imageCorr, upsampleFactor, xyShift, device="cpu"):
        """
        This performs a matrix multiply DFT around a small neighboring region of the inital
        correlation peak. By using the matrix multiply DFT to do the Fourier upsampling, the
        efficiency is greatly improved. This is adapted from the subfuction dftups found in
        the dftregistration function on the Matlab File Exchange.

        https://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation

        The matrix multiplication DFT is from:

        Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel
        image registration algorithms," Opt. Lett. 33, 156-158 (2008).
        http://www.sciencedirect.com/science/article/pii/S0045790612000778

        Args:
            imageCorr (complex valued ndarray):
                Correlation image between two images in Fourier space.
            upsampleFactor (int):
                Scalar integer of how much to upsample.
            xyShift (list of 2 floats):
                Coordinates in the UPSAMPLED GRID around which to upsample.
                These must be single-pixel IN THE UPSAMPLED GRID

        Returns:
            (ndarray):
                Upsampled image from region around correlation peak.
        """
        if device == "cpu":
            xp = np

        imageSize = imageCorr.shape
        pixelRadius = 1.5
        numRow = np.ceil(pixelRadius * upsampleFactor)
        numCol = numRow

        colKern = xp.exp(
            (-1j * 2 * np.pi / (imageSize[1] * upsampleFactor))
            * xp.outer(
                (xp.fft.ifftshift((xp.arange(imageSize[1]))) - xp.floor(imageSize[1] / 2)),
                (xp.arange(numCol) - xyShift[1]),
            )
        )

        rowKern = xp.exp(
            (-1j * 2 * np.pi / (imageSize[0] * upsampleFactor))
            * xp.outer(
                (xp.arange(numRow) - xyShift[0]),
                (xp.fft.ifftshift(xp.arange(imageSize[0])) - xp.floor(imageSize[0] / 2)),
            )
        )

        imageUpsample = xp.real(rowKern @ imageCorr @ colKern)
        return imageUpsample

    def measure_polarization(
        self,
        measure_ind: int,
        reference_ind: int,
        reference_radius: float | None = None,
        min_neighbours: int | None = 2,
        max_neighbours: int | None = None,
        plot_polarization_vectors: bool = False,
        **plot_kwargs,
    ) -> "Vector":
        """
        Measure the polarization of atoms at one site with respect to atoms at another site.
        Polarization is computed as a fractional displacement (da, db) of each atom in the
        'measure' site relative to the expected position inferred from the nearest atoms
        in the 'reference' site and the current lattice vectors. The expected position is
        the mean of neighbor positions shifted by the lattice vector transform of the
        fractional index difference.

        Parameters
        ----------
        measure_ind : int
            Index of the site whose polarization is to be measured.
            This corresponds to the index in `positions_frac` used in `add_atoms()`.
        reference_ind : int
            Index of the reference site used to calculate polarization.
            This corresponds to the index in `positions_frac` used in `add_atoms()`.
        reference_radius : float | None, default=None
            If provided, neighbors are selected by radius search (in pixels) using a KD-tree.
            Must be at least 1 pixel. If None, neighbors are selected by k-nearest search.
        min_neighbours : int | None, default=2
            Minimum number of nearest neighbors used to calculate polarization. Must be >= 2
            when using k-nearest search (i.e., when `reference_radius` is None).
        max_neighbours : int | None, default=None
            Maximum number of nearest neighbors to use. Required when `reference_radius` is None.
        plot_polarization_vectors : bool, default=False
            If True, plots the polarization vectors using `self.plot_polarization_vectors(...)`.
        **plot_kwargs
            Additional keyword arguments forwarded to the plotting function.

        Returns
        -------
        out : quantem.core.datastructures.vector.Vector
            A Vector object containing the polarizations with:
            - shape=(1,)
            - fields=("x", "y", "a", "b", "da", "db")
            - units=("px", "px", "ind", "ind", "ind", "ind")
            Here, (x, y) are positions in pixels, (a, b) are fractional indices,
            and (da, db) are fractional displacements (polarization).

        Raises
        ------
        ValueError
            - If the lattice vectors are singular (cannot invert).
            - If neither `reference_radius` nor both `min_neighbours` and `max_neighbours` are specified.
            - If `reference_radius` < 1.
            - If radius-based search fails to find at least `min_neighbours` for any atom.
            - If k-nearest search is used and `min_neighbours` or `max_neighbours` is missing.
            - If k-nearest search is used with `min_neighbours` < 2 or `max_neighbours` < 2.
            - If `min_neighbours` > `max_neighbours`.
            - If no atoms have any neighbors identified (increase `reference_radius`).
        DeprecationWarning
            If `reference_num` is provided.
            Use `max_neighbours` and `min_neighbours` instead.
        Warning
            If some atoms do not have any neighbors identified (suggests increasing `reference_radius`).

        Notes
        -----
        - Lattice vectors are taken from `self._lat` and are in pixel units.
        - Neighbor selection:
            - If `reference_radius` is provided, a radius search (KD-tree) is used and optionally
                truncated by `max_neighbours`.
            - If `reference_radius` is None, k-nearest neighbors are used with `k=max_neighbours`.
        - The expected position for each measured atom is computed as the mean over selected
        neighbors of: neighbor_position + L @ ([a - a_i, b - b_i]), where L = [u v], and
        (a, b) and (a_i, b_i) are the fractional indices of the measured atom and the neighbor,
        respectively. The polarization (da, db) is then obtained by transforming the
        Cartesian displacement back to fractional coordinates using L^{-1}.
        - If either the measure or reference site is empty, an empty Vector (with zero rows) is returned.
        """
        from scipy.spatial import cKDTree

        # This is temporary. In case any old notebooks are still using "reference_num"
        if "reference_num" in plot_kwargs:
            if max_neighbours is None:
                max_neighbours = plot_kwargs["reference_num"]
                raise DeprecationWarning(
                    "'reference_num' is deprecated. Use 'max_neighbours' and 'min_neighbours'."
                )

        # lattice vectors in pixels
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)

        measure_ind = int(measure_ind)
        reference_ind = int(reference_ind)

        def is_empty(cell):
            if cell is None:
                return True
            if isinstance(cell, list):
                return len(cell) == 0
            if isinstance(cell, dict):
                x = cell.get("x", None)
                return x is None or np.size(x) == 0
            # Fallback to numpy-like objects
            if hasattr(cell, "size"):
                return cell.size == 0
            return False

        # Check for empty cells
        A_cell = self.atoms.get_data(measure_ind)
        B_cell = self.atoms.get_data(reference_ind)
        self._pol_meas_ref_ind = (measure_ind, reference_ind)

        # Prepare a Vector with structured dtype (even for empty data)
        fields = ("x", "y", "a", "b", "da", "db")
        units = ("px", "px", "ind", "ind", "ind", "ind")

        def empty_vector():
            out = Vector.from_shape(
                shape=(1,),
                fields=fields,
                units=units,
                name="polarization",
            )
            # Create empty array with shape (0, 6) to match expected format
            empty_data = np.zeros((0, 6), dtype=float)
            out.set_data(empty_data, 0)
            return out

        if is_empty(A_cell) or is_empty(B_cell):
            return empty_vector()

        # Extract common atom data
        Ax = self.atoms[measure_ind]["x"]
        Ay = self.atoms[measure_ind]["y"]
        Aa = self.atoms[measure_ind]["a"]
        Ab = self.atoms[measure_ind]["b"]
        Bx = self.atoms[reference_ind]["x"]
        By = self.atoms[reference_ind]["y"]
        Ba = self.atoms[reference_ind]["a"]
        Bb = self.atoms[reference_ind]["b"]

        if Ax.size == 0 or Bx.size == 0:
            return empty_vector()

        # Lattice vectors: r0 (unused here), u, v
        lat = np.asarray(getattr(self, "_lat", None))
        if lat is None or lat.shape[0] < 3:
            raise ValueError("Lattice vectors (_lat) are missing or malformed.")
        _, u, v = lat[0], lat[1], lat[2]
        L = np.column_stack((u, v))
        try:
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            raise ValueError("Lattice vectors are singular and cannot be inverted.")

        query_coords = np.column_stack([Ax, Ay])
        ref_coords = np.column_stack([Bx, By])

        # Pre-allocate result array memory
        x_arr = Ax.copy().astype(float)
        y_arr = Ay.copy().astype(float)
        a_arr = Aa.copy().astype(float)
        b_arr = Ab.copy().astype(float)
        da_arr = np.zeros_like(x_arr, dtype=float)
        db_arr = np.zeros_like(x_arr, dtype=float)

        # KD-tree query
        tree = cKDTree(ref_coords)

        if max_neighbours is None and reference_radius is None:
            raise ValueError(
                "Either min_neighbours or max_neighbours or reference_radius must be passed."
            )

        # Initialize arrays for results
        dists = []
        idxs = []

        if reference_radius is not None:
            # Radius-based query
            if reference_radius < 1:
                raise ValueError(
                    f"reference_radius must be atleast 1 pixel. You have passed : {reference_radius}"
                )

            neighbor_lists = tree.query_ball_point(
                query_coords,
                r=reference_radius,
                workers=-1,
            )

            # Vectorized distance calculations where possible
            for i, neighbors in enumerate(neighbor_lists):
                if len(neighbors) == 0:
                    dists.append(np.array([]))
                    idxs.append(np.array([]))
                    continue

                # Vectorized distance calculation
                neighbor_coords = ref_coords[neighbors]
                query_point = query_coords[i]
                distances = np.linalg.norm(neighbor_coords - query_point, axis=1)

                # Vectorized sorting
                sort_idx = np.argsort(distances)
                sorted_distances = distances[sort_idx]
                sorted_indices = np.array(neighbors)[sort_idx]

                # Apply max_neighbours limit if specified
                if max_neighbours is not None and len(sorted_distances) > max_neighbours:
                    sorted_distances = sorted_distances[:max_neighbours]
                    sorted_indices = sorted_indices[:max_neighbours]

                dists.append(sorted_distances)
                idxs.append(sorted_indices)

            # Vectorized length checking
            lengths = np.array([len(row) for row in dists])
            if min_neighbours is not None and np.any(lengths < min_neighbours):
                raise ValueError(
                    "Failed to calculate enough nearest neighbours. Increase the reference_radius"
                )

        elif reference_radius is None:
            # K-nearest neighbors query
            if min_neighbours is None or max_neighbours is None:
                raise ValueError(
                    "min_neighbours and max_neighbours should be specified if reference_radius is None"
                )
            if min_neighbours < 2 or max_neighbours < 2:
                raise ValueError(
                    "Must use atleast 2 nearest neighbours to calculate the Polarization"
                )
            if min_neighbours > max_neighbours:
                raise ValueError("'min_neighbours' cannot be larger than 'max_neighbours'")

            dist_array, idx_array = tree.query(
                query_coords,
                k=max_neighbours,
                workers=-1,
            )

            # Vectorized processing of results
            finite_mask = np.isfinite(dist_array)
            for i in range(len(query_coords)):
                mask = finite_mask[i]
                dists.append(dist_array[i][mask])
                idxs.append(idx_array[i][mask])

        # Vectorized neighbor checking
        lengths = np.array([len(row) for row in dists])
        atoms_with_atleast_one_neighbour = lengths > 0

        if not np.any(atoms_with_atleast_one_neighbour):
            raise ValueError(
                "Failed to calculate nearest neighbours for all atoms. Increase reference_radius."
            )

        if not np.all(atoms_with_atleast_one_neighbour):
            missing_count = len(atoms_with_atleast_one_neighbour) - np.sum(
                atoms_with_atleast_one_neighbour
            )
            raise Warning(
                f"{missing_count} atoms do not have any neighbours identified. Try increasing reference_radius."
            )

        # Pre-allocate arrays for better performance
        da_arr = np.zeros(len(query_coords))
        db_arr = np.zeros(len(query_coords))

        # Calculate displacements with optimizations
        for i, (atom_dists, atom_idxs) in enumerate(zip(dists, idxs)):
            if len(atom_idxs) == 0:
                continue  # Arrays already initialized to 0

            # Check if we have enough neighbors
            if min_neighbours is not None and len(atom_idxs) < min_neighbours:
                continue  # Arrays already initialized to 0

            # Determine how many neighbors to use
            num_neighbors_to_use = len(atom_idxs)
            if max_neighbours is not None:
                num_neighbors_to_use = min(num_neighbors_to_use, max_neighbours)
            if min_neighbours is not None:
                num_neighbors_to_use = max(
                    num_neighbors_to_use, min(min_neighbours, len(atom_idxs))
                )

            # Select the neighbors to use (closest ones) - optimized
            if num_neighbors_to_use < len(atom_idxs):
                # Use argpartition for better performance when we don't need full sort
                closest_order = np.argpartition(atom_dists, num_neighbors_to_use)[
                    :num_neighbors_to_use
                ]
                nbr_idx = atom_idxs[closest_order].astype(int)
            else:
                nbr_idx = atom_idxs.astype(int)

            # Vectorized position calculations
            actual_pos = np.array([x_arr[i], y_arr[i]])

            # Vectorized fractional calculations
            a, b = a_arr[i], b_arr[i]
            ai, bi = Ba[nbr_idx], Bb[nbr_idx]
            xi, yi = Bx[nbr_idx], By[nbr_idx]

            # Vectorized matrix operations
            fractional_diff = np.array([a - ai, b - bi])  # (2, n_neighbors)
            neighbor_positions = np.array([xi, yi])  # (2, n_neighbors)

            # Single matrix multiplication for all neighbors
            expected_positions = neighbor_positions + L @ fractional_diff  # (2, n_neighbors)

            # Vectorized mean calculation
            expected_position = np.mean(expected_positions, axis=1)  # (2,)

            # Vectorized displacement calculations
            displacement_cartesian = actual_pos - expected_position
            displacement_fractional = L_inv @ displacement_cartesian

            # Direct assignment
            da_arr[i] = displacement_fractional[0]
            db_arr[i] = displacement_fractional[1]

        out = Vector.from_shape(
            shape=(1,),
            fields=("x", "y", "a", "b", "x_ref", "y_ref"),
            units=("px", "px", "ind", "ind", "px", "px"),
            name="polarization",
        )

        # Create structured array if needed
        if len(x_arr) > 0:
            arr = np.column_stack([x_arr, y_arr, a_arr, b_arr, da_arr, db_arr])
        else:
            # Create empty array with shape (0, 6)
            arr = np.zeros((0, 6), dtype=float)

        out.set_data(arr, 0)

        if plot_polarization_vectors:
            self.plot_polarization_vectors(out, **plot_kwargs)

        return out

    def calculate_order_parameter(
        self,
        polarization_vectors: Vector,
        num_phases: int = 2,
        phase_polarization_peak_array: NDArray | None = None,
        fix_polarization_peaks: bool = False,
        plot_order_parameter: bool = True,
        plot_gmm_visualization: bool = True,
        torch_device: str = "cpu",
        # plot_confidence_map : bool = False,
        **kwargs,
    ) -> "Lattice":
        """
        Estimate a multi-phase order parameter by fitting a Gaussian Mixture Model (GMM)
        to fractional polarization components (da, db). The order parameter for each site
        is defined as the posterior membership probabilities (responsibilities) of the
        fitted GMM components evaluated in the 2D polarization space.

        The method can optionally:
            - Use provided phase centers (polarization peaks) to initialize or fix the GMM means.
            - Visualize the mixture model in (da, db) space with KDE density, centers, and
            ~95% confidence ellipses.
            - Overlay the order parameter (probability-colored sites) on the original image grid.

        Parameters
            - polarization_vectors: Vector
            A collection holding polarization data. Only the first element
            polarization_vectors[0] is used and must provide the following keys:
                - 'x': NDArray of shape (N,), row coordinates for each site.
                - 'y': NDArray of shape (N,), column coordinates for each site.
                - 'da': NDArray of shape (N,), fractional polarization along a (e.g., du).
                - 'db': NDArray of shape (N,), fractional polarization along b (e.g., dv).
            All arrays must be one-dimensional, aligned, and of equal length N.

            - num_phases: int, default=2
            Number of Gaussian components (phases) in the mixture. Must be >= 1.
            For num_phases=1, all sites belong to a single phase (probabilities are all 1).

            - phase_polarization_peak_array: NDArray | None, default=None
            Optional array of shape (num_phases, 2) specifying phase centers (means)
            in (da, db) space:
                - If fix_polarization_peaks=False, these values initialize the GMM means.
                - If fix_polarization_peaks=True, the means are held fixed during fitting
                    and only covariances and weights are updated.

            - fix_polarization_peaks: bool, default=False
            If True, requires phase_polarization_peak_array to be provided with shape
            (num_phases, 2). The GMM means are fixed to these values throughout EM.

            - plot_order_parameter: bool, default=True
            If True, overlays sites on self._image.array and colors them by their full
            mixture probability distribution:
                - For 2 phases, adds a two-color probability bar.
                - For 3 phases, adds a ternary-style color triangle.
                - For other values, no legend is shown.

            - plot_gmm_visualization: bool, default=True
            If True, shows a visualization in (da, db) space:
                - A Gaussian KDE density (scipy.stats.gaussian_kde) on a symmetric grid
                    spanning max(abs(da), abs(db)).
                - Scatter of points colored by mixture probabilities.
                - GMM centers (means) and ~95% confidence ellipses (2 standard deviations).

            - torch_device: str, default='cpu'
            Torch device used by the TorchGMM backend. Examples: 'cpu', 'cuda',
            'cuda:0'. If a CUDA device is requested but unavailable, the underlying
            GMM implementation may raise an error.

            - **kwargs: Additional keyword arguments controlling visualization.
                When plot_gmm_visualization=True, the following keys are supported and validated:
                - contour_cmap: Matplotlib colormap name for the background contour;
                    invalid names fall back to a preset ('gray') with a warning.
                - gmm_center_colour: Color for GMM center markers;
                    invalid values fall back to a preset with a warning.
                    Presets depend on num_phases (2: 'lime'; 3-4: 'Yellow'; ≥5: 'Black').
                - gmm_ellipse_colour: Color for GMM covariance ellipses;
                    invalid values fall back to a preset with a warning.
                    Presets depend on num_phases (2: 'lime'; 3-4: 'Yellow'; ≥5: 'White').
                - scatter_colours: Colors used to map phase probabilities for scatter points
                    (and the order-parameter map). Accepted forms:
                        • callable f(i) -> RGB(A) (first 3 components used),
                        • numpy array of shape (num_phases, 3) with RGB in [0, 1],
                        • list/tuple of valid color names/values of length num_phases,
                        • single valid color (applied to all phases; prints a warning).
                    Invalid inputs fall back to a preset (site_colors) with a warning.
                    When plot_order_parameter=True,
                    scatter_colours is used to color points by phase probabilities.
                Optionally, kwargs intended for show_2d (e.g., cmap, title, vmin, vmax)
                may be provided and forwarded

        Returns
            - self:
            The same object, modified in-place.

        Side Effects
            - Sets the following attributes on self:
            - self._polarization_means: NDArray of shape (num_phases, 2),
                the fitted (or fixed) means in (da, db) space.
            - self._order_parameter_probabilities: NDArray of shape (N, num_phases),
                posterior probabilities per site.
            - Produces plots if plot_gmm_visualization or plot_order_parameter is True.

        Notes
            - The GMM uses full covariance matrices (covariance_type='full') and an EM
            implementation backed by TorchGMM (PyTorch).
            - The KDE contour limits are symmetric around the origin and set by
            max(abs(da), abs(db)).
            - In the order-parameter overlay, coordinates are plotted as:
            x-axis: 'y' (column), y-axis: 'x' (row).
            - Helper functions expected to exist:
            - create_colors_from_probabilities(probabilities, num_phases)
            - add_2phase_colorbar(ax)
            - add_3phase_color_triangle(fig, ax)
            - show_2d(image, ...)
            - Requires self._image.array to be present for the order-parameter overlay.

        Raises
            - ValueError:
            If phase_polarization_peak_array is provided with incorrect shape
            (must be (num_phases, 2)).
            - ValueError:
            If fix_polarization_peaks=True and phase_polarization_peak_array is None.
            - AttributeError:
            If plot_order_parameter=True but self._image or self._image.array is missing.
            - ImportError:
            If required plotting/scientific packages (matplotlib, scipy) are unavailable.
            - RuntimeError or ValueError (from TorchGMM):
            If the torch device is invalid or unavailable.

        Examples
            - Fit a 2-phase GMM and show both visualizations:
            lattice.calculate_order_parameter(
                polarization_vectors,
                num_phases=2,
                plot_gmm_visualization=True,
                plot_order_parameter=True
            )

            - Use fixed phase peaks:
            peaks = np.array([[0.10, -0.05],
                                [0.30,  0.07]], dtype=float)
            lattice.calculate_order_parameter(
                polarization_vectors,
                num_phases=2,
                phase_polarization_peak_array=peaks,
                fix_polarization_peaks=True
            )

            - Run on GPU (if available):
            lattice.calculate_order_parameter(
                polarization_vectors,
                num_phases=3,
                torch_device='cuda:0'
            )
        """
        # Imports
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from scipy.stats import gaussian_kde

        # Functions
        def plot_gaussian_ellipse(ax, mean, cov, n_std=2, clip_path=None, **kwargs):
            """
            Plot confidence ellipse for a 2D Gaussian

            Parameters:
            -----------
            ax : matplotlib axis
            mean : array-like, shape (2,)
                Mean of the Gaussian
            cov : array-like, shape (2, 2)
                Covariance matrix
            n_std : float
                Number of standard deviations (2 = ~95% confidence)
            clip_path : matplotlib.path.Path, optional
                Path to use for clipping the ellipse
            """
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Calculate ellipse parameters
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * n_std * np.sqrt(eigenvalues)

            # Create ellipse
            ellipse = Ellipse(mean, width, height, angle=angle, fill=False, **kwargs)

            if clip_path is not None:
                ellipse.set_clip_path(clip_path, transform=ax.transData)

            ax.add_patch(ellipse)

            return ellipse

        def to_percent(x, pos):
            """Format axis labels as percentages"""
            return f"{x * 100:.1f}%"

        # Function to validate colormap
        def is_valid_cmap(cmap_name):
            """Check if a colormap name is valid in matplotlib"""
            try:
                plt.get_cmap(cmap_name)
                return True
            except (ValueError, TypeError):
                return False

        # Function to validate color
        def is_valid_color(color):
            """Check if a color is valid in matplotlib"""
            try:
                mcolors.to_rgba(color)
                return True
            except (ValueError, TypeError):
                return False

        # Function to convert color names to RGB for scatter_cmap
        def convert_colors_to_rgb(colors, num_phases):
            """
            Convert colors to RGB array format.
            Args:
                colors: either a callable function, array of colors, or list of color names
                num_phases: number of phases/clusters
            Returns:
                numpy array of shape (num_phases, 3) with RGB values
            """
            # If it's a function (like site_colors), call it for each index
            if callable(colors):
                rgb_array = np.array([colors(i)[:3] for i in range(num_phases)])
                return rgb_array

            # If it's already an array, validate dimensions
            if isinstance(colors, np.ndarray):
                if colors.shape == (num_phases, 3):
                    return colors
                else:
                    return None

            # If it's a list/tuple of color names or values
            if isinstance(colors, (list, tuple)):
                try:
                    rgb_array = np.array([mcolors.to_rgb(c) for c in colors])
                    if rgb_array.shape == (num_phases, 3):
                        return rgb_array
                    else:
                        return None
                except (ValueError, TypeError):
                    return None

            return None

        class FixedMeansGMM(TorchGMM):
            """
            GMM variant with fixed component means.
            Means are set via fixed_means at init and held constant during EM;
            only weights and covariances are updated.
            """

            def __init__(self, fixed_means, **kwargs):
                fixed_means = np.asarray(fixed_means, dtype=np.float32)
                super().__init__(n_components=len(fixed_means), means_init=fixed_means, **kwargs)
                self.fixed_means = fixed_means

            def _m_step(self, X, r):
                """
                M-step with fixed means:
                update mixture weights and covariances from responsibilities,
                keeping means unchanged.
                """
                # Override to keep means fixed while updating weights and covariances
                N, D = X.shape
                K = self.n_components
                Nk = r.sum(dim=0) + 1e-12
                self._weights = (Nk / (N + 1e-12)).clamp_min(1e-12)

                # Keep means fixed
                self._means = self._to_tensor(self.fixed_means).clone()

                # Update covariances with fixed means
                covs = []
                for k in range(K):
                    diff = X - self._means[k]
                    cov_k = (r[:, k][:, None] * diff).T @ diff
                    cov_k = cov_k / (Nk[k] + 1e-12)
                    cov_k = cov_k + self.reg_covar * torch.eye(
                        D, device=self.device, dtype=self.dtype
                    )
                    covs.append(cov_k)
                self._covariances = torch.stack(covs, dim=0)

        x_arr = polarization_vectors[0]["x"]
        y_arr = polarization_vectors[0]["y"]

        da_arr = polarization_vectors[0]["da"]
        db_arr = polarization_vectors[0]["db"]

        d_frac_arr = np.vstack([da_arr, db_arr])
        data = np.column_stack([da_arr, db_arr])

        # Fit GMM with N Gaussians
        if phase_polarization_peak_array is None:
            gmm = TorchGMM(n_components=num_phases, covariance_type="full", device=torch_device)
        else:
            # Basic checks
            if phase_polarization_peak_array.shape != (num_phases, 2):
                raise ValueError(
                    f"phase_polarization_peak_array should have dimensions ({num_phases}, 2). You have input : {phase_polarization_peak_array.shape}"
                )
            if fix_polarization_peaks:
                gmm = FixedMeansGMM(
                    covariance_type="full",
                    fixed_means=phase_polarization_peak_array,
                    device=torch_device,
                )
            else:
                gmm = TorchGMM(
                    n_components=num_phases,
                    covariance_type="full",
                    means_init=phase_polarization_peak_array,
                    device=torch_device,
                )
        gmm.fit(data)

        # Calculate score between 0 and 1 for each point
        # Get probabilities for each Gaussian
        probabilities = gmm.predict_proba(data)  # Shape: (n_points, num_phases)

        # Create grid for contour - use max_bound to cover entire plot area
        max_bound = max(abs(da_arr).max(), abs(db_arr).max())

        x_grid = np.linspace(-max_bound, max_bound, 100)
        y_grid = np.linspace(-max_bound, max_bound, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = gaussian_kde(d_frac_arr)(positions).reshape(X.shape)

        # Save GMM data
        self._polarization_means = gmm.means_
        self._order_parameter_probabilities = probabilities

        num_components = num_phases

        # ========== Combined Plot: Scatter overlaid on Contour ==========
        if plot_gmm_visualization:
            from matplotlib.path import Path
            from matplotlib.ticker import FuncFormatter

            # Define preset colors based on num_phases
            preset_contour_cmap = "gray"
            if num_phases == 2:
                preset_gmm_center_colour = "lime"
                preset_gmm_ellipse_colour = "lime"
            elif num_phases < 5:
                preset_gmm_center_colour = "Yellow"
                preset_gmm_ellipse_colour = "Yellow"
            else:
                preset_gmm_center_colour = "Black"
                preset_gmm_ellipse_colour = "White"

            preset_scatter_colours = site_colors

            # Check and assign contour_cmap
            if "contour_cmap" in kwargs:
                if is_valid_cmap(kwargs["contour_cmap"]):
                    contour_cmap = kwargs["contour_cmap"]
                else:
                    print(
                        f"Warning: '{kwargs['contour_cmap']}' is not a valid colormap, using preset"
                    )
                    contour_cmap = preset_contour_cmap
            else:
                contour_cmap = preset_contour_cmap

            # Check and assign gmm_center_colour
            if "gmm_center_colour" in kwargs:
                if is_valid_color(kwargs["gmm_center_colour"]):
                    gmm_center_colour = kwargs["gmm_center_colour"]
                else:
                    print(
                        f"Warning: '{kwargs['gmm_center_colour']}' is not a valid color, using preset"
                    )
                    gmm_center_colour = preset_gmm_center_colour
            else:
                gmm_center_colour = preset_gmm_center_colour

            # Check and assign gmm_ellipse_colour
            if "gmm_ellipse_colour" in kwargs:
                if is_valid_color(kwargs["gmm_ellipse_colour"]):
                    gmm_ellipse_colour = kwargs["gmm_ellipse_colour"]
                else:
                    print(
                        f"Warning: '{kwargs['gmm_ellipse_colour']}' is not a valid color, using preset"
                    )
                    gmm_ellipse_colour = preset_gmm_ellipse_colour
            else:
                gmm_ellipse_colour = preset_gmm_ellipse_colour

            # Check and assign scatter_colours (with special handling)
            if "scatter_colours" in kwargs:
                scatter_colours_input = kwargs["scatter_colours"]

                # Try to convert to RGB format
                scatter_colours_rgb = convert_colors_to_rgb(scatter_colours_input, num_phases)

                if scatter_colours_rgb is not None:
                    # Successfully converted to (num_phases, 3) RGB array
                    scatter_colours = scatter_colours_rgb
                else:
                    # Check if it's a single valid color
                    if is_valid_color(scatter_colours_input):
                        # Convert single color to repeated array for indexing
                        single_color_rgb = mcolors.to_rgb(scatter_colours_input)
                        scatter_colours = np.tile(single_color_rgb, (num_phases, 1))
                        print(
                            f"Warning: Using single color '{scatter_colours_input}' for all {num_phases} phases"
                        )
                    else:
                        print(
                            "Warning: scatter_colours invalid (must be (num_phases, 3) array, list of valid colors, or callable), using preset"
                        )
                        scatter_colours = convert_colors_to_rgb(preset_scatter_colours, num_phases)
            else:
                scatter_colours = convert_colors_to_rgb(preset_scatter_colours, num_phases)

            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111)

            # Set symmetric limits centered at origin
            ax.set_xlim(-max_bound, max_bound)
            ax.set_ylim(-max_bound, max_bound)

            # Format axes as percentages
            percent_formatter = FuncFormatter(to_percent)
            ax.xaxis.set_major_formatter(percent_formatter)
            ax.yaxis.set_major_formatter(percent_formatter)

            # First: Plot contour in the background with distinct colormap
            ax.contourf(X, Y, Z, levels=15, cmap=contour_cmap, alpha=0.9)
            ax.contour(
                X, Y, Z, levels=15, cmap=contour_cmap, linewidths=0.5, alpha=0.9
            )  # FIXED: cmap instead of colors

            # Second: Overlay scatter points with classification colors
            point_colors = create_colors_from_probabilities(
                probabilities, num_components, scatter_colours
            )  # FIXED: pass scatter_colours
            ax.scatter(
                da_arr,
                db_arr,
                c=point_colors,
                alpha=0.7,
                s=20,
                edgecolors="black",
                linewidths=0.3,
                zorder=7,
            )

            # Create a clip path from the contour
            contour_path = None
            for collection in ax.collections:
                if isinstance(collection, plt.matplotlib.collections.LineCollection):
                    for path in collection.get_paths():
                        if contour_path is None:
                            contour_path = path
                        else:
                            contour_path = Path.make_compound_path(contour_path, path)

            # Plot GMM centers and ellipses using validated kwargs colors
            gmm_color = [gmm_center_colour, gmm_ellipse_colour]  # FIXED: use kwargs values

            ax.scatter(
                gmm.means_[:, 0],
                gmm.means_[:, 1],
                c=gmm_color[0],
                s=300,
                marker="x",
                linewidths=4,
                alpha=0.8,
                label="GMM Centers",
                zorder=10,
            )

            for i in range(num_components):
                plot_gaussian_ellipse(
                    ax,
                    gmm.means_[i],
                    gmm.covariances_[i],
                    n_std=2,
                    edgecolor=gmm_color[1],
                    linewidth=1.5,
                    linestyle="-",
                    alpha=0.6,
                    zorder=8,
                    clip_path=contour_path,
                )

            # Add x and y axes through origin
            ax.axhline(y=0, color="white", linewidth=1.5, linestyle="-", alpha=0.7, zorder=1)
            ax.axvline(x=0, color="white", linewidth=1.5, linestyle="-", alpha=0.7, zorder=1)

            ax.set_xlabel("du")
            ax.set_ylabel("dv")
            ax.set_title("Classification & Contour Overlay")

            # Add colorbar for contour (density)
            # plt.colorbar(contour, ax=ax, label="Density")

            # Add appropriate color reference based on number of phases
            if num_phases == 2:
                add_2phase_colorbar(ax, scatter_colours)
            elif num_phases == 3:
                add_3phase_color_triangle(fig, ax, scatter_colours)
            # For num_phases > 3 or == 1, don't add any color reference

            ax.legend(loc="best")
            # plt.tight_layout()
            plt.show()

        if plot_order_parameter:
            # Create colors from full probability distribution with custom scatter_colours
            colors = create_colors_from_probabilities(probabilities, num_phases, scatter_colours)

            fig, ax = show_2d(
                self._image.array,
                axsize=(8, 7),
                cmap="gray",
            )

            # Plot points with colormap
            ax.scatter(
                y_arr,  # col (x-axis)
                x_arr,  # row (y-axis)
                c=colors,  # color by probabilities
                s=50,  # point size
                alpha=0.8,  # slight transparency
                edgecolors="black",  # edge for visibility
                linewidth=1,
            )

            ax.set_title("Spatial phase probability map")

            # Add appropriate color reference based on number of phases
            if num_phases == 2:
                add_2phase_colorbar(ax, scatter_colours)
            elif num_phases == 3:
                add_3phase_color_triangle(fig, ax, scatter_colours)
            # For num_phases > 3 or == 1, don't add any color reference

            ax.axis("off")
            fig.tight_layout()
            fig.show()

        return self

    # --- Plotting Functions ---
    def plot_polarization_vectors(
        self,
        pol_vec: "Vector",
        length_scale: float = 1.0,
        show_image: bool = True,
        figsize=(6, 6),
        subtract_median: bool = False,
        linewidth: float = 1.0,
        tail_width: float = 1.0,
        headwidth: float = 4.0,
        headlength: float = 4.0,
        outline: bool = True,
        outline_width: float = 2.0,
        outline_color: str = "black",
        alpha: float = 1.0,
        show_ref_points: bool = False,
        chroma_boost: float = 2.0,
        use_magnitude_lightness: bool = True,
        ref_marker: str = "o",
        ref_size: float = 20.0,
        ref_edge: str = "k",
        ref_face: str = "none",
        show_colorbar: bool = True,
        disp_color_max: float | None = None,
        phase_offset_deg: float = 180.0,  # red = down
        phase_dir_flip: bool = False,  # flip color direction if desired
        **kwargs,
    ):
        import matplotlib.patheffects as pe
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import ArrowStyle, Circle, FancyArrowPatch
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from quantem.core.visualization.visualization_utils import array_to_rgba

        data = pol_vec.get_data(0)
        if isinstance(data, list) or data is None or data.size == 0:
            if show_image:
                fig, ax = show_2d(self._image.array, returnfig=True, figsize=figsize, **kwargs)
            else:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
            H, W = self._image.shape
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.set_title("polarization" + (" (median subtracted)" if subtract_median else ""))
            plt.tight_layout()
            return fig, ax

        # Fields
        xA = pol_vec[0]["x"]
        yA = pol_vec[0]["y"]
        da = pol_vec[0]["da"]
        db = pol_vec[0]["db"]

        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        L = np.column_stack((u, v))
        dr = L @ np.vstack((da, db))

        # Displacements (rows, cols)
        dr_raw = dr[0].astype(float)
        dc_raw = dr[1].astype(float)

        xR = xA - dr_raw
        yR = yA - dc_raw

        # --- Unified color mapping (identical across scripts) ---
        dr, dc, amp, disp_cap_px = _compute_polar_color_mapping(
            dr_raw,
            dc_raw,
            subtract_median=subtract_median,
            use_magnitude_lightness=use_magnitude_lightness,
            disp_color_max=disp_color_max,
        )

        # Angle mapping consistent with legend (down=0°, right=+90°, up=180°, left=-90°)
        ang = np.arctan2(dc, dr)
        if phase_dir_flip:
            ang = -ang
        ang += np.deg2rad(phase_offset_deg)

        # Colors
        rgba = array_to_rgba(amp, ang, chroma_boost=chroma_boost)
        colors = rgba.reshape(-1, 4)[:, :3] if rgba.ndim != 2 else rgba[:, :3]

        # Background
        if show_image:
            fig, ax = show_2d(self._image.array, returnfig=True, figsize=figsize, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Draw arrows (colored patch with black stroke beneath via path effects)
        arrowstyle = ArrowStyle.Simple(
            head_length=headlength, head_width=headwidth, tail_width=tail_width
        )
        for i in range(xA.size):
            x0, y0 = float(xA[i]), float(yA[i])
            x1 = x0 + float(dr[i]) * float(length_scale)
            y1 = y0 + float(dc[i]) * float(length_scale)

            arrow = FancyArrowPatch(
                (y0, x0),
                (y1, x1),
                arrowstyle=arrowstyle,
                mutation_scale=1.0,
                linewidth=linewidth,
                facecolor=colors[i],
                edgecolor=colors[i],
                alpha=alpha,
                zorder=11,
                capstyle="round",
                joinstyle="round",
                shrinkA=0.0,
                shrinkB=0.0,
            )
            if outline:
                arrow.set_path_effects(
                    [
                        pe.Stroke(linewidth=linewidth + outline_width, foreground=outline_color),
                        pe.Normal(),
                    ]
                )
            ax.add_patch(arrow)

        if show_ref_points:
            ax.scatter(
                yR,
                xR,
                s=ref_size,
                marker=ref_marker,
                facecolors=ref_face,
                edgecolors=ref_edge,
                linewidths=1.0,
                zorder=12,
            )

        H, W = self._image.shape
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title("polarization" + (" (median subtracted)" if subtract_median else ""))
        plt.tight_layout()

        # Circular legend (same mapping and label)
        if show_colorbar:
            divider = make_axes_locatable(ax)
            ax_c = divider.append_axes("right", size="28%", pad="6%")

            N = 256
            yy = np.linspace(-1, 1, N)
            xx = np.linspace(-1, 1, N)
            YY, XX = np.meshgrid(yy, xx, indexing="ij")
            rr = np.sqrt(XX**2 + YY**2)
            disk = rr <= 1.0

            ang_grid = np.arctan2(XX, -YY)
            if phase_dir_flip:
                ang_grid = -ang_grid
            ang_grid += np.deg2rad(phase_offset_deg)

            amp_grid = np.clip(rr, 0, 1)
            rgba_grid = array_to_rgba(amp_grid, ang_grid, chroma_boost=chroma_boost)
            rgba_grid[~disk] = 0.0

            ax_c.imshow(
                rgba_grid, origin="lower", extent=(-1, 1, -1, 1), interpolation="nearest", zorder=0
            )
            ax_c.set_aspect("equal")
            ax_c.axis("off")

            ring = Circle((0, 0), 0.98, facecolor="none", edgecolor="k", linewidth=1.2, zorder=3)
            ring.set_clip_on(False)
            ax_c.add_patch(ring)

            # Cardinal labels (down/right/up/left)
            ax_c.text(0.00, -1.12, "0°", ha="center", va="top", fontsize=9, color="k")
            ax_c.text(1.12, 0.00, "90°", ha="left", va="center", fontsize=9, color="k")
            ax_c.text(0.00, 1.12, "180°", ha="center", va="bottom", fontsize=9, color="k")
            ax_c.text(-1.12, 0.00, "270°", ha="right", va="center", fontsize=9, color="k")

            # Scale arrow along +x, label centered above midpoint (white)
            scale_len = 0.85
            arrow_scale = FancyArrowPatch(
                (0.0, 0.0),
                (scale_len, 0.0),
                arrowstyle=ArrowStyle.Simple(head_length=10.0, head_width=6.0, tail_width=2.0),
                mutation_scale=1.0,
                linewidth=1.2,
                facecolor="k",
                edgecolor="k",
                zorder=4,
                shrinkA=0.0,
                shrinkB=0.0,
            )
            arrow_scale.set_clip_on(False)
            ax_c.add_patch(arrow_scale)

            mid_x, mid_y = scale_len / 2.0, 0.0
            ax_c.text(
                mid_x,
                mid_y + 0.14,
                f"{disp_cap_px:.2g} px",
                ha="center",
                va="bottom",
                fontsize=9,
                color="w",
            )

            # Crosshairs & generous limits to avoid clipping
            ax_c.plot([0, 0], [-0.9, 0.9], color=(0, 0, 0, 0.15), lw=0.8, zorder=2)
            ax_c.plot([-0.9, 0.9], [0, 0], color=(0, 0, 0, 0.15), lw=0.8, zorder=2)
            ax_c.set_xlim(-1.35, 1.35)
            ax_c.set_ylim(-1.25, 1.35)

        return fig, ax

    def plot_polarization_image(
        self,
        pol_vec: "Vector",
        *,
        pixel_size: int = 16,
        padding: int = 8,
        spacing: int = 2,
        subtract_median: bool = False,
        chroma_boost: float = 2.0,
        use_magnitude_lightness: bool = True,
        disp_color_max: float | None = None,
        phase_offset_deg: float = 180.0,  # red = down (your convention)
        phase_dir_flip: bool = False,  # flip global hue mapping if desired
        aggregator: str = "mean",  # 'mean' or 'maxmag'
        plot: bool = False,  # if True, draw with show_2d and legend
        returnfig: bool = False,  # if True (and plot=True) also return (fig, ax)
        show_colorbar: bool = True,
        figsize=(6, 6),
        **kwargs,
    ):
        """
        Build and return an RGB superpixel image indexed by integer (a,b), colored by
        the same JCh cyclic mapping used for polarization vectors.

        Returns
        -------
        img_rgb : (H,W,3) float in [0,1]
        (fig, ax) : optional, only when plot=True and returnfig=True
        """
        import numpy as np
        from matplotlib.patches import ArrowStyle, Circle, FancyArrowPatch
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from quantem.core.visualization.visualization_utils import array_to_rgba
        # Requires the shared helper from the arrow script:
        # _compute_polar_color_mapping(dr, dc, subtract_median=..., use_magnitude_lightness=..., disp_color_max=...)

        # --- Extract data ---
        data = pol_vec.get_data(0)
        if isinstance(data, list) or data is None or data.size == 0:
            H = padding * 2 + pixel_size
            W = padding * 2 + pixel_size
            img_rgb = np.zeros((H, W, 3), dtype=float)
            if plot:
                fig, ax = show_2d(img_rgb, returnfig=True, figsize=figsize, **kwargs)
                ax.set_title(
                    "polarization image" + (" (median subtracted)" if subtract_median else "")
                )
                if returnfig:
                    return img_rgb, (fig, ax)
            return img_rgb

        # fields
        xA = pol_vec[0]["x"]
        yA = pol_vec[0]["y"]
        xR = pol_vec[0]["x_ref"]
        yR = pol_vec[0]["y_ref"]
        a_raw = pol_vec[0]["a"]
        b_raw = pol_vec[0]["b"]

        # displacements (rows/cols)
        dr_raw = (xA - xR).astype(float)  # down +
        dc_raw = (yA - yR).astype(float)  # right +

        # --- Unified color mapping (identical to arrow plot) ---
        dr, dc, amp, disp_cap_px = _compute_polar_color_mapping(
            dr_raw,
            dc_raw,
            subtract_median=subtract_median,
            use_magnitude_lightness=use_magnitude_lightness,
            disp_color_max=disp_color_max,
        )

        # Hue angles with your convention (down=0°, right=+90°, up=180°, left=-90°)
        ang = np.arctan2(dc, dr)
        if phase_dir_flip:
            ang = -ang
        ang += np.deg2rad(phase_offset_deg)

        # Per-sample RGB from JCh mapping
        rgba = array_to_rgba(amp, ang, chroma_boost=chroma_boost)
        colors = rgba.reshape(-1, 4)[:, :3] if rgba.ndim != 2 else rgba[:, :3]

        # Quantize to integer (a,b) tiles
        ai = np.rint(a_raw).astype(int)
        bi = np.rint(b_raw).astype(int)

        a_min, a_max = int(ai.min()), int(ai.max())
        b_min, b_max = int(bi.min()), int(bi.max())
        nrows = a_max - a_min + 1
        ncols = b_max - b_min + 1

        # Output canvas
        H = padding * 2 + nrows * pixel_size + (nrows - 1) * spacing
        W = padding * 2 + ncols * pixel_size + (ncols - 1) * spacing
        img_rgb = np.zeros((H, W, 3), dtype=float)

        # Group indices by (a,b)
        from collections import defaultdict

        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, (aa, bb) in enumerate(zip(ai, bi)):
            groups[(aa, bb)].append(idx)

        # Optional magnitude (after median subtraction) for 'maxmag' selection
        mag = np.hypot(dr, dc)

        # Fill tiles
        for (aa, bb), idx_list in groups.items():
            rr, cc = aa - a_min, bb - b_min
            r0 = padding + rr * (pixel_size + spacing)
            c0 = padding + cc * (pixel_size + spacing)

            if aggregator == "maxmag":
                j = idx_list[int(np.argmax(mag[idx_list]))]
                color = colors[j]
            else:  # 'mean'
                color = colors[idx_list].mean(axis=0)

            img_rgb[r0 : r0 + pixel_size, c0 : c0 + pixel_size, :] = color

        # --- Optional rendering with legend ---
        if plot:
            fig, ax = show_2d(img_rgb, returnfig=True, figsize=figsize, **kwargs)
            ax.set_title(
                "polarization image" + (" (median subtracted)" if subtract_median else "")
            )

            if show_colorbar:
                divider = make_axes_locatable(ax)
                ax_c = divider.append_axes("right", size="28%", pad="6%")

                N = 256
                yy = np.linspace(-1, 1, N)
                xx = np.linspace(-1, 1, N)
                YY, XX = np.meshgrid(yy, xx, indexing="ij")
                rr = np.sqrt(XX**2 + YY**2)
                disk = rr <= 1.0

                # Legend angle mapping identical to main mapping
                ang_grid = np.arctan2(XX, -YY)  # down=0 at bottom, right=+90° on +x
                if phase_dir_flip:
                    ang_grid = -ang_grid
                ang_grid += np.deg2rad(phase_offset_deg)

                amp_grid = np.clip(rr, 0, 1)
                rgba_grid = array_to_rgba(amp_grid, ang_grid, chroma_boost=chroma_boost)
                rgba_grid[~disk] = 0.0

                ax_c.imshow(
                    rgba_grid,
                    origin="lower",
                    extent=(-1, 1, -1, 1),
                    interpolation="nearest",
                    zorder=0,
                )
                ax_c.set_aspect("equal")
                ax_c.axis("off")

                # ring outline (no clipping so it isn't cut off)
                ring = Circle(
                    (0, 0), 0.98, facecolor="none", edgecolor="k", linewidth=1.2, zorder=3
                )
                ring.set_clip_on(False)
                ax_c.add_patch(ring)

                # angle labels (down/right/up/left)
                ax_c.text(0.00, -1.12, "0°", ha="center", va="top", fontsize=9, color="k")
                ax_c.text(1.12, 0.00, "90°", ha="left", va="center", fontsize=9, color="k")
                ax_c.text(0.00, 1.12, "180°", ha="center", va="bottom", fontsize=9, color="k")
                ax_c.text(-1.12, 0.00, "270°", ha="right", va="center", fontsize=9, color="k")

                # black arrow (scale) and white label centered above it
                scale_len = 0.85
                arrow = FancyArrowPatch(
                    (0.0, 0.0),
                    (scale_len, 0.0),
                    arrowstyle=ArrowStyle.Simple(head_length=10.0, head_width=6.0, tail_width=2.0),
                    mutation_scale=1.0,
                    linewidth=1.2,
                    facecolor="k",
                    edgecolor="k",
                    zorder=4,
                    shrinkA=0.0,
                    shrinkB=0.0,
                )
                arrow.set_clip_on(False)
                ax_c.add_patch(arrow)
                mid_x, mid_y = scale_len / 2.0, 0.0
                ax_c.text(
                    mid_x,
                    mid_y + 0.14,
                    f"{disp_cap_px:.2g} px",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="w",
                )

                # subtle crosshairs & generous limits to avoid clipping
                ax_c.plot([0, 0], [-0.9, 0.9], color=(0, 0, 0, 0.15), lw=0.8, zorder=2)
                ax_c.plot([-0.9, 0.9], [0, 0], color=(0, 0, 0, 0.15), lw=0.8, zorder=2)
                ax_c.set_xlim(-1.35, 1.35)
                ax_c.set_ylim(-1.25, 1.35)

            if returnfig:
                return img_rgb, (fig, ax)

        return img_rgb


# Implementing GMM using Torch (don't want skimage as a dependency)
class TorchGMM:
    """
    PyTorch Gaussian Mixture Model with full covariances optimized via EM.
    Only 'full' covariance is supported.
    Allows custom means initialization, cov regularization, and device/dtype control.
    After fit, exposes means_, covariances_, and weights_; use predict_proba for responsibilities.
    """

    def __init__(
        self,
        n_components,
        covariance_type="full",
        means_init=None,
        tol=1e-4,
        max_iter=200,
        reg_covar=1e-6,
        device=None,
        dtype=torch.float32,
    ):
        if covariance_type != "full":
            raise NotImplementedError("Only 'full' covariance_type is supported as of now.")

        # Store parameters - handle edge cases gracefully
        self.n_components = int(n_components)

        # Convert negative max_iter to 0 (or absolute value)
        self.max_iter = abs(int(max_iter))

        self.covariance_type = covariance_type
        self.means_init = None if means_init is None else np.asarray(means_init, dtype=np.float32)
        self.tol = abs(float(tol))  # Also handle negative tolerance
        self.reg_covar = float(reg_covar)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Fitted attributes (NumPy for external access)
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None

        # Internal torch parameters
        self._means = None  # [K, D]
        self._covariances = None  # [K, D, D]
        self._weights = None  # [K]

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=self.dtype, device=self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        else:
            return torch.tensor(x, dtype=self.dtype, device=self.device)

    def _kmeans_plusplus_init(self, X: torch.Tensor, K: int) -> torch.Tensor:
        """Initialize means using k-means++ algorithm for better spread."""
        N, D = X.shape

        # Work on CPU for deterministic behavior
        X_cpu = X.cpu()

        # First center: random choice
        indices = [torch.randint(0, N, (1,), device="cpu").item()]

        # Remaining centers: choose based on distance to existing centers
        for _ in range(1, K):
            # Compute distances to nearest existing center
            centers = X_cpu[indices]
            dists = torch.cdist(X_cpu, centers)  # [N, num_centers]
            min_dists = dists.min(dim=1)[0]  # [N]

            # Square distances for probability weighting
            probs = min_dists**2
            probs_sum = probs.sum()

            # Handle case where all points are identical (probs_sum == 0)
            if probs_sum > 1e-10:
                probs = probs / probs_sum
                # Sample next center
                next_idx = torch.multinomial(probs, 1).item()
            else:
                # All points are very close, just pick randomly
                next_idx = torch.randint(0, N, (1,), device="cpu").item()

            indices.append(next_idx)

        return X_cpu[indices].to(device=self.device, dtype=self.dtype)

    def _init_params(self, X: torch.Tensor) -> None:
        N, D = X.shape
        K = self.n_components

        if self.means_init is not None:
            if self.means_init.shape != (K, D):
                raise ValueError(
                    f"means_init must have shape ({K}, {D}), got {self.means_init.shape}"
                )
            self._means = self._to_tensor(self.means_init).clone()
        else:
            # Initialize means using k-means++ for better separation
            if N > 0 and K > 0:
                if N >= K:
                    self._means = self._kmeans_plusplus_init(X, K)
                else:
                    # Sample with replacement if not enough samples
                    X_cpu = X.cpu()
                    indices = torch.randint(0, N, (K,), device="cpu")
                    self._means = X_cpu[indices].clone().to(device=self.device, dtype=self.dtype)
            else:
                self._means = torch.zeros((K, D), device=self.device, dtype=self.dtype)

        # Initialize covariances with global covariance for stability
        if N > 1:
            X_centered = X - X.mean(dim=0, keepdim=True)
            global_cov = (X_centered.T @ X_centered) / (N - 1)
            # Add strong regularization for near-singular cases
            global_cov = global_cov + self.reg_covar * torch.eye(
                D, device=self.device, dtype=self.dtype
            )
        else:
            global_cov = self.reg_covar * torch.eye(D, device=self.device, dtype=self.dtype)

        # Ensure minimum eigenvalue for numerical stability
        eigenvalues = torch.linalg.eigvalsh(global_cov)
        if eigenvalues.min() < self.reg_covar:
            global_cov = global_cov + (self.reg_covar - eigenvalues.min() + 1e-6) * torch.eye(
                D, device=self.device, dtype=self.dtype
            )

        self._covariances = global_cov.unsqueeze(0).repeat(K, 1, 1).clone()

        # Initialize weights uniformly - handle K=0 case
        self._weights = torch.full(
            (K,), 1.0 / K if K > 0 else 1.0, device=self.device, dtype=self.dtype
        )

    def _log_gaussians(self, X: torch.Tensor) -> torch.Tensor:
        # X: [N, D], means: [K, D], covs: [K, D, D]
        N, D = X.shape
        K = self.n_components

        # Compute log probabilities for each component
        log_probs = []
        for k in range(K):
            # Ensure covariance is positive definite
            cov_k = self._covariances[k]

            # Check if covariance needs additional regularization
            try:
                # Try with current covariance
                dist = torch.distributions.MultivariateNormal(
                    loc=self._means[k], covariance_matrix=cov_k, validate_args=False
                )
                log_prob = dist.log_prob(X)
            except (RuntimeError, ValueError):
                # Add stronger regularization if needed
                cov_reg = cov_k + 1e-3 * torch.eye(D, device=self.device, dtype=self.dtype)
                dist = torch.distributions.MultivariateNormal(
                    loc=self._means[k], covariance_matrix=cov_reg, validate_args=False
                )
                log_prob = dist.log_prob(X)

            log_probs.append(log_prob)  # [N]

        log_comp = torch.stack(log_probs, dim=1)  # [N, K]
        return log_comp

    def _e_step(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_comp = self._log_gaussians(X)  # [N, K]
        log_weights = torch.log(self._weights.clamp_min(1e-12))  # [K]
        log_post = log_comp + log_weights[None, :]  # [N, K]
        r = torch.softmax(log_post, dim=1)  # responsibilities [N, K]
        return r, log_post

    def _m_step(self, X: torch.Tensor, r: torch.Tensor) -> None:
        N, D = X.shape
        K = self.n_components
        Nk = r.sum(dim=0).clamp_min(1e-12)  # [K]
        self._weights = (Nk / N).clamp_min(1e-12)

        # Means
        self._means = (r.T @ X) / Nk[:, None]

        # Covariances (full)
        covs = []
        for k in range(K):
            diff = X - self._means[k]  # [N, D]
            cov_k = (r[:, k][:, None] * diff).T @ diff
            cov_k = cov_k / Nk[k]

            # Add regularization
            cov_k = cov_k + self.reg_covar * torch.eye(D, device=self.device, dtype=self.dtype)

            # Ensure positive definiteness
            eigenvalues = torch.linalg.eigvalsh(cov_k)
            if eigenvalues.min() < self.reg_covar:
                cov_k = cov_k + (self.reg_covar - eigenvalues.min() + 1e-6) * torch.eye(
                    D, device=self.device, dtype=self.dtype
                )

            covs.append(cov_k)
        self._covariances = torch.stack(covs, dim=0)  # [K, D, D]

    def fit(self, data) -> "TorchGMM":
        X = self._to_tensor(data)
        if X.ndim != 2:
            raise ValueError("Input data must be 2D with shape (N, D)")

        self._init_params(X)

        prev_ll = torch.tensor(float("-inf"), device=self.device, dtype=self.dtype)

        for iteration in range(self.max_iter):
            r, _ = self._e_step(X)
            self._m_step(X, r)

            # Compute average log-likelihood of data under mixture
            log_comp = self._log_gaussians(X)
            log_weighted = log_comp + torch.log(self._weights)[None, :]
            ll = torch.logsumexp(log_weighted, dim=1).mean()

            # Check convergence
            if iteration > 0 and torch.isfinite(prev_ll) and torch.isfinite(ll):
                improvement = (ll - prev_ll).abs()
                if improvement < self.tol:
                    break
            prev_ll = ll

        # Store NumPy copies for external use (decoupled from internal tensors)
        self.means_ = self._means.detach().clone().cpu().numpy()
        self.covariances_ = self._covariances.detach().clone().cpu().numpy()
        self.weights_ = self._weights.detach().clone().cpu().numpy()
        return self

    def predict_proba(self, data) -> np.ndarray:
        X = self._to_tensor(data)
        r, _ = self._e_step(X)
        return r.detach().cpu().numpy()


# helper functions for plotting
def _compute_polar_color_mapping(
    dr: np.ndarray,
    dc: np.ndarray,
    *,
    subtract_median: bool,
    use_magnitude_lightness: bool,
    disp_color_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns (dr_adj, dc_adj, amp, disp_cap_px):
      dr_adj, dc_adj  -> components after optional median subtraction
      amp             -> [0,1] lightness (or constant if not using magnitude lightness)
      disp_cap_px     -> saturation cap (px): user value or 95th percentile
    """
    dr = np.asarray(dr, float).copy()
    dc = np.asarray(dc, float).copy()

    if subtract_median and dr.size:
        dr -= np.median(dr)
        dc -= np.median(dc)

    mag = np.hypot(dr, dc)

    if use_magnitude_lightness:
        if disp_color_max is None:
            nz = mag[mag > 0]
            disp_cap_px = float(np.percentile(nz, 95)) if nz.size else 1.0
        else:
            disp_cap_px = max(float(disp_color_max), 1e-9)
        amp = np.clip(mag / disp_cap_px, 0.0, 1.0)
    else:
        disp_cap_px = float(disp_color_max) if disp_color_max is not None else 1.0
        amp = np.full_like(mag, 0.85, dtype=float)

    return dr, dc, amp, disp_cap_px


def site_colors(number):
    """
    Map an integer 'number' to an RGB triple in [0,1].
    If 'number' is a list, array, or tuple, returns an array of RGB triples.
    Starts with the requested seed palette and cycles thereafter.
    """

    palette = [
        (1.00, 0.00, 0.00),  # 0: red
        (0.00, 0.00, 1.00),  # 1: blue
        (0.00, 1.00, 0.00),  # 2: green
        (1.00, 0.00, 1.00),  # 3: magenta
        (1.00, 0.70, 0.00),  # 4: orange
        (0.00, 0.30, 1.00),  # 5: blue-ish
        # extras to improve variety when cycling:
        (0.60, 0.20, 0.80),
        (0.30, 0.75, 0.75),
        (0.80, 0.40, 0.00),
        (0.20, 0.60, 0.20),
        (0.70, 0.70, 0.00),
        (0.00, 0.00, 0.00),  # -1: black
        # ENSURE BLACK IS ALWAYS LAST IF ADDING NEW COLORS
    ]

    # Check if input is a list, tuple, or array
    if isinstance(number, int):
        # Original behavior for single integer
        idx = int(number) % len(palette)
        return palette[idx]
    else:
        # Convert to numpy array for vectorized operations
        numbers = np.asarray(number, dtype=int)
        indices = numbers % len(palette)
        # Return array of RGB tuples
        return np.array([palette[idx] for idx in indices.flat]).reshape(numbers.shape + (3,))


def create_colors_from_probabilities(probabilities, num_phases, category_colors=None):
    """
    Create colors from probability distribution with a smooth transition to white for uncertainty.
    Smoothing is applied only when num_phases = 3.

    Parameters:
    -----------
    probabilities : array of shape (N, n_categories)
        Probabilities for each category (rows should sum to 1)
    num_phases : int
        Number of phases/categories
    category_colors : array of shape (num_phases, 3), optional
        Custom RGB colors for each category. If None, uses site_colors.

    Returns:
    --------
    colors : array of shape (N, 3)
        RGB colors for each point
    """
    import matplotlib.colors as mcolors

    # Get base colors for each category (assume 0-1 range)
    if category_colors is None:
        category_colors = np.array([site_colors(i) for i in range(num_phases)])

    # Mix colors based on probabilities
    mixed_colors = probabilities @ category_colors

    if num_phases == 3:
        # Apply smoothing for 3-phase system
        # Calculate certainty (max probability)
        certainty = np.max(probabilities, axis=1)

        # Create a smooth transition function
        def smooth_transition(x):
            return 4 * x**3 - 3 * x**4

        # Apply smooth transition to certainty
        smooth_certainty = smooth_transition(certainty)

        # Blend with white: uncertain -> white, certain -> category color
        white = np.array([1.0, 1.0, 1.0])
        final_colors = (
            smooth_certainty[:, np.newaxis] * mixed_colors
            + (1 - smooth_certainty[:, np.newaxis]) * white
        )

        # Ensure colors are in valid range [0, 1] BEFORE HSV conversion
        final_colors = np.clip(final_colors, 0, 1)

        # Convert to HSV for final adjustments
        hsv_colors = mcolors.rgb_to_hsv(final_colors)

        # Adjust saturation based on certainty
        hsv_colors[:, 1] *= smooth_certainty

        # Convert back to RGB
        final_colors = mcolors.hsv_to_rgb(hsv_colors)
    else:
        # For 2-phase system, use the original method
        # Calculate certainty (inverse of entropy)
        epsilon = 1e-10
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
        max_entropy = np.log(num_phases)

        # Certainty: 0 (uncertain) to 1 (certain)
        certainty = 1 - (entropy / max_entropy)

        # Blend with white: uncertain -> white, certain -> category color
        white = np.array([1.0, 1.0, 1.0])
        final_colors = (
            certainty[:, np.newaxis] * mixed_colors + (1 - certainty[:, np.newaxis]) * white
        )

    # Ensure final colors are in valid range [0, 1]
    final_colors = np.clip(final_colors, 0, 1)

    return final_colors


def add_2phase_colorbar(ax, scatter_colours):
    """
    Add a 1D colorbar for 2-phase system
    Creates a colormap that goes: color0 -> white (center) -> color1

    Parameters:
    -----------
    ax : matplotlib axes
        The main plot axes
    scatter_colours : array of shape (2, 3)
        RGB colors for the two phases
    """
    from matplotlib.colors import LinearSegmentedColormap

    fig = ax.get_figure()

    # Find the rightmost edge of all existing axes
    max_right = ax.get_position().x1
    for fig_ax in fig.get_axes():
        if fig_ax != ax:
            max_right = max(max_right, fig_ax.get_position().x1)

    # Calculate the position for the new colorbar
    ax_pos = ax.get_position()
    cbar_width = 0.035  # Width of the colorbar
    cbar_pad = 0.05  # Increased padding between colorbars
    cbar_left = max_right + cbar_pad
    cbar_bottom = ax_pos.y0
    cbar_height = ax_pos.height

    # Create new axes for colorbar
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

    # Get the two phase colors from scatter_colours
    color0 = scatter_colours[0]
    color1 = scatter_colours[1]

    # Create a colormap that goes: color0 -> white (center) -> color1
    colors_list = [color0, (1, 1, 1), color1]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("two_phase", colors_list, N=n_bins)
    # Create gradient
    gradient = np.linspace(0, 1, 256).reshape(256, 1)

    # Display the colorbar
    cax.imshow(gradient, aspect="auto", cmap=cmap, origin="lower")

    # Configure ticks and labels
    cax.set_xticks([])
    cax.set_yticks([0, 128, 255])
    cax.set_yticklabels(["Phase 0", "Uncertain", "Phase 1"])
    cax.yaxis.tick_right()

    return cax


def add_3phase_color_triangle(fig, ax, scatter_colours):
    """
    Add a ternary color triangle for 3-phase system

    Parameters:
    -----------
    fig : matplotlib figure
        The figure object
    ax : matplotlib axes
        The main plot axes
    scatter_colours : array of shape (3, 3)
        RGB colors for the three phases
    """

    # Check if there are existing colorbars/triangles attached to the figure
    box = ax.get_position()
    existing_elements = []

    # Find all axes that might be colorbars or previous triangles
    for fig_ax in fig.get_axes():
        if fig_ax != ax:
            pos = fig_ax.get_position()
            # Check if it's positioned to the right of the main axes
            if pos.x0 >= box.x1:
                existing_elements.append(fig_ax)

    # Calculate horizontal offset based on existing elements
    if existing_elements:
        # Find the rightmost existing element
        rightmost_x = max(elem.get_position().x1 for elem in existing_elements)
        x_offset = rightmost_x + 0.02  # Add spacing after the rightmost element
    else:
        x_offset = box.x1 + 0.02

    # Create a new axes for the triangle
    # Adjust position to account for existing colorbars
    triangle_width = box.height * 0.8
    triangle_ax = fig.add_axes([x_offset, box.y0, triangle_width, box.height * 0.8])

    # Get the three phase colors from scatter_colours
    color0 = scatter_colours[0]
    color1 = scatter_colours[1]
    color2 = scatter_colours[2]

    # Create ternary color grid
    resolution = 100
    positions = []
    probabilities_list = []

    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j

            # Probabilities (barycentric coordinates)
            p0, p1, p2 = i / resolution, j / resolution, k / resolution
            probabilities_list.append([p0, p1, p2])

            # Convert to Cartesian coordinates for ternary plot
            x = 0.5 * (2 * p1 + p2)
            y = (np.sqrt(3) / 2) * p2
            positions.append([x, y])

    positions = np.array(positions)
    probabilities_array = np.array(probabilities_list)

    # Get colors using the same function with custom scatter_colours
    colors = create_colors_from_probabilities(probabilities_array, 3, scatter_colours)

    # Plot the triangle
    triangle_ax.scatter(
        positions[:, 0], positions[:, 1], c=colors, s=20, marker="s", edgecolors="none"
    )

    # Draw triangle edges
    triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    triangle_ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], "k-", linewidth=2)

    # Add vertex markers and labels
    vertex_size = 150

    # Vertex 0 (bottom left) - Phase 0
    triangle_ax.scatter(
        0, 0, s=vertex_size, c=[color0], edgecolors="black", linewidths=2, zorder=10
    )
    triangle_ax.text(0, -0.1, "Phase 0", ha="center", va="top", fontsize=10, fontweight="bold")

    # Vertex 1 (bottom right) - Phase 1
    triangle_ax.scatter(
        1, 0, s=vertex_size, c=[color1], edgecolors="black", linewidths=2, zorder=10
    )
    triangle_ax.text(1, -0.1, "Phase 1", ha="center", va="top", fontsize=10, fontweight="bold")

    # Vertex 2 (top) - Phase 2
    triangle_ax.scatter(
        0.5, np.sqrt(3) / 2, s=vertex_size, c=[color2], edgecolors="black", linewidths=2, zorder=10
    )
    triangle_ax.text(
        0.5,
        np.sqrt(3) / 2 + 0.1,
        "Phase 2",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    # Mark center (maximum uncertainty) - white
    triangle_ax.scatter(
        0.5, np.sqrt(3) / 6, s=vertex_size, c="white", edgecolors="black", linewidths=2, zorder=10
    )
    triangle_ax.text(
        0.65,
        np.sqrt(3) / 6,
        "Uncertain\n(Equal)",
        ha="left",
        va="center",
        fontsize=8,
        style="italic",
    )

    # Set limits and styling
    triangle_ax.set_xlim(-0.15, 1.15)
    triangle_ax.set_ylim(-0.2, np.sqrt(3) / 2 + 0.15)
    triangle_ax.set_aspect("equal")
    triangle_ax.axis("off")
    triangle_ax.set_title("Probability Map", fontsize=11, pad=10)

    return triangle_ax
