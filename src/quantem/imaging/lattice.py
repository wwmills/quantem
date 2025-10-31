from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d
from scipy.ndimage import map_coordinates

from quantem.core import config

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

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
        image: Union[Dataset2d, NDArray],
        normalize_min: bool = True,
        normalize_max: bool = True,
    ) -> "Lattice":
        if isinstance(image, Dataset2d):
            ds2d = image
        else:
            arr = ensure_valid_array(image, ndim=2)
            if hasattr(Dataset2d, "from_array") and callable(getattr(Dataset2d, "from_array")):
                ds2d = Dataset2d.from_array(arr)  # type: ignore[attr-defined]
            else:
                ds2d = Dataset2d(arr)  # type: ignore[call-arg]
        if normalize_min:
            ds2d.array -= np.min(ds2d.array)
        if normalize_max:
            ds2d.array /= np.max(ds2d.array)
        return cls(image=ds2d, _token=cls._token)

    # --- Properties ---
    @property
    def image(self) -> Dataset2d:
        return self._image

    @image.setter
    def image(self, value: Union[Dataset2d, NDArray]):
        if isinstance(value, Dataset2d):
            self._image = value
        else:
            arr = ensure_valid_array(value, ndim=2)
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
        block_size: int = -1,
        plot_lattice=True,
        bound_num_vectors=None,
        input_mask=None,
        refine_lattice=True,
        refine_maxiter: int = 200,
        **kwargs,
    ):
        # Lattice
        self._lat = np.vstack(
            (
                np.array(origin),
                np.array(u),
                np.array(v),
            )
        )

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
                    input_mask_padded = np.zeros([input_mask.shape[0] + 2*pixel_buffer,input_mask.shape[1] + 2*pixel_buffer]).astype(bool)
                    input_mask_padded[pixel_buffer:-pixel_buffer, pixel_buffer:-pixel_buffer] = input_mask
                    x_round = np.round(x).astype(np.int32) + pixel_buffer
                    y_round = np.round(y).astype(np.int32) + pixel_buffer
                    valid_mask &= input_mask_padded[x_round, y_round] & input_mask_padded[x_round+1, y_round] & input_mask_padded[x_round, y_round+1] & input_mask_padded[x_round+1, y_round+1]

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
    ):
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
    ):
        import numpy as np

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
                if i1 <= i0 or j1 <= j0: # this doesn't do anything
                    continue

                patch = im[i0 : i1 + 1, j0 : j1 + 1]

                # broadcast coordinate grids to patch shape
                ii = np.arange(i0, i1 + 1)[:, None]
                jj = np.arange(j0, j1 + 1)[None, :]
                II = np.broadcast_to(ii, patch.shape)
                JJ = np.broadcast_to(jj, patch.shape)

                r2 = (II - x0) ** 2 + (JJ - y0) ** 2
                mask = r2 <= (r_fit * r_fit) # why not just square this with **? Or square root instead of r2
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

                    self.atoms_dislocation.set_data(updated, s)

        if plot_atoms:
            fig, ax = show_2d(self._image.array, figsize = (10,10),returnfig=True, **kwargs)
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
                        rgb = site_colors(int(self._numbers[s]+1))
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
        origin = None,
        u = None,
        v = None,
        positions_frac = None,
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
        intensity_radius = None,
        intensity_min: float | None = None,
        contrast_min=None,
        annulus_radii = None,
        check_uv_duplication = True,
        check_for_dislocations = False,
        merge_dislocation = False,
        **kwargs,
    ):
        self.check_for_dislocations = check_for_dislocations and check_uv_duplication
        # find all candidates above threshold
        maxima_candidates = self.get_maxima_2D(
            self.image.array, 
            subpixel = subpixel,
            upsample_factor = upsample_factor,
            sigma = sigma,
            minAbsoluteIntensity = minAbsoluteIntensity,
            minRelativeIntensity = minRelativeIntensity,
            relativeToPeak = relativeToPeak,
            minSpacing = minSpacing,
            edgeBoundary = edgeBoundary,
            maxNumPeaks = maxNumPeaks,
            )
        H, W = self._image.shape  # x=rows, y=cols

        if origin is None:
            max_intensity_index = np.argmax(maxima_candidates[:]['intensity'])
            origin_x = maxima_candidates[max_intensity_index]['x']
            origin_y = maxima_candidates[max_intensity_index]['y']
            origin = np.array([origin_x, origin_y])

        if u is None or v is None:
            num_peaks_search = 20
            num_peaks_use = 2
            center_ignore_buffer = 15
            minSpacingPeaks = 5
            uv_result_inv = self.auto_peak_finder(num_peaks_search = num_peaks_search, num_peaks_use = num_peaks_use, center_ignore_buffer = center_ignore_buffer, minSpacingPeaks = minSpacingPeaks)

            g_vector_1_c = np.array([uv_result_inv[0]['x'], uv_result_inv[0]['y']])
            g_vector_2_c = np.array([uv_result_inv[1]['x'], uv_result_inv[1]['y']])
            g_vec1 = np.zeros(2)
            g_vec1[0] = ((g_vector_1_c[0] - (0.5*H))/H)
            g_vec1[1] = ((g_vector_1_c[1] - (0.5*W))/W)
            g_vec2 = np.zeros(2)
            g_vec2[0] = ((g_vector_2_c[0] - (0.5*H))/H)
            g_vec2[1] = ((g_vector_2_c[1] - (0.5*W))/W)
            g_matrix = np.array([g_vec1, g_vec2])
            a_matrix = np.linalg.inv(g_matrix)
            a_transpose = a_matrix.T
            u = np.array([a_transpose[0,0], a_transpose[0,1]])
            v = np.array([a_transpose[1,0], a_transpose[1,1]])
            self.u = u
            self.v = v

        if positions_frac is None:
            positions_frac = np.atleast_2d(np.array((0,0)))

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
        maxima_candidates_x = maxima_candidates[:]['x']
        maxima_candidates_y = maxima_candidates[:]['y']

        pm_arr = np.array([-1,1]) # np.array([-1,0,1])
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        uv_arr = np.array([np.asarray(u),np.asarray(v)])
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
        maxima_candidates_x = maxima_candidates[:]['x']
        maxima_candidates_y = maxima_candidates[:]['y']
        maxima_candidates_intensity = maxima_candidates[:]['intensity']

        # the unique ids array is an array of the original index, candidacy, a (of a * u), and b (of b * v)
        unique_ids = np.zeros([6, len(maxima_candidates)])
        unique_ids[0,:] = np.arange(0,len(maxima_candidates))
        unique_ids[4,:] = -1*np.arange(1,1+len(maxima_candidates))
        unique_ids[5,:] -= 1

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

        radial_dist = ((maxima_candidates_x - origin[0])**2 + (maxima_candidates_y - origin[1])**2)**(0.5)
        origin_candidate_index = np.argmin(radial_dist) # use the first minima, if there are multiple
        unique_ids[1,origin_candidate_index] = 1
        unique_ids[4,origin_candidate_index] = 0

        atoms_found_this_iteration = np.zeros(len(maxima_candidates))
        atoms_found_prev_iteration = np.zeros(len(maxima_candidates))
        atoms_found_previous_iterations = np.zeros(len(maxima_candidates), dtype = bool)
        atoms_found_prev_iteration[origin_candidate_index] = 1
        found_atoms_in_prev_iteration = True
        iteration_while = 0


        def check_dislocations(
            ):
            if check_for_dislocations:
                for atom_index in range(len(maxima_candidates)):
                    if unique_ids[1,atom_index] == 1:
                        for pm in pm_arr:
                            for uv_index, lat_vec in enumerate(uv_arr):
                                position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
                                position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
                                radial_dist = ((maxima_candidates_x - position_x)**2 + (maxima_candidates_y - position_y)**2)**(0.5)
                                radial_dist[atom_index] = uv_norm * (tolerance_uv - 1) * 2 # make sure that self is outside of range
                                if (radial_dist < (uv_norm * (tolerance_uv - 1))).any():
                                    successful_candidate_index = np.argmin(radial_dist)
                                    if unique_ids[1, successful_candidate_index] == 2:
                                        atoms_found_this_iteration[successful_candidate_index] += 1
                                        unique_ids[1, successful_candidate_index] = 3 # for being found in dislocation search
                                        unique_ids[2, successful_candidate_index] = unique_ids[2, atom_index] + pm*int(uv_index == 0)
                                        unique_ids[3, successful_candidate_index] = unique_ids[3, atom_index] + pm*int(uv_index == 1)
                                        unique_ids[4, successful_candidate_index] = 0
                maxima_dislocation_x = maxima_candidates_x[unique_ids[1,:] == 3]
                maxima_dislocation_y = maxima_candidates_y[unique_ids[1,:] == 3]
                maxima_dislocation_u = unique_ids[2,unique_ids[1,:] == 3]
                maxima_dislocation_v = unique_ids[3,unique_ids[1,:] == 3]
                maxima_dislocation_intensity = maxima_candidates_intensity[unique_ids[1,:] == 3]
                arr = np.vstack(
                    (maxima_dislocation_x, maxima_dislocation_y, maxima_dislocation_u, maxima_dislocation_v, maxima_dislocation_intensity)
                ).T
                return arr

        # first, a loop that finds all of the A sites

        a0 = 0 # here we are just doing a0
        while found_atoms_in_prev_iteration is True:
            for atom_index in range(len(maxima_candidates)):
                if atoms_found_prev_iteration[atom_index] > 0:
                    for pm in pm_arr:
                        for uv_index, lat_vec in enumerate(uv_arr):
                            position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
                            position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
                            radial_dist = ((maxima_candidates_x - position_x)**2 + (maxima_candidates_y - position_y)**2)**(0.5)
                            radial_dist[atom_index] = uv_norm * (tolerance_uv - 1) * 2 # make sure that self is outside of range
                            if (radial_dist < (uv_norm * (tolerance_uv - 1))).any():
                                successful_candidate_index = np.argmin(radial_dist)
                                if unique_ids[1, successful_candidate_index] == 0:
                                    atoms_found_this_iteration[successful_candidate_index] += 1
                                    unique_ids[1, successful_candidate_index] = 1
                                    unique_ids[2, successful_candidate_index] = unique_ids[2, atom_index] + pm*int(uv_index == 0)
                                    unique_ids[3, successful_candidate_index] = unique_ids[3, atom_index] + pm*int(uv_index == 1)
                                    unique_ids[4, successful_candidate_index] = 0
                                    unique_ids[5, successful_candidate_index] = 0
            # check if any atom was somehow still found twice:
            assert np.max(atoms_found_this_iteration) < 2
            # check if any found atoms have the same uv index
            if check_uv_duplication:
                uv_pairs = unique_ids[1:6,:].T
                unique_pairs, inverse, counts = np.unique(uv_pairs, axis=0, return_inverse=True, return_counts=True)
                duplicate_groups = [np.where(inverse == k)[0] for k, c in enumerate(counts) if c > 1]
                mask_atoms_found = atoms_found_this_iteration.astype(bool)
                if len(duplicate_groups) != 0:
                    for duplicate_group in duplicate_groups:
                        duplicate_group = np.asarray(duplicate_group)
                        duplicate_atoms_index_found_previous_iterations = duplicate_group[atoms_found_previous_iterations[duplicate_group]]
                        if duplicate_atoms_index_found_previous_iterations.size > 1:
                            if origin_candidate_index not in duplicate_group:
                                raise ValueError("The duplicate atoms finding code is somehow bugged")
                            else:
                                kept_index = origin_candidate_index
                        elif duplicate_atoms_index_found_previous_iterations.size == 1:
                            kept_index = duplicate_atoms_index_found_previous_iterations
                        else:
                            kept_index = duplicate_group[mask_atoms_found[duplicate_group]][0]
                        wipe_indicies = duplicate_group[duplicate_group != kept_index]
                        unique_ids[1, wipe_indicies] = 2 # signals to not accept for this maxima anymore
                        unique_ids[2:4,wipe_indicies] = 0
                        unique_ids[5,wipe_indicies] = -1
                        unique_ids[4,wipe_indicies] = -1*(wipe_indicies+1)
                        atoms_found_previous_iterations[wipe_indicies] = False
                        mask_atoms_found[wipe_indicies] = False
                        atoms_found_this_iteration[wipe_indicies] = 0
            if np.sum(atoms_found_this_iteration) == 0:
                found_atoms_in_prev_iteration = False
                print('stopping search')

            atoms_found_previous_iterations |= atoms_found_this_iteration.astype(bool)

            atoms_found_prev_iteration = atoms_found_this_iteration.copy()
            atoms_found_this_iteration = np.zeros(len(maxima_candidates))
            iteration_while += 1

        maxima_accepted_x = maxima_candidates_x[unique_ids[1,:] == 1]
        maxima_accepted_y = maxima_candidates_y[unique_ids[1,:] == 1]

        maxima_accepted_u = unique_ids[2, unique_ids[1,:] == 1]
        maxima_accepted_v = unique_ids[3, unique_ids[1,:] == 1]

        maxima_accepted_intensity = maxima_candidates_intensity[unique_ids[1,:] == 1]

        self.atoms = Vector.from_shape(
            shape=(self._num_sites),
            fields=("x", "y", "a", "b", "int_peak"),
            units=("px", "px", "ind", "ind", "counts"),
        )
        if not merge_dislocation or not check_for_dislocations:
            arr = np.vstack(
                (maxima_accepted_x, maxima_accepted_y, maxima_accepted_u, maxima_accepted_v, maxima_accepted_intensity)
            ).T
            self.atoms.set_data(arr, 0)

        if merge_dislocation and check_for_dislocations:
            maxima_merge_x = maxima_candidates_x[np.isin(unique_ids[1, :], [1, 3])]
            maxima_merge_y = maxima_candidates_y[np.isin(unique_ids[1, :], [1, 3])]

            maxima_merge_u = unique_ids[2, np.isin(unique_ids[1, :], [1, 3])]
            maxima_merge_v = unique_ids[3, np.isin(unique_ids[1, :], [1, 3])]

            maxima_merge_intensity = maxima_candidates_intensity[np.isin(unique_ids[1, :], [1, 3])]
            arr = np.vstack(
                (maxima_merge_x, maxima_merge_y, maxima_merge_u, maxima_merge_v, maxima_merge_intensity)
            ).T
            self.atoms.set_data(arr, 0)

        # second, a loop that uses these A sites to find all other sites
        found_atoms_in_prev_iteration = True
        while found_atoms_in_prev_iteration is True:
            for atom_index in range(len(maxima_candidates)):
                if not unique_ids[5, atom_index] == 0: # skip this 'for' iteration if the atom is not an A site
                    continue
                # print(unique_ids[5, atom_index])
                for a0 in range(self._num_sites-1):
                    a0 += 1 # we don't need to go over the 0 index again
                    positions_around_A_site = self.get_xy_shifts(a0)
                    positions_around_A_site_norm = np.linalg.norm(positions_around_A_site, axis = 1)
                    for pos_index, pos_vec in enumerate(positions_around_A_site):
                        position_x = pos_vec[0] + maxima_candidates_x[atom_index]
                        position_y = pos_vec[1] + maxima_candidates_y[atom_index]
                        radial_dist = ((maxima_candidates_x - position_x)**2 + (maxima_candidates_y - position_y)**2)**(0.5)
                        radial_dist[atom_index] = positions_around_A_site_norm[pos_index] * (tolerance_uv - 1) * 2 # make sure that self is outside of range
                        if (radial_dist < (positions_around_A_site_norm[pos_index] * (tolerance_uv - 1))).any():
                            successful_candidate_index = np.argmin(radial_dist)
                            if unique_ids[1, successful_candidate_index] == 0:
                                atoms_found_this_iteration[successful_candidate_index] += 1
                                unique_ids[1, successful_candidate_index] = 1
                                unique_ids[2, successful_candidate_index] = unique_ids[2, atom_index]
                                unique_ids[3, successful_candidate_index] = unique_ids[3, atom_index]
                                unique_ids[4, successful_candidate_index] = 0
                                unique_ids[5, successful_candidate_index] = a0
                # check if any atom was somehow still found twice:
                assert np.max(atoms_found_this_iteration) < 2
                # uv duplication check won't work as is. since our uv coordinates will have to move to a floating point for extra sites, we will need to use a threshold
            if np.sum(atoms_found_this_iteration) == 0:
                found_atoms_in_prev_iteration = False
                print('stopping search')

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

        for a0 in range(self._num_sites -1):
            a0 += 1
            mask_1 = unique_ids[1,:] == 1
            mask_2 = unique_ids[5,:] == a0
            mask = mask_1.astype(bool) & mask_2.astype(bool)
            maxima_accepted_x = maxima_candidates_x[mask]
            maxima_accepted_y = maxima_candidates_y[mask]
            maxima_accepted_u = unique_ids[2, mask]
            maxima_accepted_v = unique_ids[3, mask]
            maxima_accepted_intensity = maxima_candidates_intensity[mask]
            arr = np.vstack(
                (maxima_accepted_x, maxima_accepted_y, maxima_accepted_u, maxima_accepted_v, maxima_accepted_intensity)
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
        origin = None,
        u = None,
        v = None,
        positions_frac = None,
        tolerance_uvw: float = 1.1,
        w = None,
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
        intensity_radius = None,
        intensity_min: float | None = None,
        contrast_min=None,
        annulus_radii = None,
        check_uv_duplication = True,
        check_for_dislocations = False,
        merge_dislocation = False,
        **kwargs,
    ):
        self.check_for_dislocations = check_for_dislocations and check_uv_duplication
        # find all candidates above threshold
        maxima_candidates = self.get_maxima_2D(
            self.image.array, 
            subpixel = subpixel,
            upsample_factor = upsample_factor,
            sigma = sigma,
            minAbsoluteIntensity = minAbsoluteIntensity,
            minRelativeIntensity = minRelativeIntensity,
            relativeToPeak = relativeToPeak,
            minSpacing = minSpacing,
            edgeBoundary = edgeBoundary,
            maxNumPeaks = maxNumPeaks,
            )

        H, W = self._image.shape  # x=rows, y=cols

        if origin is None:
            max_intensity_index = np.argmax(maxima_candidates[:]['intensity'])
            origin_x = maxima_candidates[max_intensity_index]['x']
            origin_y = maxima_candidates[max_intensity_index]['y']
            origin = np.array([origin_x, origin_y])

        if u is None or v is None:
            num_peaks_search = 20
            num_peaks_use = 2
            center_ignore_buffer = 15
            minSpacingPeaks = 5
            uv_result_inv = self.auto_peak_finder(num_peaks_search = num_peaks_search, num_peaks_use = num_peaks_use, center_ignore_buffer = center_ignore_buffer, minSpacingPeaks = minSpacingPeaks)

            g_vector_1_c = np.array([uv_result_inv[0]['x'], uv_result_inv[0]['y']])
            g_vector_2_c = np.array([uv_result_inv[1]['x'], uv_result_inv[1]['y']])
            g_vec1 = np.zeros(2)
            g_vec1[0] = ((g_vector_1_c[0] - (0.5*H))/H)
            g_vec1[1] = ((g_vector_1_c[1] - (0.5*W))/W)
            g_vec2 = np.zeros(2)
            g_vec2[0] = ((g_vector_2_c[0] - (0.5*H))/H)
            g_vec2[1] = ((g_vector_2_c[1] - (0.5*W))/W)
            g_matrix = np.array([g_vec1, g_vec2])
            a_matrix = np.linalg.inv(g_matrix)
            a_transpose = a_matrix.T
            u = np.array([a_transpose[0,0], a_transpose[0,1]])
            v = np.array([a_transpose[1,0], a_transpose[1,1]])
            self.u = u
            self.v = v


        if positions_frac is None:
            positions_frac = np.atleast_2d(np.array((0,0))) # 1, 1
        # if (positions_frac[0] == np.array([0,0])).all():
            # positions_frac[0] = np.atleast_2d(np.array((1,1)))

        self._positions_frac = np.atleast_2d(np.array(positions_frac, dtype=float))
        self._num_sites = self._positions_frac.shape[0]
        if numbers is None:
            self._numbers = np.arange(1, self._num_sites + 1, dtype=int)
        else:
            self._numbers = np.atleast_1d(np.array(numbers, dtype=int))

        print("numbers",self._numbers)
        print("num sites",self._num_sites)
        
        if w is None:
            if np.abs(np.rad2deg(np.arccos(np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v))))) > np.deg2rad(90):
                w = np.asarray(u)+np.asarray(v)
                w_sign = 1
            else:
                w = np.asarray(u)-np.asarray(v)
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
        maxima_candidates_x = maxima_candidates[:]['x']
        maxima_candidates_y = maxima_candidates[:]['y']

        pm_arr = np.array([-1,1])
        # pm_arr = np.array([-1,0,1])
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        w_norm = np.linalg.norm(w)
        uvw_arr = np.array([np.asarray(u),np.asarray(v),np.asarray(w)])
        uv_arr = np.array([np.asarray(u), np.asarray(v)])
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
        maxima_candidates_x = maxima_candidates[:]['x']
        maxima_candidates_y = maxima_candidates[:]['y']
        maxima_candidates_intensity = maxima_candidates[:]['intensity']

        # the unique ids array is an array of the original index, candidacy, a (of a * u), and b (of b * v), and c (of c * w)
        unique_ids = np.zeros([6, len(maxima_candidates)])
        unique_ids[0,:] = np.arange(0,len(maxima_candidates))
        unique_ids[4,:] = -1*np.arange(1,1+len(maxima_candidates))
        unique_ids[5,:] -= 1

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

        radial_dist = ((maxima_candidates_x - origin[0])**2 + (maxima_candidates_y - origin[1])**2)**(0.5)
        origin_candidate_index = np.argmin(radial_dist) # use the first minima, if there are multiple
        unique_ids[1,origin_candidate_index] = 1
        unique_ids[4,origin_candidate_index] = 0

        atoms_found_this_iteration = np.zeros(len(maxima_candidates))
        atoms_found_prev_iteration = np.zeros(len(maxima_candidates))
        atoms_found_previous_iterations = np.zeros(len(maxima_candidates), dtype = bool)
        atoms_found_prev_iteration[origin_candidate_index] = 1
        found_atoms_in_prev_iteration = True
        iteration_while = 0

        def check_dislocations():
            if check_for_dislocations:
                for atom_index in range(len(maxima_candidates)):
                    if unique_ids[1,atom_index] == 1:
                        for pm in pm_arr:
                            for uvw_index, lat_vec in enumerate(uvw_arr):
                                position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
                                position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
                                radial_dist = ((maxima_candidates_x - position_x)**2 + (maxima_candidates_y - position_y)**2)**(0.5)
                                radial_dist[atom_index] = uvw_norm * (tolerance_uvw - 1) * 2 # make sure that self is outside of range
                                if (radial_dist < (uvw_norm * (tolerance_uvw - 1))).any():
                                    successful_candidate_index = np.argmin(radial_dist)
                                    if unique_ids[1, successful_candidate_index] == 2:
                                        atoms_found_this_iteration[successful_candidate_index] += 1
                                        unique_ids[1, successful_candidate_index] = 3 # for being found in dislocation search
                                        unique_ids[2, successful_candidate_index] = unique_ids[2, atom_index] + pm*int(uvw_index == 0) + pm*int(uvw_index == 2)
                                        unique_ids[3, successful_candidate_index] = unique_ids[3, atom_index] + pm*int(uvw_index == 1) - w_sign * pm*int(uvw_index == 2)
                                        unique_ids[4, successful_candidate_index] = 0
                maxima_dislocation_x = maxima_candidates_x[unique_ids[1,:] == 3]
                maxima_dislocation_y = maxima_candidates_y[unique_ids[1,:] == 3]
                maxima_dislocation_u = unique_ids[2,unique_ids[1,:] == 3]
                maxima_dislocation_v = unique_ids[3,unique_ids[1,:] == 3]
                maxima_dislocation_intensity = maxima_candidates_intensity[unique_ids[1,:] == 3]
                arr = np.vstack(
                    (maxima_dislocation_x, maxima_dislocation_y, maxima_dislocation_u, maxima_dislocation_v, maxima_dislocation_intensity)
                ).T
                return arr

        # first, a loop that finds all of the A sites
        a0 = 0 # here we are just doing a0
        while found_atoms_in_prev_iteration is True:
            for atom_index in range(len(maxima_candidates)):
                if atoms_found_prev_iteration[atom_index] > 0:
                    for pm in pm_arr:
                        for uvw_index, lat_vec in enumerate(uvw_arr):
                            position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
                            position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
                            radial_dist = ((maxima_candidates_x - position_x)**2 + (maxima_candidates_y - position_y)**2)**(0.5)
                            radial_dist[atom_index] = uvw_norm * (tolerance_uvw - 1) * 2 # make sure that self is outside of range
                            if (radial_dist < (uvw_norm * (tolerance_uvw - 1))).any():
                                successful_candidate_index = np.argmin(radial_dist)
                                if unique_ids[1, successful_candidate_index] == 0:
                                    atoms_found_this_iteration[successful_candidate_index] += 1
                                    unique_ids[1, successful_candidate_index] = 1
                                    unique_ids[2, successful_candidate_index] = unique_ids[2, atom_index] + pm*int(uvw_index == 0) + pm*int(uvw_index == 2)
                                    unique_ids[3, successful_candidate_index] = unique_ids[3, atom_index] + pm*int(uvw_index == 1) + w_sign*pm*int(uvw_index == 2)
                                    unique_ids[4, successful_candidate_index] = 0
                                    unique_ids[5, successful_candidate_index] = a0
            # check if any atom was somehow still found twice:
            assert np.max(atoms_found_this_iteration) < 2
            # check if any found atoms have the same uv index
            if check_uv_duplication:
                uv_pairs = unique_ids[1:6,:].T
                unique_pairs, inverse, counts = np.unique(uv_pairs, axis=0, return_inverse=True, return_counts=True)
                duplicate_groups = [np.where(inverse == k)[0] for k, c in enumerate(counts) if c > 1]
                mask_atoms_found = atoms_found_this_iteration.astype(bool)
                if len(duplicate_groups) != 0:
                    for duplicate_group in duplicate_groups:
                        duplicate_group = np.asarray(duplicate_group)
                        duplicate_atoms_index_found_previous_iterations = duplicate_group[atoms_found_previous_iterations[duplicate_group]]
                        if duplicate_atoms_index_found_previous_iterations.size > 1:
                            if origin_candidate_index not in duplicate_group:
                                raise ValueError("The duplicate atoms finding code is somehow bugged")
                            else:
                                kept_index = origin_candidate_index
                        elif duplicate_atoms_index_found_previous_iterations.size == 1:
                            kept_index = duplicate_atoms_index_found_previous_iterations
                        else:
                            kept_index = duplicate_group[mask_atoms_found[duplicate_group]][0]
                        wipe_indicies = duplicate_group[duplicate_group != kept_index]
                        unique_ids[1, wipe_indicies] = 2 # this signals to not accept for this maxima anymore (and flags this as a dulpicate)
                        unique_ids[2:4,wipe_indicies] = 0
                        unique_ids[5,wipe_indicies] = -1
                        unique_ids[4,wipe_indicies] = -1*(wipe_indicies+1)
                        atoms_found_previous_iterations[wipe_indicies] = False
                        mask_atoms_found[wipe_indicies] = False
                        atoms_found_this_iteration[wipe_indicies] = 0
            if np.sum(atoms_found_this_iteration) == 0:
                found_atoms_in_prev_iteration = False
                print('stopping search')
            
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

        maxima_accepted_x = maxima_candidates_x[unique_ids[1,:] == 1]
        maxima_accepted_y = maxima_candidates_y[unique_ids[1,:] == 1]

        maxima_accepted_u = unique_ids[2, unique_ids[1,:] == 1]
        maxima_accepted_v = unique_ids[3, unique_ids[1,:] == 1]

        maxima_accepted_intensity = maxima_candidates_intensity[unique_ids[1,:] == 1]

        self.atoms = Vector.from_shape(
            shape=(self._num_sites),
            fields=("x", "y", "a", "b", "int_peak"),
            units=("px", "px", "ind", "ind", "counts"),
        )

        if not merge_dislocation or not check_for_dislocations:
            arr = np.vstack(
                (maxima_accepted_x, maxima_accepted_y, maxima_accepted_u, maxima_accepted_v, maxima_accepted_intensity)
            ).T
            self.atoms.set_data(arr, 0)

        if merge_dislocation and check_for_dislocations:

            maxima_merge_x = maxima_candidates_x[np.isin(unique_ids[1, :], [1, 3])]
            maxima_merge_y = maxima_candidates_y[np.isin(unique_ids[1, :], [1, 3])]

            maxima_merge_u = unique_ids[2, np.isin(unique_ids[1, :], [1, 3])]
            maxima_merge_v = unique_ids[3, np.isin(unique_ids[1, :], [1, 3])]

            maxima_merge_intensity = maxima_candidates_intensity[np.isin(unique_ids[1, :], [1, 3])]
            arr = np.vstack(
                (maxima_merge_x, maxima_merge_y, maxima_merge_u, maxima_merge_v, maxima_merge_intensity)
            ).T
            self.atoms.set_data(arr, 0)

        # second, a loop that uses these A sites to find all other sites
        uvw_arr_save = uvw_arr.copy()
        found_atoms_in_prev_iteration = True
        while found_atoms_in_prev_iteration is True:
            for atom_index in range(len(maxima_candidates)):
                if not unique_ids[5, atom_index] == 0: # skip this 'for' iteration if the atom is not an A site
                    continue
                # print(unique_ids[5, atom_index])
                for a0 in range(self._num_sites-1):
                    a0 += 1 # we don't need to go over the 0 index again
                    positions_around_A_site = self.get_xy_shifts(a0)
                    positions_around_A_site_norm = np.linalg.norm(positions_around_A_site, axis = 1)
                    for pos_index, pos_vec in enumerate(positions_around_A_site):
                        position_x = pos_vec[0] + maxima_candidates_x[atom_index]
                        position_y = pos_vec[1] + maxima_candidates_y[atom_index]
                        radial_dist = ((maxima_candidates_x - position_x)**2 + (maxima_candidates_y - position_y)**2)**(0.5)
                        radial_dist[atom_index] = positions_around_A_site_norm[pos_index] * (tolerance_uvw - 1) * 2 # make sure that self is outside of range
                        if (radial_dist < (positions_around_A_site_norm[pos_index] * (tolerance_uvw - 1))).any():
                            successful_candidate_index = np.argmin(radial_dist)
                            if unique_ids[1, successful_candidate_index] == 0:
                                atoms_found_this_iteration[successful_candidate_index] += 1
                                unique_ids[1, successful_candidate_index] = 1
                                unique_ids[2, successful_candidate_index] = unique_ids[2, atom_index]
                                unique_ids[3, successful_candidate_index] = unique_ids[3, atom_index]
                                unique_ids[4, successful_candidate_index] = 0
                                unique_ids[5, successful_candidate_index] = a0
                # check if any atom was somehow still found twice:
                assert np.max(atoms_found_this_iteration) < 2
                # uv duplication check won't work as is. since our uv coordinates will have to move to a floating point for extra sites, we will need to use a threshold
            if np.sum(atoms_found_this_iteration) == 0:
                found_atoms_in_prev_iteration = False
                print('stopping search')
                
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


        for a0 in range(self._num_sites -1):
            a0 += 1
            mask_1 = unique_ids[1,:] == 1
            mask_2 = unique_ids[5,:] == a0
            mask = mask_1.astype(bool) & mask_2.astype(bool)
            maxima_accepted_x = maxima_candidates_x[mask]
            maxima_accepted_y = maxima_candidates_y[mask]

            maxima_accepted_u = unique_ids[2, mask]
            maxima_accepted_v = unique_ids[3, mask]

            maxima_accepted_intensity = maxima_candidates_intensity[mask]

            arr = np.vstack(
                (maxima_accepted_x, maxima_accepted_y, maxima_accepted_u, maxima_accepted_v, maxima_accepted_intensity)
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

    def line_profile(
            self,
            origin,
            direction,
            num_samples=None
        ):

        nx, ny = self._image.array.shape
        # print(nx, ny)
        print(direction)
        # print(origin)
        x0, y0 = origin
        v = np.array(direction, dtype=float)
        v /= np.linalg.norm(v)

        if num_samples is None:
            num_samples = int(np.hypot(nx, ny))

        corners = np.array([[0,0],[nx,0],[0,ny],[nx,ny]])
        t_values = []
        for corner in corners:
            dx, dy = corner - origin
            t_values.append(np.dot([dx,dy], v))  # projection along direction
        t_min, t_max = min(t_values), max(t_values)
        # print(t_min, t_max, num_samples)
        t = np.linspace(t_min, t_max, num_samples)

        # Coordinates along the line
        x = x0 + v[0] * t
        y = y0 + v[1] * t

        # Interpolated intensities
        profile = map_coordinates(self.image.array, [x, y], order=1, mode='nearest')

        # Clip valid region (inside image bounds)
        mask = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
        return t[mask], profile[mask], x[mask], y[mask]

    # overloading this handle
    def line_profile(
        image,
        p1=None,
        p2=None,
        origin=None,
        direction=None,
        num_samples=None,
        order=1,
        mode='nearest'
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

            corners = np.array([[0,0],[nx,0],[0,ny],[nx,ny]])
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
            max_perpendicular_distance = 5,
            sigma_perp = 5,
            sigma_parallel = 5,
    ):

        atoms_arr = self.atoms.get_data(0)
        a_x = atoms_arr[:,0]
        a_y = atoms_arr[:,1]
        a_intensity = self.atoms[0]['int_peak']
        pm_arr = np.array([1,-1])

        for lat_vec in self.uv_arr:
            for atom_index in range(a_x.shape[0]):
                neighbor_x = [a_x[i] for i in self.atom_neighbor_arr[:,atom_index] if i is not None]
                neighbor_y = [a_y[i] for i in self.atom_neighbor_arr[:,atom_index] if i is not None]
                atom_x = a_x[atom_index]
                atom_y = a_y[atom_index]
                position_x = lat_vec[0] + atom_x
                position_y = lat_vec[1] + atom_y
                radial_dist = ((neighbor_x - position_x)**2 + (neighbor_y - position_y)**2)**(0.5)
                if (radial_dist < (self.uv_norm * (self.tolerance_uv - 1))).any():
                    successful_candidate_index = np.argmin(radial_dist)
                    successful_candidate_index = self.atom_neighbor_arr[successful_candidate_index, atom_index]
                    vec_x = a_x[successful_candidate_index] - atom_x
                    vec_y = a_y[successful_candidate_index] - atom_y
                    line_profile_vector = np.array([vec_x, vec_y])
                    line_profile_origin = np.array([atom_x, atom_y])
                    slice_coordinates, slice_y, x_coords_slice, y_coords_slice = self.line_profile(line_profile_origin, line_profile_vector)
                    mask, near_peaks, t_values, distances = self.select_peaks_near_line(atom_index, line_profile_vector, max_perpendicular_distance)
                    
                    t_profile = np.asarray(slice_coordinates)
                    t_values = np.asarray(t_values)
                    distances = np.asarray(distances)

                    A = a_intensity[mask]
                    A_eff = A * np.exp(-0.5 * (distances / sigma_perp) ** 2)
                    gaussian_sum = np.zeros_like(t_profile, dtype=float)
                    for t_i, A_i in zip(t_values, A_eff):
                        gaussian_sum += A_i * np.exp(-0.5 * ((t_profile - t_i) / sigma_parallel) ** 2)
                    if atom_index < 3:
                        plt.figure()
                        plt.plot()
                        plt.figure(figsize=(10,4))
                        plt.subplot(1,2,1)
                        plt.imshow(self.image.array, cmap='gray', origin='upper')
                        plt.plot(y_coords_slice, x_coords_slice, 'r-', lw=1)
                        plt.title("Line through image")

                        plt.subplot(1,2,2)
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
            origin = None,
            max_perpendicular_distance = 10,
            gaussian_smooth_data = None,
            return_fit = False,
            plot_profile = True,
    ):

        atoms_arr = self.atoms.get_data(0)
        a_intensity = self.atoms[0]['int_peak']
        bg_intensity = self.atoms[0]['int_bg']
        sigma_arr = self.atoms[0]['sigma']
        if origin is None:
            t_near_zero = np.argmin(np.abs(t))
            x_origin = x_coords[t_near_zero]
            y_origin = y_coords[t_near_zero]
        else:
            x_origin = origin[0]
            y_origin = origin[1]
        mask, near_peaks, t_values_s, distances = self.select_peaks_near_line(x_origin, y_origin, direction = direction, max_perpendicular_distance = max_perpendicular_distance)
        
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
            t_values_s,
            B,
            kind='linear',
            bounds_error=False, 
            fill_value=(B[0], B[-1])
        )

        B_interp = bg_interp_func(t_values)
        gaussian_sum += B_interp

        nx, ny = self.image.array.shape

        if plot_profile is True:
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.imshow(self.image.array, cmap='gray', origin='upper')
            plt.plot(y_coords, x_coords, 'r-', lw=1)
            plt.quiver(y_coords[0], x_coords[0], direction[1], direction[0], angles = 'xy', scale_units = 'xy',  scale = 1, color = 'red', zorder = 10)
            plt.title("Line through image")
            plt.xlim([0,nx-1])
            plt.ylim([ny-1, 0])

            plt.subplot(1,2,2)
            if gaussian_smooth_data is not None:
                plt.plot(t_values, gaussian_filter(t_profile, gaussian_smooth_data))
            else:
                plt.plot(t_values, t_profile)
            plt.plot(t_values, gaussian_sum, alpha = 0.5)
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

    def project_point_to_line_rowmajor(
        self,
        point,
        origin,
        direction
        ):
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
        a_xy = atoms_arr[:,0:2]
        selected_atoms_arr = np.zeros(a_xy.shape[0])
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

    def select_peaks_near_line(
            self,
            x_c,
            y_c,
            direction,
            max_perpendicular_distance,
    ):
        atoms_arr = self.atoms.get_data(0)
        a_xy = atoms_arr[:,0:2]
        selected_atoms_arr = np.zeros(a_xy.shape[0])
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



    def gaussian(
        self,
        s,
        amp,
        mu,
        sigma
        ):
        return amp * np.exp(-0.5 * ((s - mu) / sigma)**2)


    def get_xy_shifts(
            self,
            a0,
    ):
        position_fraction = self._positions_frac[a0]

        if (position_fraction == np.zeros([2])).all():
            p1 = np.array([1,0]) @ self.uv_arr[:2]
            p2 = np.array([-1,0]) @ self.uv_arr[:2]
            p3 = np.array([0,1]) @ self.uv_arr[:2]
            p4 = np.array([0,-1]) @ self.uv_arr[:2]
        else:
            p1 = (np.array([0,0]) + position_fraction) @ self.uv_arr[:2]
            p2 = (np.array([-1,0]) + position_fraction) @ self.uv_arr[:2]
            p3 = (np.array([0,-1]) + position_fraction) @ self.uv_arr[:2]
            p4 = (np.array([-1,-1]) + position_fraction) @ self.uv_arr[:2]

        return np.array([p1, p2, p3, p4])

    def get_a_positions_near_b_site(
            self,
    ):
        b_frac = self._positions_frac[1]
        p1 = (np.array([0,0]) - b_frac) @ self.uv_arr[:2]
        p2 = (np.array([1,0]) - b_frac) @ self.uv_arr[:2]
        p3 = (np.array([0,1]) - b_frac) @ self.uv_arr[:2]
        p4 = (np.array([1,1]) - b_frac) @ self.uv_arr[:2]
        return np.array([p1,p2,p3,p4])        

    def organize_b_neighbors(
        self,
        site_search_radius = 2,
        num_bins = 128,
        tolerance_uv = None,
        num_sites_use = 1,
        # centers = None,
    ):
        a_x_b = self.atoms.get_data(1)[:,0]
        a_y_b = self.atoms.get_data(1)[:,1]
        
        a_x_a = self.atoms.get_data(0)[:,0]
        a_y_a = self.atoms.get_data(0)[:,1]
        b_neighbor_arr = np.empty((2, 4, a_x_b.shape[0]), dtype=object)
        pm_arr = np.array([-1,1])
        a_positions_around_site = self.get_a_positions_near_b_site()
        pm_arr = np.array([-1,1])
        uvw_arr = self.uv_arr
        uvw_norm = self.uv_norm
        tolerance_uv = self.tolerance_uv
        for atom_index in range(a_x_b.shape[0]):
            # B sites near the B sites (displaced by the lattice vectors, hopefully)
            for pm in pm_arr:
                for uvw_index, lat_vec in enumerate(uvw_arr):
                    position_x = pm * lat_vec[0] + a_x_b[atom_index]
                    position_y = pm * lat_vec[1] + a_y_b[atom_index]
                    radial_dist = ((a_x_b - position_x)**2 + (a_y_b - position_y)**2)**(0.5)
                    radial_dist[atom_index] = uvw_norm * (tolerance_uv - 1) * 2 # make sure that self is outside of range
                    if (radial_dist < (uvw_norm * (tolerance_uv - 1))).any():
                        successful_candidate_index = np.argmin(radial_dist)
                        if (pm == 1 and uvw_index == 0):
                            b_neighbor_arr[1, 0,atom_index] = int(successful_candidate_index)

                        if (pm == -1 and uvw_index == 0):
                            b_neighbor_arr[1, 1,atom_index] = int(successful_candidate_index) 

                        if (pm == 1 and uvw_index == 1):
                            b_neighbor_arr[1, 2,atom_index] = int(successful_candidate_index) 

                        if (pm == -1 and uvw_index == 1):
                            b_neighbor_arr[1, 3,atom_index] = int(successful_candidate_index) 
                        # leaving out the w vector for right now...

            # A sites near the B sites
            # critically, the successful candidate indices that are written in this loop are only valid for the A site variables.
            for pos_index, pos_vec in enumerate(a_positions_around_site):
                position_x = pos_vec[0] + a_x_b[atom_index]
                position_y = pos_vec[1] + a_y_b[atom_index]
                radial_dist = ((a_x_a - position_x)**2 + (a_y_a - position_y)**2)**(0.5)
                # this line is not necessary because the coordinate of the b site is not in the a site coordinate list
                # radial_dist[atom_index] = self.uv_norm * (tolerance_uv - 1) * 2 # make sure that self is outside of range
                if (radial_dist < (self.uv_norm * (tolerance_uv - 1))).any():
                    successful_candidate_index = np.argmin(radial_dist)
                    if (pos_index == 0):
                        b_neighbor_arr[0, 0,atom_index] = int(successful_candidate_index)

                    if (pos_index == 1):
                        b_neighbor_arr[0, 1,atom_index] = int(successful_candidate_index) 

                    if (pos_index == 2):
                        b_neighbor_arr[0, 2,atom_index] = int(successful_candidate_index) 

                    if (pos_index == 3):
                        b_neighbor_arr[0, 3,atom_index] = int(successful_candidate_index)

                    if (pos_index == 4):
                        b_neighbor_arr[0, 4,atom_index] = int(successful_candidate_index) 

                    if (pos_index == 5):
                        b_neighbor_arr[0, 5,atom_index] = int(successful_candidate_index)

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
        site_search_radius = 2,
        num_bins = 128,
        tolerance_uv = None,
        num_sites_use = 1,
    ):
        if tolerance_uv is None:
            tolerance_uv = self.tolerance_uv
        for a0 in range(num_sites_use):
            atoms_arr = self.atoms.get_data(a0)
            a_x = atoms_arr[:,0]
            a_y = atoms_arr[:,1]
            pm_arr = np.array([1,-1])

            atom_neighbor_arr = np.empty((6, a_x.shape[0]), dtype=object)
            has_six_neighbors_arr = np.zeros(a_x.shape[0])
            for atom_index in range(a_x.shape[0]):
                for pm in pm_arr:
                    for uvw_index, lat_vec in enumerate(self.uv_arr):
                        position_x = pm * lat_vec[0] + a_x[atom_index]
                        position_y = pm * lat_vec[1] + a_y[atom_index]
                        radial_dist = ((a_x - position_x)**2 + (a_y - position_y)**2)**(0.5)
                        radial_dist[atom_index] = self.uv_norm * (tolerance_uv - 1) * 2 # make sure that self is outside of range
                        if (radial_dist < (self.uv_norm * (tolerance_uv - 1))).any():
                            successful_candidate_index = np.argmin(radial_dist)
                            if (pm == 1 and uvw_index == 0):
                                atom_neighbor_arr[0,atom_index] = int(successful_candidate_index)
                                has_six_neighbors_arr[atom_index] += 1

                            if (pm == -1 and uvw_index == 0):
                                atom_neighbor_arr[1,atom_index] = int(successful_candidate_index) 
                                has_six_neighbors_arr[atom_index] += 1

                            if (pm == 1 and uvw_index == 1):
                                atom_neighbor_arr[2,atom_index] = int(successful_candidate_index) 
                                has_six_neighbors_arr[atom_index] += 1

                            if (pm == -1 and uvw_index == 1):
                                atom_neighbor_arr[3,atom_index] = int(successful_candidate_index) 
                                has_six_neighbors_arr[atom_index] += 1

                            if (pm == 1 and uvw_index == 2):
                                atom_neighbor_arr[4,atom_index] = int(successful_candidate_index) 
                                has_six_neighbors_arr[atom_index] += 1

                            if (pm == -1 and uvw_index == 2):
                                atom_neighbor_arr[5,atom_index] = int(successful_candidate_index) 
                                has_six_neighbors_arr[atom_index] += 1
                
            self.has_six_neighbors_arr = has_six_neighbors_arr == 6
            self.atom_neighbor_arr = atom_neighbor_arr
        return self

    def get_next_neighborhood_layer(
        self,
        atom_index,
    ):
        neighbor_arr = np.asarray([i for i in self.atom_neighbor_arr[:, atom_index] if i is not None], dtype = int)
        neighbors_pass = [i for i in neighbor_arr if not self.added_to_neighbor_list_already[i]]

        self.added_to_neighbor_list_already[neighbors_pass] = 1

        return neighbors_pass

    def get_next_neighborhood_layer_arr(
        self,
        atom_indexes,
    ):
        atom_indexes_less_none =  [i for i in atom_indexes if i is not None]
        arr_present = False
        arr = None
        for atom_index in atom_indexes_less_none:
            if arr_present:
                arr = np.concatenate((arr, np.asarray(self.get_next_neighborhood_layer(atom_index))))
            else:
                arr = np.asarray(self.get_next_neighborhood_layer(atom_index))
                arr_present = True
        if arr is None:
            return None
        else:
            return np.asarray(arr, dtype = int)

    def get_next_neighborhood_layer_b(
        self,
        atom_index,
        a_or_b,
    ):
        neighbor_arr = np.asarray([i for i in self.b_neighbor_arr[a_or_b,:, atom_index] if i is not None], dtype = int)
        if a_or_b == 0:
            neighbors_pass = [i for i in neighbor_arr if not self.added_to_neighbor_list_already[i]]
            self.added_to_neighbor_list_already[neighbors_pass] = 1
        if a_or_b == 1:
            neighbors_pass = [i for i in neighbor_arr if not self.added_to_neighbor_list_already_b[i]]
            self.added_to_neighbor_list_already_b[neighbors_pass] = 1

        return neighbors_pass

    def get_next_neighborhood_layer_arr_b(
        self,
        atom_indexes,
        a_or_b,
    ):
        atom_indexes_less_none =  [i for i in atom_indexes if i is not None]
        arr_present = False
        arr = None
        for atom_index in atom_indexes_less_none:
            if arr_present:
                arr = np.concatenate((arr, np.asarray(self.get_next_neighborhood_layer_b(atom_index, a_or_b))))
            else:
                arr = np.asarray(self.get_next_neighborhood_layer_b(atom_index, a_or_b))
                arr_present = True
        

        # if a_or_b  == 0:
        #     arr = arr[1:] # get rid of the b index
        if arr is None:
            return None
        else:
            return np.asarray(arr, dtype = int)

    def neighborhood_unfinished(
        self,
        neighborhood_units = 2,
    ):
        
        self.neighborhood_units = neighborhood_units
        for a0 in range(self._num_sites):
            atoms_arr = self.atoms.get_data(a0)
            a_x = atoms_arr[:,0]
            atom_neighbor_list = []
            self.added_to_neighbor_list_already = np.zeros(a_x.shape[0])
            self.num_neighbors = np.zeros(a_x.shape[0])
            for atom_index in range(a_x.shape[0]):
                neighbors_search = np.asarray([atom_index])
                self.added_to_neighbor_list_already[atom_index] = 1
                neighbors_search_out = np.array([atom_index])
                for neighbor_iteration in range(neighborhood_units):
                    neighbors_search_out = self.get_next_neighborhood_layer_arr(neighbors_search_out)
                    neighbors_search = np.concatenate((neighbors_search, neighbors_search_out))
                    if neighbors_search_out.size == 0:
                        break
                atom_neighbor_list.append(neighbors_search)
                self.num_neighbors[atom_index] = np.sum(self.added_to_neighbor_list_already) - 1 # minus one because the central atom is not neighbor
                self.added_to_neighbor_list_already = np.zeros(a_x.shape[0])
        self.atom_neighbor_layer_arr = atom_neighbor_list
        return self
    
    def neighborhood_a(
        self,
        neighborhood_units = 2,
    ):
        
        self.neighborhood_units = neighborhood_units
        for a0 in range(1):
            atoms_arr = self.atoms.get_data(a0)
            a_x = atoms_arr[:,0]
            atom_neighbor_list = []
            self.added_to_neighbor_list_already = np.zeros(a_x.shape[0])
            self.num_neighbors = np.zeros(a_x.shape[0])
            for atom_index in range(a_x.shape[0]):
                neighbors_search = np.asarray([atom_index])
                self.added_to_neighbor_list_already[atom_index] = 1
                neighbors_search_out = np.array([atom_index])
                for neighbor_iteration in range(neighborhood_units):
                    neighbors_search_out = self.get_next_neighborhood_layer_arr(neighbors_search_out)
                    neighbors_search = np.concatenate((neighbors_search, neighbors_search_out))
                    if neighbors_search_out.size == 0:
                        break
                atom_neighbor_list.append(neighbors_search)
                self.num_neighbors[atom_index] = np.sum(self.added_to_neighbor_list_already) - 1 # minus one because the central atom is not neighbor
                self.added_to_neighbor_list_already = np.zeros(a_x.shape[0])
        self.atom_neighbor_layer_arr = atom_neighbor_list
        return self

    # neighborhood b collects all of the b and a neighbors in the vicinity of central b atoms
    def neighborhood_b(
            self,
            neighborhood_units = 2,
        ):

        self.neighborhood_units = neighborhood_units
        a_x_b = self.atoms.get_data(1)[:,0]
        a_x_a = self.atoms.get_data(0)[:,0]
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
                    neighbors_search_out_a = self.get_next_neighborhood_layer_arr_b(neighbors_search_out_a, 0)
                else:
                    neighbors_search_out_a = self.get_next_neighborhood_layer_arr(neighbors_search_out_a)

                neighbors_search_out_b = self.get_next_neighborhood_layer_arr_b(neighbors_search_out_b, 1)
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
            self.num_neighbors[atom_b_index] = np.sum(self.added_to_neighbor_list_already_b) + np.sum(self.added_to_neighbor_list_already) - 2 # minus one because the central atom is not neighbor, and counted twice
            self.added_to_neighbor_list_already_b = np.zeros(a_x_b.shape[0])
            self.added_to_neighbor_list_already = np.zeros(a_x_a.shape[0])
        self.atom_neighbor_layer_arr_a = atom_neighbor_list_a
        self.atom_neighbor_layer_arr_b = atom_neighbor_list_b
        return self


    # for this, the number of A site neighbors will be good probably
    def find_neighbors_in_tolerance(
            self,
            tolerance = None,
    ):
        if not hasattr(self, 'uv_norm'):
            self.uv_norm = np.mean(np.linalg.norm(self._lat[1:], axis = 1))
        if tolerance is None:
            tolerance = self.uv_norm * 1.1
        num_sites = len(self._positions_frac)
        a_x = self.atoms.get_data(0)[:,0]
        a_y = self.atoms.get_data(0)[:,1]
        count_a_neighbors = np.zeros([num_sites, 2*a_x.shape[0]])
        for site_index in range(num_sites):
            a_x_n = self.atoms.get_data(site_index)[:,0]
            a_y_n = self.atoms.get_data(site_index)[:,1]
            for atom_index in range(a_x_n.shape[0]):
                a_x_i = a_x_n[atom_index]
                a_y_i = a_y_n[atom_index]
                a_x_ai = a_x - a_x_i
                a_y_ai = a_y - a_y_i
                radial_dist = np.sqrt(a_x_ai**2 + a_y_ai**2)
                if site_index == 0:
                    radial_dist[atom_index] = tolerance * 2 # make sure that self is outside of range
                count_a_neighbors[site_index, atom_index] = np.sum(radial_dist < tolerance)
        self.count_a_neighbors = count_a_neighbors
        return self


    def remove_atoms_with_too_few_neighbors(
            self,
            min_neighbors = None,
            return_removed = False,
    ):
        if min_neighbors is None:
            min_neighbors = 2

        num_sites = len(self._positions_frac)
        removed = []
        for site_index in range(num_sites):
            site_data = self.atoms.get_data(site_index)
            keep_mask = self.count_a_neighbors[site_index, :site_data.shape[0]] >= min_neighbors
            updated = site_data[keep_mask]
            removed.append(site_data[~keep_mask])
            self.atoms.set_data(updated, site_index)
        if return_removed:
            return removed
        return self
        


    def plot_neighbors(
        self,
        centers = None,
    ):
        if centers is None:
            centers = np.arange(0,2)
        if isinstance(centers, int):
            centers = np.arange(0,centers)
        plt.figure()
        plt.imshow(self.image.array, cmap = 'gray')
        a_x_b = self.atoms.get_data(1)[:,0]
        a_y_b = self.atoms.get_data(1)[:,1]
        a_x_a = self.atoms.get_data(0)[:,0]
        a_y_a = self.atoms.get_data(0)[:,1]

        for atom_b_index in centers:
            plt.scatter(a_y_b[atom_b_index], a_x_b[atom_b_index], color = 'blue', zorder = 10)
            plt.scatter(a_y_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)], a_x_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)], color = 'red',  alpha = 0.5)
            plt.scatter(a_y_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)], a_x_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)], color = 'green')




    def intensity_neighborhood(
        self,
        neighborhood_units = 2,
        return_delta = False,
    ):
        self.neighborhood_units = neighborhood_units
        self.neighborhood_a(neighborhood_units = neighborhood_units)
        for a0 in range(self._num_sites):
            a_x = self.atoms[0]["x"]
            a_y = self.atoms[0]["y"]
            a_intensity = self.atoms[0]["int_peak"]
            delta_intensity = np.zeros([a_x.shape[0]])
            for atom_index in range(len(self.atom_neighbor_layer_arr)):
                neighbor_intensities = a_intensity[self.atom_neighbor_layer_arr[atom_index][1:]] # 1: excludes the first one, which is itself
                median_intensity = np.median(neighbor_intensities)
                delta_intensity[atom_index] = a_intensity[atom_index] - median_intensity
        self.delta_intensities = delta_intensity
        if return_delta:
            return delta_intensity, self.num_neighbors
        else:
            return self


    def neighborhood_b_circular_distance(
            self,
            circular_radius_cutoff = None,
    ):

        a_x_b = self.atoms.get_data(1)[:,0]
        a_y_b = self.atoms.get_data(1)[:,1]
        a_x_a = self.atoms.get_data(0)[:,0]
        a_y_a = self.atoms.get_data(0)[:,1]

        if circular_radius_cutoff == None:
            lattice_spacing = self.uv_norm * np.linalg.norm(self._positions_frac[1]) * 0.9
            circular_radius_cutoff = lattice_spacing * self.neighborhood_units

        for atom_b_index in range(a_x_b.shape[0]):
            b_neighbor_x = a_x_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)]
            b_neighbor_y = a_y_b[self.atom_neighbor_layer_arr_b[atom_b_index].astype(int)]
            a_neighbor_x = a_x_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)]
            a_neighbor_y = a_y_a[self.atom_neighbor_layer_arr_a[atom_b_index].astype(int)]
            b_x = a_x_b[atom_b_index]
            b_y = a_y_b[atom_b_index]
            radial_dist_b = np.sqrt((b_neighbor_x - b_x)**2 +(b_neighbor_y - b_y)**2)
            radial_dist_a = np.sqrt((a_neighbor_x - b_x)**2 +(a_neighbor_y - b_y)**2)
            mask = radial_dist_b < circular_radius_cutoff
            self.atom_neighbor_layer_arr_b[atom_b_index] = self.atom_neighbor_layer_arr_b[atom_b_index][mask]
            mask = radial_dist_a < circular_radius_cutoff
            self.atom_neighbor_layer_arr_a[atom_b_index] = self.atom_neighbor_layer_arr_a[atom_b_index][mask]

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
        a = np.cos(theta)**2/(2*xs**2) + np.sin(theta)**2/(2*ys**2)
        b = -np.cos(theta)*np.sin(theta)/(2*xs**2) + np.cos(theta)*np.sin(theta)/(2*ys**2)
        c = np.sin(theta)**2/(2*xs**2) + np.cos(theta)**2/(2*ys**2)

        # arr_pd = np.array([[a, b], [b, c]])
        # check that arr_pd is positive definite
        # condition_1 = a > 0
        # condition_2 = a * c - b**2 > 0
        assert xs > 0
        assert ys > 0

        gaussian_2d = A * np.exp(-(a*(x-xc)**2 + 2*b*(x-xc)*(y-yc) + c*(y-yc)**2)) + B
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

        a_x_b = self.atoms.get_data(1)[:,0]
        a_y_b = self.atoms.get_data(1)[:,1]
        a_s_b = self.atoms[1]['sigma']
        a_ip_b = self.atoms[1]['int_peak']

        a_x_a = self.atoms.get_data(0)[:,0]
        a_y_a = self.atoms.get_data(0)[:,1]
        a_s_a = self.atoms[0]['sigma']
        a_ip_a = self.atoms[0]['int_peak']

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

            window_fit_x_max = np.ceil(np.max(np.concatenate([a_neighbor_x, b_neighbor_x])) + window_pix).astype(int)
            window_fit_x_min =np.floor(np.min(np.concatenate([a_neighbor_x, b_neighbor_x])) - window_pix).astype(int)
            window_fit_y_max = np.ceil(np.max(np.concatenate([a_neighbor_y, b_neighbor_y])) + window_pix).astype(int)
            window_fit_y_min = np.floor(np.min(np.concatenate([a_neighbor_y, b_neighbor_y])) - window_pix).astype(int)

            window_fit_x_max = min(window_fit_x_max, H)
            window_fit_x_min = max(window_fit_x_min, 0)
            window_fit_y_max = min(window_fit_y_max, W)
            window_fit_y_min = max(window_fit_y_min, 0)

            x = np.arange(window_fit_x_min, window_fit_x_max)
            y = np.arange(window_fit_y_min, window_fit_y_max)
            xx, yy = np.meshgrid(x, y, indexing = 'ij')
            sub_window = self._image.array[window_fit_x_min:window_fit_x_max,window_fit_y_min:window_fit_y_max].copy()
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
            mask = r2 <= (r_fit * r_fit) # why not just square this with **? Or square root instead of r2
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
                    plt.axis('off')
                    plt.subplot(122)
                    plt.imshow(sub_window)
                    plt.axis('off')

        # plot the difference in values
        # intensity, xc, yc, sigmas
        delta_int = a_ip_b - updated[:,idx_amp]
        delta_xc = a_x_b - updated[:,idx_x]
        delta_yc = a_y_b - updated[:,idx_y]
        delta_sig = a_s_b - updated[:,idx_sigma]

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
        cmap_plot = 'magma'

        fig = plt.figure(figsize=(10, 10))

        ax1 = plt.subplot(221)
        ax1.imshow(self._image.array, cmap='gray')
        sc1 = ax1.scatter(updated[:, idx_y], updated[:, idx_x], c=delta_xc,
                        s=s_plot, alpha=alpha_plot, cmap=cmap_plot)
        ax1.set_title('Delta X Center')
        ax1.axis('off')
        fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = plt.subplot(222)
        ax2.imshow(self._image.array, cmap='gray')
        sc2 = ax2.scatter(updated[:, idx_y], updated[:, idx_x], c=delta_yc,
                        s=s_plot, alpha=alpha_plot, cmap=cmap_plot)
        ax2.set_title('Delta Y Center')
        ax2.axis('off')
        fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = plt.subplot(223)
        ax3.imshow(self._image.array, cmap='gray')
        sc3 = ax3.scatter(updated[:, idx_y], updated[:, idx_x], c=delta_int,
                        s=s_plot, alpha=alpha_plot, cmap=cmap_plot)
        ax3.set_title('Delta Intensity')
        ax3.axis('off')
        fig.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)

        ax4 = plt.subplot(224)
        ax4.imshow(self._image.array, cmap='gray')
        sc4 = ax4.scatter(updated[:, idx_y], updated[:, idx_x], c=delta_sig,
                        s=s_plot, alpha=alpha_plot, cmap=cmap_plot)
        ax4.set_title('Delta Sigma')
        ax4.axis('off')
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

        a_x_b = self.atoms.get_data(1)[:,0]
        a_y_b = self.atoms.get_data(1)[:,1]
        a_s_b = self.atoms[1]['sigma']
        a_ip_b = self.atoms[1]['int_peak']
        a_ib_b = self.atoms[1]['int_bg']

        a_x_a = self.atoms.get_data(0)[:,0]
        a_y_a = self.atoms.get_data(0)[:,1]
        a_s_a = self.atoms[0]['sigma']
        a_ip_a = self.atoms[0]['int_peak']
        a_ib_a = self.atoms[0]['int_bg']

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

        window_fit_x_max = np.ceil(np.max(np.concatenate([a_neighbor_x, b_neighbor_x])) + window_pix).astype(int)
        window_fit_x_min = np.floor(np.min(np.concatenate([a_neighbor_x, b_neighbor_x])) - window_pix).astype(int)
        window_fit_y_max = np.ceil(np.max(np.concatenate([a_neighbor_y, b_neighbor_y])) + window_pix).astype(int)
        window_fit_y_min = np.floor(np.min(np.concatenate([a_neighbor_y, b_neighbor_y])) - window_pix).astype(int)

        window_fit_x_max = min(window_fit_x_max, H)
        window_fit_x_min = max(window_fit_x_min, 0)
        window_fit_y_max = min(window_fit_y_max, W)
        window_fit_y_min = max(window_fit_y_min, 0)

        x = np.arange(window_fit_x_min, window_fit_x_max)
        y = np.arange(window_fit_y_min, window_fit_y_max)
        xx, yy = np.meshgrid(x, y, indexing = 'ij')
        sub_window = self._image.array[window_fit_x_min:window_fit_x_max,window_fit_y_min:window_fit_y_max].copy()
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
        if i1 <= i0 or j1 <= j0: # this doesn't do anything
            return a_x_b[atom_b_index], a_y_b[atom_b_index], a_ip_b[atom_b_index], a_s_b[atom_b_index], a_ib_b[atom_b_index]


        patch = im[i0 : i1 + 1, j0 : j1 + 1]

        # broadcast coordinate grids to patch shape
        ii = np.arange(i0, i1 + 1)[:, None]
        jj = np.arange(j0, j1 + 1)[None, :]
        II = np.broadcast_to(ii, patch.shape)
        JJ = np.broadcast_to(jj, patch.shape)

        r2 = (II - x0) ** 2 + (JJ - y0) ** 2
        mask = r2 <= (r_fit * r_fit) # why not just square this with **? Or square root instead of r2
        if not np.any(mask):
            return a_x_b[atom_b_index], a_y_b[atom_b_index], a_ip_b[atom_b_index], a_s_b[atom_b_index], a_ib_b[atom_b_index]

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
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(sub_window)
            plt.axis('off')

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
        num_peaks_search = 20,
        num_peaks_use = 2,
        center_ignore_buffer = 15,
        minSpacingPeaks = 5,
    ):
        diffraction_peaks_list = self.locate_diffraction_spots(num_peaks_search, center_ignore_buffer = center_ignore_buffer, minSpacingPeaks = minSpacingPeaks)
        if num_peaks_use == 2:
            peakA, peakB = self.locate_first_order_peaks(diffraction_peaks_list)
            diffraction_peaks_list = np.array([peakA, peakB])
        else:
            diffraction_peaks_list = np.array([[diffraction_peaks_list[i]] for i in range(1,(num_peaks_use+1))])
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
        midX = nx//2; midY = ny//2
        peakCoordinatesRespCenter = np.zeros(len(peakCoordinates), dtype=np.dtype([("x", float), ("y", float), ("intensity", float)]))
        peakCoordinatesRespCenter['x'] = peakCoordinates['x'] - midX
        peakCoordinatesRespCenter['y'] = peakCoordinates['y'] - midY
        peakRadialDistCenter = peakCoordinatesRespCenter['x']**2 + peakCoordinatesRespCenter['y']**2
        
        smallestRadiiIndices = np.argsort(peakRadialDistCenter)
        peakCoordinatesRespCenter = peakCoordinatesRespCenter[smallestRadiiIndices]
        
        # The closest peak should be the zero order peak - not interested in that.
        if peakRadialDistCenter[0] < 5:
            peakAInd = 1
            peakBInd = None
        else:
            peakAInd = 0
            peakBInd = None

        crossAWithRest = np.zeros([len(peakCoordinates)-2]) # this 2 comes from the A peak and the central peak that are excluded from consideration for the B and C peaks
        peakA_xy = self.get_xy(peakCoordinatesRespCenter[peakAInd])
        for peakIndex in np.arange(2,len(peakCoordinates)):
            currentPeak = self.get_xy(peakCoordinatesRespCenter[peakIndex])
            crossAWithRest[peakIndex-2] = np.cross(peakA_xy, currentPeak)
        threshold = 5 * (np.min(np.abs(crossAWithRest))+0.1)

        thresholdCondition = np.abs(crossAWithRest)>threshold
        if np.any(thresholdCondition):
            peakBInd = np.argmax(thresholdCondition) + 2 # returning the 2 that was subtracted above
        else:
            print('Lowering threshold B')
            threshold = 2 * (np.min(np.abs(crossAWithRest))+0.1)
            thresholdCondition = np.abs(crossAWithRest)>threshold
            peakBInd = np.argmax(thresholdCondition) + 2

        peakA = np.zeros(1, dtype=np.dtype([("x", float), ("y", float), ("intensity", float)]))
        peakB = np.zeros(1, dtype=np.dtype([("x", float), ("y", float), ("intensity", float)]))

        peakA['x'] = peakCoordinates['x'][smallestRadiiIndices[peakAInd]]; peakA['y'] = peakCoordinates['y'][smallestRadiiIndices[peakAInd]]; peakA['intensity'] = peakCoordinates['intensity'][smallestRadiiIndices[peakAInd]]
        peakB['x'] = peakCoordinates['x'][smallestRadiiIndices[peakBInd]]; peakB['y'] = peakCoordinates['y'][smallestRadiiIndices[peakBInd]]; peakB['intensity'] = peakCoordinates['intensity'][smallestRadiiIndices[peakBInd]]
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
        peakList = self.get_maxima_2D(np.abs(np.fft.fftshift(np.fft.fft2(self._image.array))), maxNumPeaks = maxNumPeaks_in, minSpacing = minSpacingPeaks)
        if center_ignore_buffer != None:
            x_dist_to_center = peakList['x'] - nx/2
            y_dist_to_center = peakList['y'] - ny/2
            rad_dist_to_center = np.sqrt(x_dist_to_center**2 + y_dist_to_center**2)
            peakList = peakList[rad_dist_to_center>center_ignore_buffer]
            zero_peak = np.zeros(1, np.dtype([("x", float), ("y", float), ("intensity", float)]))
            zero_peak['x'] = nx/2
            zero_peak['y'] = ny/2
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
        xyCoords = np.array([coords_arr['x'][0], coords_arr['y'][0]])
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
        xyCoords = np.array([coords_arr['x'], coords_arr['y']])
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
                maxima["intensity"] / maxima["intensity"][relativeToPeak]
                < minRelativeIntensity
            )
            maxima = maxima[~deletemask]

        # Remove maxima which are too close
        if minSpacing > 0:
            deletemask = np.zeros(len(maxima), dtype=bool)
            for i in range(len(maxima)):
                if deletemask[i] == False:  # noqa: E712
                    tooClose = (
                        (maxima["x"] - maxima["x"][i]) ** 2
                        + (maxima["y"] - maxima["y"][i]) ** 2
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
        elif device == "gpu":
            xp = cp

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
        except:
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
        elif device == "gpu":
            xp = cp

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
        measure_ind,
        reference_ind,
        reference_radius=None,
        reference_num=4,
        coordinates: str = "cartesian",
        plot_polarization_vectors: bool = False,
        **plot_kwargs,
    ):
        from scipy.spatial import cKDTree

        # lattice vectors in pixels
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)

        if coordinates not in ("cartesian", "fractional"):
            raise ValueError(
                f"coordinates must be 'cartesian'(default) or 'fractional'. {coordinates} is not valid."
            )

        measure_ind = int(measure_ind)
        reference_ind = int(reference_ind)

        # Check for empty cells
        A_cell = self.atoms.get_data(measure_ind)
        B_cell = self.atoms.get_data(reference_ind)

        def is_empty(cell):
            return isinstance(cell, list) or cell is None or cell.size == 0

        if is_empty(A_cell) or is_empty(B_cell):
            out = Vector.from_shape(
                shape=(1,),
                fields=("x", "y", "a", "b", "x_ref", "y_ref"),
                units=("px", "px", "ind", "ind", "px", "px"),
                name="polarization",
            )
            out.set_data(np.zeros((0, 6), float), 0)
            return out

        # Extract common atom data
        Ax = self.atoms[measure_ind]["x"]
        Ay = self.atoms[measure_ind]["y"]
        Aa = self.atoms[measure_ind]["a"]
        Ab = self.atoms[measure_ind]["b"]
        Bx = self.atoms[reference_ind]["x"]
        By = self.atoms[reference_ind]["y"]

        # Method-specific processing
        if coordinates == "cartesian":
            if reference_radius is None:
                reference_radius = float(min(np.linalg.norm(u), np.linalg.norm(v)))

            query_coords = np.column_stack([Ax, Ay])
            ref_coords = np.column_stack([Bx, By])

        elif coordinates == "fractional":
            reference_radius = 3
            L = np.column_stack((u, v))
            # try:
            #     # Not sure if we need this or not, but keeping it for now.
            #     # Also depends on whether we would be caclulating polarization
            #     # based on fractional or cartesian coordinates
            #     L_inv = np.linalg.inv(L)
            # except np.linalg.LinAlgError:
            #     raise ValueError("Lattice vectors are singular and cannot be inverted.")

            Ba = self.atoms[reference_ind]["a"]
            Bb = self.atoms[reference_ind]["b"]
            query_coords = np.column_stack([Aa, Ab])
            ref_coords = np.column_stack([Ba, Bb])

        # KD-tree query
        tree = cKDTree(ref_coords)
        k = int(max(1, reference_num))
        dists, idxs = tree.query(
            query_coords,
            k=k,
            distance_upper_bound=float(reference_radius),
            workers=-1,
        )

        # Normalize shapes for k=1 case
        if k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        # Vectorized neighbor validation
        valid_mask = np.isfinite(dists) & (idxs < len(Bx))
        valid_counts = np.sum(valid_mask, axis=1)
        atoms_with_enough_neighbors = valid_counts >= reference_num

        if not np.any(atoms_with_enough_neighbors):
            out = Vector.from_shape(
                shape=(1,),
                fields=("x", "y", "a", "b", "x_ref", "y_ref"),
                units=("px", "px", "ind", "ind", "px", "px"),
                name="polarization",
            )
            out.set_data(np.zeros((0, 6), float), 0)
            return out

        # Filter to atoms with enough neighbors
        valid_atom_indices = np.where(atoms_with_enough_neighbors)[0]
        n_valid = len(valid_atom_indices)

        # Pre-allocate result array memory
        x_arr = Ax[valid_atom_indices].astype(float)
        y_arr = Ay[valid_atom_indices].astype(float)
        a_arr = Aa[valid_atom_indices].astype(float)
        b_arr = Ab[valid_atom_indices].astype(float)
        xr_arr = np.zeros(n_valid, dtype=float)
        yr_arr = np.zeros(n_valid, dtype=float)

        if coordinates == "cartesian":
            # Vectorized reference position calculation for xy method
            for i, atom_idx in enumerate(valid_atom_indices):
                valid_neighbors = valid_mask[atom_idx]
                if np.sum(valid_neighbors) >= reference_num:
                    # Get closest reference_num neighbors
                    valid_dists = dists[atom_idx][valid_neighbors]
                    valid_idxs = idxs[atom_idx][valid_neighbors]
                    closest_order = np.argsort(valid_dists)[:reference_num]
                    nbr_idx = valid_idxs[closest_order].astype(int)

                    xr_arr[i] = np.mean(Bx[nbr_idx])
                    yr_arr[i] = np.mean(By[nbr_idx])

        else:  # coordinates == "fractional"
            # Vectorized calculation for fractional coordinates method
            for i, atom_idx in enumerate(valid_atom_indices):
                valid_neighbors = valid_mask[atom_idx]
                if np.sum(valid_neighbors) >= reference_num:
                    # Get closest reference_num neighbors
                    valid_dists = dists[atom_idx][valid_neighbors]
                    valid_idxs = idxs[atom_idx][valid_neighbors]
                    closest_order = np.argsort(valid_dists)[:reference_num]
                    nbr_idx = valid_idxs[closest_order].astype(int)

                    # Vectorized matrix operations
                    a, b = a_arr[i], b_arr[i]
                    xi, yi = Bx[nbr_idx], By[nbr_idx]
                    ai, bi = Ba[nbr_idx], Bb[nbr_idx]

                    diff_ind = np.array([a - ai, b - bi])  # (2, n_neighbors)
                    neighbor_positions = np.array([xi, yi])  # (2, n_neighbors)
                    transformed = L @ diff_ind + neighbor_positions
                    exp_pos = np.mean(transformed, axis=1)  # (2,)

                    xr_arr[i] = exp_pos[0]
                    yr_arr[i] = exp_pos[1]

        out = Vector.from_shape(
            shape=(1,),
            fields=("x", "y", "a", "b", "x_ref", "y_ref"),
            units=("px", "px", "ind", "ind", "px", "px"),
            name="polarization",
        )

        arr = np.column_stack([x_arr, y_arr, a_arr, b_arr, xr_arr, yr_arr])
        out.set_data(arr, 0)

        if plot_polarization_vectors:
            self.plot_polarization_vectors(out, **plot_kwargs)

        return out

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
        xR = pol_vec[0]["x_ref"]
        yR = pol_vec[0]["y_ref"]

        # Displacements (rows, cols)
        dr_raw = (xA - xR).astype(float)  # down +
        dc_raw = (yA - yR).astype(float)  # right +

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

        r_0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        theta_u = -np.arctan2(u[1], u[0])
        handedness = u[0] * v[1] - u[1] * v[0] > 0

        if theta_u > np.pi / 36 or theta_u < -np.pi / 36:
            from scipy.ndimage import rotate

            if not handedness:
                img_rgb = np.fliplr(img_rgb)

            img_rgb = rotate(
                img_rgb,
                -np.degrees(theta_u),
                axes=(1, 0),
                reshape=True,
                order=1,
                mode="constant",
                cval=0.0,
            )

            # Crop the image to deal with artifacts due to rotation
            mask = np.linalg.norm(img_rgb, axis=2) > 0
            rows, cols = np.where(mask)

            if len(rows) > 0 and len(cols) > 0:
                r_min, r_max = rows.min(), rows.max()
                c_min, c_max = cols.min(), cols.max()

                img_rgb = img_rgb[r_min : r_max + 1, c_min : c_max + 1, :]

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


# helper function for polar color mapping
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


def site_colors(number: int) -> tuple[float, float, float]:
    """
    Map an integer 'number' to an RGB triple in [0,1].
    Starts with the requested seed palette and cycles thereafter.
    """
    palette = [
        (0.00, 0.00, 0.00),  # 0: black
        (1.00, 0.00, 0.00),  # 1: red
        (0.00, 0.70, 1.00),  # 2: light blue (cyan-ish)
        (0.00, 0.70, 0.00),  # 3: green
        (1.00, 0.00, 1.00),  # 4: magenta
        (1.00, 0.70, 0.00),  # 5: orange
        (0.00, 0.30, 1.00),  # 6: blue-ish
        # extras to improve variety when cycling:
        (0.60, 0.20, 0.80),
        (0.30, 0.75, 0.75),
        (0.80, 0.40, 0.00),
        (0.20, 0.60, 0.20),
        (0.70, 0.70, 0.00),
    ]
    idx = int(number) % len(palette)
    return palette[idx]


####

    # def slice_along_axis(
    #         self,
    #         u = None,
    #         v = None,
    # ):
    #     uv_arr = self.uv_arr
    #     slice_vectors = self._positions_frac[1:]
        


    # def atoms_first_uvw(
    #     self,
    #     origin = None,
    #     u = None,
    #     v = None,
    #     positions_frac = None,
    #     tolerance_uvw: float = 1.1,
    #     w = None,
    #     numbers=None,
    #     edge_min_dist_px=None,
    #     subpixel: str = "poly",
    #     upsample_factor: int = 16,
    #     sigma: float = 0,
    #     minAbsoluteIntensity: float = 0,
    #     minRelativeIntensity: float = 0,
    #     relativeToPeak: float = 0,
    #     minSpacing: float = 0,
    #     edgeBoundary: int = 1,
    #     maxNumPeaks: int = 5000,
    #     plot_atoms=True,
    #     input_mask=None,
    #     refine_lattice=True,
    #     refine_maxiter: int = 200,
    #     intensity_radius = None,
    #     intensity_min: float | None = None,
    #     contrast_min=None,
    #     annulus_radii = None,
    #     check_uv_duplication = True,
    #     check_for_dislocations = False,
    #     merge_dislocation = False,
    #     **kwargs,
    # ):
    #     self.check_for_dislocations = check_for_dislocations
    #     # find all candidates above threshold
    #     maxima_candidates = self.get_maxima_2D(
    #         self.image.array, 
    #         subpixel = subpixel,
    #         upsample_factor = upsample_factor,
    #         sigma = sigma,
    #         minAbsoluteIntensity = minAbsoluteIntensity,
    #         minRelativeIntensity = minRelativeIntensity,
    #         relativeToPeak = relativeToPeak,
    #         minSpacing = minSpacing,
    #         edgeBoundary = edgeBoundary,
    #         maxNumPeaks = maxNumPeaks,
    #         )

    #     H, W = self._image.shape  # x=rows, y=cols

    #     if origin is None:
    #         max_intensity_index = np.argmax(maxima_candidates[:]['intensity'])
    #         origin_x = maxima_candidates[max_intensity_index]['x']
    #         origin_y = maxima_candidates[max_intensity_index]['y']
    #         origin = np.array([origin_x, origin_y])

    #     if u is None or v is None:
    #         num_peaks_search = 20
    #         num_peaks_use = 2
    #         center_ignore_buffer = 15
    #         minSpacingPeaks = 5
    #         uv_result_inv = self.auto_peak_finder(num_peaks_search = num_peaks_search, num_peaks_use = num_peaks_use, center_ignore_buffer = center_ignore_buffer, minSpacingPeaks = minSpacingPeaks)

    #         g_vector_1_c = np.array([uv_result_inv[0]['x'], uv_result_inv[0]['y']])
    #         g_vector_2_c = np.array([uv_result_inv[1]['x'], uv_result_inv[1]['y']])
    #         g_vec1 = np.zeros(2)
    #         g_vec1[0] = ((g_vector_1_c[0] - (0.5*H))/H)
    #         g_vec1[1] = ((g_vector_1_c[1] - (0.5*W))/W)
    #         g_vec2 = np.zeros(2)
    #         g_vec2[0] = ((g_vector_2_c[0] - (0.5*H))/H)
    #         g_vec2[1] = ((g_vector_2_c[1] - (0.5*W))/W)
    #         g_matrix = np.array([g_vec1, g_vec2])
    #         a_matrix = np.linalg.inv(g_matrix)
    #         a_transpose = a_matrix.T
    #         u = np.array([a_transpose[0,0], a_transpose[0,1]])
    #         v = np.array([a_transpose[1,0], a_transpose[1,1]])
    #         self.u = u
    #         self.v = v


    #     if positions_frac is None:
    #         positions_frac = np.atleast_2d(np.array((0,0))),

    #     self._positions_frac = np.atleast_2d(np.array(positions_frac, dtype=float))
    #     self._num_sites = self._positions_frac.shape[0]
    #     self._numbers = (
    #         np.arange(1, self._num_sites + 1, dtype=int)
    #         if numbers is None
    #         else np.atleast_1d(np.array(numbers, dtype=int))
    #     )
    #     if w is None:
    #         if np.abs(np.rad2deg(np.arccos(np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v))))) > np.deg2rad(90):
    #             w = np.asarray(u)+np.asarray(v)
    #             w_sign = 1
    #         else:
    #             w = np.asarray(u)-np.asarray(v)
    #             w_sign = -1
    #     else:
    #         w_sign = 1

    #     self._lat = np.vstack(
    #         (
    #             np.array(origin),
    #             np.array(u),
    #             np.array(v),
    #         )
    #     )

    #     im = np.asarray(self._image.array, dtype=float)
    #     r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
    #     A = np.column_stack((u, v))

    #     def _auto_radius_px() -> float:
    #         S = self._positions_frac
    #         if S.shape[0] >= 2:
    #             d = S[:, None, :] - S[None, :, :]
    #             d = d - np.round(d)
    #             same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
    #             dpix = d @ A.T
    #             dist = np.linalg.norm(dpix, axis=2)
    #             dist[same] = np.inf
    #             nn = float(np.min(dist))
    #         else:
    #             nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
    #         if not np.isfinite(nn) or nn <= 0:
    #             nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
    #         return 0.5 * nn

    #     r_px = float(intensity_radius) if intensity_radius is not None else _auto_radius_px()
    #     rin, rout = (1.5 * r_px, 3.0 * r_px) if annulus_radii is None else annulus_radii
    #     R_disk = int(np.ceil(r_px))
    #     R_ring = int(np.ceil(rout))

    #     def mean_disk(x: float, y: float) -> float:
    #         ix0, iy0 = int(np.floor(x)), int(np.floor(y))
    #         i0, i1 = max(0, ix0 - R_disk), min(H - 1, ix0 + R_disk)
    #         j0, j1 = max(0, iy0 - R_disk), min(W - 1, iy0 + R_disk)
    #         ii = np.arange(i0, i1 + 1)[:, None]
    #         jj = np.arange(j0, j1 + 1)[None, :]
    #         dx, dy = ii - x, jj - y
    #         mask_circle = (dx * dx + dy * dy) <= (r_px * r_px)
    #         vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_circle]
    #         if vals.size == 0:
    #             return float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
    #         return float(vals.mean())

    #     def mean_std_annulus(x: float, y: float) -> tuple[float, float]:
    #         ix0, iy0 = int(np.floor(x)), int(np.floor(y))
    #         i0, i1 = max(0, ix0 - R_ring), min(H - 1, ix0 + R_ring)
    #         j0, j1 = max(0, iy0 - R_ring), min(W - 1, iy0 + R_ring)
    #         ii = np.arange(i0, i1 + 1)[:, None]
    #         jj = np.arange(j0, j1 + 1)[None, :]
    #         dx, dy = ii - x, jj - y
    #         r2 = dx * dx + dy * dy
    #         mask_ring = (r2 >= rin * rin) & (r2 <= rout * rout)
    #         vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_ring]
    #         if vals.size == 0:
    #             val = float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
    #             return val, 0.0
    #         return float(vals.mean()), float(vals.std(ddof=0))

    #     # mask of where in real space maxima can occur
    #     H, W = self._image.shape  # x=rows, y=cols
    #     edge_thresh = float(edge_min_dist_px) if edge_min_dist_px is not None else 0.0

    #     DT = None
    #     if input_mask is not None:
    #         m = np.asarray(input_mask).astype(bool)
    #         if m.shape != (H, W):
    #             raise ValueError(f"mask shape {m.shape} must match image shape {(H, W)}")
    #         try:
    #             from scipy.ndimage import distance_transform_edt

    #             DT = distance_transform_edt(m)
    #         except Exception:
    #             DT = None

    #     # find the maxima closest to the origin:
    #     maxima_candidates_x = maxima_candidates[:]['x']
    #     maxima_candidates_y = maxima_candidates[:]['y']

    #     pm_arr = np.array([-1,0,1])
    #     u_norm = np.linalg.norm(u)
    #     v_norm = np.linalg.norm(v)
    #     w_norm = np.linalg.norm(w)
    #     uvw_arr = np.array([np.asarray(u),np.asarray(v),np.asarray(w)])
    #     uvw_norm = 0.5 * (u_norm + v_norm + w_norm)
    #     self.uv_norm = uvw_norm
    #     self.uv_arr = uvw_arr
    #     self.tolerance_uv = tolerance_uvw
    #     x = maxima_candidates_x
    #     y = maxima_candidates_y

    #     in_bounds = (x >= 0.0) & (x <= H - 1) & (y >= 0.0) & (y <= W - 1)
    #     border_ok = (
    #         (x - edge_thresh >= 0.0)
    #         & (x + edge_thresh <= H - 1)
    #         & (y - edge_thresh >= 0.0)
    #         & (y + edge_thresh <= W - 1)
    #     )
    #     if input_mask is not None:
    #         if DT is not None:
    #             ii = np.clip(np.round(x).astype(int), 0, H - 1)
    #             jj = np.clip(np.round(y).astype(int), 0, W - 1)
    #             mask_ok = DT[ii, jj] >= edge_thresh
    #         else:
    #             m = np.asarray(input_mask).astype(bool)
    #             mask_ok = m[
    #                 np.clip(np.round(x).astype(int), 0, H - 1),
    #                 np.clip(np.round(y).astype(int), 0, W - 1),
    #             ]
    #     else:
    #         mask_ok = np.ones_like(in_bounds, dtype=bool)

    #     int_center = np.empty(x.shape[0], dtype=float)
    #     for i in range(x.shape[0]):
    #         int_center[i] = mean_disk(x[i], y[i])

    #     keep = in_bounds & border_ok & mask_ok
    #     if intensity_min is not None:
    #         keep &= int_center >= float(intensity_min)
    #     if contrast_min is not None:
    #         bg_mean = np.empty(x.shape[0], dtype=float)
    #         for i in range(x.shape[0]):
    #             bg_mean[i], _ = mean_std_annulus(x[i], y[i])
    #         keep &= (int_center - bg_mean) >= float(contrast_min)

    #     if np.any(keep):
    #         maxima_candidates = maxima_candidates[keep]
    #     else:
    #         raise ValueError("Zero maxima candidates kept")

    #     # find the maxima closest to the origin:
    #     maxima_candidates_x = maxima_candidates[:]['x']
    #     maxima_candidates_y = maxima_candidates[:]['y']
    #     maxima_candidates_intensity = maxima_candidates[:]['intensity']

    #     # the unique ids array is an array of the original index, candidacy, a (of a * u), and b (of b * v), and c (of c * w)
    #     unique_ids = np.zeros([5, len(maxima_candidates)])
    #     unique_ids[0,:] = np.arange(0,len(maxima_candidates))
    #     unique_ids[-1,:] = -1*np.arange(1,1+len(maxima_candidates))

    #     if plot_atoms:
    #         fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
    #         if ax.images:
    #             ax.images[-1].set_zorder(0)
    #         xs = maxima_candidates_x
    #         ys = maxima_candidates_y
    #         rgb = site_colors(int(self._numbers[0]))
    #         ax.scatter(
    #             ys,
    #             xs,
    #             s=18,
    #             facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
    #             edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
    #             linewidths=0.75,
    #             marker="o",
    #             zorder=25,
    #         )
    #         ax.set_xlim(0, W)
    #         ax.set_ylim(H, 0)

    #     radial_dist = ((maxima_candidates_x - origin[0])**2 + (maxima_candidates_y - origin[1])**2)**(0.5)
    #     origin_candidate_index = np.argmin(radial_dist) # use the first minima, if there are multiple
    #     unique_ids[1,origin_candidate_index] = 1
    #     unique_ids[4,origin_candidate_index] = 0

    #     atoms_found_this_iteration = np.zeros(len(maxima_candidates))
    #     atoms_found_prev_iteration = np.zeros(len(maxima_candidates))
    #     atoms_found_previous_iterations = np.zeros(len(maxima_candidates), dtype = bool)
    #     atoms_found_prev_iteration[origin_candidate_index] = 1
    #     found_atoms_in_prev_iteration = True
    #     iteration_while = 0
    #     while found_atoms_in_prev_iteration is True:
    #         for atom_index in range(len(maxima_candidates)):
    #             if atoms_found_prev_iteration[atom_index] > 0:
    #                 for pm in pm_arr:
    #                     for uvw_index, lat_vec in enumerate(uvw_arr):
    #                         position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
    #                         position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
    #                         radial_dist = ((maxima_candidates_x - position_x)**2 + (maxima_candidates_y - position_y)**2)**(0.5)
    #                         radial_dist[atom_index] = uvw_norm * (tolerance_uvw - 1) * 2 # make sure that self is outside of range
    #                         if (radial_dist < (uvw_norm * (tolerance_uvw - 1))).any():
    #                             successful_candidate_index = np.argmin(radial_dist)
    #                             if unique_ids[1, successful_candidate_index] == 0:
    #                                 atoms_found_this_iteration[successful_candidate_index] += 1
    #                                 unique_ids[1, successful_candidate_index] = 1
    #                                 unique_ids[2, successful_candidate_index] = unique_ids[2, atom_index] + pm*int(uvw_index == 0) + pm*int(uvw_index == 2)
    #                                 unique_ids[3, successful_candidate_index] = unique_ids[3, atom_index] + pm*int(uvw_index == 1) + w_sign*pm*int(uvw_index == 2)
    #                                 unique_ids[4, successful_candidate_index] = 0
    #         # check if any atom was somehow still found twice:
    #         assert np.max(atoms_found_this_iteration) < 2
    #         # check if any found atoms have the same uv index
    #         if check_uv_duplication:
    #             uv_pairs = unique_ids[1:5,:].T
    #             unique_pairs, inverse, counts = np.unique(uv_pairs, axis=0, return_inverse=True, return_counts=True)
    #             duplicate_groups = [np.where(inverse == k)[0] for k, c in enumerate(counts) if c > 1]
    #             mask_atoms_found = atoms_found_this_iteration.astype(bool)
    #             if len(duplicate_groups) != 0:
    #                 for duplicate_group in duplicate_groups:
    #                     duplicate_group = np.asarray(duplicate_group)
    #                     duplicate_atoms_index_found_previous_iterations = duplicate_group[atoms_found_previous_iterations[duplicate_group]]
    #                     if duplicate_atoms_index_found_previous_iterations.size > 1:
    #                         if origin_candidate_index not in duplicate_group:
    #                             raise ValueError("The duplicate atoms finding code is somehow bugged")
    #                         else:
    #                             kept_index = origin_candidate_index
    #                     elif duplicate_atoms_index_found_previous_iterations.size == 1:
    #                         kept_index = duplicate_atoms_index_found_previous_iterations
    #                     else:
    #                         kept_index = duplicate_group[mask_atoms_found[duplicate_group]][0]
    #                     wipe_indicies = duplicate_group[duplicate_group != kept_index]
    #                     unique_ids[1, wipe_indicies] = 2 # this signals to not accept for this maxima anymore (and flags this as a dulpicate)
    #                     unique_ids[2:4,wipe_indicies] = 0
    #                     unique_ids[4,wipe_indicies] = -1*(wipe_indicies+1)
    #                     atoms_found_previous_iterations[wipe_indicies] = False
    #                     mask_atoms_found[wipe_indicies] = False
    #                     atoms_found_this_iteration[wipe_indicies] = 0
    #         if np.sum(atoms_found_this_iteration) == 0:
    #             found_atoms_in_prev_iteration = False
    #             print('stopping search')
            

    #         atoms_found_previous_iterations |= atoms_found_this_iteration.astype(bool)

    #         atoms_found_prev_iteration = atoms_found_this_iteration.copy()
    #         atoms_found_this_iteration = np.zeros(len(maxima_candidates))
    #         iteration_while += 1


    #     if check_for_dislocations:
    #         for atom_index in range(len(maxima_candidates)):
    #             if unique_ids[1,atom_index] == 1:
    #                 for pm in pm_arr:
    #                     for uvw_index, lat_vec in enumerate(uvw_arr):
    #                         position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
    #                         position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
    #                         radial_dist = ((maxima_candidates_x - position_x)**2 + (maxima_candidates_y - position_y)**2)**(0.5)
    #                         radial_dist[atom_index] = uvw_norm * (tolerance_uvw - 1) * 2 # make sure that self is outside of range
    #                         if (radial_dist < (uvw_norm * (tolerance_uvw - 1))).any():
    #                             successful_candidate_index = np.argmin(radial_dist)
    #                             if unique_ids[1, successful_candidate_index] == 2:
    #                                 atoms_found_this_iteration[successful_candidate_index] += 1
    #                                 unique_ids[1, successful_candidate_index] = 3 # for being found in dislocation search
    #                                 unique_ids[2, successful_candidate_index] = unique_ids[2, atom_index] + pm*int(uvw_index == 0) + pm*int(uvw_index == 2)
    #                                 unique_ids[3, successful_candidate_index] = unique_ids[3, atom_index] + pm*int(uvw_index == 1) - pm*int(uvw_index == 2)
    #                                 unique_ids[4, successful_candidate_index] = 0
    #         maxima_dislocation_x = maxima_candidates_x[unique_ids[1,:] == 3]
    #         maxima_dislocation_y = maxima_candidates_y[unique_ids[1,:] == 3]
    #         maxima_dislocation_u = unique_ids[2,unique_ids[1,:] == 3]
    #         maxima_dislocation_v = unique_ids[3,unique_ids[1,:] == 3]
    #         maxima_dislocation_intensity = maxima_candidates_intensity[unique_ids[1,:] == 3]

    #         self.atoms_dislocation = Vector.from_shape(
    #             shape=(self._num_sites),
    #             fields=("x", "y", "a", "b", "int_peak"),
    #             units=("px", "px", "ind", "ind", "counts"),
    #         )

    #         arr = np.vstack(
    #             (maxima_dislocation_x, maxima_dislocation_y, maxima_dislocation_u, maxima_dislocation_v, maxima_dislocation_intensity)
    #         ).T
    #         self.atoms_dislocation.set_data(arr, 0)



    #     maxima_accepted_x = maxima_candidates_x[unique_ids[1,:] == 1]
    #     maxima_accepted_y = maxima_candidates_y[unique_ids[1,:] == 1]

    #     maxima_accepted_u = unique_ids[2, unique_ids[1,:] == 1]
    #     maxima_accepted_v = unique_ids[3, unique_ids[1,:] == 1]

    #     maxima_accepted_intensity = maxima_candidates_intensity[unique_ids[1,:] == 1]

    #     self.atoms = Vector.from_shape(
    #         shape=(self._num_sites),
    #         fields=("x", "y", "a", "b", "int_peak"),
    #         units=("px", "px", "ind", "ind", "counts"),
    #     )

    #     if not merge_dislocation or not check_for_dislocations:
    #         arr = np.vstack(
    #             (maxima_accepted_x, maxima_accepted_y, maxima_accepted_u, maxima_accepted_v, maxima_accepted_intensity)
    #         ).T
    #         self.atoms.set_data(arr, 0)

    #     if merge_dislocation and check_for_dislocations:

    #         maxima_merge_x = maxima_candidates_x[np.isin(unique_ids[1, :], [1, 3])]
    #         maxima_merge_y = maxima_candidates_y[np.isin(unique_ids[1, :], [1, 3])]

    #         maxima_merge_u = unique_ids[2, np.isin(unique_ids[1, :], [1, 3])]
    #         maxima_merge_v = unique_ids[3, np.isin(unique_ids[1, :], [1, 3])]

    #         maxima_merge_intensity = maxima_candidates_intensity[np.isin(unique_ids[1, :], [1, 3])]
    #         arr = np.vstack(
    #             (maxima_merge_x, maxima_merge_y, maxima_merge_u, maxima_merge_v, maxima_merge_intensity)
    #         ).T
    #         self.atoms.set_data(arr, 0)


    #     if plot_atoms:
    #         fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
    #         if ax.images:
    #             ax.images[-1].set_zorder(0)
    #         xs = maxima_accepted_x
    #         ys = maxima_accepted_y
    #         rgb = site_colors(int(self._numbers[0]))
    #         ax.scatter(
    #             ys,
    #             xs,
    #             s=18,
    #             facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
    #             edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
    #             linewidths=0.75,
    #             marker="o",
    #             zorder=25,
    #         )
    #         ax.set_xlim(0, W)
    #         ax.set_ylim(H, 0)

    #     return self

    # def atoms_first_uvw_bsites(
    #     self,
    #     origin = None,
    #     u = None,
    #     v = None,
    #     positions_frac = None,
    #     tolerance_uvw: float = 1.1,
    #     w = None,
    #     numbers=None,
    #     edge_min_dist_px=None,
    #     subpixel: str = "poly",
    #     upsample_factor: int = 16,
    #     sigma: float = 0,
    #     minAbsoluteIntensity: float = 0,
    #     minRelativeIntensity: float = 0,
    #     relativeToPeak: float = 0,
    #     minSpacing: float = 0,
    #     edgeBoundary: int = 1,
    #     maxNumPeaks: int = 5000,
    #     plot_atoms=True,
    #     input_mask=None,
    #     refine_lattice=True,
    #     refine_maxiter: int = 200,
    #     intensity_radius = None,
    #     intensity_min: float | None = None,
    #     contrast_min=None,
    #     annulus_radii = None,
    #     check_uv_duplication = True,
    #     check_for_dislocations = False,
    #     merge_dislocation = False,
    #     **kwargs,
    # ):
    #     self.check_for_dislocations = check_for_dislocations
    #     # find all candidates above threshold
    #     maxima_candidates = self.get_maxima_2D(
    #         self.image.array, 
    #         subpixel = subpixel,
    #         upsample_factor = upsample_factor,
    #         sigma = sigma,
    #         minAbsoluteIntensity = minAbsoluteIntensity,
    #         minRelativeIntensity = minRelativeIntensity,
    #         relativeToPeak = relativeToPeak,
    #         minSpacing = minSpacing,
    #         edgeBoundary = edgeBoundary,
    #         maxNumPeaks = maxNumPeaks,
    #         )

    #     H, W = self._image.shape  # x=rows, y=cols

    #     if origin is None:
    #         max_intensity_index = np.argmax(maxima_candidates[:]['intensity'])
    #         origin_x = maxima_candidates[max_intensity_index]['x']
    #         origin_y = maxima_candidates[max_intensity_index]['y']
    #         origin = np.array([origin_x, origin_y])

    #     if u is None or v is None:
    #         num_peaks_search = 20
    #         num_peaks_use = 2
    #         center_ignore_buffer = 15
    #         minSpacingPeaks = 5
    #         uv_result_inv = self.auto_peak_finder(num_peaks_search = num_peaks_search, num_peaks_use = num_peaks_use, center_ignore_buffer = center_ignore_buffer, minSpacingPeaks = minSpacingPeaks)

    #         g_vector_1_c = np.array([uv_result_inv[0]['x'], uv_result_inv[0]['y']])
    #         g_vector_2_c = np.array([uv_result_inv[1]['x'], uv_result_inv[1]['y']])
    #         g_vec1 = np.zeros(2)
    #         g_vec1[0] = ((g_vector_1_c[0] - (0.5*H))/H)
    #         g_vec1[1] = ((g_vector_1_c[1] - (0.5*W))/W)
    #         g_vec2 = np.zeros(2)
    #         g_vec2[0] = ((g_vector_2_c[0] - (0.5*H))/H)
    #         g_vec2[1] = ((g_vector_2_c[1] - (0.5*W))/W)
    #         g_matrix = np.array([g_vec1, g_vec2])
    #         a_matrix = np.linalg.inv(g_matrix)
    #         a_transpose = a_matrix.T
    #         u = np.array([a_transpose[0,0], a_transpose[0,1]])
    #         v = np.array([a_transpose[1,0], a_transpose[1,1]])
    #         self.u = u
    #         self.v = v


    #     if positions_frac is None:
    #         positions_frac = np.atleast_2d(np.array((0,0))),

    #     self._positions_frac = np.atleast_2d(np.array(positions_frac, dtype=float))
    #     self._num_sites = self._positions_frac.shape[0]
    #     self._numbers = (
    #         np.arange(1, self._num_sites + 1, dtype=int)
    #         if numbers is None
    #         else np.atleast_1d(np.array(numbers, dtype=int))
    #     )
    #     if w is None:
    #         if np.abs(np.rad2deg(np.arccos(np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v))))) > np.deg2rad(90):
    #             w = np.asarray(u)+np.asarray(v)
    #             w_sign = 1
    #         else:
    #             w = np.asarray(u)-np.asarray(v)
    #             w_sign = -1
    #     else:
    #         w_sign = 1

    #     self._lat = np.vstack(
    #         (
    #             np.array(origin),
    #             np.array(u),
    #             np.array(v),
    #         )
    #     )

    #     im = np.asarray(self._image.array, dtype=float)
    #     r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
    #     A = np.column_stack((u, v))

    #     def _auto_radius_px() -> float:
    #         S = self._positions_frac
    #         if S.shape[0] >= 2:
    #             d = S[:, None, :] - S[None, :, :]
    #             d = d - np.round(d)
    #             same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
    #             dpix = d @ A.T
    #             dist = np.linalg.norm(dpix, axis=2)
    #             dist[same] = np.inf
    #             nn = float(np.min(dist))
    #         else:
    #             nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
    #         if not np.isfinite(nn) or nn <= 0:
    #             nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
    #         return 0.5 * nn

    #     r_px = float(intensity_radius) if intensity_radius is not None else _auto_radius_px()
    #     rin, rout = (1.5 * r_px, 3.0 * r_px) if annulus_radii is None else annulus_radii
    #     R_disk = int(np.ceil(r_px))
    #     R_ring = int(np.ceil(rout))

    #     def mean_disk(x: float, y: float) -> float:
    #         ix0, iy0 = int(np.floor(x)), int(np.floor(y))
    #         i0, i1 = max(0, ix0 - R_disk), min(H - 1, ix0 + R_disk)
    #         j0, j1 = max(0, iy0 - R_disk), min(W - 1, iy0 + R_disk)
    #         ii = np.arange(i0, i1 + 1)[:, None]
    #         jj = np.arange(j0, j1 + 1)[None, :]
    #         dx, dy = ii - x, jj - y
    #         mask_circle = (dx * dx + dy * dy) <= (r_px * r_px)
    #         vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_circle]
    #         if vals.size == 0:
    #             return float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
    #         return float(vals.mean())

    #     def mean_std_annulus(x: float, y: float) -> tuple[float, float]:
    #         ix0, iy0 = int(np.floor(x)), int(np.floor(y))
    #         i0, i1 = max(0, ix0 - R_ring), min(H - 1, ix0 + R_ring)
    #         j0, j1 = max(0, iy0 - R_ring), min(W - 1, iy0 + R_ring)
    #         ii = np.arange(i0, i1 + 1)[:, None]
    #         jj = np.arange(j0, j1 + 1)[None, :]
    #         dx, dy = ii - x, jj - y
    #         r2 = dx * dx + dy * dy
    #         mask_ring = (r2 >= rin * rin) & (r2 <= rout * rout)
    #         vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_ring]
    #         if vals.size == 0:
    #             val = float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
    #             return val, 0.0
    #         return float(vals.mean()), float(vals.std(ddof=0))

    #     # mask of where in real space maxima can occur
    #     H, W = self._image.shape  # x=rows, y=cols
    #     edge_thresh = float(edge_min_dist_px) if edge_min_dist_px is not None else 0.0

    #     DT = None
    #     if input_mask is not None:
    #         m = np.asarray(input_mask).astype(bool)
    #         if m.shape != (H, W):
    #             raise ValueError(f"mask shape {m.shape} must match image shape {(H, W)}")
    #         try:
    #             from scipy.ndimage import distance_transform_edt

    #             DT = distance_transform_edt(m)
    #         except Exception:
    #             DT = None

    #     # find the maxima closest to the origin:
    #     maxima_candidates_x = maxima_candidates[:]['x']
    #     maxima_candidates_y = maxima_candidates[:]['y']

    #     pm_arr = np.array([-1,0,1])
    #     u_norm = np.linalg.norm(u)
    #     v_norm = np.linalg.norm(v)
    #     w_norm = np.linalg.norm(w)
    #     uvw_arr = np.array([np.asarray(u),np.asarray(v),np.asarray(w)])
    #     uvw_norm = 0.5 * (u_norm + v_norm + w_norm)
    #     self.uv_norm = uvw_norm
    #     self.uv_arr = uvw_arr
    #     self.tolerance_uv = tolerance_uvw
    #     x = maxima_candidates_x
    #     y = maxima_candidates_y

    #     in_bounds = (x >= 0.0) & (x <= H - 1) & (y >= 0.0) & (y <= W - 1)
    #     border_ok = (
    #         (x - edge_thresh >= 0.0)
    #         & (x + edge_thresh <= H - 1)
    #         & (y - edge_thresh >= 0.0)
    #         & (y + edge_thresh <= W - 1)
    #     )
    #     if input_mask is not None:
    #         if DT is not None:
    #             ii = np.clip(np.round(x).astype(int), 0, H - 1)
    #             jj = np.clip(np.round(y).astype(int), 0, W - 1)
    #             mask_ok = DT[ii, jj] >= edge_thresh
    #         else:
    #             m = np.asarray(input_mask).astype(bool)
    #             mask_ok = m[
    #                 np.clip(np.round(x).astype(int), 0, H - 1),
    #                 np.clip(np.round(y).astype(int), 0, W - 1),
    #             ]
    #     else:
    #         mask_ok = np.ones_like(in_bounds, dtype=bool)

    #     int_center = np.empty(x.shape[0], dtype=float)
    #     for i in range(x.shape[0]):
    #         int_center[i] = mean_disk(x[i], y[i])

    #     keep = in_bounds & border_ok & mask_ok
    #     if intensity_min is not None:
    #         keep &= int_center >= float(intensity_min)
    #     if contrast_min is not None:
    #         bg_mean = np.empty(x.shape[0], dtype=float)
    #         for i in range(x.shape[0]):
    #             bg_mean[i], _ = mean_std_annulus(x[i], y[i])
    #         keep &= (int_center - bg_mean) >= float(contrast_min)

    #     if np.any(keep):
    #         maxima_candidates = maxima_candidates[keep]
    #     else:
    #         raise ValueError("Zero maxima candidates kept")

    #     # find the maxima closest to the origin:
    #     maxima_candidates_x = maxima_candidates[:]['x']
    #     maxima_candidates_y = maxima_candidates[:]['y']
    #     maxima_candidates_intensity = maxima_candidates[:]['intensity']

    #     # the unique ids array is an array of the original index, candidacy, a (of a * u), and b (of b * v), and c (of c * w)
    #     unique_ids = np.zeros([5, len(maxima_candidates)])
    #     unique_ids[0,:] = np.arange(0,len(maxima_candidates))
    #     unique_ids[-1,:] = -1*np.arange(1,1+len(maxima_candidates))

    #     if plot_atoms:
    #         fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
    #         if ax.images:
    #             ax.images[-1].set_zorder(0)
    #         xs = maxima_candidates_x
    #         ys = maxima_candidates_y
    #         rgb = site_colors(int(self._numbers[0]))
    #         ax.scatter(
    #             ys,
    #             xs,
    #             s=18,
    #             facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
    #             edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
    #             linewidths=0.75,
    #             marker="o",
    #             zorder=25,
    #         )
    #         ax.set_xlim(0, W)
    #         ax.set_ylim(H, 0)

    #     radial_dist = ((maxima_candidates_x - origin[0])**2 + (maxima_candidates_y - origin[1])**2)**(0.5)
    #     origin_candidate_index = np.argmin(radial_dist) # use the first minima, if there are multiple
    #     unique_ids[1,origin_candidate_index] = 1
    #     unique_ids[4,origin_candidate_index] = 0

    #     atoms_found_this_iteration = np.zeros(len(maxima_candidates))
    #     atoms_found_prev_iteration = np.zeros(len(maxima_candidates))
    #     atoms_found_previous_iterations = np.zeros(len(maxima_candidates), dtype = bool)
    #     atoms_found_prev_iteration[origin_candidate_index] = 1
    #     found_atoms_in_prev_iteration = True
    #     iteration_while = 0
    #     while found_atoms_in_prev_iteration is True:
    #         for atom_index in range(len(maxima_candidates)):
    #             if atoms_found_prev_iteration[atom_index] > 0:
    #                 for pm in pm_arr:
    #                     for uvw_index, lat_vec in enumerate(uvw_arr):
    #                         position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
    #                         position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
    #                         radial_dist = ((maxima_candidates_x - position_x)**2 + (maxima_candidates_y - position_y)**2)**(0.5)
    #                         radial_dist[atom_index] = uvw_norm * (tolerance_uvw - 1) * 2 # make sure that self is outside of range
    #                         if (radial_dist < (uvw_norm * (tolerance_uvw - 1))).any():
    #                             successful_candidate_index = np.argmin(radial_dist)
    #                             if unique_ids[1, successful_candidate_index] == 0:
    #                                 atoms_found_this_iteration[successful_candidate_index] += 1
    #                                 unique_ids[1, successful_candidate_index] = 1
    #                                 unique_ids[2, successful_candidate_index] = unique_ids[2, atom_index] + pm*int(uvw_index == 0) + pm*int(uvw_index == 2)
    #                                 unique_ids[3, successful_candidate_index] = unique_ids[3, atom_index] + pm*int(uvw_index == 1) + w_sign*pm*int(uvw_index == 2)
    #                                 unique_ids[4, successful_candidate_index] = 0
    #         # check if any atom was somehow still found twice:
    #         assert np.max(atoms_found_this_iteration) < 2
    #         # check if any found atoms have the same uv index
    #         if check_uv_duplication:
    #             uv_pairs = unique_ids[1:5,:].T
    #             unique_pairs, inverse, counts = np.unique(uv_pairs, axis=0, return_inverse=True, return_counts=True)
    #             duplicate_groups = [np.where(inverse == k)[0] for k, c in enumerate(counts) if c > 1]
    #             mask_atoms_found = atoms_found_this_iteration.astype(bool)
    #             if len(duplicate_groups) != 0:
    #                 for duplicate_group in duplicate_groups:
    #                     duplicate_group = np.asarray(duplicate_group)
    #                     duplicate_atoms_index_found_previous_iterations = duplicate_group[atoms_found_previous_iterations[duplicate_group]]
    #                     if duplicate_atoms_index_found_previous_iterations.size > 1:
    #                         if origin_candidate_index not in duplicate_group:
    #                             raise ValueError("The duplicate atoms finding code is somehow bugged")
    #                         else:
    #                             kept_index = origin_candidate_index
    #                     elif duplicate_atoms_index_found_previous_iterations.size == 1:
    #                         kept_index = duplicate_atoms_index_found_previous_iterations
    #                     else:
    #                         kept_index = duplicate_group[mask_atoms_found[duplicate_group]][0]
    #                     wipe_indicies = duplicate_group[duplicate_group != kept_index]
    #                     unique_ids[1, wipe_indicies] = 2 # this signals to not accept for this maxima anymore (and flags this as a dulpicate)
    #                     unique_ids[2:4,wipe_indicies] = 0
    #                     unique_ids[4,wipe_indicies] = -1*(wipe_indicies+1)
    #                     atoms_found_previous_iterations[wipe_indicies] = False
    #                     mask_atoms_found[wipe_indicies] = False
    #                     atoms_found_this_iteration[wipe_indicies] = 0
    #         if np.sum(atoms_found_this_iteration) == 0:
    #             found_atoms_in_prev_iteration = False
    #             print('stopping search')
            

    #         atoms_found_previous_iterations |= atoms_found_this_iteration.astype(bool)

    #         atoms_found_prev_iteration = atoms_found_this_iteration.copy()
    #         atoms_found_this_iteration = np.zeros(len(maxima_candidates))
    #         iteration_while += 1


    #     if check_for_dislocations:
    #         for atom_index in range(len(maxima_candidates)):
    #             if unique_ids[1,atom_index] == 1:
    #                 for pm in pm_arr:
    #                     for uvw_index, lat_vec in enumerate(uvw_arr):
    #                         position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
    #                         position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
    #                         radial_dist = ((maxima_candidates_x - position_x)**2 + (maxima_candidates_y - position_y)**2)**(0.5)
    #                         radial_dist[atom_index] = uvw_norm * (tolerance_uvw - 1) * 2 # make sure that self is outside of range
    #                         if (radial_dist < (uvw_norm * (tolerance_uvw - 1))).any():
    #                             successful_candidate_index = np.argmin(radial_dist)
    #                             if unique_ids[1, successful_candidate_index] == 2:
    #                                 atoms_found_this_iteration[successful_candidate_index] += 1
    #                                 unique_ids[1, successful_candidate_index] = 3 # for being found in dislocation search
    #                                 unique_ids[2, successful_candidate_index] = unique_ids[2, atom_index] + pm*int(uvw_index == 0) + pm*int(uvw_index == 2)
    #                                 unique_ids[3, successful_candidate_index] = unique_ids[3, atom_index] + pm*int(uvw_index == 1) - pm*int(uvw_index == 2)
    #                                 unique_ids[4, successful_candidate_index] = 0
    #         maxima_dislocation_x = maxima_candidates_x[unique_ids[1,:] == 3]
    #         maxima_dislocation_y = maxima_candidates_y[unique_ids[1,:] == 3]
    #         maxima_dislocation_u = unique_ids[2,unique_ids[1,:] == 3]
    #         maxima_dislocation_v = unique_ids[3,unique_ids[1,:] == 3]
    #         maxima_dislocation_intensity = maxima_candidates_intensity[unique_ids[1,:] == 3]

    #         self.atoms_dislocation = Vector.from_shape(
    #             shape=(self._num_sites),
    #             fields=("x", "y", "a", "b", "int_peak"),
    #             units=("px", "px", "ind", "ind", "counts"),
    #         )

    #         arr = np.vstack(
    #             (maxima_dislocation_x, maxima_dislocation_y, maxima_dislocation_u, maxima_dislocation_v, maxima_dislocation_intensity)
    #         ).T
    #         self.atoms_dislocation.set_data(arr, 0)



    #     maxima_accepted_x = maxima_candidates_x[unique_ids[1,:] == 1]
    #     maxima_accepted_y = maxima_candidates_y[unique_ids[1,:] == 1]

    #     maxima_accepted_u = unique_ids[2, unique_ids[1,:] == 1]
    #     maxima_accepted_v = unique_ids[3, unique_ids[1,:] == 1]

    #     maxima_accepted_intensity = maxima_candidates_intensity[unique_ids[1,:] == 1]

    #     self.atoms = Vector.from_shape(
    #         shape=(self._num_sites),
    #         fields=("x", "y", "a", "b", "int_peak"),
    #         units=("px", "px", "ind", "ind", "counts"),
    #     )

    #     if not merge_dislocation or not check_for_dislocations:
    #         arr = np.vstack(
    #             (maxima_accepted_x, maxima_accepted_y, maxima_accepted_u, maxima_accepted_v, maxima_accepted_intensity)
    #         ).T
    #         self.atoms.set_data(arr, 0)

    #     if merge_dislocation and check_for_dislocations:

    #         maxima_merge_x = maxima_candidates_x[np.isin(unique_ids[1, :], [1, 3])]
    #         maxima_merge_y = maxima_candidates_y[np.isin(unique_ids[1, :], [1, 3])]

    #         maxima_merge_u = unique_ids[2, np.isin(unique_ids[1, :], [1, 3])]
    #         maxima_merge_v = unique_ids[3, np.isin(unique_ids[1, :], [1, 3])]

    #         maxima_merge_intensity = maxima_candidates_intensity[np.isin(unique_ids[1, :], [1, 3])]
    #         arr = np.vstack(
    #             (maxima_merge_x, maxima_merge_y, maxima_merge_u, maxima_merge_v, maxima_merge_intensity)
    #         ).T
    #         self.atoms.set_data(arr, 0)


    #     if plot_atoms:
    #         fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
    #         if ax.images:
    #             ax.images[-1].set_zorder(0)
    #         xs = maxima_accepted_x
    #         ys = maxima_accepted_y
    #         rgb = site_colors(int(self._numbers[0]))
    #         ax.scatter(
    #             ys,
    #             xs,
    #             s=18,
    #             facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
    #             edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
    #             linewidths=0.75,
    #             marker="o",
    #             zorder=25,
    #         )
    #         ax.set_xlim(0, W)
    #         ax.set_ylim(H, 0)

    #     return self





    # def atoms_first_uvw(
    #     self,
    #     origin = None,
    #     u = None,
    #     v = None,
    #     positions_frac = None,
    #     tolerance_uvw: float = 1.1,
    #     w = None,
    #     numbers=None,
    #     edge_min_dist_px=None,
    #     subpixel: str = "poly",
    #     upsample_factor: int = 16,
    #     sigma: float = 0,
    #     minAbsoluteIntensity: float = 0,
    #     minRelativeIntensity: float = 0,
    #     relativeToPeak: float = 0,
    #     minSpacing: float = 0,
    #     edgeBoundary: int = 1,
    #     maxNumPeaks: int = 5000,
    #     plot_atoms=True,
    #     input_mask=None,
    #     refine_lattice=True,
    #     refine_maxiter: int = 200,
    #     intensity_radius = None,
    #     intensity_min: float | None = None,
    #     contrast_min=None,
    #     annulus_radii = None,
    #     check_uv_duplication = True,
    #     check_for_dislocations = False,
    #     merge_dislocation = False,
    #     **kwargs,
    # ):
    #     self.check_for_dislocations = check_for_dislocations
    #     # find all candidates above threshold
    #     maxima_candidates = self.get_maxima_2D(
    #         self.image.array, 
    #         subpixel = subpixel,
    #         upsample_factor = upsample_factor,
    #         sigma = sigma,
    #         minAbsoluteIntensity = minAbsoluteIntensity,
    #         minRelativeIntensity = minRelativeIntensity,
    #         relativeToPeak = relativeToPeak,
    #         minSpacing = minSpacing,
    #         edgeBoundary = edgeBoundary,
    #         maxNumPeaks = maxNumPeaks,
    #         )

    #     H, W = self._image.shape  # x=rows, y=cols

    #     if origin is None:
    #         max_intensity_index = np.argmax(maxima_candidates[:]['intensity'])
    #         origin_x = maxima_candidates[max_intensity_index]['x']
    #         origin_y = maxima_candidates[max_intensity_index]['y']
    #         origin = np.array([origin_x, origin_y])

    #     if u is None or v is None:
    #         num_peaks_search = 20
    #         num_peaks_use = 2
    #         center_ignore_buffer = 15
    #         minSpacingPeaks = 5
    #         uv_result_inv = self.auto_peak_finder(num_peaks_search = num_peaks_search, num_peaks_use = num_peaks_use, center_ignore_buffer = center_ignore_buffer, minSpacingPeaks = minSpacingPeaks)

    #         g_vector_1_c = np.array([uv_result_inv[0]['x'], uv_result_inv[0]['y']])
    #         g_vector_2_c = np.array([uv_result_inv[1]['x'], uv_result_inv[1]['y']])
    #         g_vec1 = np.zeros(2)
    #         g_vec1[0] = ((g_vector_1_c[0] - (0.5*H))/H)
    #         g_vec1[1] = ((g_vector_1_c[1] - (0.5*W))/W)
    #         g_vec2 = np.zeros(2)
    #         g_vec2[0] = ((g_vector_2_c[0] - (0.5*H))/H)
    #         g_vec2[1] = ((g_vector_2_c[1] - (0.5*W))/W)
    #         g_matrix = np.array([g_vec1, g_vec2])
    #         a_matrix = np.linalg.inv(g_matrix)
    #         a_transpose = a_matrix.T
    #         u = np.array([a_transpose[0,0], a_transpose[0,1]])
    #         v = np.array([a_transpose[1,0], a_transpose[1,1]])
    #         self.u = u
    #         self.v = v


    #     if positions_frac is None:
    #         positions_frac = np.atleast_2d(np.array((1,1)))
    #     if (positions_frac[0] == np.array([0,0])).all():
    #         positions_frac[0] = np.atleast_2d(np.array((1,1)))

    #     self._positions_frac = np.atleast_2d(np.array(positions_frac, dtype=float))
    #     self._num_sites = self._positions_frac.shape[0]
    #     self._numbers = (
    #         np.arange(1, self._num_sites + 1, dtype=int)
    #         if numbers is None
    #         else np.atleast_1d(np.array(numbers, dtype=int))
    #     )
    #     if w is None:
    #         if np.abs(np.rad2deg(np.arccos(np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v))))) > np.deg2rad(90):
    #             w = np.asarray(u)+np.asarray(v)
    #             w_sign = 1
    #         else:
    #             w = np.asarray(u)-np.asarray(v)
    #             w_sign = -1
    #     else:
    #         w_sign = 1





    #     self._lat = np.vstack(
    #         (
    #             np.array(origin),
    #             np.array(u),
    #             np.array(v),
    #         )
    #     )

    #     im = np.asarray(self._image.array, dtype=float)
    #     r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
    #     A = np.column_stack((u, v))

    #     def _auto_radius_px() -> float:
    #         S = self._positions_frac
    #         if S.shape[0] >= 2:
    #             d = S[:, None, :] - S[None, :, :]
    #             d = d - np.round(d)
    #             same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
    #             dpix = d @ A.T
    #             dist = np.linalg.norm(dpix, axis=2)
    #             dist[same] = np.inf
    #             nn = float(np.min(dist))
    #         else:
    #             nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
    #         if not np.isfinite(nn) or nn <= 0:
    #             nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
    #         return 0.5 * nn

    #     r_px = float(intensity_radius) if intensity_radius is not None else _auto_radius_px()
    #     rin, rout = (1.5 * r_px, 3.0 * r_px) if annulus_radii is None else annulus_radii
    #     R_disk = int(np.ceil(r_px))
    #     R_ring = int(np.ceil(rout))

    #     def mean_disk(x: float, y: float) -> float:
    #         ix0, iy0 = int(np.floor(x)), int(np.floor(y))
    #         i0, i1 = max(0, ix0 - R_disk), min(H - 1, ix0 + R_disk)
    #         j0, j1 = max(0, iy0 - R_disk), min(W - 1, iy0 + R_disk)
    #         ii = np.arange(i0, i1 + 1)[:, None]
    #         jj = np.arange(j0, j1 + 1)[None, :]
    #         dx, dy = ii - x, jj - y
    #         mask_circle = (dx * dx + dy * dy) <= (r_px * r_px)
    #         vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_circle]
    #         if vals.size == 0:
    #             return float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
    #         return float(vals.mean())

    #     def mean_std_annulus(x: float, y: float) -> tuple[float, float]:
    #         ix0, iy0 = int(np.floor(x)), int(np.floor(y))
    #         i0, i1 = max(0, ix0 - R_ring), min(H - 1, ix0 + R_ring)
    #         j0, j1 = max(0, iy0 - R_ring), min(W - 1, iy0 + R_ring)
    #         ii = np.arange(i0, i1 + 1)[:, None]
    #         jj = np.arange(j0, j1 + 1)[None, :]
    #         dx, dy = ii - x, jj - y
    #         r2 = dx * dx + dy * dy
    #         mask_ring = (r2 >= rin * rin) & (r2 <= rout * rout)
    #         vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_ring]
    #         if vals.size == 0:
    #             val = float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
    #             return val, 0.0
    #         return float(vals.mean()), float(vals.std(ddof=0))

    #     # mask of where in real space maxima can occur
    #     H, W = self._image.shape  # x=rows, y=cols
    #     edge_thresh = float(edge_min_dist_px) if edge_min_dist_px is not None else 0.0

    #     DT = None
    #     if input_mask is not None:
    #         m = np.asarray(input_mask).astype(bool)
    #         if m.shape != (H, W):
    #             raise ValueError(f"mask shape {m.shape} must match image shape {(H, W)}")
    #         try:
    #             from scipy.ndimage import distance_transform_edt

    #             DT = distance_transform_edt(m)
    #         except Exception:
    #             DT = None

    #     # find the maxima closest to the origin:
    #     maxima_candidates_x = maxima_candidates[:]['x']
    #     maxima_candidates_y = maxima_candidates[:]['y']

    #     pm_arr = np.array([-1,0,1])
    #     u_norm = np.linalg.norm(u)
    #     v_norm = np.linalg.norm(v)
    #     w_norm = np.linalg.norm(w)
    #     uvw_arr = np.array([np.asarray(u),np.asarray(v),np.asarray(w)])
    #     uv_arr = np.array([np.asarray(u), np.asarray(v)])
    #     uvw_norm = 0.5 * (u_norm + v_norm + w_norm)
    #     self.uv_norm = uvw_norm
    #     self.uv_arr = uvw_arr
    #     self.tolerance_uv = tolerance_uvw
    #     x = maxima_candidates_x
    #     y = maxima_candidates_y

    #     in_bounds = (x >= 0.0) & (x <= H - 1) & (y >= 0.0) & (y <= W - 1)
    #     border_ok = (
    #         (x - edge_thresh >= 0.0)
    #         & (x + edge_thresh <= H - 1)
    #         & (y - edge_thresh >= 0.0)
    #         & (y + edge_thresh <= W - 1)
    #     )
    #     if input_mask is not None:
    #         if DT is not None:
    #             ii = np.clip(np.round(x).astype(int), 0, H - 1)
    #             jj = np.clip(np.round(y).astype(int), 0, W - 1)
    #             mask_ok = DT[ii, jj] >= edge_thresh
    #         else:
    #             m = np.asarray(input_mask).astype(bool)
    #             mask_ok = m[
    #                 np.clip(np.round(x).astype(int), 0, H - 1),
    #                 np.clip(np.round(y).astype(int), 0, W - 1),
    #             ]
    #     else:
    #         mask_ok = np.ones_like(in_bounds, dtype=bool)

    #     int_center = np.empty(x.shape[0], dtype=float)
    #     for i in range(x.shape[0]):
    #         int_center[i] = mean_disk(x[i], y[i])

    #     keep = in_bounds & border_ok & mask_ok
    #     if intensity_min is not None:
    #         keep &= int_center >= float(intensity_min)
    #     if contrast_min is not None:
    #         bg_mean = np.empty(x.shape[0], dtype=float)
    #         for i in range(x.shape[0]):
    #             bg_mean[i], _ = mean_std_annulus(x[i], y[i])
    #         keep &= (int_center - bg_mean) >= float(contrast_min)

    #     if np.any(keep):
    #         maxima_candidates = maxima_candidates[keep]
    #     else:
    #         raise ValueError("Zero maxima candidates kept")

    #     # find the maxima closest to the origin:
    #     maxima_candidates_x = maxima_candidates[:]['x']
    #     maxima_candidates_y = maxima_candidates[:]['y']
    #     maxima_candidates_intensity = maxima_candidates[:]['intensity']

    #     # the unique ids array is an array of the original index, candidacy, a (of a * u), and b (of b * v), and c (of c * w)
    #     unique_ids = np.zeros([6, len(maxima_candidates)])
    #     unique_ids[0,:] = np.arange(0,len(maxima_candidates))
    #     unique_ids[4,:] = -1*np.arange(1,1+len(maxima_candidates))
    #     unique_ids[5,:] -= 1

    #     if plot_atoms:
    #         fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
    #         if ax.images:
    #             ax.images[-1].set_zorder(0)
    #         xs = maxima_candidates_x
    #         ys = maxima_candidates_y
    #         rgb = site_colors(int(self._numbers[0]))
    #         ax.scatter(
    #             ys,
    #             xs,
    #             s=18,
    #             facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
    #             edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
    #             linewidths=0.75,
    #             marker="o",
    #             zorder=25,
    #         )
    #         ax.set_xlim(0, W)
    #         ax.set_ylim(H, 0)

    #     radial_dist = ((maxima_candidates_x - origin[0])**2 + (maxima_candidates_y - origin[1])**2)**(0.5)
    #     origin_candidate_index = np.argmin(radial_dist) # use the first minima, if there are multiple
    #     unique_ids[1,origin_candidate_index] = 1
    #     unique_ids[4,origin_candidate_index] = 0

    #     atoms_found_this_iteration = np.zeros(len(maxima_candidates))
    #     atoms_found_prev_iteration = np.zeros(len(maxima_candidates))
    #     atoms_found_previous_iterations = np.zeros(len(maxima_candidates), dtype = bool)
    #     atoms_found_prev_iteration[origin_candidate_index] = 1
    #     found_atoms_in_prev_iteration = True
    #     iteration_while = 0
    #     def check_dislocations():
    #         if check_for_dislocations:
    #             for atom_index in range(len(maxima_candidates)):
    #                 if unique_ids[1,atom_index] == 1:
    #                     for pm in pm_arr:
    #                         for uvw_index, lat_vec in enumerate(uvw_arr):
    #                             position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
    #                             position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
    #                             radial_dist = ((maxima_candidates_x - position_x)**2 + (maxima_candidates_y - position_y)**2)**(0.5)
    #                             radial_dist[atom_index] = uvw_norm * (tolerance_uvw - 1) * 2 # make sure that self is outside of range
    #                             if (radial_dist < (uvw_norm * (tolerance_uvw - 1))).any():
    #                                 successful_candidate_index = np.argmin(radial_dist)
    #                                 if unique_ids[1, successful_candidate_index] == 2:
    #                                     atoms_found_this_iteration[successful_candidate_index] += 1
    #                                     unique_ids[1, successful_candidate_index] = 3 # for being found in dislocation search
    #                                     unique_ids[2, successful_candidate_index] = unique_ids[2, atom_index] + pm*int(uvw_index == 0) + pm*int(uvw_index == 2)
    #                                     unique_ids[3, successful_candidate_index] = unique_ids[3, atom_index] + pm*int(uvw_index == 1) - w_sign * pm*int(uvw_index == 2)
    #                                     unique_ids[4, successful_candidate_index] = 0
    #             maxima_dislocation_x = maxima_candidates_x[unique_ids[1,:] == 3]
    #             maxima_dislocation_y = maxima_candidates_y[unique_ids[1,:] == 3]
    #             maxima_dislocation_u = unique_ids[2,unique_ids[1,:] == 3]
    #             maxima_dislocation_v = unique_ids[3,unique_ids[1,:] == 3]
    #             maxima_dislocation_intensity = maxima_candidates_intensity[unique_ids[1,:] == 3]
    #             arr = np.vstack(
    #                 (maxima_dislocation_x, maxima_dislocation_y, maxima_dislocation_u, maxima_dislocation_v, maxima_dislocation_intensity)
    #             ).T
    #             return arr

    #     uvw_arr_save = uvw_arr.copy()
    #     for a0 in range(self._num_sites):
    #         unit_shifts = np.array([self._positions_frac[a0, 0], self._positions_frac[a0, 1]])
    #         found_atoms_in_prev_iteration = True
    #         while found_atoms_in_prev_iteration is True:
    #             for atom_index in range(len(maxima_candidates)):
    #                 if not((a0 >0 and unique_ids[5, atom_index] == 0) or a0 == 0):
    #                     continue
    #                 if atoms_found_prev_iteration[atom_index] > 0:
    #                     for pm in pm_arr:
    #                         for uvw_index, lat_vec in enumerate(uvw_arr):
    #                             position_x = pm * lat_vec[0] + maxima_candidates_x[atom_index]
    #                             position_y = pm * lat_vec[1] + maxima_candidates_y[atom_index]
    #                             radial_dist = ((maxima_candidates_x - position_x)**2 + (maxima_candidates_y - position_y)**2)**(0.5)
    #                             radial_dist[atom_index] = uvw_norm * (tolerance_uvw - 1) * 2 # make sure that self is outside of range
    #                             if (radial_dist < (uvw_norm * (tolerance_uvw - 1))).any():
    #                                 successful_candidate_index = np.argmin(radial_dist)
    #                                 if unique_ids[1, successful_candidate_index] == 0:
    #                                     atoms_found_this_iteration[successful_candidate_index] += 1
    #                                     unique_ids[1, successful_candidate_index] = 1
    #                                     unique_ids[2, successful_candidate_index] = unique_ids[2, atom_index] + pm*int(uvw_index == 0)*self._positions_frac[a0, 0] + pm*int(uvw_index == 2)*self._positions_frac[a0, 0]
    #                                     unique_ids[3, successful_candidate_index] = unique_ids[3, atom_index] + pm*int(uvw_index == 1)*self._positions_frac[a0, 1] + w_sign*pm*int(uvw_index == 2)*self._positions_frac[a0, 1]
    #                                     unique_ids[4, successful_candidate_index] = 0
    #                                     unique_ids[5, successful_candidate_index] = a0
    #             # check if any atom was somehow still found twice:
    #             assert np.max(atoms_found_this_iteration) < 2
    #             # check if any found atoms have the same uv index
    #             if check_uv_duplication:
    #                 uv_pairs = unique_ids[1:6,:].T
    #                 unique_pairs, inverse, counts = np.unique(uv_pairs, axis=0, return_inverse=True, return_counts=True)
    #                 duplicate_groups = [np.where(inverse == k)[0] for k, c in enumerate(counts) if c > 1]
    #                 mask_atoms_found = atoms_found_this_iteration.astype(bool)
    #                 if len(duplicate_groups) != 0:
    #                     for duplicate_group in duplicate_groups:
    #                         duplicate_group = np.asarray(duplicate_group)
    #                         duplicate_atoms_index_found_previous_iterations = duplicate_group[atoms_found_previous_iterations[duplicate_group]]
    #                         if duplicate_atoms_index_found_previous_iterations.size > 1:
    #                             if origin_candidate_index not in duplicate_group:
    #                                 raise ValueError("The duplicate atoms finding code is somehow bugged")
    #                             else:
    #                                 kept_index = origin_candidate_index
    #                         elif duplicate_atoms_index_found_previous_iterations.size == 1:
    #                             kept_index = duplicate_atoms_index_found_previous_iterations
    #                         else:
    #                             kept_index = duplicate_group[mask_atoms_found[duplicate_group]][0]
    #                         wipe_indicies = duplicate_group[duplicate_group != kept_index]
    #                         unique_ids[1, wipe_indicies] = 2 # this signals to not accept for this maxima anymore (and flags this as a dulpicate)
    #                         unique_ids[2:4,wipe_indicies] = 0
    #                         unique_ids[5,wipe_indicies] = -1
    #                         unique_ids[4,wipe_indicies] = -1*(wipe_indicies+1)
    #                         atoms_found_previous_iterations[wipe_indicies] = False
    #                         mask_atoms_found[wipe_indicies] = False
    #                         atoms_found_this_iteration[wipe_indicies] = 0
    #             if np.sum(atoms_found_this_iteration) == 0:
    #                 found_atoms_in_prev_iteration = False
    #                 print('stopping search')
                

    #             atoms_found_previous_iterations |= atoms_found_this_iteration.astype(bool)

    #             atoms_found_prev_iteration = atoms_found_this_iteration.copy()
    #             atoms_found_this_iteration = np.zeros(len(maxima_candidates))
    #             iteration_while += 1
    #         if a0 == 0:
    #             unit_shifts = np.array([self._positions_frac[a0+1, 0], self._positions_frac[a0+1, 1]])
    #             atom_arr = check_dislocations()

    #             self.atoms_dislocation = Vector.from_shape(
    #                 shape=(self._num_sites),
    #                 fields=("x", "y", "a", "b", "int_peak"),
    #                 units=("px", "px", "ind", "ind", "counts"),
    #             )
    #             self.atoms_dislocation.set_data(atom_arr, 0)

    #             atoms_found_prev_iteration = atoms_found_previous_iterations.astype(int)
    #             positive_vector = unit_shifts.copy()
    #             negative_vector = unit_shifts.copy()
    #             negative_vector[1] *= -1
    #             if np.abs(np.rad2deg(np.arccos(np.dot(positive_vector, negative_vector)/(np.linalg.norm(positive_vector) * np.linalg.norm(negative_vector))))) > np.deg2rad(90):
    #                 w_ = np.asarray(positive_vector)+np.asarray(negative_vector)
    #                 w_sign = 1
    #             else:
    #                 w_ = np.asarray(positive_vector)-np.asarray(negative_vector)
    #                 w_sign = -1
                
    #             p_v_c = positive_vector @ uv_arr
    #             n_v_c = negative_vector @ uv_arr
    #             w_v_c = w_ @ uv_arr
    #             uvw_arr = np.array([p_v_c, n_v_c, w_v_c])

    #     maxima_accepted_x = maxima_candidates_x[unique_ids[1,:] == 1]
    #     maxima_accepted_y = maxima_candidates_y[unique_ids[1,:] == 1]

    #     maxima_accepted_u = unique_ids[2, unique_ids[1,:] == 1]
    #     maxima_accepted_v = unique_ids[3, unique_ids[1,:] == 1]

    #     maxima_accepted_intensity = maxima_candidates_intensity[unique_ids[1,:] == 1]

    #     self.atoms = Vector.from_shape(
    #         shape=(self._num_sites),
    #         fields=("x", "y", "a", "b", "int_peak"),
    #         units=("px", "px", "ind", "ind", "counts"),
    #     )

    #     if not merge_dislocation or not check_for_dislocations:
    #         arr = np.vstack(
    #             (maxima_accepted_x, maxima_accepted_y, maxima_accepted_u, maxima_accepted_v, maxima_accepted_intensity)
    #         ).T
    #         self.atoms.set_data(arr, 0)

    #     if merge_dislocation and check_for_dislocations:

    #         maxima_merge_x = maxima_candidates_x[np.isin(unique_ids[1, :], [1, 3])]
    #         maxima_merge_y = maxima_candidates_y[np.isin(unique_ids[1, :], [1, 3])]

    #         maxima_merge_u = unique_ids[2, np.isin(unique_ids[1, :], [1, 3])]
    #         maxima_merge_v = unique_ids[3, np.isin(unique_ids[1, :], [1, 3])]

    #         maxima_merge_intensity = maxima_candidates_intensity[np.isin(unique_ids[1, :], [1, 3])]
    #         arr = np.vstack(
    #             (maxima_merge_x, maxima_merge_y, maxima_merge_u, maxima_merge_v, maxima_merge_intensity)
    #         ).T
    #         self.atoms.set_data(arr, 0)

    #     if plot_atoms:
    #         fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
    #         if ax.images:
    #             ax.images[-1].set_zorder(0)
    #         xs = maxima_accepted_x
    #         ys = maxima_accepted_y
    #         rgb = site_colors(int(self._numbers[0]))
    #         ax.scatter(
    #             ys,
    #             xs,
    #             s=18,
    #             facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
    #             edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
    #             linewidths=0.75,
    #             marker="o",
    #             zorder=25,
    #         )
    #         ax.set_xlim(0, W)
    #         ax.set_ylim(H, 0)

    #     return self
