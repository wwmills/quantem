import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter

from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d


class SimData(AutoSerialize):
    """
    Generating atomic resolution data for ML identification of sites.
    """

    _token = object()

    def __init__(
        self,
        u: NDArray[2],
        v: NDArray[2] | None = None,
        H: float | None = None,
        W: float | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use SimData.from_HWU() to instantiate this class.")
        self._u: NDArray = u
        if v is not None:
            self._v: NDArray = v
        if H is None and W is not None:
            H = W
        if W is None and H is not None:
            W = H
        if H is None:
            H = 512
        if W is None:
            W = 512
        self._H = H
        self._W = W

    # --- Constructors ---
    @classmethod
    def from_HWU(
        cls,
        H: int | None = None,
        W: int | None = None,
        u: NDArray[2] | None = None,
        v: NDArray[2] | None = None,
        theta: float | None = None,
        pixel_size_nm: float | None = None,
        field_size_nm: float | None = None,
        pad_fraction: float = 0.1,
    ) -> "SimData":
        """
        u, v (and every other length-like parameter across SimData's methods) are given in
        nm and converted to pixels using pixel_size_nm. Set pixel_size_nm directly, or give
        field_size_nm (the physical width of the H-pixel-wide field of view) and it's derived
        as field_size_nm / H. If neither is given, pixel_size_nm defaults to 1.0 (nm and
        pixels coincide numerically).

        H and W are the requested *output* size. Internally, the simulation runs on a larger
        canvas padded by pad_fraction on every side (to avoid edge artifacts from the lattice
        placement, PSF convolution, etc.), and get_result()/show_result() crop back down to
        exactly (H, W) at the end. Set pad_fraction=0 to disable padding.
        """
        if H is None and W is not None:
            H = W
        if W is None and H is not None:
            W = H
        if H is None:
            H = 512
        if W is None:
            W = 512
        if pixel_size_nm is None:
            pixel_size_nm = field_size_nm / H if field_size_nm is not None else 1.0
        u = ensure_valid_array(u, ndim=1) / pixel_size_nm
        if v is not None:
            v = ensure_valid_array(v, ndim=1) / pixel_size_nm
        if v is None:
            if theta is None:
                theta = np.pi / 3
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            v = (u @ rotation_matrix).T
        margin_H = int(round(pad_fraction * H))
        margin_W = int(round(pad_fraction * W))
        obj = cls(H=H + 2 * margin_H, W=W + 2 * margin_W, u=u, v=v, _token=cls._token)
        obj._pixel_size_nm = pixel_size_nm
        obj._pad_fraction = pad_fraction
        obj._H_display = H
        obj._W_display = W
        return obj

    @property
    def pixel_size_nm(self) -> float:
        return getattr(self, "_pixel_size_nm", 1.0)

    @property
    def pad_fraction(self) -> float:
        return getattr(self, "_pad_fraction", 0.0)

    def _crop_to_display(self, arr: NDArray) -> NDArray:
        H_disp = getattr(self, "_H_display", arr.shape[0])
        W_disp = getattr(self, "_W_display", arr.shape[1])
        H_cur, W_cur = arr.shape
        dh, dw = (H_cur - H_disp) // 2, (W_cur - W_disp) // 2
        if dh < 0 or dw < 0:
            raise ValueError(
                f"Current image size {arr.shape} is smaller than the requested display size "
                f"({H_disp}, {W_disp}) -- reduce prior cropping (e.g. add_noise's win) or "
                f"increase pad_fraction."
            )
        return arr[dh : dh + H_disp, dw : dw + W_disp]

    # --- Properties ---
    @property
    def atom_coordinates(self) -> Union[list[NDArray], NDArray]:
        return self._atom_coordinates

    @property
    def images(self) -> NDArray:
        if hasattr(self, "_images"):
            return self._images
        else:
            raise ValueError("Images not generated yet")

    def gauss_2D_r(
        self,
        rr,
        s,
        A,
        B,
    ):
        gaussian_2d = A * np.exp(-((rr) ** 2 / (2 * s**2))) + B
        return gaussian_2d

    def generate_mask_2D(self, array, p_s=60.0, sparsity=0.20):
        arrayShape = array.shape
        x = np.fft.fftfreq(arrayShape[0])
        y = np.fft.fftfreq(arrayShape[1])

        X, Y = np.meshgrid(y, x, indexing="ij")
        kr = np.sqrt(X**2 + Y**2)

        f_in = np.ones(arrayShape).flatten()

        A = np.exp(-p_s * kr)
        phase = np.random.randn(arrayShape[0], arrayShape[1])
        F = A * np.exp(2 * np.pi * 1j * phase)
        f = np.fft.ifftn(F)
        f_shape = np.absolute(f)

        f_shape = np.argsort(f_shape, axis=None)
        f_shape = f_shape.flatten()
        N_zero = int(np.round((array.size * (1 - sparsity))))
        f_shape[N_zero:] = f_shape[N_zero]
        f_in[f_shape] = 0

        f_in = f_in / np.amax(f_in)
        np.copyto(array, f_in.reshape(arrayShape))

    def generate_dataset_2D(self, array, p_in=30.0, p_s=60.0, sparsity=0.20):
        arrayShape = array.shape
        x = np.fft.fftfreq(arrayShape[0])
        y = np.fft.fftfreq(arrayShape[1])

        X, Y = np.meshgrid(y, x, indexing="ij")
        kr = np.sqrt(X**2 + Y**2)

        A = np.exp(-p_in * kr)
        phase = np.random.randn(arrayShape[0], arrayShape[1])
        F = A * np.exp(2 * np.pi * 1j * phase)
        f = np.fft.ifftn(F)
        f_in = np.absolute(f).copy().flatten()

        A = np.exp(-p_s * kr)
        phase = np.random.randn(arrayShape[0], arrayShape[1])
        F = A * np.exp(2 * np.pi * 1j * phase)
        f = np.fft.ifftn(F)
        f_shape = np.absolute(f)

        f_shape = np.argsort(f_shape, axis=None)
        f_shape = f_shape.flatten()

        N_zero = int(np.round((array.size * (1 - sparsity))))
        f_shape[N_zero:] = f_shape[N_zero]
        f_in[f_shape] = 0

        f_in = f_in / np.amax(f_in)
        np.copyto(array, f_in.reshape(arrayShape))

    def _lattice_candidate_positions(self, lat, positions_frac, H, W, mask, edge_min_dist_px):
        """
        Self-contained stand-in for Lattice.from_data + define_lattice + add_atoms, trimmed to
        exactly what generate_coordinates_hex needs -- no refine_lattice, no plotting, no
        intensity/contrast thresholds (add_atoms computes those for an "int_peak" column this
        class never reads). Kept in-line, rather than calling the real Lattice, so this file
        has no dependency on Lattice/Vector, which evolve independently elsewhere in quantem
        and aren't guaranteed to keep the exact method signatures used here.

        lat is the (3,2) [origin, u, v] basis; positions_frac is the (S,2) fractional site
        offsets within the unit cell. Tiles every integer lattice translation that lands
        inside the (H, W) image for each site, keeps candidates that are in-bounds, at least
        edge_min_dist_px from the image border, and at least edge_min_dist_px inside `mask`
        (by distance transform). Returns a list of (2, N) arrays (x-row, y-row), one per site.
        """
        r0, u, v = lat[0], lat[1], lat[2]
        A = np.column_stack((u, v))
        corners = np.array([[0.0, 0.0], [float(H), 0.0], [0.0, float(W)], [float(H), float(W)]])
        ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]
        a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
        b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

        edge_thresh = float(edge_min_dist_px)
        dt = distance_transform_edt(np.asarray(mask).astype(bool))

        coords = []
        for da, db in positions_frac:
            aa, bb = np.meshgrid(
                np.arange(a_min - 1 + da, a_max + 1 + da),
                np.arange(b_min - 1 + db, b_max + 1 + db),
                indexing="ij",
            )
            basis = np.vstack((np.ones(aa.size), aa.ravel(), bb.ravel())).T
            xy = basis @ lat

            x, y = xy[:, 0], xy[:, 1]
            in_bounds = (x >= 0.0) & (x <= H - 1) & (y >= 0.0) & (y <= W - 1)
            border_ok = (
                (x - edge_thresh >= 0.0)
                & (x + edge_thresh <= H - 1)
                & (y - edge_thresh >= 0.0)
                & (y + edge_thresh <= W - 1)
            )
            ii = np.clip(np.round(x).astype(int), 0, H - 1)
            jj = np.clip(np.round(y).astype(int), 0, W - 1)
            mask_ok = dt[ii, jj] >= edge_thresh

            keep = in_bounds & border_ok & mask_ok
            coords.append(np.array([x[keep], y[keep]]))
        return coords

    def _lattice_edge_atoms(self, coords, u, v, tolerance, min_neighbors):
        """
        Self-contained stand-in for Lattice.find_neighbors_in_tolerance +
        find_atoms_with_too_few_neighbors (see _lattice_candidate_positions for why this is
        inlined rather than calling Lattice). Flags atoms with too few neighbors *within
        tolerance of site 0's atoms specifically*, not their own site's -- site 0 is the
        dense reference sublattice, so proximity to it is a good edge/mask-boundary proxy for
        every site, matching the original Lattice behavior. Returns a list of boolean arrays
        (one per site), True where that atom has too few neighbors.
        """
        if tolerance is None:
            tolerance = np.mean(np.linalg.norm(np.stack([u, v]), axis=1)) * 1.1

        a_x0, a_y0 = coords[0][0], coords[0][1]
        edge_atoms = []
        for site_index in range(len(coords)):
            a_x_n, a_y_n = coords[site_index][0], coords[site_index][1]
            count_neighbors = np.empty(a_x_n.shape[0])
            for atom_index in range(a_x_n.shape[0]):
                radial_dist = np.sqrt(
                    (a_x0 - a_x_n[atom_index]) ** 2 + (a_y0 - a_y_n[atom_index]) ** 2
                )
                if site_index == 0:
                    radial_dist[atom_index] = tolerance * 2  # exclude self
                count_neighbors[atom_index] = np.sum(radial_dist < tolerance)
            edge_atoms.append(count_neighbors < min_neighbors)
        return edge_atoms

    # a way to call the lattice class within this class
    def generate_coordinates_hex(
        self,
        H=None,
        W=None,
        u=None,
        v=None,
        theta_v=np.pi / 3,
        num_sites=3,
        p_s=25.0,
        sparsity=0.9,
        plot_atoms=False,
        min_neighbors=3,
        tolerance=None,
    ):
        """
        u and v are in nm (converted via self.pixel_size_nm). p_s is in units of the lattice
        spacing |u| (not an absolute length) -- e.g. p_s=25 means void patches are roughly 25
        atomic spacings across, regardless of pixel_size_nm or the absolute field of view.
        That's what makes a single default sensible at any scale: an absolute pixel or nm
        count would need to be manually rescaled every time the resolution or physical FOV
        changes, whereas "N lattice spacings" means the same thing everywhere. tolerance is
        left in raw pixels (it's an internal neighbor-count threshold, defaults to an
        automatic 1.1x the lattice spacing if left as None). H, W, u, and v default to
        whatever was set on this instance (e.g. via SimData.from_HWU(...), or a prior call to
        this method) -- pass them explicitly here to override that.

        The lattice generation itself (tiling positions_frac across the image, filtering by
        mask/edge distance, flagging sparse-neighbor "edge" atoms) is self-contained in this
        file (_lattice_candidate_positions/_lattice_edge_atoms) rather than calling
        quantem.imaging.Lattice -- intentional, so this file has no dependency on Lattice or
        Vector, which evolve independently elsewhere in quantem.
        """
        if H is None and hasattr(self, "_H"):
            H = self._H
        if W is None and hasattr(self, "_W"):
            W = self._W
        if H is None and W is not None:
            H = W
        if W is None and H is not None:
            W = H
        if H is None:
            H = 512
        if W is None:
            W = 512
        self.num_sites = num_sites
        pixel_size_nm = self.pixel_size_nm

        theta = np.random.rand() * np.pi * 2
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        u_overridden = u is not None
        if not u_overridden and hasattr(self, "_u"):
            u = self._u
        else:
            u = np.array([16.0, 0.0]) if u is None else np.asarray(u, dtype=float)
            u = u / pixel_size_nm
        # v only inherits the stored self._v when u also isn't being overridden -- otherwise
        # a new u paired with a stale v would no longer be a consistent lattice basis.
        if v is None and not u_overridden and hasattr(self, "_v"):
            v = self._v
        elif v is not None:
            v = np.asarray(v, dtype=float) / pixel_size_nm

        mask_ = np.zeros([H, W])
        self.generate_mask_2D(mask_, p_s=p_s * np.linalg.norm(u), sparsity=sparsity)

        u = u @ rotation_matrix.T
        if v is not None:
            v = v @ rotation_matrix.T
        else:
            theta = theta_v
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            v = u @ rotation_matrix.T
        origin = np.array([0, 0])

        self._u = u
        self._v = v

        lat = np.vstack((origin, u, v)).astype(float)
        if num_sites > 3:
            raise ValueError("The number of sites automation can only do A, B, and C sites")
        positions_frac = [np.array([0, 0]), np.array([1 / 3, 1 / 3]), np.array([-1 / 3, -1 / 3])]
        positions_frac = positions_frac[:num_sites]
        coords = self._lattice_candidate_positions(
            lat,
            positions_frac,
            H,
            W,
            mask=mask_,
            edge_min_dist_px=1,
        )

        if plot_atoms:
            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            label_list = ["a", "b", "c"]
            color_list_o = ["red", "blue", "green"]
            label_list = label_list[:num_sites]
            for site_index in range(num_sites):
                a_x = coords[site_index][0]
                a_y = coords[site_index][1]
                ax[0].scatter(
                    a_y,
                    -a_x,
                    color=color_list_o[site_index],
                    s=10,
                    alpha=0.5,
                    label=label_list[site_index],
                )
            ax[0].legend()
            ax[0].set_title("Before filtering")
            ax[0].set_box_aspect(1)

        # --- Apply tolerance ---
        # note edge atoms, so that we can make them blurrier and more shifted later
        self.edge_atoms = self._lattice_edge_atoms(coords, u, v, tolerance, min_neighbors)
        if plot_atoms:
            for site_index in range(num_sites):
                a_x = coords[site_index][0]
                a_y = coords[site_index][1]
                ax[1].scatter(
                    a_y,
                    -a_x,
                    color=color_list_o[site_index],
                    s=10,
                    alpha=0.5,
                    label=label_list[site_index],
                )
            ax[1].legend()
            ax[1].set_title("After filtering")
            ax[1].set_box_aspect(1)

            for site_index in range(num_sites):
                a_x = coords[site_index][0]
                a_y = coords[site_index][1]
                ax[2].scatter(
                    a_y,
                    -a_x,
                    color=color_list_o[site_index],
                    s=10,
                    alpha=0.5,
                    label=label_list[site_index],
                )
            ax[2].set_title("Remaining after filtering")
            ax[2].set_box_aspect(1)

            for site_index in range(num_sites):
                a_x = coords[site_index][0][self.edge_atoms[site_index]]
                a_y = coords[site_index][1][self.edge_atoms[site_index]]
                ax[2].scatter(a_y, -a_x, color="black", s=10, alpha=0.5, label="edge atom")
            ax[2].legend()
            plt.tight_layout()
            plt.show()

        self._atom_coordinates = coords
        return self

    def _binary_density(self, size, density):
        if not (0.0 <= density <= 1.0):
            raise ValueError("Density must be between 0.0 and 1.0")

        num_ones = int(size * density)
        if num_ones > size:
            num_ones = size

        arr = np.zeros(size, dtype=int)
        arr[:num_ones] = 1
        np.random.shuffle(arr)

        return arr

    def _categorical_density(
        self,
        size,
        densities,
    ):
        """
        Assigns each of `size` atoms to at most one of len(densities) mutually-exclusive
        categories (labels 1, 2, ... in the order given), each covering the given fraction
        of `size`; the remainder (unassigned) are label 0. Shuffled randomly.
        """
        if sum(densities) > 1.0:
            raise ValueError("Densities must sum to <= 1.0")
        labels = np.zeros(size, dtype=int)
        idx = 0
        for cat, density in enumerate(densities, start=1):
            count = min(int(size * density), size - idx)
            labels[idx : idx + count] = cat
            idx += count
        np.random.shuffle(labels)
        return labels

    def _place_atoms_window(
        self,
        H,
        W,
        num_sites,
        window_width,
        sigma,
        u_norm,
        disorder_multiplier,
        edge_multiplier_in,
        site_multipliers,
        z_stack,
        vacancy_labels,
        substitution_labels,
        sub_multipliers,
    ):
        """
        Renders each atom into its own small window (side length window_width, sub-pixel
        centered per atom), clipped to a circular taper. O(num_atoms) Python-level loop
        iterations -- see _place_atoms_bilinear for a fully vectorized alternative.
        vacancy_labels/substitution_labels are precomputed per-atom category arrays (see
        add_sites_to_image), shared with whichever placement method is used so switching
        between 'window'/'bilinear' doesn't change the random defect realization. Also
        returns the actual rendered (post-jitter) coordinates per site, for ground truth.
        sigma is indexed per-site (sigma[a0]), so each site can have its own column width.
        """
        im__ = np.zeros([H, W])
        ww = window_width
        half_width = int(ww // 2)

        x = np.arange(0, ww) - ww // 2
        y = np.arange(0, ww) - ww // 2
        xx, yy = np.meshgrid(x, y, indexing="ij")
        circle_mask = (xx**2 + yy**2) <= (ww / 2.0) ** 2

        rendered_coordinates = []
        edge_multiplier = 1
        for a0 in range(num_sites):
            sg = sigma[a0]
            a_x_a0 = self._atom_coordinates[a0][0]
            a_y_a0 = self._atom_coordinates[a0][1]
            edge_atoms_a0 = self.edge_atoms[a0]
            vacancy_arr = vacancy_labels[a0]
            substitution_arr = substitution_labels[a0]
            rendered_x = np.empty_like(a_x_a0)
            rendered_y = np.empty_like(a_y_a0)
            for atom_index in range(a_x_a0.shape[0]):
                if edge_atoms_a0[atom_index]:
                    edge_multiplier = edge_multiplier_in
                else:
                    edge_multiplier = 1
                a_x = (
                    a_x_a0[atom_index]
                    + (np.random.rand(1)[0] - 0.5) * disorder_multiplier * u_norm * edge_multiplier
                )
                a_y = (
                    a_y_a0[atom_index]
                    + (np.random.rand(1)[0] - 0.5) * disorder_multiplier * u_norm * edge_multiplier
                )
                rendered_x[atom_index] = a_x
                rendered_y[atom_index] = a_y

                a_x_residual = a_x - np.floor(a_x)
                a_x_floor = np.floor(a_x).astype(int)
                a_y_residual = a_y - np.floor(a_y)
                a_y_floor = np.floor(a_y).astype(int)

                xc = a_x_residual
                yc = a_y_residual

                rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)

                sub_x0 = max(0, -(a_x_floor - half_width))
                sub_x1 = ww - max(0, (a_x_floor + (ww - half_width)) - H)
                sub_y0 = max(0, -(a_y_floor - half_width))
                sub_y1 = ww - max(0, (a_y_floor + (ww - half_width)) - W)

                x1 = min(H, a_x_floor + (ww - half_width))
                y1 = min(W, a_y_floor + (ww - half_width))
                x0 = max(0, a_x_floor - half_width)
                y0 = max(0, a_y_floor - half_width)

                rr_v = rr[sub_x0:sub_x1, sub_y0:sub_y1]
                mask_v = circle_mask[sub_x0:sub_x1, sub_y0:sub_y1]
                g = (
                    self.gauss_2D_r(rr_v, sg * edge_multiplier, 1, 0)
                    / 2
                    * site_multipliers[a0]
                    * 0.5
                    / edge_multiplier
                )

                g *= z_stack[a0]

                if vacancy_arr[atom_index] == 1:
                    g *= 0
                elif vacancy_arr[atom_index] == 2:
                    g *= (z_stack[a0] - 1) / z_stack[a0]

                if substitution_arr[atom_index] == 1:
                    g /= site_multipliers[a0]
                    g *= sub_multipliers[a0]

                g[~mask_v] = 0

                im__[x0:x1, y0:y1] += g
            rendered_coordinates.append(np.stack([rendered_x, rendered_y], axis=1))

        return im__, rendered_coordinates

    def _place_atoms_bilinear(
        self,
        H,
        W,
        num_sites,
        sigma,
        u_norm,
        disorder_multiplier,
        site_multipliers,
        z_stack,
        vacancy_labels,
        substitution_labels,
        sub_multipliers,
    ):
        """
        Vectorized alternative to _place_atoms_window: splats each atom's intensity onto its
        4 nearest pixels by bilinear weight (no explicit per-atom window or Python loop over
        atoms), then applies a Gaussian blur -- the same approach used in
        synthetic_wse2.ipynb. Much faster for large atom counts, and immune to the
        window/sigma undersampling pitfalls of the 'window' method, at the cost of not
        supporting per-atom edge-width widening (edge_multiplier_in has no effect here --
        every atom at a given site uses that site's sigma). Each site is splatted into its
        own accumulator and blurred with its own sigma[a0] before summing -- when sigma is
        the same for every site this is mathematically identical to blurring one combined
        accumulator once (Gaussian convolution is linear), just num_sites blur calls instead
        of one, so the common case pays no accuracy cost, only a few extra (cheap) blurs.
        vacancy_labels/substitution_labels: see _place_atoms_window. Also returns the actual
        rendered (post-jitter) coordinates per site, for ground truth.
        """
        im__ = np.zeros([H, W])
        rendered_coordinates = []
        for a0 in range(num_sites):
            a_x_a0 = self._atom_coordinates[a0][0]
            a_y_a0 = self._atom_coordinates[a0][1]
            n = a_x_a0.shape[0]

            a_x = a_x_a0 + (np.random.rand(n) - 0.5) * disorder_multiplier * u_norm
            a_y = a_y_a0 + (np.random.rand(n) - 0.5) * disorder_multiplier * u_norm
            rendered_coordinates.append(np.stack([a_x, a_y], axis=1))

            weight = np.full(n, 0.25 * site_multipliers[a0] * z_stack[a0])

            vacancy_arr = vacancy_labels[a0]
            weight[vacancy_arr == 1] = 0.0
            weight[vacancy_arr == 2] *= (z_stack[a0] - 1) / z_stack[a0]

            substitution_arr = substitution_labels[a0]
            sub_mask = substitution_arr == 1
            weight[sub_mask] = weight[sub_mask] / site_multipliers[a0] * sub_multipliers[a0]

            acc = np.zeros([H, W])
            x0f = np.floor(a_x).astype(int)
            y0f = np.floor(a_y).astype(int)
            fx = a_x - x0f
            fy = a_y - y0f
            for dx, dy, frac in [
                (0, 0, (1 - fx) * (1 - fy)),
                (1, 0, fx * (1 - fy)),
                (0, 1, (1 - fx) * fy),
                (1, 1, fx * fy),
            ]:
                xi, yi = x0f + dx, y0f + dy
                valid = (xi >= 0) & (xi < H) & (yi >= 0) & (yi < W)
                np.add.at(acc, (xi[valid], yi[valid]), (weight * frac)[valid])

            sg = sigma[a0]
            im__ += gaussian_filter(acc, sigma=sg) * (2 * np.pi * sg**2)

        return im__, rendered_coordinates

    def add_sites_to_image(
        self,
        H=None,
        W=None,
        window_width=1,
        sigma=2.0,
        disorder_multiplier=0.1,
        show_ideal=False,
        show_result=False,
        return_result=False,
        sg_c=1.0,
        gamma=1.7,
        cauchy=0.5,
        use_cauchy=False,
        use_bessel=False,
        z_el=["Ca", "Ti", "O"],
        z_stack=np.array([1, 1, 1]),
        edge_multiplier_in=1.3,
        substitution_index=np.array([0, 1, 2]),
        substitution_density=np.array([0, 0, 0]),
        substitution_species=["V", "Ti", "O"],
        vacancy_index=np.array([0, 1, 2]),
        vacancy_density=np.array([0, 0, 0]),
        partial_vacancy_density=np.array([0, 0, 0]),
        gaussian_placement="window",
    ):
        """
        gaussian_placement selects how atoms are rendered: 'window' (default) loops over
        atoms and renders each into its own small circularly-clipped window -- supports
        edge_multiplier_in's per-atom width widening, but is O(num_atoms) in Python and
        sensitive to window_width/sigma being sized correctly (see _place_atoms_window).
        'bilinear' vectorizes over all atoms via bilinear splatting + one global Gaussian
        blur (the approach used in synthetic_wse2.ipynb) -- much faster for large atom
        counts and immune to window-sizing pitfalls, but ignores edge_multiplier_in (see
        _place_atoms_bilinear).

        window_width, sigma, cauchy, and sg_c are in nm (converted via self.pixel_size_nm).
        sigma is the base Gaussian width of an atomic column; edge atoms get sigma *
        edge_multiplier_in. cauchy/sg_c are the Cauchy/Bessel point-spread-function widths.
        Unlike use_bessel (opt-in, default off), the Cauchy PSF is applied by default --
        set use_cauchy=False to exclude it (e.g. use_bessel alone, or no PSF blur at all).

        vacancy_density removes all z_stack[a0] atoms at a site (e.g. both Se in a
        z_stack=2 chalcogen column -- a "di-vacancy"). partial_vacancy_density instead
        removes exactly one atom from the stack, scaling that column's intensity by
        (z_stack[a0]-1)/z_stack[a0] rather than to zero (e.g. a single-Se "mono-vacancy"
        in an otherwise-intact Se2 column). The two densities are mutually exclusive per
        atom and are drawn from the same pool, so they should sum to <= 1.

        sigma may be a single value (shared by every site) or an array of length num_sites
        for a per-site atomic-column width (e.g. a wider column for a heavier/thicker site).
        """
        pixel_size_nm = self.pixel_size_nm

        if hasattr(self, "_H"):
            H = self._H
        if hasattr(self, "_W"):
            W = self._W
        if H is None and W is not None:
            H = W
        if W is None and H is not None:
            W = H

        if not hasattr(self, "_atom_coordinates"):
            raise ValueError(
                "No atom coordinates yet -- call generate_coordinates_hex() before "
                "add_sites_to_image()."
            )
        num_sites = self.num_sites

        window_width = max(2, int(round(window_width / pixel_size_nm)))
        sigma = np.broadcast_to(
            np.asarray(sigma, dtype=float) / pixel_size_nm, (num_sites,)
        ).copy()
        cauchy = cauchy / pixel_size_nm
        sg_c = sg_c / pixel_size_nm
        for name, val, min_px in (
            [
                ("sigma", np.min(sigma), 0.5),
                ("window_width", window_width, 8 * np.max(sigma)),
            ]
            + ([("cauchy", cauchy, 0.5)] if use_cauchy else [])
            + ([("sg_c", sg_c, 0.5)] if use_bessel else [])
        ):
            if val < min_px:
                warnings.warn(
                    f"{name}={val:.3g}px at pixel_size_nm={pixel_size_nm:.3g} is narrower than "
                    f"~{min_px:.3g}px -- a Gaussian/Cauchy/window this narrow relative to the "
                    f"pixel grid can't be sampled correctly and will give numerically wrong "
                    f"(not just 'sharper') results. Use a finer pixel_size_nm or a larger "
                    f"value in nm.",
                    stacklevel=2,
                )

        # disorder in position
        u_norm = np.linalg.norm(self._u)

        pt = self.periodic_table()
        zNums = np.asarray(list(map(lambda x: pt[x.lower()], z_el)))
        site_multipliers = zNums**gamma
        parent_mean = np.mean(site_multipliers)
        site_multipliers /= parent_mean  # normalize

        zNums = np.asarray(list(map(lambda x: pt[x.lower()], substitution_species)))
        sub_multipliers = zNums**gamma
        sub_multipliers /= parent_mean  # same normalization

        # vacancy/substitution assignment, computed ONCE and shared between placement modes
        # (previously each placement method drew its own, so switching 'window'<->'bilinear'
        # silently changed the random defect realization) -- also kept around afterward so
        # get_ground_truth() can report what actually happened to each atom.
        vacancy_labels = []
        substitution_labels = []
        for a0 in range(num_sites):
            n = self._atom_coordinates[a0][0].shape[0]
            if a0 in vacancy_index:
                vacancy_labels.append(
                    self._categorical_density(
                        n, [vacancy_density[a0], partial_vacancy_density[a0]]
                    )
                )
            else:
                vacancy_labels.append(np.zeros(n, dtype=int))
            if a0 in substitution_index:
                substitution_labels.append(self._binary_density(n, substitution_density[a0]))
            else:
                substitution_labels.append(np.zeros(n, dtype=int))

        if gaussian_placement == "window":
            im__, rendered_coordinates = self._place_atoms_window(
                H,
                W,
                num_sites,
                window_width,
                sigma,
                u_norm,
                disorder_multiplier,
                edge_multiplier_in,
                site_multipliers,
                z_stack,
                vacancy_labels,
                substitution_labels,
                sub_multipliers,
            )
        elif gaussian_placement == "bilinear":
            im__, rendered_coordinates = self._place_atoms_bilinear(
                H,
                W,
                num_sites,
                sigma,
                u_norm,
                disorder_multiplier,
                site_multipliers,
                z_stack,
                vacancy_labels,
                substitution_labels,
                sub_multipliers,
            )
        else:
            raise ValueError(
                f"gaussian_placement must be 'window' or 'bilinear', got {gaussian_placement!r}"
            )

        self._vacancy_labels = vacancy_labels
        self._substitution_labels = substitution_labels
        self._rendered_coordinates = rendered_coordinates
        self._z_el = list(z_el)
        self._substitution_species = list(substitution_species)

        if show_ideal:
            plt.figure()
            plt.imshow(im__, cmap="gray")
            plt.title("Just Gaussians")
            plt.axis("off")

        # build grid that is the size of the whole image
        x_ = np.arange(0, H) - H // 2
        y_ = np.arange(0, W) - W // 2
        xx_, yy_ = np.meshgrid(x_, y_, indexing="ij")
        rr_ = np.sqrt(xx_**2 + yy_**2)

        # convolution step
        fft_im__ = np.fft.fft2(im__)
        fft_product = fft_im__

        if use_bessel:
            B = sp.j0(rr_ / sg_c) ** 2
            B /= np.abs(B).sum()
            fft_bessel = np.fft.fft2(np.fft.ifftshift(B))
            fft_product = fft_product * fft_bessel

        if use_cauchy:
            C = 1 / (np.pi * cauchy * (1 + (rr_ / cauchy) ** 2))
            C /= C.sum()
            fft_cauchy = np.fft.fft2(np.fft.ifftshift(C))
            fft_product = fft_product * fft_cauchy

        # back to real space
        result = np.real(np.fft.ifft2(fft_product))

        if show_result:
            plt.figure()
            plt.imshow(np.real(result), cmap="gray")
            plt.title("Convolved with Cauchy")
            plt.axis("off")

        self._images = result
        if return_result:
            return result
        return self

    def _gaussian_blur(self, array, sigma, fft_threshold=20.0):
        """
        Gaussian blur, dispatching to FFT-based convolution for large sigma. scipy.ndimage.
        gaussian_filter's cost scales linearly with sigma (kernel truncation radius); FFT
        convolution costs a roughly sigma-independent O(N log N) instead. Empirically the
        crossover is around sigma~20-25px on a ~1200px image, so this picks whichever is
        faster rather than always paying scipy's cost for large blurs (e.g. contamination
        textures at a fine pixel_size_nm, where the physical sigma maps to 100+ px).
        """
        if sigma <= fft_threshold:
            return gaussian_filter(array, sigma=sigma)
        H, W = array.shape
        x = np.arange(H) - H // 2
        y = np.arange(W) - W // 2
        xx, yy = np.meshgrid(x, y, indexing="ij")
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return np.real(np.fft.ifft2(np.fft.fft2(array) * np.fft.fft2(np.fft.ifftshift(kernel))))

    def _multi_octave_field(self, shape, base_sigma_px, n_octaves=5, octave_decay=0.55):
        """Fractal-like noise: sum of Gaussian-blurred white noise at halving scales, normalized to [0,1]."""
        field = np.zeros(shape)
        amp, total_amp = 1.0, 0.0
        for octave in range(n_octaves):
            oct_sigma = max(base_sigma_px / (2**octave), 1)
            field += amp * self._gaussian_blur(np.random.randn(*shape), oct_sigma)
            total_amp += amp
            amp *= octave_decay
        field /= total_amp
        return (field - field.min()) / (field.max() - field.min() + 1e-9)

    def _nested_shells(self, field, coverage, n_layers=12, edge_spread_px=25):
        """
        Carves n_layers nested threshold shells out of `field` (a [0,1] scalar field),
        blurring each shell by progressively more so the result tapers smoothly from a
        dense core (highest-field regions) to a diffuse halo, instead of a hard-edged mask.
        Returns a [0,1]-normalized density, highest where `field` is highest.
        """
        # all n_layers quantile thresholds in one sort instead of one np.quantile call per layer
        fracs = coverage * (np.arange(n_layers) + 1) / n_layers
        thresholds = np.quantile(field, 1 - fracs)
        out = np.zeros(field.shape)
        for k in range(n_layers):
            # outer layers (larger k) cover more area and get blurred more -> diffuse halo
            mask = (field > thresholds[k]).astype(float)
            edge_sigma = 2 + edge_spread_px * (k / max(n_layers - 1, 1))
            shell = self._gaussian_blur(mask, edge_sigma)
            weight = 1.0 - k / n_layers  # inner shells denser/brighter than outer halo
            out += shell * weight
        return out / out.max()

    def add_carbon_contamination(
        self,
        coverage=0.7,
        intensity=0.6,
        base_sigma_nm=80.0,
        n_octaves=5,
        n_layers=12,
        edge_spread_nm=20.0,
        octave_decay=0.55,
    ):
        """
        Amorphous carbon contamination: a small dense core surrounded by a progressively
        softer, more diffuse halo, built by carving nested threshold shells out of a
        multi-octave (fractal-like) noise field and blurring each shell by a different
        amount. Ported from synthetic_wse2.ipynb. Additive overlay onto self._images.
        Call after add_sites_to_image and before add_noise. base_sigma_nm and edge_spread_nm
        are in nm (converted via self.pixel_size_nm).
        """
        if not hasattr(self, "_images"):
            raise ValueError("Call add_sites_to_image before add_carbon_contamination")
        H, W = self._images.shape
        pixel_size_nm = self.pixel_size_nm

        field = self._multi_octave_field(
            (H, W), base_sigma_nm / pixel_size_nm, n_octaves, octave_decay
        )
        carbon_image = self._nested_shells(
            field, coverage, n_layers, edge_spread_nm / pixel_size_nm
        )

        self._images = self._images + carbon_image * intensity
        return self

    def _sharp_mask(self, field, coverage, edge_sigma_px=3.0):
        """
        Single threshold on `field` at the given coverage fraction, blurred by ONE small,
        fixed sigma. Unlike _nested_shells (which blends many graded thresholds of the same
        field into a soft diffuse halo, with the transition width set by how smoothly
        `field` itself varies), the transition width here is set entirely by edge_sigma_px,
        independent of the field's own correlation length -- so a large base_sigma (big
        blob shapes) doesn't force a slow-looking edge. Returns a [0,1] density.
        """
        threshold = np.quantile(field, 1 - coverage)
        mask = (field > threshold).astype(float)
        envelope = self._gaussian_blur(mask, edge_sigma_px)
        return envelope / envelope.max()

    def add_textured_contamination(
        self,
        coverage=0.35,
        intensity=0.2,
        base_sigma_nm=1.5,
        edge_sigma_nm=0.05,
        n_octaves=5,
        octave_decay=0.55,
        detail_sigma_nm=0.3,
        detail_octaves=3,
        detail_decay=0.5,
        detail_strength=0.6,
    ):
        """
        Amorphous contamination with visible internal structure, rather than a smooth
        gradient: a low-frequency envelope with a genuinely sharp outer edge (one threshold
        on a multi-octave field at the given coverage, blurred by a single small
        edge_sigma_nm -- see _sharp_mask) multiplicatively gates a second, finer multi-octave
        noise field (detail_sigma_nm/detail_octaves/detail_decay) -- so the finer texture
        only shows up inside the envelope, reading as structure within the contamination
        rather than either a flat blob or noise everywhere. detail_strength in [0,1] sets
        how strongly that inner texture modulates brightness (0 = flat envelope, no internal
        structure; 1 = full contrast, down to zero in spots). Keep detail_octaves/
        detail_decay low so the inner texture forms its own soft sub-blobs rather than
        per-pixel grain.

        This method's envelope is intentionally different from add_carbon_contamination's
        nested shells, which blend many graded thresholds into a soft diffuse halo -- use
        that method instead if you want the softer look. Note base_sigma_nm only sets the
        *shape/size* of the contaminated regions here, not the edge sharpness (that's
        edge_sigma_nm) -- so unlike add_carbon_contamination, base_sigma_nm can safely be
        larger than the field of view without smearing the boundary.

        Additive overlay onto self._images. Call after add_sites_to_image and before
        add_noise. All *_nm parameters are in nm (converted via self.pixel_size_nm).
        """
        if not hasattr(self, "_images"):
            raise ValueError("Call add_sites_to_image before add_textured_contamination")
        H, W = self._images.shape
        pixel_size_nm = self.pixel_size_nm

        envelope_field = self._multi_octave_field(
            (H, W), base_sigma_nm / pixel_size_nm, n_octaves, octave_decay
        )
        envelope = self._sharp_mask(envelope_field, coverage, edge_sigma_nm / pixel_size_nm)

        detail_field = self._multi_octave_field(
            (H, W), detail_sigma_nm / pixel_size_nm, detail_octaves, detail_decay
        )
        detail = (1 - detail_strength) + 2 * detail_strength * detail_field

        texture = envelope * detail
        texture = texture / texture.max() * intensity

        self._images = self._images + texture
        return self

    def add_adatoms(
        self,
        num_adatoms=20,
        correlation=0.0,
        corr_p_in=25.0,
        corr_p_s=25.0,
        window_width=20.0,
        sigma=1.5,
        amplitude=0.5,
    ):
        """
        Off-lattice surface adatoms at random pixel positions (not tied to u/v sites).
        `correlation` blends the placement density between uniform-random (0.0) and a
        low-frequency field generated the same way as the rest of the class's noise
        fields (1.0), so adatoms can be made to cluster in some regions. Call after
        add_sites_to_image and before add_noise. window_width and sigma (the adatom blob
        size) are in nm (converted via self.pixel_size_nm). corr_p_in/corr_p_s (the
        clustering field's correlation length) are in units of the lattice spacing |u|,
        same reasoning as generate_coordinates_hex's p_s -- so the same default clusters
        adatoms on a sensible scale regardless of pixel_size_nm or field of view.
        """
        if not hasattr(self, "_images"):
            raise ValueError("Call add_sites_to_image before add_adatoms")
        H, W = self._images.shape
        pixel_size_nm = self.pixel_size_nm
        window_width = max(2, int(round(window_width / pixel_size_nm)))
        sigma = sigma / pixel_size_nm
        u_norm = np.linalg.norm(self._u)

        if correlation > 0:
            field = np.zeros([H, W])
            self.generate_dataset_2D(
                field, p_in=corr_p_in * u_norm, p_s=corr_p_s * u_norm, sparsity=1.0
            )
            field = field - field.min()
            if field.max() > 0:
                field = field / field.max()
        else:
            field = np.ones([H, W])

        density = (1 - correlation) + correlation * field
        density = density / density.sum()

        flat_idx = np.random.choice(H * W, size=num_adatoms, replace=False, p=density.ravel())
        a_x_int, a_y_int = np.unravel_index(flat_idx, (H, W))
        # sub-pixel jitter within the sampled pixel, matching the lattice atoms' floor/residual
        # centering in add_sites_to_image (otherwise every adatom would land exactly on an
        # integer pixel center)
        a_x = a_x_int + (np.random.rand(num_adatoms) - 0.5)
        a_y = a_y_int + (np.random.rand(num_adatoms) - 0.5)

        ww = window_width
        half_width = int(ww // 2)
        xx, yy = np.meshgrid(
            np.arange(-half_width, ww - half_width),
            np.arange(-half_width, ww - half_width),
            indexing="ij",
        )
        # precomputed once: circular window instead of the full square, so adatom patches
        # taper to zero on a disk rather than clipping at corners
        circle_mask = (xx**2 + yy**2) <= (ww / 2.0) ** 2

        adatom_layer = np.zeros([H, W])
        for ax, ay in zip(a_x, a_y):
            ax_floor = int(np.floor(ax))
            ay_floor = int(np.floor(ay))
            xc, yc = ax - ax_floor, ay - ay_floor
            rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
            blob = self.gauss_2D_r(rr, sigma, amplitude, 0)
            blob[~circle_mask] = 0

            x0, x1 = max(0, ax_floor - half_width), min(H, ax_floor + (ww - half_width))
            y0, y1 = max(0, ay_floor - half_width), min(W, ay_floor + (ww - half_width))
            bx0, bx1 = x0 - (ax_floor - half_width), ww - ((ax_floor + (ww - half_width)) - x1)
            by0, by1 = y0 - (ay_floor - half_width), ww - ((ay_floor + (ww - half_width)) - y1)
            adatom_layer[x0:x1, y0:y1] += blob[bx0:bx1, by0:by1]

        self._images = self._images + adatom_layer
        self._adatom_coordinates = np.stack([a_x, a_y], axis=1)
        return self

    def add_drift_blur(
        self,
        sigma_x_nm=0.03,
        sigma_y_nm=0.01,
        angle_deg=0.0,
    ):
        """
        Approximates scan-drift (or astigmatism) blur: convolution with an anisotropic 2D
        Gaussian PSF, elongated along one axis, done as a single multiplication in k-space
        -- rather than the isotropic Cauchy/Bessel PSF already applied per-atom inside
        add_sites_to_image. Real drift smears the *whole* frame the same way regardless of
        what's under it (atoms, contamination, adatoms), so this operates directly on
        self._images rather than being folded into the per-atom rendering. Call after
        surface features (contamination/adatoms) and before add_noise (drift is a real-space
        distortion of the true signal; noise is a detection-time effect that should act on
        the already-blurred signal). sigma_x_nm/sigma_y_nm are the PSF widths along the
        (unrotated) x/y axes -- unequal values give the elongated/directional smear; equal
        values reduce to an isotropic Gaussian blur. angle_deg rotates that ellipse.
        """
        if not hasattr(self, "_images"):
            raise ValueError("Call add_sites_to_image before add_drift_blur")
        H, W = self._images.shape
        pixel_size_nm = self.pixel_size_nm
        sx = sigma_x_nm / pixel_size_nm
        sy = sigma_y_nm / pixel_size_nm
        theta = np.radians(angle_deg)

        x_ = np.arange(H) - H // 2
        y_ = np.arange(W) - W // 2
        xx_, yy_ = np.meshgrid(x_, y_, indexing="ij")
        xr = xx_ * np.cos(theta) + yy_ * np.sin(theta)
        yr = -xx_ * np.sin(theta) + yy_ * np.cos(theta)
        kernel = np.exp(-(xr**2 / (2 * sx**2) + yr**2 / (2 * sy**2)))
        kernel /= kernel.sum()

        fft_image = np.fft.fft2(self._images)
        fft_kernel = np.fft.fft2(np.fft.ifftshift(kernel))
        self._images = np.real(np.fft.ifft2(fft_image * fft_kernel))
        return self

    def add_noise(
        self,
        dose=100.0,
        background_dose=0.0,
        dose_variation=0.03,
        return_counts=False,
    ):
        """
        Poisson (electron shot) noise driven by an actual electron dose, instead of an
        arbitrary rescale factor. `dose` and `background_dose` are electron doses in
        e-/nm^2 (multiply by 100 to convert to the more commonly quoted e-/A^2, e.g.
        dose=1e4 here is 100 e-/A^2). The image is normalized to a peak-intensity fraction,
        then multiplied by (dose * pixel_area_nm2) to get an expected electron count per
        pixel -- so Poisson noise is drawn on physical counts, and SNR ~ sqrt(dose) exactly
        as on a real detector. background_dose adds a flat dose everywhere (e.g. amorphous
        support / vacuum background) before the Poisson draw. dose_variation is the
        fractional amplitude of a smooth low-frequency dose non-uniformity across the field
        of view (illumination or detector-gain variation); set to 0 to disable. By default
        the result is renormalized back to the pre-noise intensity scale; set
        return_counts=True to get raw electron counts instead.

        Does not crop -- self._images stays at its current (padded) size, same as every
        other add_* method. The padding set up in from_HWU already exists to absorb edge
        artifacts from FFT-based steps like this one's dose_variation field; get_result()/
        show_result() do the one non-destructive crop down to the requested display size.
        That also means add_noise can be called repeatedly without eating into the padding
        margin each time.
        """
        pixel_area_nm2 = self.pixel_size_nm**2

        dose_map = 1.0
        if dose_variation:
            variation = np.zeros(self._images.shape)
            self.generate_dataset_2D(variation, p_in=100, p_s=100, sparsity=1)
            dose_map = 1 + variation * dose_variation

        im_norm = self._images / self._images.max()
        mean_counts = (im_norm * dose + background_dose) * dose_map * pixel_area_nm2
        counts = np.random.poisson(np.clip(mean_counts, 0, None))

        self._images = counts if return_counts else counts / (dose * pixel_area_nm2)
        return self

    def show_result(
        self,
        figsize=(10, 10),
        norm="minmax",
        **kwargs,
    ):
        """
        Extra kwargs are forwarded to show_2d -- e.g. save='path.png' to write a file, and
        dpi=... (default 300) to control that saved file's resolution. dpi only affects the
        saved copy; show_2d has no way to control the resolution of the inline/on-screen figure.
        """
        show_2d(self._crop_to_display(self._images), figsize=figsize, norm=norm, **kwargs)
        return self

    def get_result(
        self,
    ):
        return self._crop_to_display(self._images)

    def get_ground_truth(self):
        """
        Per-atom ground truth for evaluating a site-detection algorithm: the actual rendered
        (post-disorder-jitter) pixel coordinates of every lattice atom, in the SAME (x, y)
        frame as get_result()/show_result() -- already shifted for the padding crop, and
        filtered down to atoms that actually fall within that cropped frame (atoms out in
        the padded margin never appear in get_result(), so they're excluded here too).
        Call after add_sites_to_image.

        Returns a list of dicts, one per atom, with keys:
          'site'    : int, index into z_el / atom_coordinates (0, 1, ...)
          'x', 'y'  : float, pixel coordinates in get_result()'s frame
          'status'  : 'normal' | 'vacancy' | 'partial_vacancy' | 'substitution'
          'species' : element symbol actually present (None for a full vacancy)
        """
        if not hasattr(self, "_rendered_coordinates"):
            raise ValueError("Call add_sites_to_image before get_ground_truth")

        H_disp = getattr(self, "_H_display", self._images.shape[0])
        W_disp = getattr(self, "_W_display", self._images.shape[1])
        H_cur, W_cur = self._images.shape
        dh, dw = (H_cur - H_disp) // 2, (W_cur - W_disp) // 2

        records = []
        for a0, coords in enumerate(self._rendered_coordinates):
            x = coords[:, 0] - dh
            y = coords[:, 1] - dw
            in_frame = (x >= 0) & (x < H_disp) & (y >= 0) & (y < W_disp)
            vac = self._vacancy_labels[a0]
            sub = self._substitution_labels[a0]
            for i in np.nonzero(in_frame)[0]:
                if vac[i] == 1:
                    status, species = "vacancy", None
                elif vac[i] == 2:
                    status, species = "partial_vacancy", self._z_el[a0]
                elif sub[i] == 1:
                    status, species = "substitution", self._substitution_species[a0]
                else:
                    status, species = "normal", self._z_el[a0]
                records.append(
                    {
                        "site": a0,
                        "x": float(x[i]),
                        "y": float(y[i]),
                        "status": status,
                        "species": species,
                    }
                )
        return records

    def periodic_table(
        self,
    ):
        return {
            "h": 1,
            "he": 2,
            "li": 3,
            "be": 4,
            "b": 5,
            "c": 6,
            "n": 7,
            "o": 8,
            "f": 9,
            "ne": 10,
            "na": 11,
            "mg": 12,
            "al": 13,
            "si": 14,
            "p": 15,
            "s": 16,
            "cl": 17,
            "ar": 18,
            "k": 19,
            "ca": 20,
            "sc": 21,
            "ti": 22,
            "v": 23,
            "cr": 24,
            "mn": 25,
            "fe": 26,
            "co": 27,
            "ni": 28,
            "cu": 29,
            "zn": 30,
            "ga": 31,
            "ge": 32,
            "as": 33,
            "se": 34,
            "br": 35,
            "kr": 36,
            "rb": 37,
            "sr": 38,
            "y": 39,
            "zr": 40,
            "nb": 41,
            "mo": 42,
            "tc": 43,
            "ru": 44,
            "rh": 45,
            "pd": 46,
            "ag": 47,
            "cd": 48,
            "in": 49,
            "sn": 50,
            "sb": 51,
            "te": 52,
            "i": 53,
            "xe": 54,
            "cs": 55,
            "ba": 56,
            "la": 57,
            "ce": 58,
            "pr": 59,
            "nd": 60,
            "pm": 61,
            "sm": 62,
            "eu": 63,
            "gd": 64,
            "tb": 65,
            "dy": 66,
            "ho": 67,
            "er": 68,
            "tm": 69,
            "yb": 70,
            "lu": 71,
            "hf": 72,
            "ta": 73,
            "w": 74,
            "re": 75,
            "os": 76,
            "ir": 77,
            "pt": 78,
            "au": 79,
            "hg": 80,
            "tl": 81,
            "pb": 82,
            "bi": 83,
            "po": 84,
            "at": 85,
            "rn": 86,
            "fr": 87,
            "ra": 88,
            "ac": 89,
            "th": 90,
            "pa": 91,
            "u": 92,
            "np": 93,
            "pu": 94,
            "am": 95,
            "cm": 96,
            "bk": 97,
            "cf": 98,
            "es": 99,
            "fm": 100,
            "md": 101,
            "no": 102,
            "lr": 103,
            "rf": 104,
        }
