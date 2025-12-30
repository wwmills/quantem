from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d
from scipy.ndimage import map_coordinates

from quantem.imaging import Lattice

from quantem.core import config

import matplotlib.pyplot as plt

import scipy.special as sp

from scipy.ndimage import gaussian_filter

class SimData(AutoSerialize):
    """
    Generating atomic resolution data for ML identification of sites.
    """

    _token = object()

    def __init__(
        self,
        atom_coordinates: Dataset2d,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use SimData.from_coordinates() or SimData.from_HWU() to instantiate this class.")
        self._atom_coordinates: Dataset2d = atom_coordinates

    def __init__(
        self,
        u: NDArray[2],
        v: NDArray[2] | None = None,
        H: float | None = None,
        W: float | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use SimData.from_coordinates() or SimData.from_HWU() to instantiate this class.")
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
    def from_coordinates(
        cls,
        atom_coordinates: Union[list[NDArray], NDArray],
        H : int | None = None,
        W : int | None = None,
    ) -> "SimData":
        if isinstance(atom_coordinates, list[NDArray]):
            num_sites = len(atom_coordinates)
            for site_index in range(num_sites):
                atom_coordinates[site_index] = ensure_valid_array(atom_coordinates[site_index], ndim = 2)
        if isinstance(atom_coordinates, NDArray):
            atom_coordinates = ensure_valid_array(atom_coordinates, ndim = 2)
        return cls(atom_coordinates = atom_coordinates, _token = cls._token)

    @classmethod
    def from_HWU(
        cls,
        H: int | None = None,
        W: int | None = None,
        u: NDArray[2] | None = None,
        v: NDArray[2] | None = None,
        theta: float | None = None,
    ) -> "SimData":
        if H is None and W is not None: 
            H = W
        if W is None and H is not None: 
            W = H
        if H is None:
            H = 512
        if W is None:
            W = 512
        u = ensure_valid_array(u, ndim = 1)
        if v is not None:
            v = ensure_valid_array(v, ndim = 1)
        if v is None:
            if theta is None:
                theta = np.pi/3
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            v = (u@rotation_matrix).T
        return cls(H = H, W = W, u = u, v = v, _token = cls._token)


    # --- Properties ---
    @property
    def atom_coordinates(self) -> Union[list[NDArray], NDArray]:
        return self._atom_coordinates

    @property
    def images(self) -> NDArray:
        if hasattr(self, '_images'):
            return self._images
        else:
            raise ValueError('Images not generated yet')

    # @atom_coordinates.setter
    # def atom_coordinates(self, value: Union[list[NDArray], NDArray]):
    #     if isinstance(value, Dataset2d):
    #         self._image = value
    #     else:
    #         arr = ensure_valid_array(value, ndim=2)
    #         if hasattr(Dataset2d, "from_array") and callable(getattr(Dataset2d, "from_array")):
    #             self._image = Dataset2d.from_array(arr)  # type: ignore[attr-defined]
    #         else:
    #             self._image = Dataset2d(arr)  # type: ignore[call-arg]


    def gauss_2D_r(
            self,
            rr,
            s,
            A,
            B,
    ):

        gaussian_2d = A * np.exp(-((rr)**2/(2*s**2))) + B
        return gaussian_2d


    def generate_mask_2D(
            self,
            array,
            p_s=60.0,
            sparsity=0.20
        ):
        arrayShape = array.shape
        x = np.fft.fftfreq(arrayShape[0])
        y = np.fft.fftfreq(arrayShape[1])

        X, Y = np.meshgrid(y, x, indexing = 'ij')
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

    def generate_dataset_2D(
            self,
            array,
            p_in=30.0,
            p_s=60.0,
            sparsity=0.20
            ):

        arrayShape = array.shape
        x = np.fft.fftfreq(arrayShape[0])
        y = np.fft.fftfreq(arrayShape[1])

        X, Y = np.meshgrid(y, x, indexing = 'ij')
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


    # a way to call the lattice class within this class
    def generate_coordinates_hex(
            self,
            H = None,
            W = None,
            u = np.array([16,0]),
            v = None,
            theta_v = np.pi/3,
            num_sites = 3,
            p_s = 400,
            sparsity = 0.9,
            plot_atoms = False,
            min_neighbors = 3,
            tolerance = None
    ):
        if hasattr(self,'_H'):
            H = self._H
        if hasattr(self,'_W'):
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
        im = np.ones([H, W]) # dummy image array
        im = Dataset2d.from_array(im)
        lattice = Lattice.from_data(
            image=im,
            normalize_min = False,
            )
        mask_ = np.zeros([H,W])
        self.generate_mask_2D(mask_, p_s = p_s, sparsity = sparsity)

        theta = np.random.rand(1) * np.pi * 2
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

        u = np.array([16,0])
        u = u@rotation_matrix.T
        theta = np.pi/3
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        v = u@rotation_matrix.T
        origin = np.array([0,0])

        self._u = u

        lattice.define_lattice(
        origin,
        u,
        v,
        plot_lattice=False,
        input_mask=mask_,
        refine_lattice=False,
        )
        if num_sites > 3:
            raise ValueError('The number of sites automation can only do A, B, and C sites')
        positions_frac = [np.array([0,0]), np.array([1/3,1/3]), np.array([-1/3,-1/3])]
        positions_frac = positions_frac[:num_sites]
        lattice.add_atoms(
            positions_frac,
            numbers=None,
            intensity_min=None,
            intensity_radius=None,
            plot_atoms=False,
            edge_min_dist_px = 1,
            mask=mask_,
            contrast_min=None,
            annulus_radii=None,
        )

        if plot_atoms:
            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            label_list = ['a', 'b', 'c']
            color_list_o = ['red', 'blue', 'green']
            label_list = label_list[:num_sites]
            for site_index in range(num_sites):
                a_x = lattice.atoms.get_data(site_index)[:,0]
                a_y = lattice.atoms.get_data(site_index)[:,1]
                ax[0].scatter(a_y, -a_x, color = color_list_o[site_index], s=10, alpha=0.5, label = label_list[site_index])
            ax[0].legend()
            ax[0].set_title("Before filtering")
            ax[0].set_box_aspect(1)

        # --- Apply tolerance ---
        # should this be run more than once?
        lattice.find_neighbors_in_tolerance(tolerance = tolerance)
        # removed_atoms = lattice.remove_atoms_with_too_few_neighbors(min_neighbors = 3, return_removed = True)
        # note edge atoms, so that we can make them blurrier and more shifted later
        # lattice.find_neighbors_in_tolerance(tolerance = tolerance*0.8)
        self.edge_atoms = lattice.find_atoms_with_too_few_neighbors(min_neighbors = 3)
        if plot_atoms:
            color_list_r = ['#fc03db' ,'#fc7f03', '#03ecfc']
            for site_index in range(num_sites):
                a_x = lattice.atoms.get_data(site_index)[:,0]
                a_y = lattice.atoms.get_data(site_index)[:,1]
                ax[1].scatter(a_y, -a_x, color = color_list_o[site_index], s=10, alpha=0.5, label = label_list[site_index])
                # a_x_n = removed_atoms[site_index][:,0]
                # a_y_n = removed_atoms[site_index][:,1]
                # ax[1].scatter(a_y_n, -a_x_n, s = 20, alpha = 0.5, color = color_list_r[site_index], label = label_list[site_index] + '_r')
            ax[1].legend()
            ax[1].set_title("After filtering")
            ax[1].set_box_aspect(1)

            for site_index in range(num_sites):
                a_x = lattice.atoms.get_data(site_index)[:,0]
                a_y = lattice.atoms.get_data(site_index)[:,1]
                ax[2].scatter(a_y, -a_x, color = color_list_o[site_index], s=10, alpha=0.5, label = label_list[site_index])
            ax[2].set_title("Remaining after filtering")
            ax[2].set_box_aspect(1)

            for site_index in range(num_sites):
                a_x = lattice.atoms.get_data(site_index)[self.edge_atoms[site_index],0]
                a_y = lattice.atoms.get_data(site_index)[self.edge_atoms[site_index],1]
                ax[2].scatter(a_y, -a_x, color = 'black', s=10, alpha=0.5, label = 'edge atom')
            ax[2].legend()
            plt.tight_layout()
            plt.show()

        self.lattice_obj = lattice
        self._atom_coordinates = []
        for site_index in range(num_sites):
            self._atom_coordinates.append(np.array([lattice.atoms.get_data(site_index)[:,0], lattice.atoms.get_data(site_index)[:,1]]))
        return self

    # def make_layer_map(
    #         self,
    #         mask_in = None,
    # ):
    #     if mask_in is None:
    #         mask_in = np.zeros([self._H,self._W])
    #         self.generate_mask_2D(
    #             mask_in,
    #             p_s = 400,
    #             sparsity = 0.9,
    #             )


        # from lattice code
                ##################
                # self._positions_frac = np.atleast_2d(np.array(positions_frac, dtype=float))
                # self._num_sites = self._positions_frac.shape[0]
                # self._numbers = (
                #     np.arange(1, self._num_sites + 1, dtype=int)
                #     if numbers is None
                #     else np.atleast_1d(np.array(numbers, dtype=int))
                # )

                # im = np.asarray(self._image.array, dtype=float)
                # H, W = self._image.shape  # x=rows, y=cols
                # r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
                # A = np.column_stack((u, v))

                # corners = np.array(
                #     [[0.0, 0.0], [float(H), 0.0], [0.0, float(W)], [float(H), float(W)]], dtype=float
                # )
                # ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]
                # a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
                # b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

                # def _auto_radius_px() -> float:
                #     S = self._positions_frac
                #     if S.shape[0] >= 2:
                #         d = S[:, None, :] - S[None, :, :]
                #         d = d - np.round(d)
                #         same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
                #         dpix = d @ A.T
                #         dist = np.linalg.norm(dpix, axis=2)
                #         dist[same] = np.inf
                #         nn = float(np.min(dist))
                #     else:
                #         nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
                #     if not np.isfinite(nn) or nn <= 0:
                #         nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
                #     return 0.5 * nn

                # r_px = float(intensity_radius) if intensity_radius is not None else _auto_radius_px()
                # rin, rout = (1.5 * r_px, 3.0 * r_px) if annulus_radii is None else annulus_radii
                # R_disk = int(np.ceil(r_px))
                # R_ring = int(np.ceil(rout))
                # edge_thresh = float(edge_min_dist_px) if edge_min_dist_px is not None else 0.0

                # DT = None
                # if mask is not None:
                #     m = np.asarray(mask).astype(bool)
                #     if m.shape != (H, W):
                #         raise ValueError(f"mask shape {m.shape} must match image shape {(H, W)}")
                #     try:
                #         from scipy.ndimage import distance_transform_edt

                #         DT = distance_transform_edt(m)
                #     except Exception:
                #         DT = None

                # def mean_disk(x: float, y: float) -> float:
                #     ix0, iy0 = int(np.floor(x)), int(np.floor(y))
                #     i0, i1 = max(0, ix0 - R_disk), min(H - 1, ix0 + R_disk)
                #     j0, j1 = max(0, iy0 - R_disk), min(W - 1, iy0 + R_disk)
                #     ii = np.arange(i0, i1 + 1)[:, None]
                #     jj = np.arange(j0, j1 + 1)[None, :]
                #     dx, dy = ii - x, jj - y
                #     mask_circle = (dx * dx + dy * dy) <= (r_px * r_px)
                #     vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_circle]
                #     if vals.size == 0:
                #         return float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
                #     return float(vals.mean())

                # def mean_std_annulus(x: float, y: float) -> tuple[float, float]:
                #     ix0, iy0 = int(np.floor(x)), int(np.floor(y))
                #     i0, i1 = max(0, ix0 - R_ring), min(H - 1, ix0 + R_ring)
                #     j0, j1 = max(0, iy0 - R_ring), min(W - 1, iy0 + R_ring)
                #     ii = np.arange(i0, i1 + 1)[:, None]
                #     jj = np.arange(j0, j1 + 1)[None, :]
                #     dx, dy = ii - x, jj - y
                #     r2 = dx * dx + dy * dy
                #     mask_ring = (r2 >= rin * rin) & (r2 <= rout * rout)
                #     vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_ring]
                #     if vals.size == 0:
                #         val = float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
                #         return val, 0.0
                #     return float(vals.mean()), float(vals.std(ddof=0))

                # self.atoms = Vector.from_shape(
                #     shape=(self._num_sites,),
                #     fields=("x", "y", "a", "b", "int_peak"),
                #     units=("px", "px", "ind", "ind", "counts"),
                # )

                # for a0 in range(self._num_sites):
                #     da, db = self._positions_frac[a0, 0], self._positions_frac[a0, 1]
                #     aa, bb = np.meshgrid(
                #         np.arange(a_min - 1 + da, a_max + 1 + da),
                #         np.arange(b_min - 1 + db, b_max + 1 + db),
                #         indexing="ij",
                #     )
                #     basis = np.vstack((np.ones(aa.size), aa.ravel(), bb.ravel())).T
                #     xy = basis @ self._lat  # (N,2) in (x,y)

                #     x, y = xy[:, 0], xy[:, 1]
                #     in_bounds = (x >= 0.0) & (x <= H - 1) & (y >= 0.0) & (y <= W - 1)
                #     border_ok = (
                #         (x - edge_thresh >= 0.0)
                #         & (x + edge_thresh <= H - 1)
                #         & (y - edge_thresh >= 0.0)
                #         & (y + edge_thresh <= W - 1)
                #     )

                #     if mask is not None:
                #         if DT is not None:
                #             ii = np.clip(np.round(x).astype(int), 0, H - 1)
                #             jj = np.clip(np.round(y).astype(int), 0, W - 1)
                #             mask_ok = DT[ii, jj] >= edge_thresh
                #         else:
                #             m = np.asarray(mask).astype(bool)
                #             mask_ok = m[
                #                 np.clip(np.round(x).astype(int), 0, H - 1),
                #                 np.clip(np.round(y).astype(int), 0, W - 1),
                #             ]
                #     else:
                #         mask_ok = np.ones_like(in_bounds, dtype=bool)

                #     int_center = np.empty(xy.shape[0], dtype=float)
                #     for i in range(xy.shape[0]):
                #         int_center[i] = mean_disk(x[i], y[i])

                #     keep = in_bounds & border_ok & mask_ok
                #     if intensity_min is not None:
                #         keep &= int_center >= float(intensity_min)
                #     if contrast_min is not None:
                #         bg_mean = np.empty(xy.shape[0], dtype=float)
                #         for i in range(xy.shape[0]):
                #             bg_mean[i], _ = mean_std_annulus(x[i], y[i])
                #         keep &= (int_center - bg_mean) >= float(contrast_min)

                #     if np.any(keep):
                #         arr = np.vstack(
                #             (x[keep], y[keep], basis[keep, 1], basis[keep, 2], int_center[keep])
                #         ).T
                #     else:
                #         arr = np.zeros((0, 5), dtype=float)

                #     # --- Correct API usage ---
                #     self.atoms.set_data(arr, a0)
        ###########

    def _binary_density(
            self,
            size,
            density
            ):
        if not (0.0 <= density <= 1.0):
            raise ValueError("Density must be between 0.0 and 1.0")

        num_ones = int(size * density)
        if num_ones > size:
            num_ones = size

        arr = np.zeros(size, dtype=int)
        arr[:num_ones] = 1
        np.random.shuffle(arr)

        return arr

    def add_sites_to_image(
            self,
            H = None,
            W = None,
            window_width = 20,
            disorder_multiplier = 0.1,
            show_ideal = False,
            show_result = True,
            return_result = False,
            sg_c = 1,
            gamma = 1.7,
            cauchy = 0.5,
            use_bessel = False,
            z_el = ['Ca', 'Ti', 'O'],
            z_stack = np.array([1,1,1]),
            edge_multiplier_in = 3,
            substitution_index = np.array([0, 1, 2]),
            substitution_density = np.array([0,0,0]),
            substitution_species = ['V', 'Ti', 'O'],
            vacancy_index = np.array([0, 1, 2]),
            vacancy_density = np.array([0,0,0]),
    ):
        
        # Automatically set the number of pixels needed
        # sometimes this may be overkill if one coordinate is slightly over the edge of one atom
        if hasattr(self, '_H'):
            H = self._H
        if hasattr(self, '_W'):
            W = self._W
        if H is None and W is not None: 
            H = W
        if W is None and H is not None: 
            W = H
        if H is None:
            default_size_arr_pow = np.arange(0,15)
            default_size_arr_px = 2**default_size_arr_pow
            # find x max
            num_sites = len(self._atom_coordinates)
            print(num_sites)
            max_x = -1
            for site_index in range(num_sites):
                max_x = max(max_x, np.max(self._atom_coordinates[site_index][:,0]))
            argmin_num_px = np.argmin(np.abs(default_size_arr_px - max_x))
            px_count_t = default_size_arr_px[argmin_num_px]
            if px_count_t - max_x < 0:
                argmin_num_px += 1
            H = default_size_arr_px[argmin_num_px]
        if W is None:
            default_size_arr_pow = np.arange(0,15)
            default_size_arr_px = 2**default_size_arr_pow
            # find x max
            num_sites = len(self._atom_coordinates)
            max_y = -1
            for site_index in range(num_sites):
                max_y = max(max_y, np.max(self._atom_coordinates[site_index][:,1]))
            argmin_num_px = np.argmin(np.abs(default_size_arr_px - max_y))
            px_count_t = default_size_arr_px[argmin_num_px]
            if px_count_t - max_y < 0:
                argmin_num_px += 1
            W = default_size_arr_px[argmin_num_px]

        
        mask_ = np.zeros([H,W])
        self.generate_mask_2D(
            mask_,
            p_s = 400,
            sparsity = 0.9,
            )
        
        # num_sites = 3
        num_sites =self.num_sites
        print(num_sites)
        if not hasattr(self, '_atom_coordinates'):
            self.generate_coordinates_hex(
                H,
                W,
                u = np.array([16,0]),
                v = None,
                theta_v = np.pi/3,
                num_sites = num_sites,
                p_s = 400,
                sparsity = 0.9,
                plot_atoms = False,
                min_neighbors = 3,

                )

        im__ = np.zeros([H,W])

        ww = window_width
        half_width = int(ww//2)

        x = np.arange(0, ww) - ww // 2
        y = np.arange(0, ww) - ww // 2
        xx, yy = np.meshgrid(x, y, indexing = 'ij')
        sg = 2
        r = 0

        # disorder in position
        u_norm = np.linalg.norm(self._u)

        # a0
        self._atom_coordinates

        pt = self.periodic_table()
        zNums = np.asarray(list(map(lambda x: pt[x.lower()], z_el)))
        site_multipliers = zNums ** gamma
        parent_mean = np.mean(site_multipliers)
        site_multipliers /= parent_mean # normalize

        zNums = np.asarray(list(map(lambda x: pt[x.lower()], substitution_species)))
        sub_multipliers = zNums ** gamma
        sub_multipliers /= parent_mean # same normalization

        # now add the contamination
        contamination = np.zeros([H,W])
        self.generate_dataset_2D(contamination, p_in = 500, p_s = 100, sparsity = 0.1)

        edge_multiplier = 1
        for a0 in range(num_sites):
            a_x_a0 = self._atom_coordinates[a0][0]
            a_y_a0 = self._atom_coordinates[a0][1]
            edge_atoms_a0 = self.edge_atoms[a0]
            if a0 in vacancy_index:
                vacancy_arr = self._binary_density(a_x_a0.shape[0], vacancy_density[a0])
            if a0 in substitution_index:
                substitution_arr = self._binary_density(a_x_a0.shape[0], substitution_density[a0])
            for atom_index in range(a_x_a0.shape[0]):
                if edge_atoms_a0[atom_index]:
                    edge_multiplier = edge_multiplier_in
                else:
                    edge_multiplier = 1
                a_x = a_x_a0[atom_index] + (np.random.rand(1)[0]-0.5) * disorder_multiplier * u_norm * edge_multiplier
                a_y = a_y_a0[atom_index] + (np.random.rand(1)[0]-0.5) * disorder_multiplier * u_norm * edge_multiplier

                a_x_residual = a_x - np.floor(a_x)
                a_x_floor = np.floor(a_x).astype(int)
                a_y_residual = a_y - np.floor(a_y)
                a_y_floor = np.floor(a_y).astype(int)

                xc = a_x_residual
                yc = a_y_residual

                rr = np.sqrt((xx - xc)**2 + (yy - yc) ** 2)

                sub_x0 = max(0, - (a_x_floor - half_width))
                sub_x1 = ww - max(0, (a_x_floor + half_width) - H)
                sub_y0 = max(0, - (a_y_floor - half_width))
                sub_y1 = ww - max(0, (a_y_floor + half_width) - W)

                # image index (in image coordinates)
                x1 = min(H, a_x_floor + half_width)
                y1 = min(W, a_y_floor + half_width)
                x0 = max(0, a_x_floor - half_width)
                y0 = max(0, a_y_floor - half_width)


                rr_v = rr[sub_x0:sub_x1, sub_y0:sub_y1]
                g = self.gauss_2D_r(rr_v, sg * edge_multiplier, 1, 0) /2 * site_multipliers[a0] * (contamination[np.round(a_x).astype(int), np.round(a_y).astype(int)] + 1) *0.5 / edge_multiplier

                g *= z_stack[a0]

                if a0 in vacancy_index:
                    if vacancy_arr[atom_index] == 1:
                        g *= 0
                
                if a0 in substitution_index:
                    if substitution_arr[atom_index] == 1:
                        g /= site_multipliers[a0]
                        g *= sub_multipliers[a0]
                
                g[rr_v<np.min(rr_v)/np.sqrt(2)] = 0
                g -= np.min(g)


                im__[x0:x1, y0:y1] += g

        if show_ideal:
            plt.figure()
            plt.imshow(im__, cmap = 'gray')
            plt.title('Just Gaussians')
            plt.axis('off')

        # build grid that is the size of the whole image
        x_ = np.arange(0, H) - H//2
        y_ = np.arange(0, W) - W//2
        xx_, yy_ = np.meshgrid(x_, y_, indexing = 'ij')
        rr_ = np.sqrt(xx_**2 + yy_**2)

        # convolution step
        fft_im__ = np.fft.fft2(im__)

        if use_bessel:
            sg_c = 1
            B = sp.j0(rr_/sg_c) ** 2; B /= np.abs(B).sum()
            fft_bessel = np.fft.fft2(np.fft.ifftshift(B))

        C = 1 / (np.pi * cauchy * (1 + (rr_ / cauchy)**2)); C /= C.sum()
        fft_cauchy = np.fft.fft2(np.fft.ifftshift(C))

        # multiply in k-space:
        if use_bessel:
            fft_product = fft_im__ * fft_bessel * fft_cauchy
        else:
            fft_product = fft_im__ * fft_cauchy

        # back to real space
        result = np.real(np.fft.ifft2(fft_product))

        if show_result:
            plt.figure()
            plt.imshow(np.real(result), cmap = 'gray')
            plt.title('Convolved with Cauchy')
            plt.axis('off')
        
        self._images = result
        if return_result:
            return result
        return self
        

    def add_noise(
            self,
            bg = 0.1,
            rescale = 100,
            win = 16
    ):
        low_frequency_variation = np.zeros([self._H,self._W])
        self.generate_dataset_2D(low_frequency_variation, p_in = 100, p_s = 100, sparsity = 1)
        # plt.imshow(low_frequency_variation, cmap = 'gray')

        im_bg = self._images + bg + low_frequency_variation * 0.03

        im_max = np.max(im_bg)
        self._images = np.random.poisson(im_bg / im_max * rescale) * im_max / rescale # noisy image
        self._crop_in(win = win)
        return self

    def _crop_in(
            self,
            win = 16,
    ):
        self._images = self._images[win:-win, win:-win]

    def show_result(
            self,
    ):
        plt.figure()
        plt.imshow(self._images, cmap = 'gray')
        plt.axis('off')
        return self

    def get_result(
            self,
    ):
        return self._images

    def periodic_table(
            self,
    ):
        return { 'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8, 'f': 9, 'ne': 10, 'na': 11, 
        'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16, 'cl': 17, 'ar': 18, 'k': 19, 'ca': 20, 'sc': 21, 'ti': 22, 
        'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29, 'zn': 30, 'ga': 31, 'ge': 32, 'as': 33, 'se': 34, 
        'br': 35, 'kr': 36, 'rb': 37, 'sr': 38, 'y': 39, 'zr': 40, 'nb': 41, 'mo': 42, 'tc': 43, 'ru': 44, 'rh': 45, 'pd': 46,
        'ag': 47, 'cd': 48, 'in': 49, 'sn': 50, 'sb': 51, 'te': 52, 'i': 53, 'xe': 54, 'cs': 55, 'ba': 56, 'la': 57, 'ce': 58, 
        'pr': 59, 'nd': 60, 'pm': 61, 'sm': 62, 'eu': 63, 'gd': 64, 'tb': 65, 'dy': 66, 'ho': 67, 'er': 68, 'tm': 69, 'yb': 70,
        'lu': 71, 'hf': 72, 'ta': 73, 'w': 74, 're': 75, 'os': 76, 'ir': 77, 'pt': 78, 'au': 79, 'hg': 80, 'tl': 81, 'pb': 82, 
        'bi': 83, 'po': 84, 'at': 85, 'rn': 86, 'fr': 87, 'ra': 88, 'ac': 89, 'th': 90, 'pa': 91, 'u': 92, 'np': 93, 'pu': 94,
        'am': 95, 'cm': 96, 'bk': 97, 'cf': 98, 'es': 99, 'fm': 100, 'md': 101, 'no': 102, 'lr': 103, 'rf': 104}


    # need the following abilities in simulated data:
    
    # point defect modifier: strain neighbors inwards, chance of removing neighbor
    # fixing up the way to interact with substitutions and vacancies


    # adding layers and edges

    # multiple image sizes and FOVs

    # can I get an edge disloation in here?


    # def create_defects(
    #     self,
    #     z_el = ['Ca', 'Ti', 'O'],
    #     z_stack = np.array([1,1,1]),
    #     edge_multiplier_in = 3,
    #     substitution_index = np.array([0, 1, 2]),
    #     substitution_density = np.array([0,0,0]),
    #     substitution_species = ['V', 'Ti', 'O'],
    #     vacancy_index = np.array([0, 1, 2]),
    #     vacancy_density = np.array([0,0,0]),
    # ):
    #     num_sites = self.num_sites

    #     if len(z_el) != num_sites:
    #         raise ValueError('The number of specified elements must match the number of z numbers given')
    #     if z_stack.shape[0] != num_sites:
    #         raise ValueError('The number of specified elements must match the number vertically overlapping atoms given')
        
    #     if z_stack.shape[0] != num_sites:
    #         raise ValueError('The number of specified elements must match the number vertically overlapping atoms given')
        
    


    # def strain_defects(
    #         self,
    # ):
    