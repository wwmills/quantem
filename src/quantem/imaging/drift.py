from collections.abc import Sequence
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from tqdm import tqdm

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.compound_validators import (
    validate_list_of_dataset2d,
    validate_pad_value,
)
from quantem.core.utils.imaging_utils import (
    bilinear_kde,
    cross_correlation_shift,
    fourier_cropping,
)
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d
from mpire import WorkerPool
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from mpire import WorkerPool
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

class DriftCorrection(AutoSerialize):
    """
    DriftCorrection provides translation, affine, and non-rigid drift correction for
    sequential 2D images using scan direction metadata and flexible spatial interpolation.

    This class supports input data as numpy arrays, Dataset2d, or Dataset3d instances,
    with various padding strategies and configurable spline interpolation of scanline
    trajectories via Bézier knot control.

    Features
    --------
    - Load data from arrays or files
    - Apply initial scanline resampling using Bézier curves
    - Align images using translation, affine, or non-rigid optimization
    - Visualize intermediate and final results with optional knot overlays
    - Serialize state with `.save()` and restore with `.load()`

    Parameters (via `from_data` or `from_file`)
    -------------------------------------------
    images : list of 2D arrays, Dataset2d, Dataset3d, or file names, or a 3D numpy array
        The image stack to correct for drift.
    scan_direction_degrees : list of float
        The scan direction angle (in degrees) for each image, measured relative to vertical.
    pad_fraction : float, default 0.25
        Fraction of padding to add around each image during interpolation.
    pad_value : str, float, or list of float, default 'median'
        How to pad outside the image area during warping. Can be:
        - One of: 'median', 'mean', 'min', 'max'
        - A float quantile value (e.g., 0.25)
        - A list of per-image float values
    number_knots : int, default 1
        Number of knots to use for Bézier interpolation of scanline trajectories.
        We strongly recommend using `number_knots = 1` unless the fast scan direction is
        expected to vary within the image.

    Example
    -------
    Instantiate the DriftCorrection class, run preprocessing and alignment, and save/load results:

    >>> drift = DriftCorrection.from_data(
    ...     images=[
    ...         image0,  # 2D numpy array or Dataset2d
    ...         image1,
    ...     ],
    ...     scan_direction_degrees=[0, 90],
    ... ).preprocess(
    ...     pad_fraction=0.25,
    ...     pad_value='median',
    ...     number_knots=1,
    ... )

    >>> drift.align_affine()
    >>> drift.align_nonrigid()
    >>> drift.plot_merged_images()
    >>> image_corr = drift.generate_corrected_image()

    >>> drift.save("drift_result.zip")
    >>> drift_reloaded = quantem.io.load("drift_result.zip")

    >>> image_corr.save("image_corrected.zip")
    >>> image_corr_reloaded = quantem.io.load("image_corrected.zip")

    Notes
    -----
    - Use `align_translation()` for rigid shifts, `align_affine()` for scan-shear or uniform drift,
      and `align_nonrigid()` for flexible per-row or per-image correction.
    - The class stores resampled images in `self.images_warped` and the control knots in `self.knots`.
    - Interactive visualization is supported through `plot_merged_images()` and `plot_transformed_images()`.
    """

    _token = object()

    def __init__(
        self,
        images: List[Dataset2d],
        scan_direction_degrees: NDArray,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use DriftCorrection.from_data() or .from_file() to instantiate this class."
            )

        self._images = images
        self.scan_direction_degrees = scan_direction_degrees
        self.print_thing = True

    @classmethod
    def from_file(
        cls,
        file_paths: Sequence[str],
        scan_direction_degrees: Union[Sequence[float], NDArray],
        file_type: str | None = None,
    ) -> "DriftCorrection":
        image_list = [Dataset2d.from_file(fp, file_type=file_type) for fp in file_paths]
        return cls.from_data(
            image_list,
            scan_direction_degrees,
        )

    @classmethod
    def from_data(
        cls,
        images: Union[List[Dataset2d], List[NDArray], Dataset3d, NDArray],
        scan_direction_degrees: Union[List[float], NDArray],
    ) -> "DriftCorrection":
        validated_images = validate_list_of_dataset2d(images)

        return cls(
            images=validated_images,
            scan_direction_degrees=scan_direction_degrees,
            _token=cls._token,
        )

    # --- Properties ---
    @property
    def images(self) -> List[Dataset2d]:
        return self._images

    @images.setter
    def images(self, value: Union[List[Dataset2d], List[NDArray], Dataset3d, NDArray]):
        self._images = validate_list_of_dataset2d(value)
        self.pad_value = self.pad_value

    @property
    def pad_value(self) -> List[float]:
        return self._pad_value

    @pad_value.setter
    def pad_value(self, value: Union[float, str, List[float]]):
        self._pad_value = validate_pad_value(value, self.images)

    @property
    def scan_direction_degrees(self) -> NDArray:
        return self._scan_direction_degrees

    @scan_direction_degrees.setter
    def scan_direction_degrees(self, value: Union[List[float], NDArray]):
        self._scan_direction_degrees = ensure_valid_array(value, ndim=1)

    @property
    def pad_fraction(self) -> float:
        return self._pad_fraction

    @pad_fraction.setter
    def pad_fraction(self, value: float):
        self._pad_fraction = float(value)

    @property
    def kde_sigma(self) -> float:
        return self._kde_sigma

    @kde_sigma.setter
    def kde_sigma(self, value: float):
        self._kde_sigma = float(value)

    @property
    def number_knots(self) -> int:
        return self._number_knots

    @number_knots.setter
    def number_knots(self, value: float):
        self._number_knots = int(value)

    def preprocess(
        self,
        pad_fraction: float = 0.25,
        pad_value: Union[float, str, List[float]] = "median",
        kde_sigma: float = 0.5,
        number_knots: int = 1,
        show_merged: bool = False,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        # Validators
        validated_pad_value = validate_pad_value(pad_value, self._images)

        # print('validated pad value:', validate_pad_value(pad_value, self._images))
        # print('actual median 0:', np.median(self._images[0].array))
        # print('actual median 1:', np.median(self._images[1].array))

        # Input data
        self.pad_fraction = pad_fraction
        self._pad_value = validated_pad_value
        self.kde_sigma = kde_sigma
        self.number_knots = number_knots

        # Derived data
        self.scan_direction = np.deg2rad(self.scan_direction_degrees)
        self.scan_fast = np.stack(
            [
                np.sin(-self.scan_direction),
                np.cos(-self.scan_direction),
            ],
            axis=1,
        )
        self.scan_slow = np.stack(
            [
                np.cos(-self.scan_direction),
                -np.sin(-self.scan_direction),
            ],
            axis=1,
        )
        self.shape = (
            len(self.images),
            int(np.round(self.images[0].shape[0] * (1 + self.pad_fraction) / 2) * 2),
            int(np.round(self.images[1].shape[1] * (1 + self.pad_fraction) / 2) * 2),
        )

        # Initialize Bezier knots and scan vectors for scanlines
        self.knots = []
        for a0 in range(self.shape[0]):
            shape = self.images[a0].shape

            v_slow = np.linspace(-(shape[0] - 1) / 2, (shape[0] - 1) / 2, shape[0])
            u_fast = np.linspace(-(shape[1] - 1) / 2, (shape[1] - 1) / 2, self.number_knots)

            xa = (
                (self.shape[1] - 1) / 2
                + u_fast[None, :] * self.scan_fast[a0, 0]
                + v_slow[:, None] * self.scan_slow[a0, 0]
            )
            ya = (
                (self.shape[2] - 1) / 2
                + u_fast[None, :] * self.scan_fast[a0, 1]
                + v_slow[:, None] * self.scan_slow[a0, 1]
            )

            self.knots.append(np.stack([xa, ya], axis=0))
        # print("Preprocess knots:", self.knots[0])
        # plt.figure()
        # plt.plot(self.knots[0][1,:], self.knots[0][0,:])
        # plt.plot(self.knots[1][1,:], self.knots[1][0,:])
        # Precompute the interpolator for all images
        self.interpolator = []
        for a0 in range(self.shape[0]):
            self.interpolator.append(
                DriftInterpolator(
                    input_shape=self.images[a0].shape,
                    output_shape=self.shape[1:],
                    scan_fast=self.scan_fast[a0],
                    scan_slow=self.scan_slow[a0],
                    pad_value=self.pad_value[a0],
                    kde_sigma=self.kde_sigma,
                    scan_2_fast = self.scan_fast[1],
                    input_2_shape = self.images[1].shape,
                    rot = self.scan_direction[0],
                    rot_2 = self.scan_direction[1],
                )
            )


        # print('input_shape',self.images[a0].shape)
        # print('output_shape',self.shape[1:])

        # Generate initial resampled images
        self.images_warped = Dataset3d.from_shape(self.shape)
        self.weights_warped = Dataset3d.from_shape(self.shape)
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )


        # print('median after interpolating', np.median(self.images_warped.array[0]))


        # Error tracking
        self.calculate_error(0)

        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(show_knots=show_knots, title="Merged: initial", **kwargs)

            # self.plot_merged_images(show_knots=show_knots, title="Merged: initial", **kwargs)

        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: initial" for i in range(self.shape[0])],
                **kwargs,
            )

        return self
    # import ipywidgets as widgets
    # from IPython.display import display, clear_output

    # # intended for 0 90 pairs
    # def hand_align_translation(
    #         self,
    # ):
    #     img_ref = self.images[0]
    #     img_mov = self.images[1]
    #     fig, ax = figax

    #     # Output area for dynamic display
    #     out = widgets.Output()
    #     display(out)

    #     # Shift storage
    #     shift_vals = {'sr': 0.0, 'sc': 0.0}

    #     # Sliders
    #     sr_slider = widgets.FloatSlider(
    #         description="Row shift",
    #         min=-max_shift, max=max_shift, step=1, value=0,
    #     )
    #     sc_slider = widgets.FloatSlider(
    #         description="Col shift",
    #         min=-max_shift, max=max_shift, step=1, value=0,
    #     )

    #     # Initial previous values for computing deltas (if wanted)
    #     prev_sr = 0.0
    #     prev_sc = 0.0

    #     # Update function
    #     def update(_):
    #         nonlocal prev_sr, prev_sc

    #         sr = sr_slider.value
    #         sc = sc_slider.value

    #         shift_vals['sr'] = sr
    #         shift_vals['sc'] = sc

    #         # Shift the moving image
    #         shifted = np.roll(img_mov, shift=(sr, sc), axis = (0,1))

    #         # Redraw in OUTPUT area
    #         with out:
    #             clear_output(wait=True)
    #             ax.clear()
    #             ax.imshow(img_ref, cmap='gray', alpha=1)
    #             ax.imshow(shifted, cmap='plasma', alpha=0.5)
    #             ax.set_title(f"Manual shift → row={sr}, col={sc}")
    #             # plt.tight_layout()
    #             display(fig)

    #     # Attach update callbacks
    #     sr_slider.observe(update, names='value')
    #     sc_slider.observe(update, names='value')

    #     # Display UI
    #     ui = widgets.VBox([sr_slider, sc_slider])
    #     display(ui)

    #     # Initial render
    #     # update(None)

    #     print("Adjust sliders until aligned. The returned dict updates live.")
    #     return shift_vals


    # Translation alignment
    def align_translation(
        self,
        upsample_factor: int = 8,
        min_image_shift: Optional[float] = None,
        max_image_shift: float = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        return_images_warped = False,
        return_images_shift = False,
        **kwargs,
    ):
        """
        Solve for the translation between all images in DriftCorrection.images_warped
        """

        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()

        # init
        dxy = np.zeros((self.shape[0], 2))

        # loop over images
        # crop and pad images warped
        # im_w_0_original = self.images_warped.array[0].copy()

        buff_r = int(int(np.round(self.images[0].shape[0] * (1 + self.pad_fraction) / 2) * 2) - self.images[0].shape[0])//2
        buff_c = int(int(np.round(self.images[1].shape[1] * (1 + self.pad_fraction) / 2) * 2) - self.images[0].shape[0])//2
        self.buff_r = buff_r
        self.buff_c = buff_c
        # print('buff_r:',buff_r)
        # print('buff_c:',buff_c)
        # buff = self.images_warped.array[0].shape[0]//8
        # print('buff row', buff_r)
        # print('buff col', buff_c)
        # # print('buff ', buff)
        # buff = 256
        # cropped_1 = self.images_warped.array[1].copy()[buff_r:-buff_r, buff_c:-buff_c]
        # padded_1 = np.pad(cropped_1, ((buff_r, buff_r), (buff_c, buff_c)), mode = 'constant', constant_values=np.median(cropped_1))
        # self.images_warped.array[1] = padded_1

        for a0 in range(self.shape[0]):
            cropped = self.images_warped.array[a0].copy()[buff_r:-buff_r, buff_c:-buff_c]
            padded = np.pad(cropped, ((buff_r, buff_r), (buff_c, buff_c)), mode = 'constant', constant_values=np.median(cropped))
            self.images_warped.array[a0] = padded
            self.interpolator[a0].set_pad_value(
                np.median(cropped)
            )


        # plt.figure(figsize = (15,5), dpi = 300)
        # plt.subplot(131)
        # plt.imshow(im_w_0_original, cmap = 'gray')
        # plt.colorbar()
        # plt.axis('off')
        # plt.subplot(132)
        # plt.imshow(cropped_0, cmap = 'gray')
        # plt.colorbar()
        # plt.axis('off')
        # plt.subplot(133)
        # plt.imshow(padded_0, cmap = 'gray')
        # plt.colorbar()
        # plt.axis('off')

        # plt.figure()
        # plt.subplot(121)
        # # plt.imshow(padded_1, cmap = 'gray')
        # plt.imshow(self.images_warped.array[0], cmap = 'gray')
        # plt.title('0 image handed to cc')
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(self.images_warped.array[1], cmap = 'gray')
        # plt.title('1 image handed to cc')
        # plt.axis('off')
        # print('corner pixel',self.images_warped.array[0,0,0])
        
        F_ref = np.fft.fft2(self.images_warped.array[0])
        for ind in range(1, self.shape[0]):
            shifts, image_shift = cross_correlation_shift(
                F_ref,
                np.fft.fft2(self.images_warped.array[ind]),
                upsample_factor=upsample_factor,
                max_shift=max_image_shift,
                min_shift=min_image_shift,
                fft_input=True,
                fft_output=True,
                return_shifted_image=True,
            )

            dxy[ind, :] = shifts
            F_ref = F_ref * ind / (ind + 1) + image_shift / (ind + 1)

        # Normalize dxy
        dxy -= np.mean(dxy, axis=0)

        # Minimum image shift
        # if min_image_shift is not None:
        #     if np.linalg.norm(dxy[ind]) < min_image_shift:
        #         dxy[ind] = 0.0




        # Apply shifts to knots
        for ind in range(self.shape[0]):
            self.knots[ind][0] += dxy[ind, 0]
            self.knots[ind][1] += dxy[ind, 1]

        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )

        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(show_knots=show_knots, title="Merged: translation", **kwargs)

            # self.plot_merged_images(show_knots=show_knots, title="Merged: translation", **kwargs)

        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: translation" for i in range(self.shape[0])],
                **kwargs,
            )

        if return_images_shift:
            images_shift = np.zeros((len(self.images), self.images[0].shape[0], self.images[0].shape[1]))
            # dxy are our shifts
            print(dxy)
            for ind in range(self.shape[0]): # over all, not 1:
                images_shift[ind] = np.roll(np.pad(self.images[ind].array, ((max_image_shift, max_image_shift), (max_image_shift, max_image_shift)), mode = 'median'), dxy[ind], axis = (0,1))[max_image_shift:-max_image_shift,max_image_shift:-max_image_shift]
            return images_shift, self.images_warped.array


        if return_images_warped:
            return self.images_warped.array

        return self

    # Affine alignment
    def align_affine(
        self,
        step: float = 0.01,
        num_tests: int = 9,
        refine: bool = True,
        upsample_factor: int = 8,
        max_image_shift: float | None = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Estimate affine drift from the first 2 images.
        """

        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()

        if num_tests % 2 == 0:
            raise ValueError("num_tests should be odd.")

        # Potential drift vectors
        vec = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
        xx, yy = np.meshgrid(vec, vec, indexing="ij")
        keep = xx**2 + yy**2 <= (num_tests / 2) ** 2
        dxy = (
            np.vstack(
                (
                    xx[keep],
                    yy[keep],
                )
            ).T
            * step
        )
        # print("dxy total shape:",dxy.shape)
        # Measure cost function for linear drift vectors
        cost = np.zeros(dxy.shape[0])
        for a0 in tqdm(range(dxy.shape[0]), desc="Solving affine drift"):
            # updated knots
            knot_0 = self.knots[0].copy()
            u = np.arange(knot_0.shape[1]) - (knot_0.shape[1] - 1) / 2
            knot_0[0] += dxy[a0, 0] * u[:, None]
            knot_0[1] += dxy[a0, 1] * u[:, None]

            knot_1 = self.knots[1].copy()
            u = np.arange(knot_1.shape[1]) - (knot_1.shape[1] - 1) / 2
            knot_1[0] += dxy[a0, 0] * u[:, None]
            knot_1[1] += dxy[a0, 1] * u[:, None]

            im0, w0 = self.interpolator[0].warp_image(
                self.images[0].array,
                knot_0,
            )
            im1, w1 = self.interpolator[1].warp_image(
                self.images[1].array,
                knot_1,
            )
            # Cross correlation alignment
            shifts, image_shift = cross_correlation_shift(
                im0,
                im1,
                upsample_factor=upsample_factor,
                fft_input=False,
                fft_output=False,
                return_shifted_image=True,
                max_shift=max_image_shift,
            )
            # print(a0, shifts)

            cost[a0] = np.mean(np.abs(im0 - image_shift))
            # plt.figure()
            # plt.subplot(131)
            # plt.imshow(im0)
            # plt.axis('off')
            # plt.title(str(dxy[a0, 0]))
            # plt.subplot(132)
            # plt.imshow(image_shift)
            # plt.axis('off')
            # plt.title(str(dxy[a0, 1]))
            # plt.subplot(133)
            # plt.imshow(np.abs(im0 - image_shift))
            # plt.axis('off')
            # plt.title(str(np.round(cost[a0], 2)))
        # update all knots
        ind = np.argmin(cost)
        for a0 in range(self.shape[0]):
            u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
            self.knots[a0][0] += dxy[ind, 0] * u[:, None]
            self.knots[a0][1] += dxy[ind, 1] * u[:, None]
        print(dxy[ind])
        # Regenerate images
        for ind in range(self.shape[0]):
            # print("a0:",a0)
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )
        # for ind in range(self.shape[0]):
        #     self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
        #         ind
        #     ].warp_image(
        #         self.images[ind].array,
        #         self.knots[ind],
        #     )
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(self.images_warped.array[0])
        # plt.subplot(122)
        # plt.imshow(self.images_warped.array[1])
        # Translation alignment
        self.align_translation(
            max_image_shift=max_image_shift,
            show_images=False,
            show_merged=False,
            show_knots=False,
        )

        # Error tracking
        self.calculate_error(1)

        if refine:
            # Potential drift vectors
            dxy /= num_tests - 1

            # Measure cost function
            cost = np.zeros(dxy.shape[0])
            for a0 in tqdm(range(dxy.shape[0]), desc="Refining affine drift"):
                # updated knots

                knot_0 = self.knots[0].copy()
                u = np.arange(knot_0.shape[1]) - (knot_0.shape[1] - 1) / 2
                knot_0[0] += dxy[a0, 0] * u[:, None]
                knot_0[1] += dxy[a0, 1] * u[:, None]

                knot_1 = self.knots[1].copy()
                u = np.arange(knot_1.shape[1]) - (knot_1.shape[1] - 1) / 2
                knot_1[0] += dxy[a0, 0] * u[:, None]
                knot_1[1] += dxy[a0, 1] * u[:, None]

                im0, w0 = self.interpolator[0].warp_image(
                    self.images[0].array,
                    knot_0,
                )
                im1, w1 = self.interpolator[1].warp_image(
                    self.images[1].array,
                    knot_1,
                )
                # Cross correlation alignment
                shifts, image_shift = cross_correlation_shift(
                    im0,
                    im1,
                    upsample_factor=upsample_factor,
                    fft_input=False,
                    fft_output=False,
                    return_shifted_image=True,
                    max_shift=max_image_shift,
                )
                cost[a0] = np.mean(np.abs(im0 - image_shift))

            # update all knots
            ind = np.argmin(cost)
            for a0 in range(self.shape[0]):
                u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
                self.knots[a0][0] += dxy[ind, 0] * u[:, None]
                self.knots[a0][1] += dxy[ind, 1] * u[:, None]
            print(dxy[ind])
        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )

        # Translation alignment
        self.align_translation(
            max_image_shift=max_image_shift,
            show_images=False,
            show_merged=False,
            show_knots=False,
        )

        # Error tracking
        self.calculate_error(1)

        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: affine",
                **kwargs,
            )
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: affine" for i in range(self.shape[0])],
                **kwargs,
            )

        return self






    # Affine alignment
    def align_affine_2(
        self,
        step: float = 0.01,
        num_tests: int = 9,
        refine: bool = True,
        upsample_factor: int = 8,
        max_image_shift: float | None = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Estimate affine drift from the first 2 images.
        """

        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()

        if num_tests % 2 == 0:
            raise ValueError("num_tests should be odd.")

        # Potential drift vectors
        vec = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
        xx, yy = np.meshgrid(vec, vec, indexing="ij")
        keep = xx**2 + yy**2 <= (num_tests / 2) ** 2
        dxy = (
            np.vstack(
                (
                    xx[keep],
                    yy[keep],
                )
            ).T
            * step
        )

        # Affine drift refinement
        self.affine_cost_list = []
        with WorkerPool(n_jobs = 2) as pool:
            def cost_affine(dxy):
                def interpolate_one_image(image_index):
                    knot = self.knots[image_index].copy()
                    u = np.arange(knot.shape[1]) - (knot.shape[1] - 1) / 2
                    knot[0] += dxy[0] * u[:, None]
                    knot[1] += dxy[1] * u[:, None]
                    im0, w0 = self.interpolator[image_index].warp_image(
                        self.images[image_index].array,
                        knot,
                    )
                    return im0
                mpire_result = pool.map(interpolate_one_image, [0,1])
                im0, im1 = mpire_result[:self.shape[1],:], mpire_result[self.shape[1]:,:]
                shifts, image_shift = cross_correlation_shift(
                    im0,
                    im1,
                    upsample_factor=upsample_factor,
                    fft_input=False,
                    fft_output=False,
                    return_shifted_image=True,
                    max_shift=max_image_shift,
                )
                affine_cost = np.mean(np.abs(im0 - image_shift))
                self.affine_cost_list.append(affine_cost)
                return(affine_cost)
        import time
        tic = time.time()
        optimization_result = minimize(
            cost_affine,
            x0 = [0.0,0.0],
            method = "Powell",
            options={
                "maxiter": 50,
                "maxfev": 100,
                "xtol": 1e-3,
                "ftol": 1e-3,
            })
        toc = time.time()
        print(f"Affine elapsed time: {toc - tic:.3f} seconds")
        if not optimization_result.success:
            raise RuntimeError(
                f"Affine optimization failed: {optimization_result.message}"
            )
        dxy = optimization_result.x
        print("Affine dxy:",dxy)
        # update all knots
        for a0 in range(self.shape[0]):
            u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
            self.knots[a0][0] += dxy[0] * u[:, None]
            self.knots[a0][1] += dxy[1] * u[:, None]

        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )

        # Translation alignment
        self.align_translation(
            max_image_shift=max_image_shift,
            show_images=False,
            show_merged=False,
            show_knots=False,
        )

        # Error tracking
        self.calculate_error(1)
        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: affine",
                **kwargs,
            )
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: affine" for i in range(self.shape[0])],
                **kwargs,
            )

        return self



    # Affine alignment
    def align_affine_3(
        self,
        step: float = 0.01,
        num_tests: int = 9,
        refine: bool = True,
        upsample_factor: int = 8,
        max_image_shift: float | None = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Estimate affine drift from the first 2 images.
        """

        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()

        if num_tests % 2 == 0:
            raise ValueError("num_tests should be odd.")

        # Potential drift vectors
        vec = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
        xx, yy = np.meshgrid(vec, vec, indexing="ij")
        keep = xx**2 + yy**2 <= (num_tests / 2) ** 2
        dxy = (
            np.vstack(
                (
                    xx[keep],
                    yy[keep],
                )
            ).T
            * step
        )

        # Affine drift refinement
        self.affine_cost_list = []
        # with WorkerPool(n_jobs = 2) as pool:
        def cost_affine(dxy):
            def interpolate_one_image(image_index):
                knot = self.knots[image_index].copy()
                u = np.arange(knot.shape[1]) - (knot.shape[1] - 1) / 2
                knot[0] += dxy[0] * u[:, None]
                knot[1] += dxy[1] * u[:, None]
                im0, w0 = self.interpolator[image_index].warp_image(
                    self.images[image_index].array,
                    knot,
                )
                return im0
            # mpire_result = pool.map(interpolate_one_image, [0,1])
            # im0, im1 = mpire_result[:self.shape[1],:], mpire_result[self.shape[1]:,:]
            im0 = interpolate_one_image(0)
            im1 = interpolate_one_image(1)
            shifts, image_shift = cross_correlation_shift(
                im0,
                im1,
                upsample_factor=upsample_factor,
                fft_input=False,
                fft_output=False,
                return_shifted_image=True,
                max_shift=max_image_shift,
            )
            affine_cost = np.mean(np.abs(im0 - image_shift))
            self.affine_cost_list.append(affine_cost)
            return(affine_cost)
        import time
        tic = time.time()
        optimization_result = minimize(
            cost_affine,
            x0 = [0.0,0.0],
            method = "Powell",
            options={
                "maxiter": 50,
                "maxfev": 100,
                "xtol": 1e-3,
                "ftol": 1e-3,
            })
        toc = time.time()
        print(f"Affine elapsed time: {toc - tic:.3f} seconds")
        if not optimization_result.success:
            raise RuntimeError(
                f"Affine optimization failed: {optimization_result.message}"
            )
        dxy = optimization_result.x
        print("Affine dxy:",dxy)
        # update all knots
        for a0 in range(self.shape[0]):
            u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
            self.knots[a0][0] += dxy[0] * u[:, None]
            self.knots[a0][1] += dxy[1] * u[:, None]

        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )

        # Translation alignment
        self.align_translation(
            max_image_shift=max_image_shift,
            show_images=False,
            show_merged=False,
            show_knots=False,
        )

        # Error tracking
        self.calculate_error(1)
        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: affine",
                **kwargs,
            )
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: affine" for i in range(self.shape[0])],
                **kwargs,
            )

        return self






    # Affine alignment
    def align_affine_2(
        self,
        step: float = 0.01,
        num_tests: int = 9,
        refine: bool = True,
        upsample_factor: int = 8,
        max_image_shift: float | None = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Estimate affine drift from the first 2 images.
        """

        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()

        if num_tests % 2 == 0:
            raise ValueError("num_tests should be odd.")

        # Potential drift vectors
        vec = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
        xx, yy = np.meshgrid(vec, vec, indexing="ij")
        keep = xx**2 + yy**2 <= (num_tests / 2) ** 2
        dxy = (
            np.vstack(
                (
                    xx[keep],
                    yy[keep],
                )
            ).T
            * step
        )

        # Affine drift refinement
        self.affine_cost_list = []
        with WorkerPool(n_jobs = 2) as pool:
            def cost_affine(dxy):
                def interpolate_one_image(image_index):
                    knot = self.knots[image_index].copy()
                    u = np.arange(knot.shape[1]) - (knot.shape[1] - 1) / 2
                    knot[0] += dxy[0] * u[:, None]
                    knot[1] += dxy[1] * u[:, None]
                    im0, w0 = self.interpolator[image_index].warp_image(
                        self.images[image_index].array,
                        knot,
                    )
                    return im0
                mpire_result = pool.map(interpolate_one_image, [0,1])
                im0, im1 = mpire_result[:self.shape[1],:], mpire_result[self.shape[1]:,:]
                shifts, image_shift = cross_correlation_shift(
                    im0,
                    im1,
                    upsample_factor=upsample_factor,
                    fft_input=False,
                    fft_output=False,
                    return_shifted_image=True,
                    max_shift=max_image_shift,
                )
                affine_cost = np.mean(np.abs(im0 - image_shift))
                self.affine_cost_list.append(affine_cost)
                return(affine_cost)
        import time
        tic = time.time()
        optimization_result = minimize(
            cost_affine,
            x0 = [0.0,0.0],
            method = "Powell",
            options={
                "maxiter": 50,
                "maxfev": 100,
                "xtol": 1e-3,
                "ftol": 1e-3,
            })
        toc = time.time()
        print(f"Affine elapsed time: {toc - tic:.3f} seconds")
        if not optimization_result.success:
            raise RuntimeError(
                f"Affine optimization failed: {optimization_result.message}"
            )
        dxy = optimization_result.x
        print("Affine dxy:",dxy)
        # update all knots
        for a0 in range(self.shape[0]):
            u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
            self.knots[a0][0] += dxy[0] * u[:, None]
            self.knots[a0][1] += dxy[1] * u[:, None]

        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )

        # Translation alignment
        self.align_translation(
            max_image_shift=max_image_shift,
            show_images=False,
            show_merged=False,
            show_knots=False,
        )

        # Error tracking
        self.calculate_error(1)
        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: affine",
                **kwargs,
            )
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: affine" for i in range(self.shape[0])],
                **kwargs,
            )

        return self



    # Affine alignment
    def align_affine_3(
        self,
        step: float = 0.01,
        num_tests: int = 9,
        refine: bool = True,
        upsample_factor: int = 8,
        max_image_shift: float | None = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Estimate affine drift from the first 2 images.
        """

        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()

        if num_tests % 2 == 0:
            raise ValueError("num_tests should be odd.")

        # Potential drift vectors
        vec = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
        xx, yy = np.meshgrid(vec, vec, indexing="ij")
        keep = xx**2 + yy**2 <= (num_tests / 2) ** 2
        dxy = (
            np.vstack(
                (
                    xx[keep],
                    yy[keep],
                )
            ).T
            * step
        )

        # Affine drift refinement
        self.affine_cost_list = []
        # with WorkerPool(n_jobs = 2) as pool:
        def cost_affine(dxy):
            def interpolate_one_image(image_index):
                knot = self.knots[image_index].copy()
                u = np.arange(knot.shape[1]) - (knot.shape[1] - 1) / 2
                knot[0] += dxy[0] * u[:, None]
                knot[1] += dxy[1] * u[:, None]
                im0, w0 = self.interpolator[image_index].warp_image(
                    self.images[image_index].array,
                    knot,
                )
                return im0
            # mpire_result = pool.map(interpolate_one_image, [0,1])
            # im0, im1 = mpire_result[:self.shape[1],:], mpire_result[self.shape[1]:,:]
            im0 = interpolate_one_image(0)
            im1 = interpolate_one_image(1)
            shifts, image_shift = cross_correlation_shift(
                im0,
                im1,
                upsample_factor=upsample_factor,
                fft_input=False,
                fft_output=False,
                return_shifted_image=True,
                max_shift=max_image_shift,
            )
            affine_cost = np.mean(np.abs(im0 - image_shift))
            self.affine_cost_list.append(affine_cost)
            return(affine_cost)
        import time
        tic = time.time()
        optimization_result = minimize(
            cost_affine,
            x0 = [0.0,0.0],
            method = "Powell",
            options={
                "maxiter": 50,
                "maxfev": 100,
                "xtol": 1e-3,
                "ftol": 1e-3,
            })
        toc = time.time()
        print(f"Affine elapsed time: {toc - tic:.3f} seconds")
        if not optimization_result.success:
            raise RuntimeError(
                f"Affine optimization failed: {optimization_result.message}"
            )
        dxy = optimization_result.x
        print("Affine dxy:",dxy)
        # update all knots
        for a0 in range(self.shape[0]):
            u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
            self.knots[a0][0] += dxy[0] * u[:, None]
            self.knots[a0][1] += dxy[1] * u[:, None]

        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )

        # Translation alignment
        self.align_translation(
            max_image_shift=max_image_shift,
            show_images=False,
            show_merged=False,
            show_knots=False,
        )

        # Error tracking
        self.calculate_error(1)
        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: affine",
                **kwargs,
            )
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: affine" for i in range(self.shape[0])],
                **kwargs,
            )

        return self






    # non-rigid alignment
    def align_nonrigid_original(
        self,
        num_iterations: int = 8,
        max_optimize_iterations: int = 10,
        regularization_sigma_px: float = 16.0,
        regularization_poly_order: int = 1,
        regularization_max_image_shift_px: Optional[float] = None,
        regularization_update_step_size: Optional[float] = 0.8,
        solve_individual_rows: bool = True,
        min_image_shift: Optional[float] = None,
        max_image_shift: float | None = 32.0,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Non-rigid drift correction.
        """

        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()
        self.cost_nonrigid_list = []
        for iterations in tqdm(
            range(num_iterations),
            desc="Solving nonrigid drift",
        ):
            for ind in range(self.shape[0]):
                image_ref = np.delete(self.images_warped.array, ind, axis=0).mean(axis=0)

                knots_init = self.knots[ind]
                shape_knots = knots_init.shape

                if solve_individual_rows:
                    knots_updated = np.zeros_like(knots_init)

                    for row_ind in range(knots_init.shape[1]):
                        x0 = knots_init[:, row_ind, :].ravel()

                        def cost_function(x):
                            knots_row = x.reshape(shape_knots[0], shape_knots[2])
                            xa, ya = self.interpolator[ind].transform_rows(knots_row)
                            xf = np.clip(np.floor(xa).astype(int), 0, self.shape[1] - 2)
                            yf = np.clip(np.floor(ya).astype(int), 0, self.shape[2] - 2)
                            dx = xa - xf
                            dy = ya - yf

                            warped = (
                                image_ref[xf, yf] * (1 - dx) * (1 - dy)
                                + image_ref[xf + 1, yf] * dx * (1 - dy)
                                + image_ref[xf, yf + 1] * (1 - dx) * dy
                                + image_ref[xf + 1, yf + 1] * dx * dy
                            )

                            residual = warped - self.images[ind].array[row_ind, :]
                            # residual = warped - self.images_warped.array[ind][row_ind + self.buff_r, self.buff_c:-self.buff_c]
                            # residual = warped - warped_self
                            cost_nonrigid = np.sum(residual**2)
                            self.cost_nonrigid_list.append(cost_nonrigid)
                            return cost_nonrigid

                        # Run optimization
                        options = (
                            {"maxiter": max_optimize_iterations}
                            if max_optimize_iterations is not None
                            else {}
                        )

                        result = minimize(cost_function, x0, method="L-BFGS-B", options=options)                            
                        knots_updated[:, row_ind, :] = result.x.reshape((2, -1))

                else:
                    x0 = knots_init.ravel()

                    def cost_function(x):
                        knots = x.reshape(shape_knots)
                        xa, ya = self.interpolator[ind].transform_coordinates(knots)

                        xf = np.clip(np.floor(xa).astype(int), 0, self.shape[1] - 2)
                        yf = np.clip(np.floor(ya).astype(int), 0, self.shape[2] - 2)
                        dx = xa - xf
                        dy = ya - yf

                        warped = (
                            image_ref[xf, yf] * (1 - dx) * (1 - dy)
                            + image_ref[xf + 1, yf] * dx * (1 - dy)
                            + image_ref[xf, yf + 1] * (1 - dx) * dy
                            + image_ref[xf + 1, yf + 1] * dx * dy
                        )

                        residual = warped - self.images[ind].array
                        cost_nonrigid = np.sum(residual**2)
                        self.cost_nonrigid_list.append(cost_nonrigid)
                        return cost_nonrigid

                    # Run optimization
                    options = (
                        {"maxiter": max_optimize_iterations}
                        if max_optimize_iterations is not None
                        else {}
                    )
                    result = minimize(cost_function, x0, method="L-BFGS-B", options=options)
                    knots_updated = result.x.reshape(shape_knots)

                # apply max shift regularization if needed
                if regularization_max_image_shift_px is not None:
                    knots_shift = knots_updated - self.knots[ind]
                    knots_dist = np.sqrt(np.sum(knots_shift**2, axis=0))
                    sub = knots_dist > regularization_max_image_shift_px
                    knots_updated[0][sub] = (
                        self.knots[ind][0][sub]
                        + knots_shift[0][sub] * regularization_max_image_shift_px / knots_dist[sub]
                    )
                    knots_updated[1][sub] = (
                        self.knots[ind][1][sub]
                        + knots_shift[1][sub] * regularization_max_image_shift_px / knots_dist[sub]
                    )

                # apply smoothness regularization if needed
                if regularization_sigma_px is not None and regularization_sigma_px > 0:
                    knots_smoothed = knots_updated.copy()

                    for dim in range(knots_updated.shape[0]):
                        x = np.arange(knots_updated.shape[1])
                        for knot_ind in range(knots_updated.shape[2]):
                            y = knots_updated[dim, :, knot_ind]

                            coefs = np.polyfit(x, y, deg=regularization_poly_order)
                            trend = np.polyval(coefs, x)

                            # Remove trend, filter, add back
                            residual = y - trend
                            residual_smooth = gaussian_filter(
                                residual, sigma=regularization_sigma_px
                            )
                            knots_smoothed[dim, :, knot_ind] = residual_smooth + trend

                    knots_updated = knots_smoothed

                # Apply step size if needed
                if regularization_update_step_size is not None:
                    knots_updated = (
                        self.knots[ind]
                        + (knots_updated - self.knots[ind]) * regularization_update_step_size
                    )

                # Update knots with optimized values
                self.knots[ind] = knots_updated

            # Update images
            for ind in range(self.shape[0]):
                self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                    ind
                ].warp_image(
                    self.images[ind].array,
                    self.knots[ind],
                )

            # Translation alignment
            self.align_translation(
                min_image_shift=min_image_shift,
                max_image_shift=max_image_shift,
                show_images=False,
                show_merged=False,
                show_knots=False,
            )

            # Error tracking
            self.calculate_error(2)

        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: non-rigid",
                **kwargs,
            )

        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: non-rigid" for i in range(self.shape[0])],
                **kwargs,
            )

        return self


    # non-rigid alignment
    def align_nonrigid(
        self,
        num_iterations: int = 8,
        max_optimize_iterations: int = 10,
        regularization_sigma_px: float = 16.0,
        regularization_poly_order: int = 1,
        regularization_max_image_shift_px: Optional[float] = None,
        regularization_update_step_size: Optional[float] = 0.8,
        solve_individual_rows: bool = True,
        min_image_shift: Optional[float] = None,
        max_image_shift: float | None = 32.0,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Non-rigid drift correction.
        """

        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()
        self.cost_nonrigid_list = []
        for iterations in tqdm(
            range(num_iterations),
            desc="Solving nonrigid drift",
        ):
            for ind in range(self.shape[0]):
                image_ref = np.delete(self.images_warped.array, ind, axis=0).mean(axis=0)

                knots_init = self.knots[ind]
                shape_knots = knots_init.shape
                prev_prev_row_ind = knots_init.shape[1]//2 + ((0+1)//2)
                prev_row_ind = knots_init.shape[1]//2 + ((0+1)//2)

                if solve_individual_rows:
                    # knots_updated = np.zeros_like(knots_init)

                    pm_row_ind = 1
                    knots_updated = knots_init.copy()
                    for row_ind in range(knots_init.shape[1]):
                        row_ind_carpet = knots_init.shape[1]//2 + ((row_ind+1)//2) * pm_row_ind 
                        pm_row_ind *= -1
                        

                        # if row_ind > 1000:
                        #     print('row ind:', row_ind)
                        #     print('row ind carpet:', row_ind_carpet)
                        
                        row_ind = row_ind_carpet
                        # handle any mismatches:
                        if row_ind < 0:
                            row_ind = knots_init.shape[1]
                        if row_ind > knots_init.shape[1]:
                            row_ind = 0
                        
                        

                        # x0 = knots_init[:, row_ind, :].ravel()
                        # x0 = knots_updated[:, row_ind, :].ravel()


                        def cost_function(x):
                            knots_row = x.reshape(shape_knots[0], shape_knots[2])
                            xa, ya = self.interpolator[ind].transform_rows(knots_row)

                            # xf = np.clip(np.floor(xa).astype(int), self.buff_r, self.shape[1] - 2 - self.buff_r)
                            # yf = np.clip(np.floor(ya).astype(int), self.buff_c, self.shape[1] - 2 - self.buff_c)
                            xf = np.clip(np.floor(xa).astype(int), 0, self.shape[1] - 2)
                            yf = np.clip(np.floor(ya).astype(int), 0, self.shape[2] - 2)
                            dx = xa - xf
                            dy = ya - yf


                            # knots_row_0 = x_init.reshape(shape_knots[0], shape_knots[2])
                            # xa_0, ya_0 = self.interpolator[ind].transform_rows(knots_row_0)

                            # # xf_0 = np.clip(np.floor(xa_0).astype(int), self.buff_r, self.shape[1] - 2 - self.buff_r)
                            # # yf_0 = np.clip(np.floor(ya_0).astype(int), self.buff_c, self.shape[2] - 2  - self.buff_c)
                            # xf_0 = np.clip(np.floor(xa_0).astype(int), 0, self.shape[1] - 2)
                            # yf_0 = np.clip(np.floor(ya_0).astype(int), 0, self.shape[2] - 2)
                            # dx_0 = xa_0 - xf_0
                            # dy_0 = ya_0 - yf_0

                            warped = (
                                image_ref[xf, yf] * (1 - dx) * (1 - dy)
                                + image_ref[xf + 1, yf] * dx * (1 - dy)
                                + image_ref[xf, yf + 1] * (1 - dx) * dy
                                + image_ref[xf + 1, yf + 1] * dx * dy
                            )

                            # warped_self = (
                            #     self.images_warped.array[ind][xf_0, yf_0] * (1 - dx_0) * (1 - dy_0)
                            #     + self.images_warped.array[ind][xf_0 + 1, yf_0] * dx_0 * (1 - dy_0)
                            #     + self.images_warped.array[ind][xf_0, yf_0 + 1] * (1 - dx_0) * dy_0
                            #     + self.images_warped.array[ind][xf_0 + 1, yf_0 + 1] * dx_0 * dy_0
                            # )

                            # plt.figure()
                            # plt.imshow()
                            # if self.print_thing is True:
                                # plt.figure(figsize = (10,5), dpi = 300)
                                # plt.subplot(131)
                                # plt.imshow(self.images[0].array, cmap = 'gray')
                                # plt.title('original image 0')
                                # plt.subplot(132)
                                # plt.imshow(image_ref[128:-128, 128:-128], cmap = 'gray')
                                # plt.title('mean of other images in stack')
                                # plt.subplot(133)
                                # plt.imshow(self.images[0].array, cmap = 'gray')
                                # plt.imshow(image_ref[128:-128, 128:-128], cmap = 'magma', alpha = 0.5)


                                # just plot the current image overlays
                                # plt.figure()
                                # plt.plot(self.images_warped.array[ind][row_ind+self.buff_r, self.buff_c:-self.buff_c], label = 'ind 0', zorder = 5)
                                # plt.plot(image_ref[row_ind+self.buff_r, self.buff_c:-self.buff_c], label = 'ind 1', zorder = 4)
                                # plt.plot(gaussian_filter(self.images[ind].array[row_ind,:],10), label = 'filtered original image', zorder = 3)
                                # plt.plot(self.images[ind].array[row_ind,:], label = 'original image', zorder = 2)
                                # plt.legend()
                                # plt.savefig('nonrigid_row_access_current_alignment.png')

                                # # now see if using the coordinates generated using the knots above work
                                # row_ind_custom = xf[0,0]
                                # plt.figure()
                                # plt.plot(self.images_warped.array[ind][row_ind_custom, self.buff_c:-self.buff_c], label = 'ind 0', zorder = 5)
                                # plt.plot(image_ref[row_ind_custom, self.buff_c:-self.buff_c], label = 'ind 1', zorder = 4)
                                # plt.plot(gaussian_filter(self.images[ind].array[row_ind,:],10), label = 'filtered original image', zorder = 3)
                                # plt.plot(self.images[ind].array[row_ind,:], label = 'original image', zorder = 2)
                                # plt.plot(warped.T, label = 'warped')
                                # plt.legend()
                                # plt.savefig('nonrigid_row_access_current_alignment.png')


                                # plt.figure()
                                # plt.plot(warped.T, label = 'warped ref')
                                # plt.plot(warped_self.T, label = 'warped self')
                                # plt.legend()
                                # plt.savefig('nonrigid_warped_ref_warped_self.png')




                                # print('xf', xf)
                                # print('yf', yf)
                                # print('xf_0', xf_0)
                                # print('yf_0', yf_0)
                                # plt.figure()
                                # plt.plot(warped.T, label = 'warped')
                                # # plt.plot(self.images[ind].array[row_ind,:], label = 'original image')
                                # plt.plot(gaussian_filter(self.images[ind].array[row_ind,:],10), label = 'filtered original image')
                                # plt.legend()
                                # plt.title('row index' + str(row_ind))
                                # plt.savefig('nonrigid_internal_warped_vs_og_im_plot.png')


                                # plt.figure(figsize = (10,5), dpi = 300)
                                # plt.subplot(131)
                                # plt.imshow(self.images_warped.array[ind][self.buff_r:-self.buff_r, self.buff_c:-self.buff_c], cmap = 'gray')
                                # plt.title('original image 0')
                                # plt.subplot(132)
                                # plt.imshow(image_ref[self.buff_r:-self.buff_r, self.buff_c:-self.buff_c], cmap = 'gray')
                                # plt.title('mean of other images in stack')
                                # plt.subplot(133)
                                # plt.imshow(self.images_warped.array[ind][self.buff_r:-self.buff_r, self.buff_c:-self.buff_c], cmap = 'gray')
                                # plt.imshow(image_ref[self.buff_r:-self.buff_r, self.buff_c:-self.buff_c], cmap = 'magma', alpha = 0.5)
                                # plt.title('row ind' + str(row_ind))
                                # # plt.imshow(warped, cmap = 'gray')
                                # # plt.title('after doing sub-pixel warping..')
                                # plt.savefig('nonrigid_image_internal.png')
                                # # print('warped stack average shape', warped.shape)
                                # plt.figure(figsize=(6, 3))
                                # plt.plot(self.images[ind].array[row_ind, :], label="original image")
                                # plt.plot(self.images_warped.array[ind][row_ind + self.buff_r, self.buff_c:-self.buff_c], label="ind 0")
                                # plt.plot(warped.T, label="warped stack average")
                                # plt.legend()
                                # plt.title("Row residual after affine")
                                # plt.savefig('nonrigid_plot_internal.png')

                            #     # print(xa)
                            #     # print(xf)
                            #     print(ya)
                            #     print(yf)
                                # self.print_thing = False
                            # plt.figure()
                            # plt.plot(warped[0,:])
                            
                            # plt.figure()
                            # plt.imshow()
                            if self.print_thing is True:
                                # print(xa)
                                # print(xf)
                                print(ya)
                                print(yf)
                                self.print_thing = False
                            # plt.figure()
                            # plt.plot(warped[0,:])
                            residual = warped - self.images[ind].array[row_ind, :]
                            cost_nonrigid = np.sum(residual**2)
                            self.cost_nonrigid_list.append(cost_nonrigid)
                            return cost_nonrigid

                        # Run optimization
                        options = (
                            {"maxiter": max_optimize_iterations}
                            if max_optimize_iterations is not None
                            else {}
                        )
                        x0 = (knots_updated[:, prev_prev_row_ind, :] - knots_init[:, prev_prev_row_ind, :])*0.4 + knots_init[:, row_ind, :]
                        x0 = x0.ravel()
                        # x0 = knots_init[:, row_ind, :].ravel()
                        # x_init = knots_init[:, row_ind, :].ravel()
                        # x0 = knots_init[:, row_ind, :].ravel()

                        result = minimize(cost_function, x0, method="L-BFGS-B", options=options)
                        if self.print_thing:
                            print('prev prev',prev_prev_row_ind)
                            print('prev',prev_row_ind)
                            print('row ind',row_ind)
                            print('prev prev sol',knots_updated[:, prev_prev_row_ind, :])
                            print('prev prev init', knots_init[:, prev_prev_row_ind, :])
                            print('curr init',knots_init[:, row_ind, :])
                            # print(x0)
                            # print(knots_updated[:, prev_prev_row_ind, :] - knots_init[:, prev_prev_row_ind, :] + knots_init[:, row_ind, :])
                            # print('curr init guess new',(knots_updated[:, prev_prev_row_ind, :] - knots_init[:, prev_prev_row_ind, :] + knots_init[:, row_ind, :]).ravel())
                            print('curr init guess new',x0)
                            print('result', result.x.reshape((2, -1)))
                            if row_ind < 508:
                                self.print_thing = False
                            
                        knots_updated[:, row_ind, :] = result.x.reshape((2, -1))
                        prev_prev_row_ind = prev_row_ind
                        prev_row_ind = row_ind

                else:
                    x0 = knots_init.ravel()

                    def cost_function(x):
                        knots = x.reshape(shape_knots)
                        xa, ya = self.interpolator[ind].transform_coordinates(knots)

                        xf = np.clip(np.floor(xa).astype(int), 0, self.shape[1] - 2)
                        yf = np.clip(np.floor(ya).astype(int), 0, self.shape[2] - 2)
                        dx = xa - xf
                        dy = ya - yf

                        warped = (
                            image_ref[xf, yf] * (1 - dx) * (1 - dy)
                            + image_ref[xf + 1, yf] * dx * (1 - dy)
                            + image_ref[xf, yf + 1] * (1 - dx) * dy
                            + image_ref[xf + 1, yf + 1] * dx * dy
                        )

                        residual = warped - self.images[ind].array
                        cost_nonrigid = np.sum(residual**2)
                        self.cost_nonrigid_list.append(cost_nonrigid)
                        return cost_nonrigid

                    # Run optimization
                    options = (
                        {"maxiter": max_optimize_iterations}
                        if max_optimize_iterations is not None
                        else {}
                    )
                    result = minimize(cost_function, x0, method="L-BFGS-B", options=options)
                    knots_updated = result.x.reshape(shape_knots)






                # apply max shift regularization if needed
                if regularization_max_image_shift_px is not None:
                    knots_shift = knots_updated - self.knots[ind]
                    knots_dist = np.sqrt(np.sum(knots_shift**2, axis=0))
                    sub = knots_dist > regularization_max_image_shift_px
                    knots_updated[0][sub] = (
                        self.knots[ind][0][sub]
                        + knots_shift[0][sub] * regularization_max_image_shift_px / knots_dist[sub]
                    )
                    knots_updated[1][sub] = (
                        self.knots[ind][1][sub]
                        + knots_shift[1][sub] * regularization_max_image_shift_px / knots_dist[sub]
                    )

                # apply smoothness regularization if needed
                if regularization_sigma_px is not None and regularization_sigma_px > 0:
                    knots_smoothed = knots_updated.copy()

                    for dim in range(knots_updated.shape[0]):
                        x = np.arange(knots_updated.shape[1])
                        for knot_ind in range(knots_updated.shape[2]):
                            y = knots_updated[dim, :, knot_ind]

                            coefs = np.polyfit(x, y, deg=regularization_poly_order)
                            trend = np.polyval(coefs, x)

                            # Remove trend, filter, add back
                            residual = y - trend
                            residual_smooth = gaussian_filter(
                                residual, sigma=regularization_sigma_px
                            )
                            knots_smoothed[dim, :, knot_ind] = residual_smooth + trend

                    knots_updated = knots_smoothed

                # Apply step size if needed
                if regularization_update_step_size is not None:
                    knots_updated = (
                        self.knots[ind]
                        + (knots_updated - self.knots[ind]) * regularization_update_step_size
                    )

                # Update knots with optimized values
                self.knots[ind] = knots_updated

            # Update images
            for ind in range(self.shape[0]):
                self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                    ind
                ].warp_image(
                    self.images[ind].array,
                    self.knots[ind],
                )

            # Translation alignment
            self.align_translation(
                min_image_shift=min_image_shift,
                max_image_shift=max_image_shift,
                show_images=False,
                show_merged=False,
                show_knots=False,
            )

            # Error tracking
            self.calculate_error(2)

        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: non-rigid",
                **kwargs,
            )

        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: non-rigid" for i in range(self.shape[0])],
                **kwargs,
            )

        return self

    def generate_corrected_image(
        self,
        upsample_factor: int = 2,
        output_original_shape: bool = True,
        fourier_filter: bool = True,
        filter_midpoint: float = 0.5,
        kde_sigma: float = 0.5,
        show_image: bool = True,
        **kwargs,
    ):
        """
        Generate the final drift-corrected image after aligning a stack of input images.

        Parameters
        ----------
        upsample_factor : int, default 2
            Factor to upsample the output image for enhanced interpolation accuracy.
        output_original_shape : bool, default True
            If True, crop the output image back to the original input dimensions after processing.
        fourier_filter : bool, default True
            Whether to apply Fourier-based directional filtering to merge corrected images.
        filter_midpoint : float, default 0.5
            Midpoint for the sigmoid-based Fourier weighting filter, determining transition smoothness.
            Setting this to a low value close to 0 will include more signal but also more slow scan artifacts.
            If using 2 images at 0 and 90 degrees scan angles, any value >0.75 will be unstable.
            Only use larger values (close to 1.0) if multiple images covering many scan angles are used.
        kde_sigma : float, default 0.5
            Standard deviation for kernel density estimation used during image interpolation. Defaults
            to the object's stored kde_sigma if set to None.
        show_image : bool, default True
            Whether to display the final corrected image after processing.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function when displaying the image.

        Returns
        -------
        image_corr : Dataset2d
            The final drift-corrected output image encapsulated in a Dataset2d object.

        Notes
        -----
        - The function applies per-frame warping using knot-based interpolation and optionally
          performs directional Fourier filtering to blend multiple warped images.
        - The Fourier filter suppresses directional artifacts by weighting image contributions based
          on their scan angles, utilizing a bounded sine sigmoid for smooth transition.
        - Upsampling enhances interpolation precision but may increase computational cost.
        """

        # init
        stack_corr = np.zeros(
            (
                self.shape[0],
                np.round(self.shape[1] * upsample_factor).astype("int"),
                np.round(self.shape[2] * upsample_factor).astype("int"),
            )
        )

        if kde_sigma is None:
            kde_sigma = self.kde_sigma

        # Update images
        for ind in range(self.shape[0]):
            stack_corr[ind], _ = self.interpolator[ind].warp_image(
                self.images[ind].array,
                self.knots[ind],
                kde_sigma=kde_sigma,
                upsample_factor=upsample_factor,
            )

        if fourier_filter:
            # Apply fourier filtering
            kx = np.fft.fftfreq(stack_corr.shape[1])[:, None]
            ky = np.fft.fftfreq(stack_corr.shape[2])[None, :]
            # kr = np.sqrt(kx**2 + ky**2)
            kt = np.arctan2(ky, kx)

            stack_fft = np.fft.fft2(stack_corr)
            weights = np.zeros_like(stack_corr)

            for ind in range(stack_corr.shape[0]):
                # Calculate weights as a function of angle
                weights[ind] = np.abs(
                    np.mod((kt - self.scan_direction[ind]) / np.pi + 0.5, 1.0) - 0.5
                ) / (1 / 2)
                weights[ind][0, 0] = 1.0

                # Apply sigmoid to weighting function
                weights[ind] = bounded_sine_sigmoid(
                    weights[ind],
                    midpoint=filter_midpoint,
                )

                # Weight the fourier transformed images
                stack_fft[ind] *= weights[ind]

            weights_sum = np.sum(weights, axis=0)
            image_corr_fft = np.divide(
                np.sum(stack_fft, axis=0),
                weights_sum,
                where=weights_sum > 0.0,
            )

        else:
            image_corr_fft = np.fft.fft2(np.mean(stack_corr, axis=0))

        if output_original_shape:
            image_corr_fft = fourier_cropping(image_corr_fft, self.shape[-2:]) / upsample_factor**2

        image_corr = Dataset2d.from_array(
            np.real(np.fft.ifft2(image_corr_fft)),
            name="drift corrected image",
            origin=self.images[0].origin,
            sampling=self.images[0].sampling,
            units=self.images[0].units,
        )

        print(image_corr.array.shape)
        print(image_corr.array)

        if show_image:
            fig, ax = image_corr.show(**kwargs)

        return image_corr

    def calculate_error(
        self,
        mode,
    ):
        # Mask for error estimate
        mask = np.prod(self.weights_warped.array, axis=0)

        # Estimate current error
        images_mean = np.mean(self.images_warped.array, axis=0)
        sig_diff = np.mean(
            mask[None, :, :] * np.abs(self.images_warped.array - images_mean[None, :, :]),
            axis=(1, 2),
        ) / np.sum(mask)

        # Error vector
        error_current = np.hstack((mode, np.mean(sig_diff), sig_diff))

        # Initialize or append to error tracking array
        if not hasattr(self, "error_track"):
            self.error_track = error_current[None, :]  # initialize with first row
        else:
            self.error_track = np.vstack((self.error_track, error_current))

    def plot_transformed_images(self, show_knots: bool = True, **kwargs):
        fig, ax = show_2d(
            list(self.images_warped.array),
            **kwargs,
        )
        if show_knots:
            for a0 in range(self.shape[0]):
                x = self.knots[a0][0]
                y = self.knots[a0][1]
                ax[a0].plot(
                    y,
                    x,
                    color="r",
                )

    def plot_error(
        self,
        figsize=None,
        show_minimization = False,
        **kwargs,
    ):
        """
        Plot the convergence of the drift correction.
        """
        error_types_present = np.unique(self.error_track[:,0]).astype(int)

        stage_colors = {0: "red", 1: "blue", 2: "green"}
        stage_labels = {0: "Translation", 1: "Affine", 2: "Non-Rigid"}
        nrows = 1
        affine_in_track = 1 in error_types_present
        nonrigid_in_track = 2 in error_types_present
        do_affine_min_plot = False
        do_nonrigid_min_plot = False
        if affine_in_track and show_minimization and hasattr(self, "affine_cost_list"):
            nrows += 1
            do_affine_min_plot = True
        if nonrigid_in_track and show_minimization and hasattr(self, "nonrigid_cost_list"):
            nrows += 1
            do_nonrigid_min_plot = True
        error = self.error_track[:, 1]
        it = np.arange(error.shape[0])
        if figsize is None:
            figsize = (5, 2 * nrows)

        fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=False)
        if nrows == 1:
            axes = [axes]  # iterable


        # error_track across stages
        ax = axes[0]
        error = self.error_track[:, 1]
        it = np.arange(error.shape[0])

        # for stage in error_types_present:
        #     mask = self.error_track[:, 0] == stage
        #     ax.plot(
        #         it[mask],
        #         error[mask],
        #         marker="o",
        #         color=stage_colors.get(stage, "black"),
        #         linestyle="-",
        #         label=stage_labels.get(stage, f"Stage {stage}"),
        #         **kwargs,
        #     )

        # ax.set_xlabel("Iteration")
        # ax.set_ylabel("Mean Error [%]")
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        # ax.legend()
        zorder_plot1 = 10
        for stage in error_types_present:
            mask = self.error_track[:,0] == stage

            # Find the indices for this stage
            idx = np.where(mask)[0]

            # Extend range so line connects to previous stage (if not the very first stage)
            if len(idx) > 0 and idx[0] > 0:
                idx = np.insert(idx, 0, idx[0] - 1)

            ax.plot(
                it[idx],
                self.error_track[idx,1],
                color=stage_colors.get(stage, "black"),
                marker="o",
                linestyle="-",
                label=stage_labels[stage],
                zorder = zorder_plot1-stage
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax.legend()






        panel_idx = 1
        if do_affine_min_plot:
            ax = axes[panel_idx]
            ax.plot(
                np.arange(len(self.affine_cost_list)),
                self.affine_cost_list,
                marker="o",
                color="blue",
                linestyle="-",
                label="Affine Minimization",
            )
            ax.set_xlabel("Affine Function Evaluations")
            ax.set_ylabel("Cost")
            ax.legend()
            panel_idx += 1
        if do_nonrigid_min_plot:
            ax = axes[panel_idx]
            ax.plot(
                np.arange(len(self.cost_nonrigid_list)),
                self.cost_nonrigid_list,
                marker="o",
                color="green",
                linestyle="-",
                label="Non-Rigid Minimization",
            )
            ax.set_xlabel("Non-Rigid Function Evaluations")
            ax.set_ylabel("Cost")
            ax.legend()
        plt.tight_layout()
        return self

    # def plot_error(
    #     self,
    #     figsize=(8, 3),
    #     **kwargs,
    # ):
    #     """
    #     Plot the convergence of the drift correction.
    #     """
    #     sub = np.abs(self.error_track[:, 0] - 2) < 0.1
    #     error = self.error_track[:, 1]
    #     it = np.arange(error.shape[0])

    #     from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    #     fig, ax = plt.subplots(1, 2, figsize=figsize)
    #     color = (1, 0, 0)  # red

    #     # Plot Affine
    #     if np.any(~sub):
    #         ax[0].plot(
    #             it[~sub],
    #             100 * error[~sub],
    #             marker="o",
    #             color=color,
    #             linestyle="-",
    #             label="Affine",
    #             **kwargs,
    #         )
    #         ax[0].set_xlabel("Affine Iterations")
    #         ax[0].set_ylabel("Mean Error [%]")
    #         ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    #         ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    #     else:
    #         ax[0].axis("off")

    #     # Plot Non-Rigid
    #     if np.any(sub):
    #         first_true = np.argmax(sub)
    #         if first_true > 0:
    #             sub[first_true - 1] = True

    #         ax[1].plot(
    #             it[sub],
    #             100 * error[sub],
    #             marker="o",
    #             color=color,
    #             linestyle="-",
    #             label="Non-Rigid",
    #             **kwargs,
    #         )
    #         ax[1].set_xlabel("Non-Rigid Iterations")
    #         ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    #         ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    #     else:
    #         ax[1].axis("off")

    #     plt.tight_layout()

    #     return self

    def plot_merged_images(self, show_knots: bool = True, **kwargs):
        """
        Plot the current transformed images, with knot overlays.
        """
        fig, ax = show_2d(
            self.images_warped.array.mean(0),
            **kwargs,
        )
        if show_knots:
            for a0 in range(self.shape[0]):
                x = self.knots[a0][0]
                y = self.knots[a0][1]
                ax.plot(
                    y,
                    x,
                )



# scratch

    # Affine alignment
    def align_affine_wm(
        self,
        step: float = 0.01,
        num_tests: int = 9,
        refine: bool = True,
        upsample_factor: int = 8,
        max_image_shift: float | None = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Estimate affine drift from the first 2 images.
        """
        # WM: check for knots, if none defined, we need some
        if not hasattr(self, "knots"):
            print("\033[91mNo knots found — running .preprocess() with default settings.\033[0m")
            self.preprocess()

        # WM: the radius of the search? this should probably be renamed. The number must be odd because we are centering on a pixel.
        # could just resolve this automatically + include a message
        if num_tests % 2 == 0:
            # raise ValueError("num_tests should be odd.")
            print("num_tests should be odd, decreasing num_tests to", num_tests-1)
            num_tests -= 1
        
        # WM: Build an array with 0 at the center, use this to make an x and y meshgrid (square aspect ratio).
        # make r**2, essentially, and threshold it by the radius of num_tests/2. This will make a circular binary mask.
        # then build an array of coordinates, where the first index gives a pair, and the second gives x or y
        # Potential drift vectors
        vec = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
        xx, yy = np.meshgrid(vec, vec, indexing="ij")
        keep = xx**2 + yy**2 <= (num_tests / 2) ** 2
        dxy = (
            np.vstack(
                (
                    xx[keep],
                    yy[keep],
                )
            ).T
            * step
        )

        # plt.figure()
        # plt.subplot(121)
        # plt.scatter(dxy[0], dxy[1])
        # print(dxy)
        # plt.subplot(122)
        # plt.imshow(dxy[1])
        # instantiate cost variable that has an entry for each set of coordinates
        # start a for loop that goes through each drift vector and measures the cost.
        # knots, at this point, is a list that is equal to the number of images
        # when we index the list self.knots, we are picking the 0th and 1st image in the list
        # the idea is that we get the shape of the image (ideally the same size... this code may accept differing sizes though) and use this to generate the u vector.
        # the u vector is a vector with length equal to the number of rows in the given image, and the origin is at the center of u.
        # we index into dxy to get a single row column pair of coordinates. we then multiply this by the entire u vector
        # this creates a stretching/compression along the row direction (rows further apart or closer together).
        # we add this to the knot, so that we shift the row of pixels up or down accordingly.
        # this is also applied to the columns of pixels. this means that each knot can stretch in both x and y
        # apply this so the knots that are from the second image. Though the code does not explicitly require it, it would be nice if we had a way to ensure that the image we use has a non-parallel scan direction.
        # we then call warp_image from the interpolator. the interpolator has the same shape as the input images, but uses the coordinates from knot_0 to interpolate the pixel values to the new pixel coordinates
        # this has wrap around, so actually it is a good idea to have a padding region (I believe this is already the case)
        # Measure cost function for linear drift vectors
        # print(self.pad_value)
        im1_pad = np.ones(self.shape[1:]) * self.pad_value[1]
        pad_pixels = int((self.shape[1] - self.images[0].shape[0])/2)
        self.pad_pixels = pad_pixels
        # print("Pad pixels:",pad_pixels)
        im1_pad[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels] = self.images[1].array
        im1_pad_filtered = gaussian_filter(im1_pad, self.kde_sigma)
        cost = np.zeros(dxy.shape[0])
        for a0 in tqdm(range(dxy.shape[0]), desc="Solving affine drift"):
            # updated knots
            knot_0 = self.knots[0].copy()
            u = np.arange(knot_0.shape[1]) - (knot_0.shape[1] - 1) / 2
            # print("Knot 0 before:", knot_0)
            knot_0[0] += dxy[a0, 0] * u[:, None]
            knot_0[1] += dxy[a0, 1] * u[:, None]
            # print("Knot 0 after:", knot_0)

            # print(dxy[a0])
            knot_1 = self.knots[1].copy()
            # knot_1_p = self.knots[1].copy()
            u = np.arange(knot_1.shape[1]) - (knot_1.shape[1] - 1) / 2
            knot_1[0] += dxy[a0, 0] * u[:, None]
            knot_1[1] += dxy[a0, 1] * u[:, None]
            # print("knot 1 shape:",knot_1.shape)
            # knot_1 = self.knots[1].copy()
            # u = np.arange(knot_1.shape[1]) - (knot_1.shape[1] - 1) / 2
            # knot_1[0] += dxy[a0, 0] * u[:, None]
            # knot_1[1] += dxy[a0, 1] * u[:, None]
            # knot_all = np.concatenate((knot_0,knot_1))
            im0, w0 = self.interpolator[0].warp_image_wm(
                self.images[0].array,
                knot_0,
                # knot_1-self.knots[1].copy(),
                knot_1,#-self.knots[1].copy(),
                dxy[a0],
                pad_pixels,
            )
            # im1, w1 = self.interpolator[1].warp_image(
            #     self.images[1].array,
            #     knot_1_p,
            # )
            
            # Cross correlate the images to calculate the error that they have with respect to each other.
            # Cross correlation alignment
            pad_pixels = int((self.shape[1] - self.images[0].shape[0])/2)
            shifts, image_shift = cross_correlation_shift(
                im0,
                im1_pad_filtered,
                upsample_factor=upsample_factor,
                fft_input=False,
                fft_output=False,
                return_shifted_image=True,
                max_shift=max_image_shift,
            )
            cost[a0] = np.mean(np.abs(im0 - image_shift))
            plt.figure()
            plt.subplot(131)
            plt.imshow(im0)
            plt.axis('off')
            plt.title(str(dxy[a0, 0]))
            plt.subplot(132)
            plt.imshow(image_shift)
            plt.axis('off')
            plt.title(str(dxy[a0, 1]))
            plt.subplot(133)
            plt.imshow(np.abs(im0 - image_shift))
            plt.axis('off')
            plt.title(str(np.round(cost[a0], 2)))







            # knot_0 = self.knots[0].copy()
            # u = np.arange(knot_0.shape[1]) - (knot_0.shape[1] - 1) / 2
            # knot_0[0] += dxy[a0, 0] * u[:, None]
            # knot_0[1] += dxy[a0, 1] * u[:, None]

            # knot_1 = self.knots[1].copy()
            # u = np.arange(knot_1.shape[1]) - (knot_1.shape[1] - 1) / 2
            # knot_1[0] -= dxy[a0, 0] * u[:, None]
            # knot_1[1] -= dxy[a0, 1] * u[:, None]

            # knot_all = np.concatenate((knot_0,knot_1))
            # im0, w0 = self.interpolator[0].warp_image(
            #     self.images[0].array,
            #     knot_all,
            # )
            # # Cross correlation alignment
            # pad_pixels = int((self.shape[1] - self.images[0].shape[0])/2)
            # shifts, image_shift = cross_correlation_shift(
            #     im0,
            #     im1_pad_filtered,
            #     upsample_factor=upsample_factor,
            #     fft_input=False,
            #     fft_output=False,
            #     return_shifted_image=True,
            #     max_shift=max_image_shift,
            # )




        # the above loop operated for all of the possible shifts in the search region, so now there is a large vector of costs.
        # find the cost minimizer and index into dxy. apply these changes to the actual knots (not a copy, like before)
        # update all knots
        ind = np.argmin(cost)
        print(dxy[ind])
        for a0 in range(self.shape[0]):
            u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
            self.knots[a0][0] += dxy[ind, 0] * u[:, None]
            self.knots[a0][1] += dxy[ind, 1] * u[:, None]

        # we have dumped the interpolations from before, which adds one more interpolation.
        # in principle, we could have saved the best one as we went instead of regenerating it, but whatever.
        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )

        # we do a translation alignment step which requires a cross correlation... would we want to do this first? or no
        # 
        # Translation alignment
        self.align_translation(
            max_image_shift=max_image_shift,
            show_images=False,
            show_merged=False,
            show_knots=False,
        )

        # calculate the error between the images. the 1 is not a computational parameter, but is rather a signal to the user or other functions that this measurement is post affine.
        # Error tracking
        self.calculate_error(1)

        # now comes the second refinement step. The dxys are rescaled by the original radius. They already were scaled down by the "step" parameter, but now they are futher scaled down.
        # this is really a fine tuning step for the affine alignment. all of the code between the hashes is a direct copy from above
        # Affine drift refinement
        if refine:
            # Potential drift vectors
            dxy /= num_tests - 1
            ######
            # Measure cost function
            cost = np.zeros(dxy.shape[0])
            for a0 in tqdm(range(dxy.shape[0]), desc="Refining affine drift"):
                # updated knots

                knot_0 = self.knots[0].copy()
                u = np.arange(knot_0.shape[1]) - (knot_0.shape[1] - 1) / 2
                knot_0[0] += dxy[a0, 0] * u[:, None]
                knot_0[1] += dxy[a0, 1] * u[:, None]


                knot_1 = self.knots[1].copy()
                u = np.arange(knot_1.shape[1]) - (knot_1.shape[1] - 1) / 2
                knot_1[0] += dxy[a0, 0] * u[:, None]
                knot_1[1] += dxy[a0, 1] * u[:, None]

                im0, w0 = self.interpolator[0].warp_image(
                    self.images[0].array,
                    knot_0,
                )
                im1, w1 = self.interpolator[1].warp_image(
                    self.images[1].array,
                    knot_1,
                )
                # Cross correlation alignment
                shifts, image_shift = cross_correlation_shift(
                    im0,
                    im1,
                    upsample_factor=upsample_factor,
                    fft_input=False,
                    fft_output=False,
                    return_shifted_image=True,
                    max_shift=max_image_shift,
                )
                cost[a0] = np.mean(np.abs(im0 - image_shift))

            # update all knots
            ind = np.argmin(cost)
            # print(dxy[ind])
            for a0 in range(self.shape[0]):
                u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
                self.knots[a0][0] += dxy[ind, 0] * u[:, None]
                self.knots[a0][1] += dxy[ind, 1] * u[:, None]

        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
                ind
            ].warp_image(
                self.images[ind].array,
                self.knots[ind],
            )

        # Translation alignment
        self.align_translation(
            max_image_shift=max_image_shift,
            show_images=False,
            show_merged=False,
            show_knots=False,
        )

        # Error tracking
        self.calculate_error(1)
        ######
        # above, the second and final affine error is recorded.

        # if desired, plot the images, nothing too fancy.
        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: affine",
                **kwargs,
            )
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: affine" for i in range(self.shape[0])],
                **kwargs,
            )

        return self


class DriftInterpolator:
    def __init__(
        self,
        input_shape,
        output_shape,
        scan_fast,
        scan_slow,
        pad_value,
        kde_sigma,
        scan_2_fast,
        input_2_shape,
        rot,
        rot_2,
    ):
        self.input_shape = input_shape
        self.input_2_shape = input_2_shape
        self.output_shape = output_shape
        self.scan_fast = scan_fast
        self.scan_2_fast = scan_2_fast
        self.scan_slow = scan_slow
        self.pad_value = pad_value
        self.kde_sigma = kde_sigma
        self.rot = rot
        self.rot_2 = rot_2
        self.print_thing = True
        self.rows_input = np.arange(input_shape[0])
        self.cols_input = np.arange(input_shape[1])
        self.u = np.linspace(0, 1, input_shape[1])
        self.u_2 = np.linspace(0, 1, input_2_shape[1])

    def transform_rows(
        self,
        knots_row: NDArray,
    ):
        num_knots = knots_row.shape[-1]
        basis = np.linspace(0, 1, num_knots)

        if num_knots == 1:
            xa = knots_row[0] + self.u[None, :] * self.scan_fast[0] * (self.input_shape[0] - 1)
            ya = knots_row[1] + self.u[None, :] * self.scan_fast[1] * (self.input_shape[1] - 1)
        elif num_knots == 2:
            xa = interp1d(basis, knots_row[0], kind="linear", assume_sorted=True)(self.u)
            ya = interp1d(basis, knots_row[1], kind="linear", assume_sorted=True)(self.u)
        else:
            kind = "quadratic" if num_knots == 3 else "cubic"
            xa = interp1d(
                basis,
                knots_row[0],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)
            ya = interp1d(
                basis,
                knots_row[1],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)

        return xa, ya

    def transform_rows_wm(
        self,
        knots_row: NDArray,
        knots_col: NDArray,
        pad_pixels:int,
        dxy,
    ):
        num_knots = knots_row.shape[-1]
        basis = np.linspace(0, 1, num_knots)


        center_x = (self.input_shape[0]+pad_pixels*2)/2
        center_y = (self.input_shape[1]+pad_pixels*2)/2
        # print("input shape:",self.input_shape)
        # print("input shape[0]:",self.input_shape[0])
        # print("pad_pixels:",pad_pixels)
        # print("center_x shape:",center_x.shape)
        # print("center_y shape:",center_y.shape)

        if num_knots == 1:
            xa = knots_row[0] + self.u[None, :] * self.scan_fast[0] * (self.input_shape[0] - 1)
            xa_col = knots_col[0] + self.u_2[None, :] * self.scan_2_fast[0] * (self.input_2_shape[0] - 1)
            xa = np.squeeze(xa)
            xa_col = np.squeeze(xa_col)
            xa_col = xa_col - center_x
            xa_shift = xa - center_x

            ya = knots_row[1] + self.u[None, :] * self.scan_fast[1] * (self.input_shape[1] - 1)
            ya_col = knots_col[1] + self.u_2[None, :] * self.scan_2_fast[1] * (self.input_2_shape[1] - 1)
            ya_col = np.squeeze(ya_col)
            ya_col = ya_col - center_y
            ya = np.squeeze(ya)
            ya_shift = ya - center_y
            
            rotate_im_angle = self.rot - self.rot_2
            rotation_arr = np.array([[np.cos(rotate_im_angle), -np.sin(rotate_im_angle)], [np.sin(rotate_im_angle), np.cos(rotate_im_angle)]])
            coords = np.stack([xa_shift.ravel(), ya_shift.ravel()], axis=0)  # shape (2, N)
            rot_coords = rotation_arr @ coords
            xa_rot = rot_coords[0, :].reshape(xa.shape)
            ya_rot = rot_coords[1, :].reshape(ya.shape)



            # xa = xa_rot + center_x - (knots_col[0])[None,:]
            # ya = ya_rot + center_y - (knots_col[1].T)[None,:]
            theta = np.radians(self.rot_2)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            drift_vector_rot = R @ dxy
            num_rows = self.input_shape[0]
            num_cols = self.input_shape[1]
            row_pixels = np.linspace(-(num_rows - 1) / 2, (num_rows - 1) / 2, num_rows)
            col_pixels = np.linspace(-(num_cols - 1) / 2, (num_cols - 1) / 2, num_cols)
            rrow, ccol = np.meshgrid(row_pixels, col_pixels, indexing = 'ij')
            # xa = xa_rot + center_x #- (drift_vector_rot[0] * (self.u-0.5)[None, :] * (self.input_shape[0] - 1))
            # ya = ya_rot + center_y #- (drift_vector_rot[1] * (self.u-0.5)[None, :] * (self.input_shape[0] - 1))
            # plt.figure()
            # plt.subplot(141)
            # plt.imshow(rrow * drift_vector_rot[0])
            # plt.colorbar()
            # plt.subplot(142)
            # plt.imshow(ccol)
            # plt.colorbar()
            # plt.subplot(143)
            # plt.imshow(xa_rot)
            # plt.colorbar()
            # plt.subplot(144)
            # plt.imshow(ya_rot)
            # plt.colorbar()

            # xa = xa_rot + center_x + xa_col
            # ya = ya_rot + center_y + xa_col
            # xa = xa_rot + center_x + (drift_vector_rot[0] * xa_col)
            # ya = ya_rot + center_y + (drift_vector_rot[1] * xa_col)
            xa = xa_rot + center_x + (drift_vector_rot[1] * ccol)
            ya = ya_rot + center_y + (drift_vector_rot[0] * ccol)

            # if self.print_thing is True:
            #     plt.figure()
            #     plt.subplot(121)
            #     plt.imshow(ccol)
            #     plt.subplot(122)
            #     plt.imshow(xa_col)
            #     self.print_thing = False
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(xa)
            # plt.subplot(122)
            # plt.imshow(ya)
        elif num_knots == 2:
            xa = interp1d(basis, knots_row[0], kind="linear", assume_sorted=True)(self.u)
            ya = interp1d(basis, knots_row[1], kind="linear", assume_sorted=True)(self.u)
        else:
            kind = "quadratic" if num_knots == 3 else "cubic"
            xa = interp1d(
                basis,
                knots_row[0],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)
            ya = interp1d(
                basis,
                knots_row[1],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)

        return xa, ya

    def transform_rows_wm_2(
        self,
        knots_row: NDArray,
        knots_col: NDArray,
    ):
        num_knots = knots_row.shape[-1]
        basis = np.linspace(0, 1, num_knots)


        if num_knots == 1:
            xa = knots_row[0] + self.u[None, :] * self.scan_fast[0] * (self.input_shape[0] - 1) - knots_col[0, :, 0][None, :]
            ya = knots_row[1] + self.u[None, :] * self.scan_fast[1] * (self.input_shape[1] - 1) - knots_col[1, :, 0][None, :]

        elif num_knots == 2:
            xa = interp1d(basis, knots_row[0], kind="linear", assume_sorted=True)(self.u)
            ya = interp1d(basis, knots_row[1], kind="linear", assume_sorted=True)(self.u)
        else:
            kind = "quadratic" if num_knots == 3 else "cubic"
            xa = interp1d(
                basis,
                knots_row[0],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)
            ya = interp1d(
                basis,
                knots_row[1],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)

        return xa, ya









    def transform_coordinates(
        self,
        knots: NDArray,
    ):
        num_knots = knots.shape[-1]

        if num_knots == 1:
            # vectorized version for speed
            xa, ya = self.transform_rows(knots)
        else:
            xa = np.zeros(self.input_shape)
            ya = np.zeros(self.input_shape)
            for i in range(self.input_shape[0]):
                xa[i], ya[i] = self.transform_rows(knots[:, i])

        return xa, ya

    def transform_coordinates_wm(
        self,
        knots_0: NDArray,
        knots_1: NDArray,
        pad_pixels:int,
        dxy,
    ):
        num_knots = knots_0.shape[-1]

        if num_knots == 1:
            # vectorized version for speed
            xa, ya = self.transform_rows_wm(knots_0, knots_1, pad_pixels, dxy)
        else:
            xa = np.zeros(self.input_shape)
            ya = np.zeros(self.input_shape)
            for i in range(self.input_shape[0]):
                xa[i], ya[i] = self.transform_rows(knots_0[:, i],knots_1[:, i])

        return xa, ya

    def warp_image_wm(
        self,
        image: NDArray,
        knots_0: NDArray,  # shape: (2, rows, num_knots)
        knots_1: NDArray,  # shape: (2, rows, num_knots)
        dxy,
        pad_pixels: int,
        kde_sigma=None,
        output_shape=None,
        pad_value=None,
        upsample_factor=None,
    ) -> NDArray:
        xa, ya = self.transform_coordinates_wm(
            knots_0, knots_1, pad_pixels, dxy
        )

        if kde_sigma is None:
            kde_sigma = self.kde_sigma

        if output_shape is None:
            output_shape = self.output_shape

        if pad_value is None:
            pad_value = self.pad_value

        if upsample_factor is None:
            upsample_factor = 1.0

        image_interp, weight_interp = bilinear_kde(
            xa=xa * upsample_factor,  # rows
            ya=ya * upsample_factor,  # cols
            values=image,
            output_shape=np.round(np.array(output_shape) * upsample_factor).astype("int"),
            kde_sigma=kde_sigma * upsample_factor,
            pad_value=pad_value,
            return_pix_count=True,
        )

        return image_interp, weight_interp
    
    def warp_image(
        self,
        image: NDArray,
        knots: NDArray,  # shape: (2, rows, num_knots)
        kde_sigma=None,
        output_shape=None,
        pad_value=None,
        upsample_factor=None,
    ) -> NDArray:
        xa, ya = self.transform_coordinates(
            knots,
        )

        if kde_sigma is None:
            kde_sigma = self.kde_sigma

        if output_shape is None:
            output_shape = self.output_shape

        if pad_value is None:
            pad_value = self.pad_value

        if upsample_factor is None:
            upsample_factor = 1.0

        image_interp, weight_interp = bilinear_kde(
            xa=xa * upsample_factor,  # rows
            ya=ya * upsample_factor,  # cols
            values=image,
            output_shape=np.round(np.array(output_shape) * upsample_factor).astype("int"),
            kde_sigma=kde_sigma * upsample_factor,
            pad_value=pad_value,
            return_pix_count=True,
        )

        return image_interp, weight_interp
    
    def set_pad_value(
            self,
            pad_value,
    ):
        self.pad_value = pad_value


def bounded_sine_sigmoid(x, midpoint=0.5, width=1.0):
    """
    Piecewise bounded sigmoid: zero, raised sine squared, one.

    Parameters
    ----------
    x : array-like, shape (...,)
        Input values in [0, 1].
    midpoint : float
        Center of the sigmoid transition.
    width : float
        Width of the sigmoid (range over which it ramps from 0 to 1).
    Returns
    -------
    y : array-like
        Output in [0, 1], same shape as x.
    """
    x = np.asarray(x)
    # Truncate width if midpoint too close to edge
    left_max = midpoint - width / 2
    right_min = midpoint + width / 2
    if left_max < 0:
        width = 2 * midpoint  # can't start below zero
    if right_min > 1:
        width = 2 * (1 - midpoint)  # can't extend above one
    # Recalculate edges
    left = midpoint - width / 2
    right = midpoint + width / 2

    y = np.zeros_like(x, dtype=float)
    in_band = (x >= left) & (x <= right)
    # Map [left, right] to [0, pi/2]
    t = (x[in_band] - left) / width  # goes from 0 to 1
    y[in_band] = np.sin(t * np.pi / 2) ** 2
    y[x > right] = 1.0
    return y


# sandbox


        # # print("scan fast:",self.scan_fast)
        # # print("scan 2 fast:",self.scan_2_fast)
        # # print(self.u)
        # # print(self.u.shape)
        # if num_knots == 1:
        #     xa = knots_row[0] + self.u[None, :] * self.scan_fast[0] * (self.input_shape[0] - 1)
        #     # xa = knots_row[0] + self.u[None, :] * (self.input_shape[0] - 1)
        #     # xa -= knots_col[0] + self.v[None,:] * self.scan_2_fast[0] * (self.input_shape[1] - 1)
        #     # print(xa[:,0].shape)
        #     # print(knots_col[0].shape)
        #     # print("knots col:", knots_col)
        #     # print("knots col shape:", knots_col.shape)
        #     # print("col shift shape:", (20*(1/xa.shape[1])* (10-xa.shape[1]/2)))
        #     # print(self.scan_2_fast[0])
        #     for column in range(xa.shape[1]):
        #         # xa[:,column] -= knots_col[0,0]# + self.u * self.scan_2_fast[0] * (self.input_shape[1] - 1)
        #         # xa[:,column] -= 20*(1/xa.shape[1])* (column-xa.shape[1]/2) #self.u * self.scan_2_fast[0] * (self.input_shape[1] - 1)
        #         xa[:,column] -= self.scan_2_fast[0] * knots_col[0,column,0]
        #         # xa[:,column] -= self.scan_2_fast[0] * knots_col[1,column,0]
        #         # print(knots_col[0,column,0]) 
        #     ya = knots_row[1] + self.u[None, :] * self.scan_fast[1] * (self.input_shape[1] - 1)
        #     for row in range(ya.shape[0]):
        #         xa[:,row] -= self.scan_2_fast[1] * knots_col[1,row,0]
        #     # ya = knots_row[1] + self.u[None, :] * (self.input_shape[1] - 1)
        #     # print(ya[0,:].shape)
        #     # for row in range(ya.shape[0]):
        #     #     ya[row,:] -= knots_col[1,0] #+ self.u * self.scan_2_fast[1] * (self.input_shape[1] - 1)
        #     # print(xa[-1,-1])
        #     # print(ya[-1,-1])
        #     # print(self.u)
        #     # print(self.scan_fast[0])
        #     # plt.figure()
        #     # plt.subplot(121)
        #     # plt.imshow(xa)
        #     # plt.subplot(122)
        #     # plt.imshow(ya)
            
            
            

    # def transform_rows_wm(
    #     self,
    #     knots_row: NDArray,
    #     knots_col: NDArray,
    # ):
    #     num_knots = knots_row.shape[-1]
    #     basis = np.linspace(0, 1, num_knots)

    #     # center_x = self.input_shape[0]/2
    #     # center_y = self.input_shape[1]/2
    #     # print("input shape:",self.input_shape)
    #     # center_x = 512/2
    #     center_x = (512+128)/2
    #     center_y = (512+128)/2
        

    #     if num_knots == 1:
    #         # xa = knots_row[0] + self.u[None, :] * self.scan_fast[0] * (self.input_shape[0] - 1) - knots_col[0, :, 0][None, :]
    #         # xa = knots_row[0] + self.u[None, :] * self.scan_fast[0] * (self.input_shape[0] - 1) - knots_col[0, :, 0][None, :] - self.u_2[None, :] * self.scan_2_fast[0] * (self.input_2_shape[0] - 1)
    #         # xa = knots_row[0] + self.u[None, :] * self.scan_fast[0] * (self.input_shape[0] - 1) - knots_col[0] - self.u_2[None, :] * self.scan_2_fast[0] * (self.input_2_shape[0] - 1)
    #         xa = knots_row[0] + self.u[None, :] * self.scan_fast[0] * (self.input_shape[0] - 1)  #- self.u_2[None, :] * self.scan_2_fast[0] * (self.input_2_shape[0] - 1)
    #         xa = np.squeeze(xa)
    #         xa_shift = xa - center_x
            
    #         # plt.figure()
    #         # plt.subplot(121)
    #         # plt.imshow(knots_row[0] + self.u[None, :] * self.scan_fast[0] * (self.input_shape[0] - 1))
    #         # plt.subplot(122)
    #         # plt.imshow(- knots_col[0] - self.u_2[None, :] * self.scan_2_fast[0] * (self.input_2_shape[0] - 1))
    #         # for column in range(xa.shape[1]):
    #         #     # xa[:,column] -= knots_col[0,column,0]
    #         #     xa[:,column] -= self.scan_2_fast[0] * knots_col[0,column,0]

    #         # plt.figure()
    #         # plt.plot(knots_row[0,:,0], label = "knots row 0")
    #         # plt.plot(knots_row[1,:,0], label = "knots row 1")
    #         # plt.plot(knots_col[0,:,0], label = "knots col 0")
    #         # plt.plot(knots_col[1,:,0], label = "knots col 1")
    #         # plt.legend()
    #         # ya = knots_row[1] + self.u[None, :] * self.scan_fast[1] * (self.input_shape[1] - 1) - knots_col[1, :, 0][None, :]
    #         # ya = knots_row[1] + self.u[None, :] * self.scan_fast[1] * (self.input_shape[1] - 1) - knots_col[1, :, 0][None, :]  - self.u_2[None, :] * self.scan_2_fast[1] * (self.input_2_shape[1] - 1)
    #         # ya = knots_row[1] + self.u[None, :] * self.scan_fast[1] * (self.input_shape[1] - 1) - knots_col[1]  - self.u_2[None, :] * self.scan_2_fast[1] * (self.input_2_shape[1] - 1)
    #         ya = knots_row[1] + self.u[None, :] * self.scan_fast[1] * (self.input_shape[1] - 1)# - self.u_2[None, :] * self.scan_2_fast[1] * (self.input_2_shape[1] - 1)
    #         ya = np.squeeze(ya)
    #         ya_shift = ya - center_y
            
    #         rotation_arr = np.array([[np.cos(-self.rot_2), -np.sin(-self.rot_2)], [np.sin(-self.rot_2), np.cos(-self.rot_2)]])
    #         coords = np.stack([xa_shift.ravel(), ya_shift.ravel()], axis=0)  # shape (2, N)
    #         rot_coords = rotation_arr @ coords
    #         xa_rot = rot_coords[0, :].reshape(xa.shape)
    #         ya_rot = rot_coords[1, :].reshape(ya.shape)
    #         xa = xa_rot + center_x - (knots_col[0].T)[None,:]
    #         ya = ya_rot + center_y - (knots_col[1])[None,:]
            
    #         # plt.figure()
    #         # plt.subplot(121)
    #         # plt.imshow(xa_rot)
    #         # plt.subplot(122)
    #         # plt.imshow((xa_rot - knots_col[0,:,0][None,:])[:,:])
    #         # for column in range(ya.shape[1]):
    #         #     # ya[:,row] -= knots_col[1,row,0]
    #         #     ya[:,column] -= self.scan_2_fast[1] * knots_col[1,column,0]

    #     elif num_knots == 2:
    #         xa = interp1d(basis, knots_row[0], kind="linear", assume_sorted=True)(self.u)
    #         ya = interp1d(basis, knots_row[1], kind="linear", assume_sorted=True)(self.u)
    #     else:
    #         kind = "quadratic" if num_knots == 3 else "cubic"
    #         xa = interp1d(
    #             basis,
    #             knots_row[0],
    #             kind=kind,
    #             fill_value="extrapolate",
    #             assume_sorted=True,
    #         )(self.u)
    #         ya = interp1d(
    #             basis,
    #             knots_row[1],
    #             kind=kind,
    #             fill_value="extrapolate",
    #             assume_sorted=True,
    #         )(self.u)

    #     return xa, ya
