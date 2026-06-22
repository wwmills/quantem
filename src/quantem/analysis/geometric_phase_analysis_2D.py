from collections.abc import Sequence
from typing import List, Union

import numpy as np
from numpy.typing import NDArray
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from skimage import restoration as skr
from skimage.color import lab2rgb

from quantem.core import config
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.compound_validators import (
    validate_list_of_dataset2d,
    validate_pad_value,
)
from quantem.core.visualization import show_2d

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp

import copy

import matplotlib.pyplot as plt


def create_lattice(
    n_rows: int,
    n_cols: int,
    a_rows: int,
    a_cols: int,
):
    """
    This function creates a square image with lattice parameters a1 and a2.
    This can be used to create a simple lattice for demonstration purposes.

    n_rows:int
        Number of lattice sites in the row direction  (the x direction, by convention)
    n_cols:int
        Number of lattice sites in the column direction  (the y direction, by convention)
    a_rows:int
        Lattice parameter in the row direction (the x direction, by convention)
    a_cols:int
        Lattice parameter in the column direction (the y direction, by convention)

    Returns:
        coords: np.ndarray, (n_rows*n_cols, 2)
            Coordinates with the row (x) coordinates in the first column and the column (y) coordinates in the second column
    """

    ind = 0
    coords = np.zeros([n_rows * n_cols, 2])
    a_rows_array = np.array([a_rows, 0])
    a_cols_array = np.array([0, a_cols])
    for row in range(n_rows):
        for col in range(n_cols):
            coords[ind] = row * a_rows_array + col * a_cols_array
            ind += 1
    return coords


class geometric_phase_analysis_2D(AutoSerialize):
    """
    A class for performing geometric phase retrieval on 2D real space images using Gaussian fitting and Fourier transforms.

    This can be used to retrieve atomic displacements and strain maps.
    """

    _token = object()

    def __init__(
        self,
        images: List[Dataset2d],
        _token: object | None = None,  ####
    ):
        """
        Parameters
        ----------
        image: (nx, ny) np.ndarray
            A 2D image in real space.

        """
        if _token is not self._token:
            raise RuntimeError(
                "Use geometric_phase_analysis_2D.from_data() or .from_file() to instantiate this class."
            )

        self._images = images
        self.dtype = np.dtype([("x", float), ("y", float), ("intensity", float)])
        self._FFT = []
        self.global_image_index = 0
        self.calculated_phase = False
        self.calculated_displacement = False

    @classmethod
    def from_file(
        cls,
        file_paths: Sequence[str],
        file_type: str | None = None,
    ) -> "geometric_phase_analysis_2D":
        image_list = [Dataset2d.from_file(fp, file_type=file_type) for fp in file_paths]
        return cls.from_data(
            image_list,
        )

    @classmethod
    def from_data(
        cls,
        images: Union[List[Dataset2d], List[NDArray], Dataset3d, NDArray],
    ) -> "geometric_phase_analysis_2D":
        if isinstance(images, np.ndarray):
            images = [images]
        validated_images = validate_list_of_dataset2d(images)
        return cls(
            images=validated_images,
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
    def pad_fraction(self) -> float:
        return self._pad_fraction

    @pad_fraction.setter
    def pad_fraction(self, value: float):
        self._pad_fraction = float(value)

    @property
    def FFT(self) -> List[Dataset2d]:
        return self._FFT

    @FFT.setter
    def FFT(self, value: List[Dataset2d]):
        self._FFT = value

    def preprocess(
        self,
        pad_fraction: float = 0.125,
        pad_value: Union[float, str, List[float]] = "median",
        show_images: bool = False,
        edgeblendPixels=32,
        **kwargs,
    ):
        self.pad_fraction = pad_fraction
        validated_pad_value = validate_pad_value(pad_value, self._images)
        self._pad_value = validated_pad_value
        # self.pad_value = pad_value
        self.shape = (
            len(self._images),
            int(np.round(self.images[0].shape[0] * (1 + self.pad_fraction) / 2) * 2),
            int(np.round(self.images[0].shape[1] * (1 + self.pad_fraction) / 2) * 2),
        )
        self.mask_size = None
        self.defined_peaks = np.zeros(
            self.shape[0], dtype=bool
        )  # set to true when the peaks have been set
        if show_images:
            self.plot_original_image(
                title=[f"Image {i}: initial" for i in range(self.shape[0])],
                **kwargs,
            )

        self.shape_init = []
        for a0 in range(len(self._images)):
            image = self.images[a0].array
            median_val = np.median(image)
            shape_init = image.shape
            self.shape_init.append(image.shape)

            pad_widths = (
                (
                    int(np.round(shape_init[0] * pad_fraction / 2) * 2),
                    int(np.round(shape_init[0] * pad_fraction / 2) * 2),
                ),
                (
                    int(np.round(shape_init[1] * pad_fraction / 2) * 2),
                    int(np.round(shape_init[1] * pad_fraction / 2) * 2),
                ),
            )

            xP = np.arange(0, shape_init[0]) + 1
            yP = np.arange(0, shape_init[1]) + 1
            xP, yP = np.meshgrid(xP, yP, indexing="ij")
            edgeBlend = (
                np.cos(
                    (np.pi / 2)
                    * (
                        np.maximum(
                            1
                            - (1 + shape_init[0]) / (2 * max(edgeblendPixels, 1))
                            + np.abs(xP - (1 + shape_init[0]) / 2) / max(edgeblendPixels, 1),
                            0,
                        )
                    )
                )
                ** 2
                * np.cos(
                    (np.pi / 2)
                    * (
                        np.maximum(
                            1
                            - (1 + shape_init[1]) / (2 * max(edgeblendPixels, 1))
                            + np.abs(yP - (1 + shape_init[1]) / 2) / max(edgeblendPixels, 1),
                            0,
                        )
                    )
                )
                ** 2
            )

            self.images[a0].array = np.pad(
                image * edgeBlend + (1 - edgeBlend) * median_val, pad_widths, mode=pad_value
            )

        for a0 in range(len(self.images)):
            self._FFT.append(
                Dataset2d.from_array(
                    fftshift(fft2(self.images[a0].array)),
                    name="fourier transform",
                )
            )
        return self

    def plot_original_image(self, **kwargs):
        fig, ax = show_2d(
            list(self.images[a0].array for a0 in range(len(self.images))),
            **kwargs,
        )

    def fourier_filter(
        self,
        data: np.ndarray,
        image_index: int = 0,
        threshold: int = 1,
        show_plot: bool = False,
    ):
        """
        Calculates a mask based on the low frequency structure in real space. Signal is set to one, vacuum is set to zero.

        Parameters
        ----------
        data: (nx, ny) np.ndarray
            An image in real space that matches the dimensions of self.image.
        threshold: int
            A Fourier threshold value for the real space amplitude after filtering. Defaults to 1.
        show_plot: bool
            Controls if the mask and original real space are shown. Defaults to False.

        Returns
        -------
        self.fourier_mask * data: (nx, ny) np.ndarray
            The input data multiplied by a binary mask.
        """
        if data.shape != self.images[image_index].shape:
            print("Input shape does not match that of original image")
            return 0
        nx, ny = self.images[image_index].shape
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        1 / (nx)
        1 / (ny)

        center = np.array(self.images[image_index].shape) / 2
        mask_size = 10
        gaussCoords = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / mask_size**2
        del xx, yy
        mask = np.exp(-0.5 * gaussCoords, dtype=np.float32)
        del gaussCoords
        self.fourier_mask = np.abs((ifft2(self._FFT * mask))) * 100
        self.fourier_mask[self.fourier_mask < threshold] = 0
        self.fourier_mask[self.fourier_mask > 0] = 1
        if show_plot:
            show_2d(
                [
                    self.fourier_mask,
                    self.fourier_mask * data,
                ],
                title=["Mask", "Masked Data"],
            )
        shape_mask: tuple[int, int] = (nx, ny)
        self.masked_data = Dataset2d.from_shape(shape_mask)
        self.masked_data = self.fourier_mask * data
        return self

    def plot_masked_images(self, **kwargs):
        """
        Plot the current transformed images, with knot overlays.
        """
        fig, ax = show_2d(
            self.masked_data.array,
            **kwargs,
        )

    def phase_im_lab(
        self,
        phaseIM: np.ndarray,
        brightness: int = 60,
        saturation: int = 60,
    ):
        """
        Display an input phase image using color; because phase is bound to a range spanning 2pi, a color wrap is used.

        Parameters
        ----------
        phaseIM: np.ndarray
            The phase image.
        brightness: int
            The brightness of the output color image. Defaults to 60.
        saturation: int
            The saturation of the output color image. Defaults to 60.

        Returns
        -------
        im_pha_gp: np.ndarray
            The phase represented by 3 color channels. The dimensionality is im_pha_gp.shape = phaseIM.shape, 3.
        """
        L = brightness * (1 + np.zeros(phaseIM.shape))  # Brightness
        a = saturation * np.cos(phaseIM)  # Saturation
        b = saturation * np.sin(phaseIM)  # Saturation
        im_pha_gp = lab2rgb(np.dstack((L, a, b)))
        return im_pha_gp

    def image_normalizer(
        self,
        image: np.ndarray,
    ):
        """
        Normalizing input image.

        Parameters
        ----------
        image: np.ndarray
            The original image to be normalized

        Returns
        -------
        image_out: np.ndarray
            Normalized image
        """
        image_out = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
        return image_out

    def precise_peak_location(
        self,
        peakCoordinates: np.dtype([("x", float), ("y", float), ("intensity", float)]),
        image_index: int = 0,
        subImageHalfLength: int = 20,
        show_result=False,
    ):
        """
        Zero in on peak location by fitting with a Gaussian.

        Parameters
        ----------
        peakCoordinates: np.dtype([("x", float), ("y", float), ("intensity", float)])
            One set of peak coordinates.
        subImageHalfLength: int
            Half of side length of sub image for peak fitting. Defaults to 20.

        Returns
        -------
        peakCoordinatesPrecise: np.dtype([("x", float), ("y", float), ("intensity", float)])
            A more precise estimate of the Bragg peak location.
        """

        px = float(peakCoordinates["x"].flat[0])
        py = float(peakCoordinates["y"].flat[0])
        decimal_x = px - int(px)
        decimal_y = py - int(py)
        subIm = np.abs(self._FFT[image_index].array)[
            int(px - subImageHalfLength) : int(px + subImageHalfLength),
            int(py - subImageHalfLength) : int(py + subImageHalfLength),
        ]
        GaussianFit = self.fit_diffraction_center(subIm, plot_results=show_result)
        peakCoordinatesPrecise = np.zeros(1, dtype=self.dtype)
        peakCoordinatesPrecise["x"] = (
            peakCoordinates["x"] + GaussianFit[2] - subImageHalfLength - decimal_x
        )
        peakCoordinatesPrecise["y"] = (
            peakCoordinates["y"] + GaussianFit[3] - subImageHalfLength - decimal_y
        )
        return peakCoordinatesPrecise

    def gauss2D(
        self,
        xdata: np.ndarray,
        A: float,
        B: float,
        xc: float,
        yc: float,
        sx: float,
        sy: float,
        th: float,
    ):
        """
        A 2D Gaussian (with rotation).

        Parameters
        ----------
        xdata: (nx, ny, 2) np.ndarray
            The input subimage coordinates. X and Y coordinates should be present.
        A: float
            The amplitude multiplier of the Gaussian.
        B: float
            The scalar offset of the Gaussian.
        xc: float
            The central coordinate of the Gaussian in the X (row) direction.
        yc: float
            The central coordinate of the Gaussian in the Y (column).
        sx: float
            The standard deviation of the Gaussian in the X (row) direction.
        sy: float
            The standard deviation of the Gaussian in the Y (column) direction.
        th: float
            The rotation (in radians) of the Gaussian in the Z direction.

        Returns
        -------
        G: (nx, ny) np.ndarray
            The 2D Gaussian.
        """
        xx = xdata[:, :, 0]
        yy = xdata[:, :, 1]
        a = np.cos(th) ** 2 / (2 * sx**2) + np.sin(th) ** 2 / (2 * sy**2)
        b = -np.sin(2 * th) / (4 * sx**2) + np.sin(2 * th) / (4 * sy**2)
        c = np.sin(th) ** 2 / (2 * sx**2) + np.cos(th) ** 2 / (2 * sy**2)
        G = (
            A * np.exp(-(a * (xx - xc) ** 2 + 2 * b * (xx - xc) * (yy - yc) + c * (yy - yc) ** 2))
            + B
        )
        return G.ravel()

    def fit_diffraction_center(
        self,
        subIm: np.ndarray,
        plot_results: bool = True,
    ):
        """
        Given a subimage of the Fourier transform, use curve fitting to improve the estimate of the Bragg peak's central coordinates.

        Parameters
        ----------
        subIm: (nx, ny) np.ndarray
            The subimage of the Fourier transform. This should ideally contain a single strongest Bragg peak close to the center of the image.
        plot_results: bool
            Plot the subimage and Gaussian fit of the Bragg peak. Defaults to True.

        Returns
        -------
        popt: (7) np.ndarray
            An array of the optimal values returned by the curve fit algorithm. These entries have the following identities: [amplitude, offset, center x, center y, std x, std y, theta].
        """

        # Create Grid for Curve Fit
        (xx, yy) = np.meshgrid(np.arange(subIm.shape[0]), np.arange(subIm.shape[1]), indexing="ij")
        xdata = np.dstack((xx, yy))

        # Default Parameters for 2D Curve Fit
        A0 = np.max(subIm)
        B0 = np.min(subIm)
        x0 = np.array([A0, B0, subIm.shape[0] / 2, subIm.shape[1] / 2, 0.5, 0.5, 0])
        lb = np.array([A0 / 4, -A0, subIm.shape[0] * 3 / 8, subIm.shape[1] * 3 / 8, 0, 0, -np.pi])
        ub = np.array([2 * A0, A0, subIm.shape[0] * 5 / 8, subIm.shape[1] * 5 / 8, 15, 5, np.pi])

        # Fit Parameters
        popt, pcov = curve_fit(self.gauss2D, xdata, subIm.ravel(), p0=x0, bounds=(lb, ub))

        # Display fit
        if plot_results:
            data_fitted = self.gauss2D(xdata, *popt)
            plt.figure()
            plt.imshow(np.log(subIm), cmap="gray")
            plt.title("Log Abs of FFT Sub Window and Gaussian Fit")
            plt.contour(
                xdata[:, :, 1], xdata[:, :, 0], data_fitted.reshape(subIm.shape[0], subIm.shape[1])
            )  # like scatter, contour follows these rules: "len(X) == N is the number of columns in Z and len(Y) == M is the number of rows in Z."

        return popt

    def define_diffraction_peaks(
        self,
        diffraction_peaks_dict=None,
        num_peaks_search=20,
        num_peaks_use=2,
        center_ignore_buffer=15,
        minSpacingPeaks=5,
        outer_ignore_buffer=None,
        sub_image_half_length=20,
        show_result=True,
    ):
        if diffraction_peaks_dict is not None:
            diffraction_peaks_dict = copy.deepcopy(diffraction_peaks_dict)
        if diffraction_peaks_dict is not None and self.pad_fraction != 0:
            shape_init = self.shape_init[self.global_image_index]
            shape_padded = self.images[self.global_image_index].array.shape
            print("shape init", shape_init)
            print("shape padded", shape_padded)
            print(diffraction_peaks_dict)
            for diffraction_peak in diffraction_peaks_dict:
                diffraction_peak["x"] = (diffraction_peak["x"] - shape_init[0] / 2) * shape_padded[
                    0
                ] / shape_init[0] + shape_padded[0] / 2
                diffraction_peak["y"] = (diffraction_peak["y"] - shape_init[1] / 2) * shape_padded[
                    1
                ] / shape_init[1] + shape_padded[1] / 2
            print(diffraction_peaks_dict)

        self.phase_container_containers = []
        if diffraction_peaks_dict is None:
            for a0 in range(self.shape[0]):
                self.global_image_index = a0
                self.nx, self.ny = self.images[self.global_image_index].shape
                self.auto_peak_finder(
                    num_peaks_search=num_peaks_search,
                    num_peaks_use=num_peaks_use,
                    center_ignore_buffer=center_ignore_buffer,
                    outer_ignore_buffer=outer_ignore_buffer,
                    minSpacingPeaks=minSpacingPeaks,
                    show_result=show_result,
                )

        else:
            for a0 in range(self.shape[0]):
                self.defined_peaks[a0] = diffraction_peaks_dict[a0] is not None
                self.global_image_index = a0
                self.nx, self.ny = self.images[self.global_image_index].shape
                if diffraction_peaks_dict[a0] is not None:
                    for peak_index in range(diffraction_peaks_dict[a0].shape[0]):
                        diffraction_peaks_dict[a0][peak_index] = self.precise_peak_location(
                            diffraction_peaks_dict[a0][peak_index],
                            a0,
                            sub_image_half_length,
                            show_result=show_result,
                        )
                    self.create_peak_objects(diffraction_peaks_dict[a0])
                else:
                    self.auto_peak_finder(
                        num_peaks_search=num_peaks_search,
                        num_peaks_use=num_peaks_use,
                        center_ignore_buffer=center_ignore_buffer,
                        outer_ignore_buffer=outer_ignore_buffer,
                        minSpacingPeaks=minSpacingPeaks,
                        show_result=show_result,
                    )
        return self

    def auto_peak_finder(
        self,
        num_peaks_search=20,
        num_peaks_use=2,
        center_ignore_buffer=15,
        outer_ignore_buffer=None,
        minSpacingPeaks=5,
        show_result=False,
    ):
        diffraction_peaks_list = self.locate_diffraction_spots(
            num_peaks_search,
            center_ignore_buffer=center_ignore_buffer,
            outer_ignore_buffer=outer_ignore_buffer,
            minSpacingPeaks=minSpacingPeaks,
            show_result=show_result,
        )
        if num_peaks_use == 2:
            peakA, peakB = self.locate_first_order_peaks(diffraction_peaks_list)
            diffraction_peaks_list = np.array([peakA, peakB])
            self.create_peak_objects(diffraction_peaks_list)
        else:
            diffraction_peaks_list_clipped = np.array(
                [[diffraction_peaks_list[i]] for i in range(1, (num_peaks_use + 1))]
            )
            self.create_peak_objects(diffraction_peaks_list_clipped)
        return self

    def create_peak_objects(
        self,
        diffraction_peaks_list,
    ):
        num_peaks = diffraction_peaks_list.shape[0]
        self.phase_containers = []
        for peak_ind in range(num_peaks):
            # print(diffraction_peaks_list[peak_ind])
            self.phase_containers.append(self.phase_container(diffraction_peaks_list[peak_ind]))
        self.phase_container_containers.append(self.phase_containers)

    class phase_container:
        def __init__(
            self,
            peak_coordinates,
        ):
            self._peak_coordinates = peak_coordinates
            self.phase_map_: NDArray | None = None
            self.refined_phase_map_: NDArray | None = None
            self.amplitude_map_: NDArray | None = None
            self.refined_amplitude_map_: NDArray | None = None
            self.displacementX_: NDArray | None = None
            self.displacementY_: NDArray | None = None

        @property
        def peak_coordinates(self):
            return self._peak_coordinates

        @peak_coordinates.setter
        def peak_coordinates(self, value: NDArray):
            self._peak_coordinates = value

        @property
        def phase_map(self):
            return self.phase_map_

        @phase_map.setter
        def phase_map(self, value: NDArray):
            self.phase_map_ = value

        @property
        def refined_phase_map(self):
            return self.refined_phase_map_

        @refined_phase_map.setter
        def refined_phase_map(self, value: NDArray):
            self.refined_phase_map_ = value

        @property
        def amplitude_map(self):
            return self.amplitude_map_

        @amplitude_map.setter
        def amplitude_map(self, value: NDArray):
            self.amplitude_map_ = value

        @property
        def refined_amplitude_map(self):
            return self.refined_amplitude_map_

        @refined_amplitude_map.setter
        def refined_amplitude_map(self, value: NDArray):
            self.refined_amplitude_map_ = value

        @property
        def displacement_maps(self):
            return self.displacementX, self.displacementY

        @displacement_maps.setter
        def displacement_maps(self, value: list[NDArray]):
            self.displacementX = value[0]
            self.displacementY = value[1]

    def calculate_phase_maps(
        self,
        inputMaskSize: Union[float, NDArray[float]],
        gaussianMask: bool = True,
        distanceMask: bool = False,
        useHamming: bool = False,
        show_result: bool = True,
        display_window_width: int | None = None,
        amplitude_mask_result: bool = False,
        threshold_mask: float = 0.4,
        showAmplitude=False,
        center_ignore_buffer=None,
        outer_ignore_buffer=None,
        minSpacingPeaks=None,
    ):
        for a0 in range(self.shape[0]):
            if np.isscalar(inputMaskSize):
                self.mask_size = inputMaskSize
            elif inputMaskSize.shape[0] != self.shape[0]:
                raise RuntimeError(
                    f"The list of mask sizes must be a scalar or have length equal to {self.shape[0]}"
                )
            else:
                self.mask_size = inputMaskSize[a0]
            self.global_image_index = a0
            self.phase_containers = self.phase_container_containers[self.global_image_index]
            for peak_index in range(len(self.phase_containers)):
                self.calculate_phase_map(
                    peakIndex=peak_index,
                    inputMaskSize=self.mask_size,
                    gaussianMask=gaussianMask,
                    distanceMask=distanceMask,
                    useHamming=useHamming,
                    show_result=show_result,
                    display_window_width=display_window_width,
                    amplitude_mask_result=amplitude_mask_result,
                    threshold_mask=threshold_mask,
                    showAmplitude=showAmplitude,
                    center_ignore_buffer=center_ignore_buffer,
                    outer_ignore_buffer=outer_ignore_buffer,
                    minSpacingPeaks=minSpacingPeaks,
                )
        self.calculated_phase = True
        return self

    def calculate_phase_map(
        self,
        peakIndex: int,
        inputMaskSize: float,
        gaussianMask: bool = True,
        distanceMask: bool = False,
        useHamming: bool = False,
        show_result: bool = True,
        display_window_width: int | None = None,
        amplitude_mask_result: bool = False,
        refine: bool = False,
        threshold_mask: float = 0.4,
        showAmplitude=False,
        center_ignore_buffer=None,
        outer_ignore_buffer=None,
        minSpacingPeaks=None,
    ):
        """
        Calculate the geometric phase for a single Bragg peak.

        Parameters
        ----------
        peakCoordinates: np.dtype([("x", float), ("y", float), ("intensity", float)])
            Coordinates of the Bragg peak.
        inputMaskSize: float
            The size of the input mask. Highly tunable. Lower values correspond to larger convolution kernel and lower resolution.
        guassianMask: bool
            Control for whether to use a Gaussian mask or circular binary mask. Defaults to True (Gaussian).
        useHamming: bool
            Control whether to use a Hamming window in k-space. Defaults to False.
        show_result: bool
            Show the real space geometric phase alongside the shifted Fourier transform and the Gaussian mask. Defaults to True.

        Returns
        -------
        G_matrix: (nx, ny) np.ndarray
            A 2D array of the geometric phase corresponding to the input peak.
        """
        peakCoordinates = self.phase_containers[peakIndex].peak_coordinates
        # print(peakCoordinates)
        peakCoordinates_xy = self.get_xy_2(peakCoordinates)

        # initialize display cropping
        shape = (self.nx, self.ny)
        if display_window_width is not None:
            d_lower_x = int(np.floor(self.nx / 2 - display_window_width / 2))
            d_upper_x = int(display_window_width + d_lower_x)
            d_lower_y = int(np.floor(self.ny / 2 - display_window_width / 2))
            d_upper_y = int(display_window_width + d_lower_y)

        else:
            d_lower_x = 0
            d_upper_x = self.nx
            d_lower_y = 0
            d_upper_y = self.ny

        # get r_all
        # peak_all = np.zeros([len(self.phase_containers)-1, 2])
        # peak_index_arr = 0
        # for peak_index_other in range(len(self.phase_containers)):
        #     if peak_index_other != peakIndex:
        #         peakCoordinates = self.phase_containers[peak_index_other].peak_coordinates
        #         peakCoordinates_xy_other = self.get_xy_2(peakCoordinates)
        #         peak_all[peak_index_arr] = peakCoordinates_xy_other
        #         peak_index_arr += 1

        # Construct Fourier Coordinates
        xx, yy = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        dkx = 1 / (self.nx)
        dky = 1 / (self.ny)

        # Shift Bragg Peak to Center
        center = np.array(self.images[self.global_image_index].shape) / 2
        shift = np.array(center - peakCoordinates_xy)
        shift = shift * [dkx, dky]
        shift_phase = np.exp(1j * 2 * np.pi * (shift[0] * xx + shift[1] * yy))

        if gaussianMask and not distanceMask:  # Create Mask with Gaussian Kernel
            xx, yy = np.mgrid[0 : self.nx, 0 : self.ny]
            gg = (((xx - center[0]) ** 2) + ((yy - center[1]) ** 2)) / inputMaskSize**2
            mask = np.exp((-0.5) * gg)
        elif distanceMask and gaussianMask:
            # need to search for other peaks in the vicinity of peakCoordinates_xy
            # peaks = self.locate_diffraction_spots()
            shift = np.round(np.array(center - peakCoordinates_xy))
            peak_all = self.locate_diffraction_spots(
                5,
                center_ignore_buffer=center_ignore_buffer,
                outer_ignore_buffer=outer_ignore_buffer,
                minSpacingPeaks=minSpacingPeaks,
                shift_fft=shift,
            )
            # peak_all_xy = self.get_xy(peak_all)
            peak_all_xy = np.zeros([peak_all.shape[0], 2])
            for peak_index in range(peak_all.shape[0]):
                # peak_all_i_xy = self.get_xy(peak_all[peak_index])
                peak_all_i_xy = np.array([peak_all["x"][peak_index], peak_all["y"][peak_index]])
                peak_all_xy[peak_index] = peak_all_i_xy
            # print(peak_all_xy)
            # fig, ax = plt.subplots(1,1, figsize = (5,5))

            # peak_all_xy += shift
            shape = (self.nx, self.ny)
            # ax.scatter(peak_all_xy[:,1], peak_all_xy[:,0])
            # ax.scatter(center[1], center[0])
            mask = self._make_mask(
                r0=np.array([center[0], center[1]]),
                r_all=peak_all_xy,
                sigma=inputMaskSize,
                shape=shape,
            )
        else:  # Create a Hard Circle Mask
            circ_rad = np.amin(
                inputMaskSize * np.asarray(self.images[self.global_image_index].shape)
            )
            mask = (
                self.make_circle(
                    self.images[self.global_image_index].shape, self.nx / 2, self.ny / 2, circ_rad
                )
            ).astype(bool)

        if useHamming:
            ham_x = np.hamming(self.nx)[:, None]
            ham_y = np.hamming(self.ny)[None, :]
            ham = np.sqrt(ham_x * ham_y)
            G_matrix = ifft2(
                ifftshift(
                    mask
                    * fftshift(
                        fft2(self.images[self.global_image_index].array * ham * shift_phase)
                    )
                )
            )  # With hamming in 2D. Original methods would take the phase immediately, but the amplitude is also useful.
        else:
            G_matrix = ifft2(
                ifftshift(
                    mask * fftshift(fft2(self.images[self.global_image_index].array * shift_phase))
                )
            )  # Without hamming

        if show_result:
            if amplitude_mask_result:
                if isinstance(threshold_mask, float):
                    max_thresh = threshold_mask
                    min_thresh = 0
                elif hasattr(threshold_mask, "__iter__"):
                    arr = np.asarray(threshold_mask)
                    if arr.size == 2:
                        min_thresh, max_thresh = arr
                    else:
                        raise ValueError("threshold_mask must have exactly two entries")
                else:
                    raise ValueError(
                        "threshold_mask must be a scalar or an iterable with two entries"
                    )
                amplitude_mask = np.abs(G_matrix)
                amplitude_map_norm = amplitude_mask / np.max(amplitude_mask)
                mask_rescaled = (amplitude_map_norm - min_thresh) / (max_thresh - min_thresh)
                mask_rescaled[mask_rescaled < 0] = 0
                mask_rescaled[mask_rescaled > 1] = 1

                mask_soft = np.sin(np.pi / 2 * mask_rescaled) ** 2
                # amplitude_map_norm[amplitude_map_norm < threshold_mask] = 0
                # amplitude_map_norm[amplitude_map_norm > 0] = 1
                # def normalize_arr(array):
                #     array-=np.min(array)
                #     array /= np.max(array)
                #     return array
                # amplitude_mask = np.abs(G_matrix)
                # amplitude_map_norm = normalize_arr(amplitude_mask)
                # amplitude_map_norm[amplitude_map_norm < threshold_mask] = 0
                # amplitude_map_norm[amplitude_map_norm > 0] = 1
                im_pha_gp = self.phase_im_lab(np.angle(G_matrix) * mask_soft)
            else:
                im_pha_gp = self.phase_im_lab(np.angle(G_matrix))
            imFFT = fftshift(fft2(self.images[self.global_image_index].array * shift_phase))
            if showAmplitude:
                (_, axs) = plt.subplots(1, 3, figsize=(15, 30))
                axs[0].imshow(im_pha_gp, origin="upper")
                axs[0].axis("off")
                axs[1].imshow(np.abs(G_matrix), origin="upper")
                axs[1].axis("off")
                axs[2].imshow(
                    np.log(np.abs(imFFT) + 1)[d_lower_x:d_upper_x, d_lower_y:d_upper_y],
                    cmap="gray",
                    origin="upper",
                )
                plt.imshow(
                    mask[d_lower_x:d_upper_x, d_lower_y:d_upper_y], alpha=0.4, origin="upper"
                )
                axs[2].axis("off")
            else:
                (_, axs) = plt.subplots(1, 2, figsize=(15, 30))
                axs[0].imshow(im_pha_gp, origin="upper")
                axs[0].axis("off")
                axs[1].imshow(
                    np.log(np.abs(imFFT) + 1)[d_lower_x:d_upper_x, d_lower_y:d_upper_y],
                    cmap="gray",
                    origin="upper",
                )
                plt.imshow(
                    mask[d_lower_x:d_upper_x, d_lower_y:d_upper_y], alpha=0.4, origin="upper"
                )
                axs[1].axis("off")
        if refine:
            self.phase_containers[peakIndex].refined_phase_map = np.angle(G_matrix)
            self.phase_containers[peakIndex].refined_amplitude_map = np.abs(G_matrix)
        else:
            self.phase_containers[peakIndex].phase_map = np.angle(G_matrix)
            self.phase_containers[peakIndex].amplitude_map = np.abs(G_matrix)
        return self

    def _make_mask(
        self,
        r0,
        r_all,
        sigma,
        shape,
        show_mask=False,
        center_min_dist=2,
    ):
        dists = np.linalg.norm(r_all - r0, axis=1)
        r_all = r_all[dists >= center_min_dist]

        xa, ya = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            indexing="ij",
        )

        mask = np.exp(((xa - r0[0]) ** 2 + (ya - r0[1]) ** 2) / (-2 * sigma**2))

        # filter neighbors

        for r_index in range(r_all.shape[0]):
            r1 = r_all[r_index, :]
            v = r0 - r1
            v_norm = np.linalg.norm(v)
            v /= v_norm
            dist = np.clip(
                (xa - r1[0]) * v[0] + (ya - r1[1]) * v[1] - v_norm / 2 + 0.5,
                0,
                1,
            )
            mask *= dist

            # dist = ((r1[1]- r0[1])*xa - (r1[0]- r0[0])*ya + r1[0]*r0[1]- r1[1]*r0[0]) / \
            #     np.linalg.norm((r0-r1))

        if show_mask:
            fig, ax = plt.subplots()
            ax.imshow(mask, cmap="gray")
            ax.scatter(r_all[:, 1], r_all[:, 0])
            ax.scatter(r0[1], r0[0], c="red")
        return mask

    def get_phase_container(self, image_index=0):
        return self.phase_container_containers[image_index]

    def make_circle(
        self,
        size_circ: np.ndarray,
        center_x: float,
        center_y: float,
        radius: float,
    ):
        """
        Make a circle Mask

        Parameters
        ----------
        size_circ: ndarray
                2 element array giving the size of the output matrix
        center_x: float
                x position of circle center
        center_y: float
                y position of circle center
        radius: float
                radius of the circle

        Returns
        -------
        circle: ndarray
                p X q sized array where the it is 1
                inside the circle and 0 outside
        """
        p = size_circ[0]
        q = size_circ[1]
        yV, xV = np.mgrid[0:p, 0:q]
        sub = ((((yV - center_y) ** 2) + ((xV - center_x) ** 2)) ** 0.5) < radius
        circle = np.asarray(sub, dtype=np.float64)
        return circle

    def get_a_matrix(
        self,
        g_vector_1: np.ndarray,
        g_vector_2: np.ndarray,
    ):
        """
        Retrieve the inverse of the g matrix. The g matrix has reciprocal lattice vectors along its rows.
        The following is true: [[g1x g1y], [g2x g2y]]^-1 = [[a1x a2x], [a1y a2y]].
        The three reciprocal lattice vectors should be linearly independent.

        Parameters
        ----------
        g_vector_1: (2) np.ndarray
            The first reciprocal lattice vector.
        g_vector_2: (2) np.ndarray
            The second reciprocal lattice vector.

        Returns
        -------
        a_matrix: (2, 2) np.ndarray
            The transpose of the real space lattice vector matrix. The entries are organized like this: [[a1x a2x], [a1y a2y]].
        """
        g_matrix = np.array([g_vector_1, g_vector_2])
        a_matrix = np.linalg.inv(g_matrix)
        return a_matrix

    def get_u_matrices(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        a_matrix: np.ndarray,
    ):
        """
        Retrieve the displacment (U) matrices using two phase matrices.

        Parameters
        ----------
        P1: (nx, ny) np.ndarray
            The first phase matrix.
        P2: (nx, ny) np.ndarray
            The second phase matrix.
        a_matrix: (2, 2) np.ndarray
            The transpose of the real space lattice vector matrix. The entries are organized like this: [[a1x a2x], [a1y a2y]].

        Returns
        -------
        ux: (nx, ny) np.ndarray
            The atomic displacement map in the X (row) direction.
        uy: (nx, ny) np.ndarray
            The atomic displacement map in the Y (column) direction.
        """
        P1 = skr.unwrap_phase(P1)
        P2 = skr.unwrap_phase(P2)
        rolled_p = np.asarray((np.reshape(P1, -1), np.reshape(P2, -1)))
        u_matrix = -1 / (2 * np.pi) * np.matmul(a_matrix, rolled_p)
        u_x = np.reshape(u_matrix[0, :], P1.shape)
        u_y = np.reshape(u_matrix[1, :], P2.shape)
        return u_x, u_y

    def circ_to_G(self, circ_pos: np.ndarray):
        """
        Convert peak coordinates from absolute pixel location to centered k-space units.

        Parameters
        ----------
        circ_pos: (2) np.ndarray
            The position of the peak given in absolute coordinates (measured from corner origin) in pixels.

        Returns
        -------
        g_vec: (2) np.ndarray
            The position of the peak given in centered (self.image.shape/2) k-space coordinates (frequency units).
        """
        g_vec = np.zeros(2)
        g_vec[0] = (circ_pos[0] - (0.5 * self.nx)) / self.nx
        g_vec[1] = (circ_pos[1] - (0.5 * self.ny)) / self.ny
        return g_vec

    def G_to_circ(
        self,
        g_vec: np.ndarray,
    ):
        """
        Convert peak coordinates from centered k-space units to absolute pixel location.

        Parameters
        ----------
        g_vec: (2) np.ndarray
            The position of the peak given in centered (self.image.shape/2) k-space coordinates (frequency units).

        Returns
        -------
        circ_pos: (2) np.ndarray
            The position of the peak given in absolute coordinates (measured from corner origin) in pixels.
        """
        circ_pos = np.zeros(2)
        circ_pos[0] = (g_vec[0] * self.nx) + (0.5 * self.nx)
        circ_pos[1] = (g_vec[1] * self.ny) + (0.5 * self.ny)
        return circ_pos

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

    def calculate_displacement_maps(
        self,
        num_peaks: int = 2,
        show_result: bool = False,
        amplitude_mask_result: bool = False,
        threshold_mask: float = 0.4,
    ):
        #         peakCoordinatesA: np.dtype([("x", float), ("y", float), ("intensity", float)]),
        # peakCoordinatesB: np.dtype([("x", float), ("y", float), ("intensity", float)]),
        # phaseA: np.ndarray,
        # phaseB: np.ndarray,
        """
        Use the phase maps and peak coordinates to retrieve the x and y displacement maps.

        Parameters
        ----------
        peakCoordinatesA: np.dtype([("x", float), ("y", float), ("intensity", float)])
            The absolute pixel coordinates of the first selected peak.
        peakCoordinatesB: np.dtype([("x", float), ("y", float), ("intensity", float)])
            The absolute pixel coordinates of the second selected peak.
        phaseA: (nx, ny) np.ndarray
            The geometric phase corresponding to the first selected peak.
        phaseB: (nx, ny) np.ndarray
            The geometric phase corresponding to the second selected peak.
        show_result: bool
            Show the real space displacement. Defaults to False.

        Returns
        -------
        displacementX: (nx, ny) np.ndarray
            A 2D array that maps the X (row offset) displacement within the lattice.
        displacementY: (nx, ny) np.ndarray
            A 2D array that maps the Y (column offset) displacement within the lattice.
        """

        peak_coords = []
        phases = []
        amplitudes = []

        for i in range(num_peaks):
            pc = self.phase_containers[i]
            peak_coords.append(self.circ_to_G(self.get_xy_2(pc.peak_coordinates)))
            phases.append(pc.phase_map)
            if amplitude_mask_result:
                amplitudes.append(pc.amplitude_map)
            else:
                amplitudes.append(np.ones_like(pc.phase_map))

        A = np.array([[p[0], p[1]] for p in peak_coords])

        phases = np.stack(phases, axis=0)
        amplitudes = np.stack(amplitudes, axis=0)

        if amplitude_mask_result:
            # amp_mins = np.min(amplitudes, axis=(1, 2), keepdims=True)
            # amp_maxs = np.max(amplitudes, axis=(1, 2), keepdims=True)
            # amplitude_map_norm = (amplitudes - amp_mins) / (amp_maxs - amp_mins)
            # amplitude_map_norm = (amplitude_map_norm >= threshold_mask).astype(amplitudes.dtype)

            if isinstance(threshold_mask, float):
                max_thresh = threshold_mask
                min_thresh = 0
            elif hasattr(threshold_mask, "__iter__"):
                arr = np.asarray(threshold_mask)
                if arr.size == 2:
                    min_thresh, max_thresh = arr
                else:
                    raise ValueError("threshold_mask must have exactly two entries")
            else:
                raise ValueError("threshold_mask must be a scalar or an iterable with two entries")
            amplitude_map_norm = amplitudes / np.max(amplitudes, axis=(1, 2), keepdims=True)
            mask_rescaled = (amplitude_map_norm - min_thresh) / (max_thresh - min_thresh)
            mask_rescaled[mask_rescaled < 0] = 0
            mask_rescaled[mask_rescaled > 1] = 1

            mask_soft = np.sin(np.pi / 2 * mask_rescaled) ** 2
            weighted_phases = phases * mask_soft
        else:
            weighted_phases = phases

        n, nx, ny = weighted_phases.shape
        Phi = weighted_phases.reshape(n, -1)

        W = 1 / np.array([np.linalg.norm(p) for p in peak_coords])
        W = np.diag(W)

        AtW = A.T @ W
        A_pinv = np.linalg.inv(AtW @ A) @ AtW

        U = A_pinv @ Phi

        displacementX = U[0].reshape(nx, ny)
        displacementY = U[1].reshape(nx, ny)

        if show_result:
            show_2d(
                [displacementX, displacementY],
                title=[
                    "Displacement Along X Direction (rows)",
                    "Displacement Along Y Direction (columns)",
                ],
            )
        # if num_peaks == 2:
        self.phase_containers[0].displacement_maps = displacementX, displacementY
        self.displacement_map_index = 0
        self.calculated_displacement = True
        return self

    def calculate_strain_map(
        self,
        num_peaks=2,
        amplitude_mask_result=False,
        threshold_mask: float = 0.4,
        show_result=True,
        use_phase_directly=True,
        amplitude_mask_uniform=None,
    ):
        if not self.calculated_phase:
            inputMaskSizes = np.zeros(self.shape[0])
            for a0 in range(self.shape[0]):
                inputMaskSizes[a0] = np.mean(self.images[a0].shape) / 10
            self.calculate_phase_maps(
                inputMaskSizes,
                gaussianMask=True,
                show_result=show_result,
                amplitude_mask_result=amplitude_mask_result,
                threshold_mask=threshold_mask,
            )
        if use_phase_directly:
            for a0 in range(self.shape[0]):
                self.global_image_index = a0
                self.phase_containers = self.phase_container_containers[self.global_image_index]
                self.calculate_strain_map_phase(
                    num_peaks,
                    amplitude_mask_result,
                    show_result=show_result,
                    threshold_mask=threshold_mask,
                    amplitude_mask_uniform=amplitude_mask_uniform,
                )
        else:
            if not self.calculated_displacement:
                self.calculate_displacement_maps(
                    num_peaks=num_peaks,
                    show_result=show_result,
                    amplitude_mask_result=amplitude_mask_result,
                    threshold_mask=threshold_mask,
                )
            for a0 in range(self.shape[0]):
                self.global_image_index = a0
                self.phase_containers = self.phase_container_containers[self.global_image_index]
                self.calculate_strain_map_displacement(
                    show_result=show_result,
                    amplitude_mask_uniform=amplitude_mask_uniform,
                )
        return self

    def calculate_strain_map_displacement(
        self,
        show_result: bool = True,
        amplitude_mask_uniform=None,
    ):
        """
        Using the x and y displacement maps, calculate the strain maps.

        Parameters
        ----------
        displacementX: (nx, ny) np.ndarray
            A 2D array that maps the X (row offset) displacement within the lattice.
        displacementY: (nx, ny) np.ndarray
            A 2D array that maps the Y (column offset) displacement within the lattice.
        show_result: bool
            Show the real space strain. Defaults to True.

        Returns
        -------
        e_mat: (2, 2, nx, ny) np.ndarray
            The 2x2 tensor of strain maps.
        """
        displacementX, displacementY = self.phase_containers[
            self.displacement_map_index
        ].displacement_maps
        e_xx, e_xy = self.phase_diff(displacementX)
        e_yx, e_yy = self.phase_diff(displacementY)
        e_mat = np.array([[e_xx, e_xy], [e_yx, e_yy]])

        e_xx, e_yy = self.get_axial_strain(e_mat)
        e_th_xy, e_dg_xy = self.get_rot_and_diag_strain(e_mat)
        if show_result:
            (fig, axs) = plt.subplots(2, 2, figsize=(20, 25))
            show_2d(
                [
                    self.images[self.global_image_index].array,
                    e_xx,
                ],
                figax=(fig, axs[:1]),
                title=["", "$Strain_{xx}$"],
                cmap=["gray", "BrBG_r"],
                cbar=[False, True],
            )
            show_2d(
                [
                    e_yy,
                    e_dg_xy,
                ],
                figax=(fig, axs[1:]),
                title=["$Strain_{yy}$", "$Strain_{xy}$"],
                cmap="BrBG_r",
                cbar=[True, True],
            )
            vmin = min(e_xx.min(), e_yy.min(), e_dg_xy.min())
            vmax = min(e_xx.max(), e_yy.max(), e_dg_xy.max())
            axs[0, 1].images[0].set_clim(vmin, vmax)
            axs[1, 0].images[0].set_clim(vmin, vmax)
            axs[1, 1].images[0].set_clim(vmin, vmax)
            fig.tight_layout()

        return e_mat

    def calculate_strain_map_phase(
        self,
        num_peaks,
        amplitude_mask_result=False,
        amplitude_mask_each=False,
        threshold_mask=0.4,
        show_result=True,
        amplitude_mask_uniform=None,
        mean_center=True,
        show_angle=True,
    ):
        peak_coords = []
        nx_, ny_ = self.phase_containers[
            0
        ].phase_map.shape  # get the rows and columns of the phase map, which does not have padding here

        phase_derivatives = np.zeros([num_peaks, 2, nx_, ny_])

        # print("phase containers", len(self.phase_containers))
        # print("test", i)
        amp_sum = np.zeros(num_peaks)
        for i in range(num_peaks):
            pc = self.phase_containers[i]
            peak_coords.append(self.circ_to_G(self.get_xy_2(pc.peak_coordinates)))
            exp_matrix1 = np.exp(-1j * pc.phase_map)
            exp_matrix2 = np.exp(1j * pc.phase_map)
            if amplitude_mask_result:
                # def normalize_arr(array):
                #     array-=np.min(array)
                #     array /= np.max(array)
                #     return array
                # amplitude_map_norm = normalize_arr(pc.amplitude_map)
                # amplitude_map_norm[amplitude_map_norm < threshold_mask] = 0
                # amplitude_map_norm[amplitude_map_norm > 0] = 1

                if isinstance(threshold_mask, float):
                    max_thresh = threshold_mask
                    min_thresh = 0
                elif hasattr(threshold_mask, "__iter__"):
                    arr = np.asarray(threshold_mask)
                    if arr.size == 2:
                        min_thresh, max_thresh = arr
                    else:
                        raise ValueError("threshold_mask must have exactly two entries")
                else:
                    raise ValueError(
                        "threshold_mask must be a scalar or an iterable with two entries"
                    )
                amplitude_mask = pc.amplitude_map
                amplitude_map_norm = amplitude_mask / np.max(amplitude_mask)
                mask_rescaled = (amplitude_map_norm - min_thresh) / (max_thresh - min_thresh)
                mask_rescaled[mask_rescaled < 0] = 0
                mask_rescaled[mask_rescaled > 1] = 1

                mask_soft = np.sin(np.pi / 2 * mask_rescaled) ** 2

                amp_sum[i] = np.sum(amplitude_map_norm)
                if amplitude_mask_each:
                    phase_derivatives[i, 0] = (
                        np.imag(np.multiply(exp_matrix1, np.gradient(exp_matrix2, axis=0)))
                        * mask_soft
                    )  # phaseA_dx
                    phase_derivatives[i, 1] = (
                        np.imag(np.multiply(exp_matrix1, np.gradient(exp_matrix2, axis=1)))
                        * mask_soft
                    )  # phaseA_dy
                else:
                    phase_derivatives[i, 0] = np.imag(
                        np.multiply(exp_matrix1, np.gradient(exp_matrix2, axis=0))
                    )  # phaseA_dx
                    phase_derivatives[i, 1] = np.imag(
                        np.multiply(exp_matrix1, np.gradient(exp_matrix2, axis=1))
                    )  # phaseA_dy
            else:
                phase_derivatives[i, 0] = np.imag(
                    np.multiply(exp_matrix1, np.gradient(exp_matrix2, axis=0))
                )  # phaseA_dx
                phase_derivatives[i, 1] = np.imag(
                    np.multiply(exp_matrix1, np.gradient(exp_matrix2, axis=1))
                )  # phaseA_dy

        if amplitude_mask_uniform is not None:
            if amplitude_mask_uniform in np.arange(0, num_peaks):
                amplitude_mask_index = amplitude_mask_uniform
            else:
                amplitude_mask_index = np.argmin(amp_sum)
            amplitude_map = self.phase_containers[amplitude_mask_index].amplitude_map
            # def normalize_arr(array):
            #     array-=np.min(array)
            #     array /= np.max(array)
            #     return array
            # amplitude_map_norm = normalize_arr(amplitude_map)
            # amplitude_map_norm[amplitude_map_norm < threshold_mask] = 0
            # amplitude_map_norm[amplitude_map_norm > 0] = 1
            if isinstance(threshold_mask, float):
                max_thresh = threshold_mask
                min_thresh = 0
            elif hasattr(threshold_mask, "__iter__"):
                arr = np.asarray(threshold_mask)
                if arr.size == 2:
                    min_thresh, max_thresh = arr
                else:
                    raise ValueError("threshold_mask must have exactly two entries")
            else:
                raise ValueError("threshold_mask must be a scalar or an iterable with two entries")
            amplitude_mask = amplitude_map
            amplitude_map_norm = amplitude_mask / np.max(amplitude_mask)
            mask_rescaled = (amplitude_map_norm - min_thresh) / (max_thresh - min_thresh)
            mask_rescaled[mask_rescaled < 0] = 0
            mask_rescaled[mask_rescaled > 1] = 1
            # plt.figure()
            # plt.imshow(mask_rescaled)

            mask_soft = np.sin(np.pi / 2 * mask_rescaled) ** 2
            self.strain_mask_soft = mask_soft
            phase_derivatives = phase_derivatives * mask_soft

        A = np.array([[p[0], p[1]] for p in peak_coords])

        W = 1 / np.array([np.linalg.norm(p) for p in peak_coords])
        W = np.diag(W)

        AtW = A.T @ W
        A_pinv = np.linalg.inv(AtW @ A) @ AtW

        # e_mat = A_pinv @ phase_derivatives
        e_mat = -1 / (2 * np.pi) * np.einsum("ij,jkab->ikab", A_pinv, phase_derivatives)
        self.e_mat = e_mat

        e_xx, e_yy = self.get_axial_strain(e_mat)
        omega, e_dg_xy = self.get_rot_and_diag_strain(e_mat)

        eps_uu, eps_vv, eps_uv, u, v = self.get_uv_strain(e_mat, num_peaks)

        # crop_in_mask = np.zeros(eps_uu.shape)
        # win_c = 60
        # crop_in_mask[win_c:-win_c, win_c:-win_c] = 1
        # crop_in_mask = gaussian_filter(crop_in_mask, 20)

        # eps_uu *= crop_in_mask
        # eps_uv *= crop_in_mask
        # eps_vv *= crop_in_mask
        # omega_uv *= crop_in_mask
        if show_result:
            # (fig,axs) = plt.subplots(2,2, figsize = (20,25))

            # show_2d(
            #     [self.images[self.global_image_index].array,
            #     e_xx * 100,  # x100 for percent units
            #     ],
            #     figax = (fig, axs[:1]),
            #     norm=[{}, {"vmin": vmin, "vmax": vmax}],
            #     title = ["", "$Strain_{xx}$"],
            #     cmap = ["gray", "BrBG_r"],
            #     cbar = [False, True],
            # )
            # fig.axes[-1].set_ylabel("Percent strain")
            # show_2d(
            #     [e_yy * 100,  # x100 for percent units
            #     e_dg_xy * 100,
            #     ],
            #     figax = (fig, axs[1:]),
            #     norm=[{"vmin": vmin, "vmax": vmax}, {"vmin": vmin, "vmax": vmax}],
            #     title = ["$Strain_{yy}$", "$Strain_{xy}$"],
            #     cmap = "BrBG_r",
            #     cbar = [True, True],
            # )
            # for cax, label in zip(fig.axes[-2:], ["Percent strain", "Percent strain"]):
            #     cax.set_ylabel(label)
            # axs[0,1].images[0].set_clim(vmin, vmin)
            # axs[1,0].images[0].set_clim(vmin, vmax)
            # axs[1,1].images[0].set_clim(vmin, vmax)
            # fig.tight_layout()
            # fig.canvas.draw_idle()

            # vmin = min(e_xx.min(), e_yy.min(), e_dg_xy.min())
            # vmax = min(e_xx.max(), e_yy.max(), e_dg_xy.max())

            # (fig2, ax2) = plt.subplots(2, 2, figsize=(20, 25))

            # ax2[0,0].imshow(self.images[self.global_image_index].array, cmap = 'gray')

            # im01 = ax2[0,1].imshow(e_xx * 100, cmap="BrBG_r", vmin=vmin, vmax=vmax)
            # ax2[0,1].set_title("Strain_xx")

            # im10 = ax2[1,0].imshow(e_yy * 100, cmap="BrBG_r", vmin=vmin, vmax=vmax)
            # ax2[1,0].set_title("Strain_yy")

            # im11 = ax2[1,1].imshow(e_dg_xy * 100, cmap="BrBG_r", vmin=vmin, vmax=vmax)
            # ax2[1,1].set_title("Strain_xy")

            # fig2.colorbar(im01, ax=ax2[0,1])
            # fig2.colorbar(im10, ax=ax2[1,0])
            # fig2.colorbar(im11, ax=ax2[1,1])
            # fig2.tight_layout()
            if not show_angle:
                if mean_center:
                    eps_uu -= np.mean(eps_uu)
                    eps_vv -= np.mean(eps_vv)
                    eps_uv -= np.mean(eps_uv)

                vmin = min(np.min(eps_uu), np.min(eps_vv), np.min(eps_uv)) * 100
                vmax = max(np.max(eps_uu), np.max(eps_vv), np.max(eps_uv)) * 100

                vmax = max(np.abs(vmin), vmax)
                vmin = -vmax

                fig2 = plt.figure(figsize=(20, 20))

                ax1 = plt.subplot(221)
                ax1.imshow(self.images[self.global_image_index].array, cmap="gray")
                ax1.set_title("")

                ax2 = plt.subplot(222)
                im12 = ax2.imshow(eps_uu * 100, cmap="BrBG_r", vmin=vmin, vmax=vmax)
                ax2.set_title("$Strain_{uu}$")
                # fig2.colorbar(im12, ax=ax2)

                ax3 = plt.subplot(223)
                im13 = ax3.imshow(eps_vv * 100, cmap="BrBG_r", vmin=vmin, vmax=vmax)
                ax3.set_title("$Strain_{vv}$")
                # fig2.colorbar(im13, ax=ax3)

                ax4 = plt.subplot(224)
                im14 = ax4.imshow(eps_uv * 100, cmap="BrBG_r", vmin=vmin, vmax=vmax)
                ax4.set_title("$Strain_{uv}$")
                # fig2.colorbar(im14, ax=ax4)

                def add_matching_colorbar(fig, ax, im, label, width=0.015, pad=0.01):
                    pos = ax.get_position()
                    cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
                    fig.colorbar(im, cax=cax, label=label)

                add_matching_colorbar(fig2, ax2, im12, "Percent Strain")
                add_matching_colorbar(fig2, ax3, im13, "Percent Strain")
                add_matching_colorbar(fig2, ax4, im14, "Percent Strain")

                ax1.axis("off")
                ax2.axis("off")
                ax3.axis("off")
                ax4.axis("off")

                peak_coords_px = []
                for i in range(num_peaks):
                    pc = self.phase_containers[i]
                    peak_coords_px.append(self.get_xy_2(pc.peak_coordinates))

                peak_coords_px_arr = np.array(peak_coords_px)

                inset_width = 1 / 3
                inset_height = 1 / 3
                inset_ax = ax1.inset_axes(
                    [1 - inset_width, 1 - inset_height, inset_width, inset_height]
                )
                nx, ny = self._FFT[self.global_image_index].array.shape
                win_x = nx // 3
                win_y = ny // 3
                inset_ax.imshow(
                    np.log1p(
                        np.abs(
                            self._FFT[self.global_image_index].array[win_x:-win_x, win_y:-win_y]
                        )
                    ),
                    cmap="gray",
                )
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])

                # inset_ax.set_xticks([])
                # inset_ax.set_yticks([])

                for spine in inset_ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)
                    spine.set_color("black")

                inset_ax.scatter(
                    peak_coords_px_arr[:, 1] - win_y,
                    peak_coords_px_arr[:, 0] - win_x,
                    s=10,
                    facecolors=(1, 0, 0, 0.6),
                    edgecolors=(1, 1, 1, 1),
                    linewidths=0.5,
                )
                inset_origin_x = nx / 2 - win_x - 1
                inset_origin_y = ny / 2 - win_y + 1
                inset_ax.scatter(
                    inset_origin_y,
                    inset_origin_x,
                    s=50,
                    facecolors=(0, 0, 0, 0.6),
                    # edgecolors=(1,1,1,1),
                    linewidths=0.5,
                    zorder=10,
                )

                for x, y in peak_coords_px_arr:
                    inset_ax.annotate(
                        "",
                        xy=(y - win_y, x - win_x),
                        xytext=(inset_origin_y, inset_origin_x),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                    )
                labels = ["u", "v"]
                for (x, y), label in zip(peak_coords_px_arr[:2], labels):
                    inset_ax.text(
                        y - win_y,
                        x - win_x + 30,
                        label,
                        color="white",
                        fontsize=10,
                        ha="left",
                        va="bottom",
                    )

            if show_angle:
                if mean_center:
                    eps_uu -= np.mean(eps_uu)
                    eps_vv -= np.mean(eps_vv)
                    eps_uv -= np.mean(eps_uv)
                    omega -= np.mean(omega)

                vmin = min(np.min(eps_uu), np.min(eps_vv), np.min(eps_uv), np.min(omega)) * 100
                vmax = max(np.max(eps_uu), np.max(eps_vv), np.max(eps_uv), np.max(omega)) * 100

                vmax = max(np.abs(vmin), vmax)
                vmin = -vmax

                fig2 = plt.figure(figsize=(20, 30))

                ax1 = plt.subplot(321)
                ax1.imshow(self.images[self.global_image_index].array, cmap="gray")
                ax1.set_title("")

                ax2 = plt.subplot(322)
                im12 = ax2.imshow(eps_uu * 100, cmap="BrBG_r", vmin=vmin, vmax=vmax)
                ax2.set_title("$Strain_{uu}$")
                # fig2.colorbar(im12, ax=ax2)

                ax3 = plt.subplot(323)
                im13 = ax3.imshow(eps_vv * 100, cmap="BrBG_r", vmin=vmin, vmax=vmax)
                ax3.set_title("$Strain_{vv}$")
                # fig2.colorbar(im13, ax=ax3)

                ax4 = plt.subplot(324)
                im14 = ax4.imshow(eps_uv * 100, cmap="BrBG_r", vmin=vmin, vmax=vmax)
                ax4.set_title("$Strain_{uv}$")
                # fig2.colorbar(im14, ax=ax4)

                ax5 = plt.subplot(325)
                im15 = ax5.imshow(omega * 100, cmap="BrBG_r", vmin=vmin, vmax=vmax)
                ax5.set_title(r"$\omega$")

                def add_matching_colorbar(fig, ax, im, label, width=0.015, pad=0.01):
                    pos = ax.get_position()
                    cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
                    fig.colorbar(im, cax=cax, label=label)

                add_matching_colorbar(fig2, ax2, im12, "Percent Strain")
                add_matching_colorbar(fig2, ax3, im13, "Percent Strain")
                add_matching_colorbar(fig2, ax4, im14, "Percent Strain")
                add_matching_colorbar(fig2, ax5, im15, "Percent Strain")

                ax1.axis("off")
                ax2.axis("off")
                ax3.axis("off")
                ax4.axis("off")
                ax5.axis("off")

                peak_coords_px = []
                for i in range(num_peaks):
                    pc = self.phase_containers[i]
                    peak_coords_px.append(self.get_xy_2(pc.peak_coordinates))

                peak_coords_px_arr = np.array(peak_coords_px)

                inset_width = 1 / 3
                inset_height = 1 / 3
                inset_ax = ax1.inset_axes(
                    [1 - inset_width, 1 - inset_height, inset_width, inset_height]
                )
                nx, ny = self._FFT[self.global_image_index].array.shape
                win_x = nx // 3
                win_y = ny // 3
                inset_ax.imshow(
                    np.log1p(
                        np.abs(
                            self._FFT[self.global_image_index].array[win_x:-win_x, win_y:-win_y]
                        )
                    ),
                    cmap="gray",
                )
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])

                # inset_ax.set_xticks([])
                # inset_ax.set_yticks([])

                for spine in inset_ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)
                    spine.set_color("black")

                inset_ax.scatter(
                    peak_coords_px_arr[:, 1] - win_y,
                    peak_coords_px_arr[:, 0] - win_x,
                    s=10,
                    facecolors=(1, 0, 0, 0.6),
                    edgecolors=(1, 1, 1, 1),
                    linewidths=0.5,
                )
                inset_origin_x = nx / 2 - win_x - 1
                inset_origin_y = ny / 2 - win_y + 1
                inset_ax.scatter(
                    inset_origin_y,
                    inset_origin_x,
                    s=50,
                    facecolors=(0, 0, 0, 0.6),
                    # edgecolors=(1,1,1,1),
                    linewidths=0.5,
                    zorder=10,
                )

                for x, y in peak_coords_px_arr:
                    inset_ax.annotate(
                        "",
                        xy=(y - win_y, x - win_x),
                        xytext=(inset_origin_y, inset_origin_x),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                    )
                labels = ["p1", "p2"]
                for (x, y), label in zip(peak_coords_px_arr[:2], labels):
                    inset_ax.text(
                        y - win_y,
                        x - win_x + 30,
                        label,
                        color="white",
                        fontsize=10,
                        ha="left",
                        va="bottom",
                    )

        return e_mat

    def show_strain_maps(
        self,
        e_mat=None,
        crop_width=None,
        mean_center=True,
        num_peaks=2,
        amplitude_mask_uniform=None,
        v_lim=None,
        fft_win=None,
        u=None,
        scalebar=None,
        scalebar_i=None,
        return_figax=False,
        # dilation = False,
        # rotation = True,
    ):
        if e_mat is None:
            e_mat = self.e_mat
        eps_uu, eps_vv, eps_uv, u, v = self.get_uv_strain(e_mat, num_peaks, u=u)
        omega, _ = self.get_rot_and_diag_strain(e_mat)

        if crop_width is None:
            crop_width = (
                np.array(self.images[0].array.shape).astype(int)
                - np.array(self.shape_init[0]).astype(int)
            ) / 2
            print(crop_width)
            crop_width = crop_width.astype(int)

        if isinstance(crop_width, (int, float)):
            crop_width = (int(np.round(crop_width)), int(np.round(crop_width)))

        black_mask = self.strain_mask_soft[
            crop_width[0] : -crop_width[0], crop_width[1] : -crop_width[1]
        ]

        # phase image, after fitting carrier wave
        # iteratively fit a plane wave to it
        # periodic mapping?
        # i - i_mean
        # i_mean = <I * m>  / <m>
        # i_mean = np.mean(eps_uu) / (np.mean(black_mask)) # here the black mask is already applied to eps_uu
        if mean_center:
            eps_uu -= np.mean(eps_uu) / np.mean(black_mask)
            eps_vv -= np.mean(eps_vv) / np.mean(black_mask)
            eps_uv -= np.mean(eps_uv) / np.mean(black_mask)
            omega -= np.mean(omega) / np.mean(black_mask)

        if v_lim is not None:
            vmin = -np.abs(v_lim)
            vmax = np.abs(v_lim)

        if v_lim is None:
            vmin = min(np.min(eps_uu), np.min(eps_vv), np.min(eps_uv), np.min(omega)) * 100
            vmax = max(np.max(eps_uu), np.max(eps_vv), np.max(eps_uv), np.max(omega)) * 100

            vmax = max(np.abs(vmin), vmax)
            vmin = -vmax

        # if amplitude_mask_uniform is not None:
        #     if amplitude_mask_uniform in np.arange(0, num_peaks):
        #         amplitude_mask_index = amplitude_mask_uniform
        #     amplitude_map = self.phase_containers[amplitude_mask_index].amplitude_map
        #     # def normalize_arr(array):
        #     #     array-=np.min(array)
        #     #     array /= np.max(array)
        #     #     return array
        #     # amplitude_map_norm = normalize_arr(amplitude_map)
        #     # amplitude_map_norm[amplitude_map_norm < threshold_mask] = 0
        #     # amplitude_map_norm[amplitude_map_norm > 0] = 1
        #     if isinstance(threshold_mask, float):
        #         max_thresh = threshold_mask
        #         min_thresh = 0
        #     elif hasattr(threshold_mask, "__iter__"):
        #         arr = np.asarray(threshold_mask)
        #         if arr.size == 2:
        #             min_thresh, max_thresh = arr
        #         else:
        #             raise ValueError("threshold_mask must have exactly two entries")
        #     else:
        #         raise ValueError("threshold_mask must be a scalar or an iterable with two entries")
        #     amplitude_mask = amplitude_map
        #     amplitude_map_norm = amplitude_mask/np.max(amplitude_mask)
        #     mask_rescaled = (amplitude_map_norm - min_thresh) / (max_thresh - min_thresh)
        #     mask_rescaled[mask_rescaled<0] = 0
        #     mask_rescaled[mask_rescaled>1] = 1

        # # eps_uu[crop_width[0]:-crop_width[0], crop_width[1]:-crop_width[1]] * 100
        # masked_data = np.ma.array(eps_uu, mask = mask < 0.5)   # mask "outside" region

        # cmap = plt.cm.BrBG_r.copy()
        # cmap.set_bad(color='black')   # masked values will be pure black
        # plt.figure()
        # plt.imshow(masked_data, cmap=cmap)

        # plt.figure()
        # plt.imshow(eps_uu, cmap="BrBG_r")
        # print(mask)

        fig2 = plt.figure(figsize=(20, 30))
        plt.rcParams["font.size"] = 20
        # plt.rcParams["axes.titlesize"] = 400
        # plt.rcParams["axes.labelsize"] = 400
        # plt.rcParams["xtick.labelsize"] = 400
        # plt.rcParams["ytick.labelsize"] = 400
        # plt.rcParams["legend.fontsize"] = 400
        # plt.rcParams["figure.titlesize"] = 400

        ax1 = plt.subplot(321)
        ax1.imshow(
            self.images[self.global_image_index].array[
                crop_width[0] : -crop_width[0], crop_width[1] : -crop_width[1]
            ],
            cmap="gray",
        )
        ax1.set_title("")

        from quantem.core.visualization.visualization_utils import (
            _resolve_scalebar,
            add_scalebar_to_ax,
        )

        if scalebar is not None:
            scalebar_config = _resolve_scalebar(scalebar)
            if scalebar_config is not None:
                add_scalebar_to_ax(
                    ax1,
                    self.images[self.global_image_index].array.shape[1],
                    scalebar_config.sampling,
                    scalebar_config.length,
                    scalebar_config.units,
                    scalebar_config.width_px,
                    scalebar_config.pad_px,
                    scalebar_config.color,
                    scalebar_config.loc,
                    scalebar_config.font_size,
                )

        ax2 = plt.subplot(322)
        im12 = ax2.imshow(
            eps_uu[crop_width[0] : -crop_width[0], crop_width[1] : -crop_width[1]] * 100,
            cmap="BrBG_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax2.set_title("$Strain_{uu}$")
        # fig2.colorbar(im12, ax=ax2)

        ax3 = plt.subplot(323)
        im13 = ax3.imshow(
            eps_vv[crop_width[0] : -crop_width[0], crop_width[1] : -crop_width[1]] * 100,
            cmap="BrBG_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax3.set_title("$Strain_{vv}$")
        # fig2.colorbar(im13, ax=ax3)

        ax4 = plt.subplot(324)
        im14 = ax4.imshow(
            eps_uv[crop_width[0] : -crop_width[0], crop_width[1] : -crop_width[1]] * 100,
            cmap="BrBG_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax4.set_title("$Strain_{uv}$")
        # fig2.colorbar(im14, ax=ax4)

        # print('eps_uv 0:2,0:2 pixel values', (eps_uv[crop_width[0]:-crop_width[0], crop_width[1]:-crop_width[1]] * 100)[0:2,0:2])

        ax5 = plt.subplot(325)
        im15 = ax5.imshow(
            omega[crop_width[0] : -crop_width[0], crop_width[1] : -crop_width[1]] * 100,
            cmap="BrBG_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax5.set_title(r"$\omega$")

        ax6 = plt.subplot(326)
        im16 = ax6.imshow(
            (eps_uu + eps_vv)[crop_width[0] : -crop_width[0], crop_width[1] : -crop_width[1]]
            * 100,
            cmap="BrBG_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax6.set_title("$Dilation$")

        ax2.imshow(1 - black_mask, cmap="gray_r", alpha=1 - black_mask)
        ax3.imshow(1 - black_mask, cmap="gray_r", alpha=1 - black_mask)
        ax4.imshow(1 - black_mask, cmap="gray_r", alpha=1 - black_mask)
        ax5.imshow(1 - black_mask, cmap="gray_r", alpha=1 - black_mask)
        ax6.imshow(1 - black_mask, cmap="gray_r", alpha=1 - black_mask)

        def add_matching_colorbar(fig, ax, im, label, width=0.015, pad=0.01):
            pos = ax.get_position()
            cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
            fig.colorbar(im, cax=cax, label=label)

        add_matching_colorbar(fig2, ax2, im12, "Percent Strain")
        add_matching_colorbar(fig2, ax3, im13, "Percent Strain")
        add_matching_colorbar(fig2, ax4, im14, "Percent Strain")
        add_matching_colorbar(fig2, ax5, im15, "Percent Strain")
        add_matching_colorbar(fig2, ax6, im16, "Percent Strain")

        ax1.axis("off")
        ax2.axis("off")
        ax3.axis("off")
        ax4.axis("off")
        ax5.axis("off")
        ax6.axis("off")

        peak_coords_px = []
        for i in range(num_peaks):
            pc = self.phase_containers[i]
            peak_coords_px.append(self.get_xy_2(pc.peak_coordinates))

        peak_coords_px_arr = np.array(peak_coords_px)

        inset_width = 1 / 3
        inset_height = 1 / 3
        inset_ax = ax1.inset_axes([1 - inset_width, 1 - inset_height, inset_width, inset_height])
        nx, ny = self._FFT[self.global_image_index].array.shape
        if fft_win is not None:
            if isinstance(fft_win, (int, float)):
                fft_win = (fft_win, fft_win)
            win_x = fft_win[0]
            win_y = fft_win[1]

        if fft_win is None:
            win_x = nx // 3
            win_y = ny // 3
        inset_ax.imshow(
            np.log1p(np.abs(self._FFT[self.global_image_index].array[win_x:-win_x, win_y:-win_y])),
            cmap="gray",
        )
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

        # inset_ax.set_xticks([])
        # inset_ax.set_yticks([])

        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("black")

        inset_ax.scatter(
            peak_coords_px_arr[:, 1] - win_y,
            peak_coords_px_arr[:, 0] - win_x,
            s=10,
            facecolors=(1, 0, 0, 0.6),
            edgecolors=(1, 1, 1, 1),
            linewidths=0.5,
        )
        inset_origin_x = nx / 2 - win_x - 1
        inset_origin_y = ny / 2 - win_y + 1
        inset_ax.scatter(
            inset_origin_y,
            inset_origin_x,
            s=50,
            facecolors=(0, 0, 0, 0.6),
            # edgecolors=(1,1,1,1),
            linewidths=0.5,
            zorder=10,
        )

        for x, y in peak_coords_px_arr:
            inset_ax.annotate(
                "",
                xy=(y - win_y, x - win_x),
                xytext=(inset_origin_y, inset_origin_x),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            )
        labels = ["p1", "p2"]
        for (x, y), label in zip(peak_coords_px_arr[:2], labels):
            inset_ax.text(
                y - win_y,
                x - win_x + 40,
                label,
                color="white",
                fontsize=10,
                ha="left",
                va="bottom",
            )
        if scalebar_i is not None:
            scalebar_config = _resolve_scalebar(scalebar_i)
            if scalebar_config is not None:
                add_scalebar_to_ax(
                    inset_ax,
                    self.images[self.global_image_index].array.shape[1],
                    scalebar_config.sampling,
                    scalebar_config.length,
                    scalebar_config.units,
                    scalebar_config.width_px,
                    scalebar_config.pad_px,
                    scalebar_config.color,
                    scalebar_config.loc,
                    scalebar_config.font_size,
                )

        uv_arr = np.array([u, v])
        origin_x = 7 / 8 * nx - crop_width[0] * 2
        origin_y = 7 / 8 * ny - crop_width[1] * 2
        for x, y in uv_arr:
            ax1.annotate(
                "",
                xy=(
                    y * (ny - crop_width[1] * 2) / 10 + origin_y,
                    x * (nx - crop_width[0] * 2) / 10 + origin_x,
                ),
                xytext=(origin_y, origin_x),
                arrowprops=dict(arrowstyle="->", color="red", lw=4),
            )
        labels = ["u", "v"]
        for (x, y), label in zip(uv_arr, labels):
            ax1.text(
                y * (ny - crop_width[1] * 2) / 10 + origin_y + 20,
                x * (nx - crop_width[0] * 2) / 10 + origin_x + 40,
                label,
                color="white",
                fontsize=25,
                ha="left",
                va="bottom",
            )
        ax1.scatter(
            origin_y,
            origin_x,
            s=100,
            facecolors=(0, 0, 0, 0.9),
            # edgecolors=(1,1,1,1),
            linewidths=0.5,
            zorder=10,
        )

        plt.rcParams["font.size"] = 12

        # now show the direction of the u and v vectors
        # inset_ax_2 = ax1.inset_axes([1-inset_width, 1-inset_height, inset_width, inset_height])
        # nx, ny = self._FFT[self.global_image_index].array.shape
        # if fft_win is not None:
        #     if isinstance(fft_win, (int, float)):
        #         fft_win = (fft_win, fft_win)
        #     win_x = fft_win[0]
        #     win_y = fft_win[1]

        # if fft_win is None:
        #     win_x = nx//3
        #     win_y = ny//3
        # inset_ax_2.imshow(np.log1p(np.abs(self._FFT[self.global_image_index].array[win_x:-win_x, win_y:-win_y])), cmap="gray")
        # inset_ax_2.set_xticks([])
        # inset_ax_2.set_yticks([])

        # for spine in inset_ax_2.spines.values():
        #     spine.set_visible(True)
        #     spine.set_linewidth(1.5)
        #     spine.set_color("black")

        # inset_ax_2.scatter(
        #     peak_coords_px_arr[:,1] - win_y,
        #     peak_coords_px_arr[:,0] - win_x,
        #     s=10,
        #     facecolors=(1,0,0,0.6),
        #     edgecolors=(1,1,1,1),
        #     linewidths = 0.5,
        # )
        # inset_origin_x = nx/2 - win_x-1
        # inset_origin_y = ny/2 - win_y+1
        # inset_ax_2.scatter(
        #     inset_origin_y,
        #     inset_origin_x,
        #     s=50,
        #     facecolors=(0,0,0,0.6),
        #     # edgecolors=(1,1,1,1),
        #     linewidths = 0.5,
        #     zorder = 10,
        # )

        # for (x, y) in peak_coords_px_arr:
        #     inset_ax_2.annotate(
        #         "",
        #         xy=(y - win_y, x - win_x),
        #         xytext=(inset_origin_y, inset_origin_x),
        #         arrowprops=dict(
        #             arrowstyle="->",
        #             color="red",
        #             lw=1.5
        #         ),
        #     )
        # labels = ["u", "v"]
        # for (x, y), label in zip(peak_coords_px_arr[:2], labels):
        #     inset_ax_2.text(
        #         y - win_y,
        #         x - win_x+80,
        #         label,
        #         color="white",
        #         fontsize=10,
        #         ha="left",
        #         va="bottom"
        #     )

        if return_figax:
            return fig2

        return self

    def get_rot_and_diag_strain(
        self,
        e_mat: np.ndarray,
    ):
        """
        Unpack and build the rotation and shear strain components.

        Parameters
        ----------
        e_mat: (2,2, nx, ny)
            The strain maps.

        Returns
        -------
        e_th_xy: (nx, ny) np.ndarray
            The xy rotation matrix.
        e_dg_xy: (nx, ny) np.ndarray
            The xy shear strain matrix.
        """
        e_th_xy = 0.5 * (e_mat[0, 1] - e_mat[1, 0])

        e_dg_xy = 0.5 * (e_mat[0, 1] + e_mat[1, 0])
        return e_th_xy, e_dg_xy

    def get_axial_strain(
        self,
        e_mat: np.ndarray,
    ):
        """
        Unpack and build the axial strain components.

        Parameters
        ----------
        e_mat: (2, 2, nx, ny)
            The strain maps.

        Returns
        -------
        e_mat[0,0]: (nx, ny) np.ndarray
            The x (row) axial strain.
        e_mat[1,1]: (nx, ny) np.ndarray
            The y (column) axial strain.
        """
        return e_mat[0, 0], e_mat[1, 1]

    def get_uv_strain(
        self,
        e_mat,
        num_peaks,
        u=None,
        v=None,
        theta=None,
    ):
        eps_xx, eps_yy = self.get_axial_strain(e_mat)
        _, eps_xy = self.get_rot_and_diag_strain(e_mat)
        shape = np.array(eps_xx.shape)
        if theta is not None:
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            u = (R @ np.array([1, 0]).T).T
            v = (R @ np.array([0, 1]).T).T

        elif u is None:
            peak_coords_px = []
            for i in range(num_peaks):
                pc = self.phase_containers[i]
                peak_coords_px.append(self.get_xy_2(pc.peak_coordinates))
            peak_coords_px_arr = np.array(peak_coords_px)
            u = peak_coords_px_arr[0] - shape / 2
            if v is None:
                theta = np.pi / 2
                R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                v = (R @ u.T).T
        elif u is not None:
            theta = np.pi / 2
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            v = (R @ u.T).T
        # elif u is None:
        #     peak_coords_px = []
        #     for i in range(num_peaks):
        #         pc = self.phase_containers[i]
        #         peak_coords_px.append(self.get_xy_2(pc.peak_coordinates))
        #     peak_coords_px_arr = np.array(peak_coords_px)
        #     u = peak_coords_px_arr[0] - shape/2
        #     if v is None:
        #         v = peak_coords_px_arr[1] - shape/2

        # print(u, v)
        # normalize
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        ux = u[0]
        uy = u[1]
        vx = v[0]
        vy = v[1]
        eps_uu = ux * ux * eps_xx + 2 * ux * uy * eps_xy + uy * uy * eps_yy
        eps_vv = vx * vx * eps_xx + 2 * vx * vy * eps_xy + vy * vy * eps_yy
        eps_uv = ux * vx * eps_xx + (ux * vy + uy * vx) * eps_xy + uy * vy * eps_yy

        return eps_uu, eps_vv, eps_uv, u, v

    def refine_phases(
        self,
        reference_center,
        reference_radius,
        iterations,
        useGaussMask,
        show_result,
        mask_size,
        amplitude_mask_result=False,
        threshold_mask: float = 0.4,
    ):
        if mask_size is None and self.mask_size is not None:
            mask_size = self.mask_size
        elif mask_size is None and self.mask_size is None:
            raise RuntimeError("No mask size set")
        refMatrix = self.set_reference_matrix_s(reference_center, reference_radius)
        for peak_index in range(len(self.phase_containers)):
            phase = self.phase_containers[peak_index].phase_map
            peak = self.phase_containers[peak_index].phase_map
            peakRefined, phaseRefined = self.refine_phase(
                phase,
                peak,
                peak_index,
                refMatrix,
                mask_size,
                iterations,
                useGaussMask,
                show_result,
                amplitude_mask_result,
                threshold_mask=threshold_mask,
            )
            self.phase_containers[peak_index].phase_refined = phaseRefined
            self.phase_containers[peak_index].peak_refined = peakRefined
        return self

    def define_reference(
        self,
        x1: int,
        x2: int,
        y1: int,
        y2: int,
    ):
        """
        Locate visually the unstrained reference region.

        Parameters
        ----------
        x1:      Left limiting line
        x2:     Right limiting line
        y1:     Bottom limiting line
        y2:     Top limiting line

        Returns
        -------
        ref_reg: np.ndarray
                Boolean indices marking the reference region in 2D
        """
        xx, yy = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        ref_reg = np.logical_and(
            np.logical_and(xx > x1, xx < x2), np.logical_and(yy > y1, yy < y2)
        )

        A = (x1, y2)
        B = (x2, y2)
        C = (x2, y1)
        D = (x1, y1)

        plt.figure(figsize=(15, 15))
        plt.imshow(
            self.image_normalizer(self.images[self.global_image_index].array) + 0.33 * ref_reg,
            origin="upper",
        )
        plt.annotate(
            A,
            (A[0] / self.nx, (1 - A[1] / self.ny)),
            textcoords="axes fraction",
            size=15,
            color="w",
        )
        plt.annotate(
            B,
            (B[0] / self.nx, (1 - B[1] / self.ny)),
            textcoords="axes fraction",
            size=15,
            color="w",
        )
        plt.annotate(
            C,
            (C[0] / self.nx, (1 - C[1] / self.ny)),
            textcoords="axes fraction",
            size=15,
            color="w",
        )
        plt.annotate(
            D,
            (D[0] / self.nx, (1 - D[1] / self.ny)),
            textcoords="axes fraction",
            size=15,
            color="w",
        )
        plt.scatter(
            A[1], A[0]
        )  # scatter uses column row ordering, so we must put these in reverse order.
        plt.scatter(B[1], B[0])
        plt.scatter(C[1], C[0])
        plt.scatter(D[1], D[0])
        plt.axis("off")
        return ref_reg

    def set_reference_matrix(
        self,
        planeX1: int,
        planeX2: int,
        planeY1: int,
        planeY2: int,
    ):
        """
        Dictate a region of ideal (minimally distorted) lattice using 4 lines. This region will be a rectangle.

        Parameters
        ----------
        planeX1: int
            The lower bounding x plane for the region.
        planeX2: int
            The upper bounding x plane for the region.
        planeY1: int
            The lower bounding y plane for the region.
        planeY2: int
            The upper bounding y plane for the region.

        Returns
        -------
        referenceMatrix: np.ndarray, bool
            The reference (ideal) region of the crystal.
        """
        referenceMatrix = self.define_reference(planeX1, planeX2, planeY1, planeY2)
        return referenceMatrix

    def set_reference_matrix_s(
        self,
        centerOfReferenceRegion: int,
        radiusOfReferenceRegion: int,
    ):
        """
        Dictate a region of ideal (minimally distorted) lattice using a center and square half side length. This region will be a square.

        Parameters
        ----------
        centerOfReferenceRegion: (2) np.ndarray
            An array with 2 entries giving the X (row) and Y (column) coordinates of the center of the region. These coordinates should be absolute and in pixels.
        radiusOfReferenceRegion: int
            An integer value that gives the half side length of the square defining the reference region.

        Returns
        -------
        referenceMatrix: np.ndarray, bool
            The reference (ideal) region of the crystal.
        """
        planeX1 = centerOfReferenceRegion[0] - radiusOfReferenceRegion
        planeX2 = centerOfReferenceRegion[0] + radiusOfReferenceRegion
        planeY1 = centerOfReferenceRegion[1] - radiusOfReferenceRegion
        planeY2 = centerOfReferenceRegion[1] + radiusOfReferenceRegion
        referenceMatrix = self.define_reference(planeX1, planeX2, planeY1, planeY2)
        return referenceMatrix

    def refine_phase(
        self,
        phaseMap: np.ndarray,
        peakCoordinates: np.dtype([("x", float), ("y", float), ("intensity", float)]),
        peak_index,
        referenceMatrix: np.ndarray,
        maskSize: float,
        iterations: int,
        useGaussMask: bool,
        amplitude_mask_result=False,
        show_result: bool = True,
        threshold_mask: np.ndarray | float = 0.4,
    ):
        """
        Refine the geometric phase according to the user-defined reference (ideal) region of the crystal.

        Parameters
        ----------
        phaseMap: (nx, ny) np.ndarray
            A 2D geometric phase map (to be refined).
        peakCoordinates: np.dtype([("x", float), ("y", float), ("intensity", float)])
            The coordinates for the peak used to generate the phase map.
        referenceMatrix: np.ndarray, bool
            The user-defined reference (ideal) region of the crystal.
        maskSize: float
            The size of the input mask. Highly tunable. Lower values correspond to larger convolution kernel and lower resolution.
        iterations: int
            The number of iterations of phase refinement.
        useGaussMask: bool
            Control for whether to use a Gaussian mask or circular binary mask. Defaults to True (Gaussian).
        show_result: bool
            Show the real space geometric phase alongside the shifted Fourier transform and the Gaussian mask. Defaults to True.

        Returns
        -------
        peakCoordinatesRefined_dtype: np.dtype([("x", float), ("y", float), ("intensity", float)])
            The refined peak coordinates in a custom dtype.
        phaseMapRefined: (nx, ny) np.ndarray
            The 2D phase map after refinement.
        """
        peakCoordinates_xy = self.get_xy_2(peakCoordinates)
        ry = np.arange(start=-self.nx / 2, stop=self.nx / 2, step=1)
        rx = np.arange(start=-self.ny / 2, stop=self.ny / 2, step=1)
        rx, ry = np.meshgrid(rx, ry, indexing="ij")

        peakCoordinatesRefined = peakCoordinates_xy.copy()
        phaseMapRefined = phaseMap.copy()
        for _ in range(int(iterations)):
            G_x, G_y = self.phase_diff(phaseMapRefined)
            G_nabla = G_x + G_y
            g_r = G_nabla / (2 * np.pi)
            del_g = np.asarray(
                (
                    np.median(g_r[referenceMatrix] / rx[referenceMatrix]),
                    np.median(g_r[referenceMatrix] / ry[referenceMatrix]),
                )
            )
            peakCoordinatesRefined += del_g
            peakCoordinatesRefined_dtype = np.zeros(1, dtype=self.dtype)
            peakCoordinatesRefined_dtype["x"] = peakCoordinatesRefined[0]
            peakCoordinatesRefined_dtype["y"] = peakCoordinatesRefined[1]
            self.calculate_phase_map(
                peak_index,
                gaussianMask=useGaussMask,
                inputMaskSize=maskSize,
                show_result=False,
                amplitude_mask_result=False,
                refine=True,
            )
            phaseMapRefined = self.phase_containers[peak_index].refined_phase_map
            amplitudeMapRefined = self.phase_containers[peak_index].refined_amplitude_map

        if show_result:
            if amplitude_mask_result:
                if isinstance(threshold_mask, float):
                    max_thresh = threshold_mask
                    min_thresh = 0
                elif hasattr(threshold_mask, "__iter__"):
                    arr = np.asarray(threshold_mask)
                    if arr.size == 2:
                        min_thresh, max_thresh = arr
                    else:
                        raise ValueError("threshold_mask must have exactly two entries")
                else:
                    raise ValueError(
                        "threshold_mask must be a scalar or an iterable with two entries"
                    )
                amplitude_map_norm = amplitudeMapRefined / np.max(amplitudeMapRefined)
                mask_rescaled = (amplitude_map_norm - min_thresh) / (max_thresh - min_thresh)
                mask_rescaled[mask_rescaled < 0] = 0
                mask_rescaled[mask_rescaled > 1] = 1

                mask_soft = np.sin(np.pi / 2 * mask_rescaled) ** 2
                # amplitude_map_norm[amplitude_map_norm < threshold_mask] = 0
                # amplitude_map_norm[amplitude_map_norm > 0] = 1
                im_pha_gp = self.phase_im_lab(phaseMapRefined * mask_soft)
            else:
                im_pha_gp = self.phase_im_lab(phaseMapRefined)
            if show_result:
                show_2d(
                    [
                        self.images[self.global_image_index].array,
                        im_pha_gp,
                    ],
                    title=["", "$Strain_{xx}$"],
                    cmap=["gray", "BrBG_r"],
                )

        peakCoordinatesRefined_dtype = np.zeros(1, dtype=self.dtype)
        peakCoordinatesRefined_dtype["x"] = peakCoordinatesRefined[0]
        peakCoordinatesRefined_dtype["y"] = peakCoordinatesRefined[1]
        return peakCoordinatesRefined_dtype, phaseMapRefined

    def phase_diff(
        self,
        angle_image: np.ndarray,
    ):
        """
        Differentiate the complex exponential of the phase image, and then obtain the
        differentiation result by multiplying the differential with
        the conjugate of the complex phase image.
        Here, the image is 2D.

        Parameters
        ----------
        angle_image:  np.ndarray
                    Wrapped phase image

        Returns
        -------
        diff_x: np.ndarray
                X difference of the phase image
        diff_y: np.ndarray
                Y difference of the phase image
        """
        imaginary_image = np.exp(1j * angle_image)

        diff_imaginary_x = np.zeros(imaginary_image.shape, dtype=complex)
        diff_imaginary_x[0:-1, :] = np.diff(imaginary_image, axis=0)
        diff_imaginary_y = np.zeros(imaginary_image.shape, dtype=complex)
        diff_imaginary_y[:, 0:-1] = np.diff(imaginary_image, axis=1)

        conjugate_imaginary = np.conj(imaginary_image)
        diff_complex_x = np.multiply(conjugate_imaginary, diff_imaginary_x)
        diff_complex_y = np.multiply(conjugate_imaginary, diff_imaginary_y)

        diff_x = np.imag(diff_complex_x)
        diff_y = np.imag(diff_complex_y)

        return diff_x, diff_y

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
        midX = self.nx // 2
        midY = self.ny // 2
        peakCoordinatesRespCenter = np.zeros(len(peakCoordinates), dtype=self.dtype)
        peakCoordinatesRespCenter["x"] = peakCoordinates["x"] - midX
        peakCoordinatesRespCenter["y"] = peakCoordinates["y"] - midY
        peakRadialDistCenter = (
            peakCoordinatesRespCenter["x"] ** 2 + peakCoordinatesRespCenter["y"] ** 2
        )

        smallestRadiiIndices = np.argsort(peakRadialDistCenter)
        peakCoordinatesRespCenter = peakCoordinatesRespCenter[smallestRadiiIndices]

        # The closest peak should be the zero order peak - not interested in that.
        peakAInd = 1
        peakBInd = None
        ####
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

        peakA = np.zeros(1, dtype=self.dtype)
        peakB = np.zeros(1, dtype=self.dtype)

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
        outer_ignore_buffer: int | None = None,
        shift_fft=np.array([0, 0]),
        show_result=False,
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
        # plt.figure(figsize = (10,10),dpi = 300)
        # plt.subplot(121)
        # plt.imshow(np.log(np.abs(np.roll(self._FFT[self.global_image_index].array, shift = shift_fft, axis = (0,1)))), cmap = 'gray')
        # plt.subplot(122)
        # win = 400
        # plt.imshow(np.log(np.abs(np.roll(self._FFT[self.global_image_index].array, shift = shift_fft, axis = (0,1))[win:-win, win:-win])), cmap = 'gray')

        shape = self._FFT[self.global_image_index].array.shape
        lower_x = 0
        lower_y = 0
        upper_x = shape[0]
        upper_y = shape[1]
        if outer_ignore_buffer is not None:
            lower_x = int(np.floor(shape[0] / 2 - outer_ignore_buffer - 3))
            upper_x = int(np.ceil(shape[0] / 2 + outer_ignore_buffer + 3))
            lower_y = int(np.floor(shape[1] / 2 - outer_ignore_buffer - 3))
            upper_y = int(np.ceil(shape[1] / 2 + outer_ignore_buffer + 3))
        peakList = self.get_maxima_2D(
            np.abs(
                np.roll(self._FFT[self.global_image_index].array, shift=shift_fft, axis=(0, 1))[
                    lower_x : upper_x + 1, lower_y : upper_y + 1
                ]
            ),
            maxNumPeaks=maxNumPeaks_in,
            minSpacing=minSpacingPeaks,
        )

        peak_xy_all = np.zeros([peakList.shape[0], 2])
        for peak_index in range(peakList.shape[0]):
            peak_xy = np.array(
                [peakList[peak_index]["x"] + lower_x, peakList[peak_index]["y"] + lower_y]
            )
            peak_xy_all[peak_index] = peak_xy
        # plt.scatter(peak_xy_all[:,1]-win, peak_xy_all[:,0]-win, c = 'red', s=10, alpha = 0.5)
        # print(peakList)
        peakList["x"] += lower_x
        peakList["y"] += lower_y
        if center_ignore_buffer is not None:
            x_dist_to_center = peakList["x"] - self.nx / 2
            y_dist_to_center = peakList["y"] - self.ny / 2
            rad_dist_to_center = np.sqrt(x_dist_to_center**2 + y_dist_to_center**2)
            peakList = peakList[rad_dist_to_center > center_ignore_buffer]
            zero_peak = np.zeros(1, np.dtype([("x", float), ("y", float), ("intensity", float)]))
            zero_peak["x"] = self.nx / 2
            zero_peak["y"] = self.ny / 2
            peakList = np.append(zero_peak, peakList)
        if outer_ignore_buffer is not None:
            x_dist_to_center = peakList["x"] - self.nx / 2
            y_dist_to_center = peakList["y"] - self.ny / 2
            rad_dist_to_center = np.sqrt(x_dist_to_center**2 + y_dist_to_center**2)
            peakList = peakList[rad_dist_to_center < outer_ignore_buffer]
        return peakList

    # Functions from py4DSTEM for peak finding.
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
        except Exception:
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
