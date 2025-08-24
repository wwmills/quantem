from typing import Literal

import numpy as np
from tqdm import trange

from quantem.core import config
from quantem.core.datastructures import Dataset
from quantem.core.io.serialize import AutoSerialize
from quantem.core.datastructures.dataset3d import Dataset3d
from scipy.fftpack import fft, fftshift, ifftshift, ifft, fft2, ifft2, fftn, ifftn
from skimage import restoration as skr
from scipy.optimize import curve_fit
from skimage.color import lab2rgb
from scipy import ndimage as scnd
import numba
from scipy.ndimage import gaussian_filter

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp
    
import matplotlib.pyplot as plt

def create_lattice(
    n_rows: int,
    n_cols: int,
    n_stacks: int,
    a_rows: int,
    a_cols: int,
    a_stacks: int,
    ):  
    """
    This function creates a square image with lattice parameters a1 and a2.
    This can be used to create a simple lattice for demonstration purposes.
    
    Parameters
    ----------
    n_rows: int
        Number of lattice sites in the row direction  (the x direction, by convention)
    n_cols: int
        Number of lattice sites in the column direction  (the y direction, by convention)
    n_stacks: int
        Number of lattice sites in the z direction
    a_rows: int
        Lattice parameter in the row direction (the x direction, by convention)
    a_cols: int
        Lattice parameter in the column direction (the y direction, by convention)
    a_stacks: int
        Lattice parameter in the z direction
    
    Returns
    -------
    coords: (n_rows*n_cols*n_stacks, 3) np.ndarray
        Coordinates with the row (x) coordinates in the first column,
        the column (y) coordinates in the second column, and the z
        coordinates in the third column.
    """
    ind = 0
    coords = np.zeros([n_rows*n_cols*n_stacks,3])

    a_rows_array = np.array([a_rows, 0,0])
    a_cols_array = np.array([0,a_cols,0])
    a_stacks_array = np.array([0,0,a_stacks])
    for row in range(n_rows):
        for col in range(n_cols):
            for stack in range(n_stacks):
                coords[ind] = row * a_rows_array + col * a_cols_array + stack * a_stacks_array
                ind += 1
    return coords


class geometric_phase_analysis_3D(AutoSerialize):
    """
    A class for performing geometric phase retrieval on 3D real space volumes using Gaussian fitting and Fourier transforms.
    
    This can be used to retrieve atomic displacements and strain maps.
    """

    def __init__(
        self,
        volume: Dataset3d,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        """
        Parameters
        ----------
        volume: (nx, ny, nz) np.ndarray
            A 3D volume in real space.
        device: string
            The device to use, either cpu or gpu. The default is cpu
        """
        # self.xp = cp if device == "gpu" else np
        self.volume = np.asarray(volume)
        [self.nx, self.ny, self.nz] = self.volume.shape
        self.device = device
        self.imFFT = fftshift(fftn(self.volume))
        self.dtype = np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)]) # the py4DSTEM data type for real space data, extended to 3D

    def get_FFT(
        self,
    ):
        """
        Returns the FFT of the input dataset.

        Returns
        -------
        self.imFFT: (nx, ny, nz) np.ndarray, complex128
            The FFT of the input data.
        """
        return self.imFFT
    
    def fourier_filter(
        self, 
        data: np.ndarray, 
        threshold: int = 1,
        show_plot: bool = False,
    ):
        """
        Calculates a mask based on the low frequency structure in real space. Signal is set to one, vacuum is set to zero.
        
        Parameters
        ----------
        data: (nx, ny, nz) np.ndarray
            A 3D volume in real space that matches the dimensions of self.volume.
        threshold: int
            A Fourier threshold value for the real space amplitude after filtering. Defaults to 1.
        show_plot: bool
            Controls if the mask and original real space are shown. Defaults to False.
        
        Returns
        -------
        self.fourier_mask * data: (nx, ny, nz) np.ndarray
            The input data multiplied by a binary mask.
        """
        if data.shape != self.volume.shape:
            print("Input shape does not match that of original image")
            return 0
        xx,yy,zz = np.meshgrid(np.arange(self.nx),np.arange(self.ny),np.arange(self.nz), indexing = 'ij')
        dkx = 1/(self.nx); dky = 1/(self.ny); dkz = 1/(self.nz)

        center = np.array(self.imFFT.shape)/2
        mask_size = 10
        gaussCoords = ((xx - center[0])**2 + (yy - center[1])**2 + (zz - center[2])**2) / mask_size**2
        del xx, yy, zz
        mask = np.exp( -0.5 * gaussCoords, dtype=np.float32 )
        del gaussCoords
        self.fourier_mask = np.abs((ifftn(self.imFFT*mask)))*100
        self.fourier_mask[self.fourier_mask<threshold] = 0
        self.fourier_mask[self.fourier_mask>0] = 1
        if show_plot:
            plt.figure(figsize = (5,10))
            plt.subplot(121)
            plt.imshow(self.fourier_mask[self.nx//2,:,:], origin = 'upper'); plt.axis('off')
            plt.subplot(122)
            plt.imshow(self.fourier_mask[self.nx//2,:,:] * data[self.nx//2], origin = 'upper'); plt.axis('off')        
        return self.fourier_mask * data
    

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
            The phase image, which may be of any dimensionality.
        brightness: int
            The brightness of the output color image. Defaults to 60.
        saturation: int
            The saturation of the output color image. Defaults to 60.
        
        Returns
        -------
        im_pha_gp: np.ndarray
            The phase represented by 3 color channels. The dimensionality is im_pha_gp.shape = phaseIM.shape, 3. 
        """
        L = brightness * (1 + np.zeros(phaseIM.shape))     # Brightness
        a = saturation * np.cos(phaseIM)                   # Saturation
        b = saturation * np.sin(phaseIM)                   # Saturation
        im_pha_gp = lab2rgb(np.dstack((L,a,b)))
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
        peakCoordinates: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)]), 
        subImageHalfLength: int = 20,
        ):
        """
        Zero in on peak location by fitting with a Gaussian.
        
        Parameters
        ----------
        peakCoordinates: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            One set of peak coordinates.
        subImageHalfLength: int
            Half of side length of sub volume for peak fitting. Defaults to 20. 
        
        Returns
        -------
        peakCoordinatesPrecise: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            A more precise estimate of the Bragg peak location.
        """
        decimal_x = peakCoordinates['x'] - int(peakCoordinates['x'])
        decimal_y = peakCoordinates['y'] - int(peakCoordinates['y'])
        decimal_z = peakCoordinates['z'] - int(peakCoordinates['z'])
        subIm = np.abs(self.imFFT)[int(peakCoordinates['x']-subImageHalfLength):int(peakCoordinates['x']+subImageHalfLength), 
                                                    int(peakCoordinates['y']-subImageHalfLength):int(peakCoordinates['y']+subImageHalfLength),
                                                    int(peakCoordinates['z']-subImageHalfLength):int(peakCoordinates['z']+subImageHalfLength),]
        GaussianFit = self.fit_diffraction_center(subIm)
        peakCoordinatesPrecise = np.zeros(1, dtype=self.dtype)
        peakCoordinatesPrecise['x'] = peakCoordinates['x'] + GaussianFit[2] - subImageHalfLength - decimal_x
        peakCoordinatesPrecise['y'] = peakCoordinates['y'] + GaussianFit[3] - subImageHalfLength - decimal_y
        peakCoordinatesPrecise['z'] = peakCoordinates['z'] + GaussianFit[4] - subImageHalfLength - decimal_z
        return peakCoordinatesPrecise

    def gauss3D(
        self,
        xdata: np.ndarray,
        A: float,
        B: float,
        xc: float,
        yc: float,
        zc: float,
        sx: float,
        sy: float,
        sz: float,
        ):
        
        """
        A 3D Gaussian (without any rotation effects).
        
        Parameters
        ----------
        xdata: (nx, ny, nz, 3) np.ndarray
            The input subvolume coordinates. X, Y, and Z coordinates should be present.
        A: float
            The amplitude multiplier of the Gaussian.
        B: float
            The scalar offset of the Gaussian.
        xc: float
            The central coordinate of the Gaussian in the X (row) direction.
        yc: float
            The central coordinate of the Gaussian in the Y (column).
        zc: float
            The central coordinate of the Gaussian in the Z.
        sx: float
            The standard deviation of the Gaussian in the X (row) direction.
        sy: float
            The standard deviation of the Gaussian in the Y (column) direction.
        sz: float
            The standard deviation of the Gaussian in the Z direction.
            
        Returns
        -------
        G: (nx, ny, nz) np.ndarray
            The 3D Gaussian.
        """
        xx = xdata[:,:,:,0]; yy = xdata[:,:,:,1]; zz = xdata[:,:,:,2]
        position_array = np.array([(xx-xc).flatten(), (yy-yc).flatten(), (zz-zc).flatten()])
        G = A*np.exp(-0.5*((position_array[0]/sx)**2 +(position_array[1]/sy)**2 + (position_array[2]/sz)**2)) + B
        return G.ravel()


    def fit_diffraction_center(
        self,
        subIm: np.ndarray,
        plot_results: bool =True,
        ):
        
        """
        Given a sub volume of the Fourier transform, use curve fitting to improve the estimate of the Bragg peak's central coordinates.
        
        Parameters
        ----------
        subIm: (nx, ny, nz) np.ndarray
            The subvolume of the Fourier transform. This should ideally contain a single strongest Bragg peak close to the center of the volume.
        plot_results: bool
            Plot the subvolume and Gaussian fit of the Bragg peak. Defaults to True.
            
        Returns
        -------
        popt: (8) np.ndarray
            An array of the optimal values returned by the curve fit algorithm. These entries have the following identities: [amplitude, offset, center x, center y, center z, std x, std y, std z].
        """
        (xx,yy,zz) = np.meshgrid(np.arange(subIm.shape[0]), np.arange(subIm.shape[1]), np.arange(subIm.shape[2]), indexing = 'ij')
        xdata = np.stack((xx,yy,zz), axis=-1)

        A0 = np.max(subIm); B0 = np.min(subIm)
        x0 = np.array([A0,   B0, subIm.shape[0]/2,   subIm.shape[1]/2, subIm.shape[2]/2,    0.5,   0.5, 0.5])
        lb = np.array([A0/4, 0,  subIm.shape[0]*3/8,   subIm.shape[1]*3/8, subIm.shape[2]*3/8,   0, 0, 0])
        ub = np.array([2*A0, A0, subIm.shape[0]*5/8, subIm.shape[1]*5/8, subIm.shape[2]*5/8, 5,   5, 5])

        popt, pcov = curve_fit(self.gauss3D, xdata, subIm.ravel(), p0=x0, bounds=(lb,ub))

        if plot_results:
            data_fitted = self.gauss3D(xdata,*popt)
            data_fitted = data_fitted.reshape(*subIm.shape)
            plt.figure()
            plt.imshow(subIm[:,:,int(subIm.shape[2]/2)],cmap='gray', origin = 'upper')
            plt.contour(xdata[:,:,int(subIm.shape[2]/2), 1],xdata[:,:, int(subIm.shape[2]/2), 0], data_fitted[:,:,int(subIm.shape[2]/2)]) # 1 and 0 are flipped here, because contour has column row ordering.
        return popt

    def calculate_phase_map(
        self,
        peakCoordinates: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)]), 
        inputMaskSize: float, 
        gaussianMask: bool = True,
        useHamming: bool = False,
        showResult: bool = True,
        ):
        """
        Calculate the geometric phase for a single Bragg peak.
        
        Parameters
        ----------
        peakCoordinates: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            Coordinates of the Bragg peak.
        inputMaskSize: float
            The size of the input mask. Highly tunable. Lower values correspond to larger convolution kernel and lower resolution.
        guassianMask: bool
            Control for whether to use a Gaussian mask or circular binary mask. Defaults to True (Gaussian).
        useHamming: bool
            Control whether to use a Hamming window in k-space. Defaults to False.
        showResult: bool
            Show the real space geometric phase alongside the shifted Fourier transform and the Gaussian mask. Defaults to True.
            
        Returns
        -------
        G_matrix: (nx, ny, nz) np.ndarray
            A 3D array of the geometric phase corresponding to the input peak.
        """
        peakCoordinates_xyz = self.get_xyz_2(peakCoordinates)

        # Construct Fourier Coordinates
        xx,yy,zz = np.meshgrid(np.arange(self.nx),np.arange(self.ny),np.arange(self.nz), indexing = 'ij')
        dkx = 1/(self.nx); dky = 1/(self.ny); dkz = 1/(self.nz)

        # Shift Bragg Peak to Center
        center = np.array(self.volume.shape)/2
        shift = np.array( center - peakCoordinates_xyz)
        shift = shift * [dkx, dky, dkz]
        shift_phase = np.exp(1j*2*np.pi*(shift[0]*xx+shift[1]*yy+shift[2]*zz))

        if gaussianMask: # Create Mask with Gaussian Kernel
            xx,yy,zz = np.mgrid[0:self.nx,0:self.ny,0:self.nz]
            gg = (((xx - center[0])**2) + ((yy - center[1])**2) + ((zz-center[2])**2))/inputMaskSize
            mask = np.exp((-0.5)*gg)
        else:             # Create a Hard Circle Mask
            circ_rad = np.amin(inputMaskSize*np.asarray(self.volume.shape))
            mask = (self.make_sphere(self.volume.shape,self.nx/2,self.ny/2, self.nz/2,circ_rad)).astype(bool)

        if useHamming:
            ham_x = np.hamming(self.nx)[:, None, None]
            ham_y = np.hamming(self.ny)[None, :, None]
            ham_z = np.hamming(self.nz)[None, None, :]
            ham = np.sqrt(ham_x * ham_y * ham_z)
            G_matrix = ifftn(ifftshift(mask*fftshift(fftn(self.volume*ham*shift_phase))))    # With hamming in 3D. Original methods would take the phase immediately, but the amplitude is also useful.
        else:
            G_matrix = ifftn(ifftshift(mask*fftshift(fftn(self.volume*shift_phase))))    # Without hamming


        if showResult:
            im_pha_gp = self.phase_im_lab(np.angle(G_matrix[:,:,int(self.nz//2)]))
            imFFT = fftshift(fftn(self.volume*shift_phase))
            (_,axs) = plt.subplots(1,2,figsize=(15,30))
            axs[0].imshow(im_pha_gp, origin = 'upper')#; axs[0].axis('off')
            axs[1].imshow(np.log(np.abs(imFFT[:,:,self.nz//2])+1),cmap='gray', origin = 'upper'); plt.imshow(mask[:,:,self.nz//2],alpha=0.4, origin = 'upper'); axs[1].axis('off')

        return G_matrix

    def make_sphere(
        self,
        size_sphere: np.ndarray,
        center_x: float,
        center_y: float,
        center_z: float,
        radius: float,
        ):
        """
        Make a sphere mask.
        
        Parameters
        ----------
        size_sphere: (3) np.ndarray
                3 element array giving the size of the output matrix
        center_x: float
                x position of sphere center
        center_y: float
                y position of sphere center
        center_z: float
                z position of sphere center
        radius: float
                radius of the sphere
        
        Returns
        -------
        sphere: (nx, ny nz) np.ndarray, bool
                Binary circular mask.
        """
        p = size_circ[0]
        q = size_circ[1]
        s = size_circ[2]
        yV, xV, zV = np.mgrid[0:p, 0:q, 0:s]
        sub = ((((yV - center_y) ** 2) + ((xV - center_x) ** 2) + ((zV - center_z) ** 2)) ** 0.5) < radius
        sphere = np.asarray(sub,dtype=np.float64)
        return sphere

    def calculate_displacement_map(
        self,
        peakCoordinatesA: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)]), 
        peakCoordinatesB: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)]), 
        peakCoordinatesC: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)]), 
        phaseA: np.ndarray, 
        phaseB: np.ndarray, 
        phaseC: np.ndarray, 
        showResult: bool = False,
        ):
        """
        Use the phase maps and peak coordinates to retrieve the x, y, and z displacement maps.
        
        Parameters
        ----------
        peakCoordinatesA: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            The absolute pixel coordinates of the first selected peak.
        peakCoordinatesB: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            The absolute pixel coordinates of the second selected peak.
        peakCoordinatesC: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            The absolute pixel coordinates of the third selected peak.
        phaseA: (nx, ny, nz) np.ndarray
            A 3D volume, the geometric phase corresponding to the first selected peak.
        phaseB: (nx, ny, nz) np.ndarray
            A 3D volume, the geometric phase corresponding to the second selected peak.
        phaseC: (nx, ny, nz) np.ndarray
            A 3D volume, the geometric phase corresponding to the third selected peak.
        showResult: bool
            Show the real space displacement. Defaults to False.
            
        Returns
        -------
        displacementX: (nx, ny, nz) np.ndarray
            A 3D array that maps the X (row offset) displacement within the lattice.
        displacementY: (nx, ny, nz) np.ndarray
            A 3D array that maps the Y (column offset) displacement within the lattice.
        displacementZ: (nx, ny, nz) np.ndarray
            A 3D array that maps the Z displacement within the lattice.
        """
        center_coords = np.asarray(self.volume.shape)//2
        peakCoordinatesA_G = self.circ_to_G(self.get_xyz_2(peakCoordinatesA))
        peakCoordinatesB_G = self.circ_to_G(self.get_xyz_2(peakCoordinatesB))
        peakCoordinatesC_G = self.circ_to_G(self.get_xyz_2(peakCoordinatesC))
        peakMatrix = self.get_a_matrix(peakCoordinatesA_G, peakCoordinatesB_G, peakCoordinatesC_G)
        displacementX, displacementY, displacementZ = self.get_u_matrices(phaseA, phaseB, phaseC, peakMatrix)
        if showResult == True:
            (fig, axs) = plt.subplots(1,3,figsize = (15,10))
            axs[0].imshow(displacementX[:,:,self.nz//2], origin = 'upper'); axs[0].set_title('Displacement Along X Direction (rows)'); axs[0].axis('off')
            axs[1].imshow(displacementY[:,:,self.nz//2], origin = 'upper'); axs[1].set_title('Displacement Along Y Direction (columns)'); axs[1].axis('off')
            axs[2].imshow(displacementZ[:,:,self.nz//2], origin = 'upper'); axs[2].set_title('Displacement Along Z Direction'); axs[2].axis('off')
        return displacementX, displacementY, displacementZ

    def get_a_matrix(
        self,
        g_vector_1: np.ndarray,
        g_vector_2: np.ndarray, 
        g_vector_3: np.ndarray,
        ):
        """
        Retrieve the inverse of the g matrix. The g matrix has reciprocal lattice vectors along its rows.
        The following is true: [[g1x g1y g1z], [g2x g2y g2z], [g3x g3y g3z]]^-1 = [[a1x a2x a3x], [a1y a2y a3y], [a1z a2z a3z]].
        The three reciprocal lattice vectors should be linearly independent.
        
        Parameters
        ----------
        g_vector_1: (3) np.ndarray
            The first reciprocal lattice vector.
        g_vector_2: (3) np.ndarray
            The second reciprocal lattice vector.
        g_vector_3: (3) np.ndarray
            The third reciprocal lattice vector.
        
        Returns
        -------
        a_matrix: (3, 3) np.ndarray
            The transpose of the real space lattice vector matrix. The entries are organized like this: [[a1x a2x a3x], [a1y a2y a3y], [a1z a2z a3z]].
        """
        g_matrix = np.array([g_vector_1, g_vector_2, g_vector_3])
        a_matrix = np.linalg.inv(g_matrix)
        return a_matrix

    def get_u_matrices(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        P3: np.ndarray,
        a_matrix: np.ndarray,
        ):
        """
        Retrieve the displacment (U) matrices using three phase matrices.
        
        Parameters
        ----------
        P1: (nx, ny, nz) np.ndarray
            The first phase matrix.
        P2: (nx, ny, nz) np.ndarray
            The second phase matrix.
        P3: (nx, ny, nz) np.ndarray
            The third phase matrix.
        a_matrix: (3, 3) np.ndarray
            The transpose of the real space lattice vector matrix. The entries are organized like this: [[a1x a2x a3x], [a1y a2y a3y], [a1z a2z a3z]].

        Returns
        -------
        ux: (nx, ny, nz) np.ndarray
            The atomic displacement map in the X (row) direction.
        uy: (nx, ny, nz) np.ndarray
            The atomic displacement map in the Y (column) direction.
        uz: (nx, ny, nz) np.ndarray
            The atomic displacement map in the Z direction.
        """
        P1 = skr.unwrap_phase(P1)
        P2 = skr.unwrap_phase(P2)
        P3 = skr.unwrap_phase(P3)
        rolled_p = np.asarray((np.reshape(P1,-1),np.reshape(P2,-1), np.reshape(P3,-1)))
        u_matrix = -1/(2*np.pi)*np.matmul(a_matrix,rolled_p)
        u_x = np.reshape(u_matrix[0,:],P1.shape)
        u_y = np.reshape(u_matrix[1,:],P2.shape)
        u_z = np.reshape(u_matrix[2,:],P3.shape)
        return u_x,u_y,u_z

    def circ_to_G(
        self,
        circ_pos: np.ndarray
        ):
        """
        Convert peak coordinates from absolute voxel location to centered k-space units.
        
        Parameters
        ----------
        circ_pos: (3) np.ndarray
            The position of the peak given in absolute coordinates (measured from corner origin) in voxels.
        
        Returns
        -------
        g_vec: (3) np.ndarray
            The position of the peak given in centered (self.volume.shape/2) k-space coordinates (frequency units).
        """
        g_vec = np.zeros(3)
        g_vec[0] = ((circ_pos[0] - (0.5*self.nx))/self.nx)
        g_vec[1] = ((circ_pos[1] - (0.5*self.ny))/self.ny)
        g_vec[2] = ((circ_pos[2] - (0.5*self.nz))/self.nz)
        return g_vec

    def G_to_circ(
        self,
        g_vec: np.ndarray,
        ):
        """
        Convert peak coordinates from centered k-space units to absolute voxel location.
        
        Parameters
        ----------
        g_vec: (3) np.ndarray
            The position of the peak given in centered (self.volume.shape/2) k-space coordinates (frequency units).
        
        Returns
        -------
        circ_pos: (3) np.ndarray
            The position of the peak given in absolute coordinates (measured from corner origin) in voxels.
        """
        circ_pos = np.zeros(3)
        circ_pos[0] = (g_vec[0]*self.nx) + (0.5*self.nx)
        circ_pos[1] = (g_vec[1]*self.ny) + (0.5*self.ny)
        circ_pos[2] = (g_vec[2]*self.nz) + (0.5*self.nz)
        return circ_pos

    def get_xyz_2(
        self,
        coords_arr: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)]),
        ):
        """
        Converts the custom dtype to an np.ndarray.
        
        Parameters
        ----------
        coords_arr: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            A single set of peak coordinates that has not already been indexed.
        
        Returns
        -------
        xyzCoords: (3) np.ndarray
            A simple array with three entries giving the x (row), y (column), and z coordinates of the input peak.
        """
        xyzCoords = np.array([coords_arr['x'][0], coords_arr['y'][0], coords_arr['z'][0]])
        return xyzCoords
    
    def get_xyz(
        self,
        coords_arr: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)]),
        ):
        """
        Converts the custom dtype to an np.ndarray.
        
        Parameters
        ----------
        coords_arr: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            A single set of peak coordinates.
            
        Returns
        -------
        xyzCoords: (3) np.ndarray
            A simple array with three entries giving the x (row), y (column), and z coordinates of the input peak.
        """
        xyzCoords = np.array([coords_arr['x'], coords_arr['y'], coords_arr['z']])
        return xyzCoords

    def calculate_strain_map(
        self,
        displacementX: np.ndarray, 
        displacementY: np.ndarray,
        displacementZ: np.ndarray, 
        showResult: bool = True,
        ):
        """
        Using the x, y, and z displacement maps, calculate the strain maps.
        
        Parameters
        ----------
        displacementX: (nx, ny, nz) np.ndarray
            A 3D array that maps the X (row offset) displacement within the lattice.
        displacementY: (nx, ny, nz) np.ndarray
            A 3D array that maps the Y (column offset) displacement within the lattice.
        displacementZ: (nx, ny, nz) np.ndarray
            A 3D array that maps the Z displacement within the lattice.
        showResult: bool
            Show the real space strain. Defaults to True.
            
        Returns
        -------
        e_mat: (3, 3, nx, ny, nz) np.ndarray
            The 3x3 tensor of strain maps.
        """
        e_xx,e_xy,e_xz = self.phase_diff(displacementX)
        e_yx,e_yy,e_yz = self.phase_diff(displacementY)
        e_zx,e_zy,e_zz = self.phase_diff(displacementZ)
        e_mat = np.array([
            [e_xx, e_xy, e_xz],
            [e_yx, e_yy, e_yz],
            [e_zx, e_zy, e_zz],
        ])
        
        e_xx, e_yy, e_zz = self.get_axial_strain(e_mat)
        e_th_xy, e_th_xz, e_th_yz, e_dg_xy, e_dg_xz, e_dg_yz = self.get_rot_and_diag_strain(e_mat)
        if showResult == True:
            (fig,axs) = plt.subplots(4,3, figsize = (25,20))
            axs = axs.flatten()
            axs[0].imshow(self.volume[self.nx//2,:,:], cmap = 'gray', origin = 'upper'); axs[0].axis('off')
            axs[1].imshow(self.volume[:,self.ny//2,:], cmap = 'gray', origin = 'upper'); axs[1].axis('off')
            axs[2].imshow(self.volume[:,:,self.nz//2], cmap = 'gray', origin = 'upper'); axs[2].axis('off')
            axs[3].imshow(e_xx[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[3].set_title('$Strain_{xx}$', fontsize = 20); axs[3].axis('off')
            axs[4].imshow(e_yy[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[4].set_title('$Strain_{yy}$', fontsize = 20); axs[4].axis('off')
            axs[5].imshow(e_zz[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[5].set_title('$Strain_{zz}$', fontsize = 20); axs[5].axis('off')
            axs[6].imshow(e_dg_yz[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[6].set_title('$Strain_{yz}$', fontsize = 20); axs[6].axis('off')
            axs[7].imshow(e_dg_xz[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[7].set_title('$Strain_{xz}$', fontsize = 20); axs[7].axis('off')
            axs[8].imshow(e_dg_xy[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[8].set_title('$Strain_{xy}$', fontsize = 20); axs[8].axis('off')
            axs[9].imshow(e_th_yz[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[9].set_title('$Theta Strain_{yz}$', fontsize = 20); axs[9].axis('off')
            axs[10].imshow(e_th_xz[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[10].set_title('$Theta Strain_{xz}$', fontsize = 20); axs[10].axis('off')
            axs[11].imshow(e_th_xy[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[11].set_title('$Theta Strain_{xy}$', fontsize = 20); axs[11].axis('off')
        fig.tight_layout()
        return e_mat

    def calculate_strain_map_phase(
        self,
        peakCoordinatesA,
        peakCoordinatesB,
        peakCoordinatesC,
        phaseA,
        phaseB,
        phaseC,
        showResult = True
    ):

        center_coords = np.asarray(self.volume.shape)//2
        peakCoordinatesA_G = self.circ_to_G(self.get_xyz_2(peakCoordinatesA))
        peakCoordinatesB_G = self.circ_to_G(self.get_xyz_2(peakCoordinatesB))
        peakCoordinatesC_G = self.circ_to_G(self.get_xyz_2(peakCoordinatesC))
        peakMatrix = self.get_a_matrix(peakCoordinatesA_G, peakCoordinatesB_G, peakCoordinatesC_G)

        phase_derivative = np.zeros([3,3,self.nx, self.ny, self.nz])

        expA_matrix1 = np.exp(-1j*phaseA)
        expA_matrix2 = np.exp(1j*phaseA)
        phase_derivative[0, 0] = np.imag(np.multiply(expA_matrix1,np.gradient(expA_matrix2, axis=0))) # phaseA_dx 
        phase_derivative[0, 1] = np.imag(np.multiply(expA_matrix1,np.gradient(expA_matrix2, axis=1))) # phaseA_dy
        phase_derivative[0, 2] = np.imag(np.multiply(expA_matrix1,np.gradient(expA_matrix2, axis=2))) # phaseA_dz

        expB_matrix1 = np.exp(-1j*phaseB)
        expB_matrix2 = np.exp(1j*phaseB)
        phase_derivative[1, 0] = np.imag(np.multiply(expB_matrix1,np.gradient(expB_matrix2, axis=0))) # phaseB_dx
        phase_derivative[1, 1] = np.imag(np.multiply(expB_matrix1,np.gradient(expB_matrix2, axis=1))) # phaseB_dy
        phase_derivative[1, 2] = np.imag(np.multiply(expB_matrix1,np.gradient(expB_matrix2, axis=2))) # phaseB_dz
        
        expC_matrix1 = np.exp(-1j*phaseC)
        expC_matrix2 = np.exp(1j*phaseC)
        phase_derivative[2, 0] = np.imag(np.multiply(expC_matrix1,np.gradient(expC_matrix2, axis=0))) # phaseC_dx
        phase_derivative[2, 1] = np.imag(np.multiply(expC_matrix1,np.gradient(expC_matrix2, axis=1))) # phaseC_dy
        phase_derivative[2, 2] = np.imag(np.multiply(expC_matrix1,np.gradient(expC_matrix2, axis=2))) # phaseC_dz

        e_mat = -1/(2*np.pi) * np.einsum('ij,jkabc->ikabc', peakMatrix, phase_derivative)

        e_xx, e_yy, e_zz = self.get_axial_strain(e_mat)
        e_th_xy, e_th_xz, e_th_yz, e_dg_xy, e_dg_xz, e_dg_yz = self.get_rot_and_diag_strain(e_mat)
        if showResult == True:
            (fig,axs) = plt.subplots(4,3, figsize = (25,20))
            axs = axs.flatten()
            axs[0].imshow(self.volume[self.nx//2,:,:], cmap = 'gray', origin = 'upper'); axs[0].axis('off')
            axs[1].imshow(self.volume[:,self.ny//2,:], cmap = 'gray', origin = 'upper'); axs[1].axis('off')
            axs[2].imshow(self.volume[:,:,self.nz//2], cmap = 'gray', origin = 'upper'); axs[2].axis('off')
            axs[3].imshow(e_xx[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[3].set_title('$Strain_{xx}$', fontsize = 20); axs[3].axis('off')
            axs[4].imshow(e_yy[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[4].set_title('$Strain_{yy}$', fontsize = 20); axs[4].axis('off')
            axs[5].imshow(e_zz[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[5].set_title('$Strain_{zz}$', fontsize = 20); axs[5].axis('off')
            axs[6].imshow(e_dg_yz[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[6].set_title('$Strain_{yz}$', fontsize = 20); axs[6].axis('off')
            axs[7].imshow(e_dg_xz[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[7].set_title('$Strain_{xz}$', fontsize = 20); axs[7].axis('off')
            axs[8].imshow(e_dg_xy[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[8].set_title('$Strain_{xy}$', fontsize = 20); axs[8].axis('off')
            axs[9].imshow(e_th_yz[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[9].set_title('$Theta Strain_{yz}$', fontsize = 20); axs[9].axis('off')
            axs[10].imshow(e_th_xz[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[10].set_title('$Theta Strain_{xz}$', fontsize = 20); axs[10].axis('off')
            axs[11].imshow(e_th_xy[:,:,self.nz//2], cmap = 'BrBG', origin = 'upper'); axs[11].set_title('$Theta Strain_{xy}$', fontsize = 20); axs[11].axis('off')
        fig.tight_layout()

        return e_mat

    def get_rot_and_diag_strain(
        self, 
        e_mat: np.ndarray,
        ):
        """
        Unpack and build the rotation and shear strain components.
        
        Parameters
        ----------
        e_mat: (3,3, nx, ny, nz)
            The strain maps.
        
        Returns
        -------
        e_th_xy: (nx, ny, nz) np.ndarray
            The xy rotation matrix.
        e_th_xz: (nx, ny, nz) np.ndarray
            The xz rotation matrix.
        e_th_yz: (nx, ny, nz) np.ndarray
            The yz rotation matrix.
        e_dg_xy: (nx, ny, nz) np.ndarray
            The xy shear strain matrix.
        e_dg_xz: (nx, ny, nz) np.ndarray
            The xz shear strain matrix.
        e_dg_yz: (nx, ny, nz) np.ndarray
            The yz shear strain matrix.
        """
        e_th_xy = 0.5*(e_mat[0,1] - e_mat[1,0])
        e_th_xz = 0.5*(e_mat[0,2] - e_mat[2,0])
        e_th_yz = 0.5*(e_mat[1,2] - e_mat[2,1])
        
        e_dg_xy = 0.5*(e_mat[0,1] + e_mat[1,0])
        e_dg_xz = 0.5*(e_mat[0,2] + e_mat[2,0])
        e_dg_yz = 0.5*(e_mat[1,2] + e_mat[2,1])
        return e_th_xy, e_th_xz, e_th_yz, e_dg_xy, e_dg_xz, e_dg_yz

    def get_axial_strain(
        self,
        e_mat: np.ndarray,
        ):
        """
        Unpack and build the axial strain components.
        
        Parameters
        ----------
        e_mat: (3, 3, nx, ny, nz)
            The strain maps.
        
        Returns
        -------
        e_mat[0,0]: (nx, ny, nz) np.ndarray
            The x (row) axial strain.
        e_mat[1,1]: (nx, ny, nz) np.ndarray
            The y (column) axial strain.
        e_mat[2,2]: (nx, ny, nz) np.ndarray
            The z axial strain.
        """
        return e_mat[0,0], e_mat[1,1], e_mat[2,2]


    # Advanced reference region functions
    
    def define_reference(
        self,
        x1: int,
        x2: int,
        y1: int,
        y2: int,
        z1: int,
        z2: int,
        ):
        """
        Locate visually the unstrained reference region.
        
        Parameters
        ----------
        x1:      Left limiting plane
        x2:     Right limiting plane
        y1:     Bottom limiting plane
        y2:     Top limiting plane
        z1:     Lower limiting plane
        z2:     Upper limiting plane
        
        Returns
        -------
        ref_reg: np.ndarray
                Boolean indices marking the reference region in 3D
        """
        xx,yy,zz = np.meshgrid(np.arange(self.nx),np.arange(self.ny),np.arange(self.nz), indexing = 'ij')
        ref_reg = np.logical_and(np.logical_and(np.logical_and(zz>z1, zz<z2), np.logical_and(xx>x1, xx<x2)), np.logical_and(yy>y1, yy<y2))

        A = (x1, y2, z2)
        B = (x2, y2, z2)
        C = (x2, y1, z2)
        D = (x1, y1, z2)
        
        plt.figure(figsize=(15,15))
        plt.imshow(self.image_normalizer(self.volume[:,:,z2-1])+0.33*ref_reg[:,:,z2-1], origin = 'upper')
        plt.annotate(A, (A[0]/self.nx, (1 - A[1]/self.ny)), textcoords='axes fraction', size=15,color='w')
        plt.annotate(B, (B[0]/self.nx, (1 - B[1]/self.ny)), textcoords='axes fraction', size=15,color='w')
        plt.annotate(C, (C[0]/self.nx, (1 - C[1]/self.ny)), textcoords='axes fraction', size=15,color='w')
        plt.annotate(D, (D[0]/self.nx, (1 - D[1]/self.ny)), textcoords='axes fraction', size=15,color='w')
        plt.scatter(A[1],A[0]) # scatter uses column row ordering, so we must put these in reverse order.
        plt.scatter(B[1],B[0])
        plt.scatter(C[1],C[0])
        plt.scatter(D[1],D[0])
        plt.axis('off')
        return ref_reg

    def set_reference_matrix(
        self,
        planeX1: int,
        planeX2: int,
        planeY1: int,
        planeY2: int,
        planeZ1: int,
        planeZ2: int,
        ):
        """
        Dictate a region of ideal (minimally distorted) lattice using 6 plane locations. This region will be a rectangular prism.
        
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
        planeZ1: int
            The lower bounding z plane for the region.
        planeZ2: int
            The upper bounding z plane for the region.
            
        Returns
        -------
        referenceMatrix: np.ndarray, bool
            The reference (ideal) region of the crystal.
        """
        referenceMatrix = self.define_reference(planeX1, planeX2, planeY1, planeY2, planeZ1, planeZ2)
        return referenceMatrix

    def set_reference_matrix(
        self,
        centerOfReferenceRegion: int,
        radiusOfReferenceRegion: int,
        ):
        """
        Dictate a region of ideal (minimally distorted) lattice using a center and cube half side length. This region will be a cube.
        
        Parameters
        ----------
        centerOfReferenceRegion: (3) np.ndarray
            An array with 3 entries giving the X (row), Y (column), and Z coordinates of the center of the region. These coordinates should be absolute and in pixels.
        radiusOfReferenceRegion: int
            An integer value that gives the half side length of the cube defining the reference region.

        Returns
        -------
        referenceMatrix: np.ndarray, bool
            The reference (ideal) region of the crystal.
        """
        planeX1 = centerOfReferenceRegion[0] - radiusOfReferenceRegion
        planeX2 = centerOfReferenceRegion[0] + radiusOfReferenceRegion
        planeY1 = centerOfReferenceRegion[1] - radiusOfReferenceRegion
        planeY2 = centerOfReferenceRegion[1] + radiusOfReferenceRegion
        planeZ1 = centerOfReferenceRegion[2] - radiusOfReferenceRegion
        planeZ2 = centerOfReferenceRegion[2] + radiusOfReferenceRegion
        referenceMatrix = self.define_reference(planeX1, planeX2, planeY1, planeY2, planeZ1, planeZ2)
        return referenceMatrix
        
    def refine_phase(
        self,
        phaseMap: np.ndarray,
        peakCoordinates: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)]),
        referenceMatrix: np.ndarray,
        maskSize: float,
        iterations: int,
        useGaussMask: bool,
        showResult: bool = True,
        ):
        """
        Refine the geometric phase according to the user-defined reference (ideal) region of the crystal.

        Parameters
        ----------
        phaseMap: (nx, ny, nz) np.ndarray
            A 3D geometric phase map (to be refined).
        peakCoordinates: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            The coordinates for the peak used to generate the phase map.
        referenceMatrix: np.ndarray, bool
            The user-defined reference (ideal) region of the crystal.
        maskSize: float
            The size of the input mask. Highly tunable. Lower values correspond to larger convolution kernel and lower resolution.
        iterations: int
            The number of iterations of phase refinement.
        useGaussMask: bool
            Control for whether to use a Gaussian mask or circular binary mask. Defaults to True (Gaussian).
        showResult: bool
            Show the real space geometric phase alongside the shifted Fourier transform and the Gaussian mask. Defaults to True.
        
        Returns
        -------
        peakCoordinatesRefined_dtype: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            The refined peak coordinates in a custom dtype.
        phaseMapRefined: (nx, ny, nz) np.ndarray
            The 3D phase map after refinement.
        """
        peakCoordinates_xyz = self.get_xyz_2(peakCoordinates)        
        ry = np.arange(start=-self.nx/2,stop=self.nx/2,step=1)
        rx = np.arange(start=-self.ny/2,stop=self.ny/2,step=1)
        rz = np.arange(start=-self.nz/2,stop=self.nz/2,step=1)
        rx,ry,rz = np.meshgrid(rx,ry,rz, indexing = 'ij')
        
        peakCoordinatesRefined = peakCoordinates_xyz.copy(); phaseMapRefined = phaseMap.copy()
        for _ in range(int(iterations)):
            G_x,G_y,G_z = self.phase_diff(phaseMapRefined)
            G_nabla = G_x + G_y + G_z
            g_r = G_nabla/(2*np.pi)
            del_g = np.asarray((np.median(g_r[referenceMatrix]/rx[referenceMatrix]),np.median(g_r[referenceMatrix]/ry[referenceMatrix]), np.median(g_r[referenceMatrix]/rz[referenceMatrix])))
            peakCoordinatesRefined += del_g
            peakCoordinatesRefined_dtype = np.zeros(1, dtype=self.dtype)
            peakCoordinatesRefined_dtype["x"] = peakCoordinatesRefined[0]
            peakCoordinatesRefined_dtype["y"] = peakCoordinatesRefined[1]
            peakCoordinatesRefined_dtype["z"] = peakCoordinatesRefined[2]
            phaseMapRefined = np.angle(self.calculate_phase_map(peakCoordinatesRefined_dtype,gaussianMask=useGaussMask,inputMaskSize=maskSize,showResult=False))
        
        if showResult:
            im_pha_gp = self.phase_im_lab(phaseMapRefined[:,:,int(self.nz//2)])
            (_,axs) = plt.subplots(1,2,figsize=(15,30))
            axs[0].imshow(self.volume[:,:,int(self.nz/2)],cmap='gray', origin = 'upper'); axs[0].axis('off')
            axs[1].imshow(im_pha_gp, origin = 'upper'); axs[1].axis('off')

        peakCoordinatesRefined_dtype = np.zeros(1, dtype=self.dtype)
        peakCoordinatesRefined_dtype['x'] = peakCoordinatesRefined[0]
        peakCoordinatesRefined_dtype['y'] = peakCoordinatesRefined[1]
        peakCoordinatesRefined_dtype['z'] = peakCoordinatesRefined[2]
        return peakCoordinatesRefined_dtype, phaseMapRefined
        

    def phase_diff(
        self, 
        angle_image: np.ndarray,
        ):
        """
        Differentiate the complex exponential of the phase image, and then obtain the 
        differentiation result by multiplying the differential with 
        the conjugate of the complex phase image.
        Here, the image is 3D.
        
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
        diff_z: np.ndarray
                z difference of the phase image
        """
        imaginary_image = np.exp(1j * angle_image)
        
        diff_imaginary_x = np.zeros(imaginary_image.shape,dtype=complex)
        diff_imaginary_x[0:-1,:,:] = np.diff(imaginary_image,axis=0)
        diff_imaginary_y = np.zeros(imaginary_image.shape,dtype=complex)
        diff_imaginary_y[:,0:-1, :] = np.diff(imaginary_image,axis=1)

        diff_imaginary_z = np.zeros(imaginary_image.shape,dtype=complex)
        diff_imaginary_z[:,:,0:-1] = np.diff(imaginary_image,axis=2)
        
        conjugate_imaginary = np.conj(imaginary_image)
        diff_complex_x = np.multiply(conjugate_imaginary,diff_imaginary_x)
        diff_complex_y = np.multiply(conjugate_imaginary,diff_imaginary_y)
        diff_complex_z = np.multiply(conjugate_imaginary,diff_imaginary_z)
        
        diff_x = np.imag(diff_complex_x)
        diff_y = np.imag(diff_complex_y)
        diff_z = np.imag(diff_complex_z)
        
        return diff_x,diff_y, diff_z

    def locate_first_order_peaks(
        self,
        peakCoordinates: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)]),
        ):
        """
        Locate three low-order linearly independent peaks in k-space.

        Parameters
        ----------
        peakCoordinates: (number of peaks) np.ndarrary, np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            An array of input peaks. This array should contain at least 3 linearly independent Bragg vectors.
            
        Returns
        -------
        peakA: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            The first peak (closest to central peak).
        peakB: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            The first peak (second closest to central peak).
        peakC: np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            The first peak (third closest to central peak).
        """
        midX = self.nx//2; midY = self.ny//2; midZ = self.nz//2
        peakCoordinatesRespCenter = np.zeros(len(peakCoordinates), dtype=self.dtype)
        peakCoordinatesRespCenter['x'] = peakCoordinates['x'] - midX
        peakCoordinatesRespCenter['y'] = peakCoordinates['y'] - midY
        peakCoordinatesRespCenter['z'] = peakCoordinates['z'] - midZ
        peakRadialDistCenter = peakCoordinatesRespCenter['x']**2 + peakCoordinatesRespCenter['y']**2 + peakCoordinatesRespCenter['z']**2
        
        smallestRadiiIndices = np.argsort(peakRadialDistCenter)
        peakCoordinatesRespCenter = peakCoordinatesRespCenter[smallestRadiiIndices]
        
        # The closest peak should be the zero order peak - not interested in that.
        peakAInd = 1
        peakBInd = None
        peakCInd = None
        ####
        crossAWithRest = np.zeros([len(peakCoordinates)-2, 3]) # this 2 comes from the A peak and the central peak that are excluded from consideration for the B and C peaks
        peakA_xyz = self.get_xyz(peakCoordinatesRespCenter[peakAInd])
        for peakIndex in np.arange(2,len(peakCoordinates)):
            currentPeak = self.get_xyz(peakCoordinatesRespCenter[peakIndex])
            crossAWithRest[peakIndex-2] = np.cross(peakA_xyz, currentPeak)
        threshold = 5 * (np.min(np.abs(np.linalg.norm(crossAWithRest, axis = 1)))+0.1)

        thresholdCondition = np.abs(np.linalg.norm(crossAWithRest, axis = 1))>threshold
        if np.any(thresholdCondition):
            peakBInd = np.argmax(thresholdCondition) + 2 # returning the 2 that was subtracted above
        else:
            print('Lowering threshold B')
            threshold = 2 * (np.min(np.abs(np.linalg.norm(crossAWithRest, axis = 1)))+0.1)
            thresholdCondition = np.abs(np.linalg.norm(crossAWithRest, axis = 1))>threshold
            peakBInd = np.argmax(thresholdCondition) + 2

        # now need to find a peak that is not in the plane of the first two vectors.
        # This could be achieved with the triple scalar product.
        peakB_xyz = self.get_xyz(peakCoordinatesRespCenter[peakBInd])
        crossAWithB = np.cross(peakA_xyz, peakB_xyz)
        tripleScalarProduct = np.zeros(len(peakCoordinates)-2)
        for peakIndex in np.arange(2, len(peakCoordinates)):
            currentPeak = self.get_xyz(peakCoordinatesRespCenter[peakIndex])
            tripleScalarProduct[peakIndex-2] = np.dot(crossAWithB, currentPeak)
        threshold = 5 * (np.min(np.abs(tripleScalarProduct)) + 0.1)
        print(np.abs(tripleScalarProduct))
        thresholdCondition = np.abs(tripleScalarProduct)>threshold
        if np.any(thresholdCondition):
            peakCInd = np.argmax(thresholdCondition) + 2
        else:
            print('Lowering threshold C')
            threshold = 2 * (np.min(np.abs(tripleScalarProduct)) + 0.1)
            thresholdCondition = np.abs(tripleScalarProduct)>threshold
            peakCInd = np.argmax(thresholdCondition) + 2

        peakA = np.zeros(1, dtype=self.dtype)
        peakB = np.zeros(1, dtype=self.dtype)
        peakC = np.zeros(1, dtype=self.dtype)

        peakA['x'] = peakCoordinates['x'][smallestRadiiIndices[peakAInd]]; peakA['y'] = peakCoordinates['y'][smallestRadiiIndices[peakAInd]]; peakA['z'] = peakCoordinates['z'][smallestRadiiIndices[peakAInd]]; peakA['intensity'] = peakCoordinates['intensity'][smallestRadiiIndices[peakAInd]]
        peakB['x'] = peakCoordinates['x'][smallestRadiiIndices[peakBInd]]; peakB['y'] = peakCoordinates['y'][smallestRadiiIndices[peakBInd]]; peakB['z'] = peakCoordinates['z'][smallestRadiiIndices[peakBInd]]; peakB['intensity'] = peakCoordinates['intensity'][smallestRadiiIndices[peakBInd]]
        peakC['x'] = peakCoordinates['x'][smallestRadiiIndices[peakCInd]]; peakC['y'] = peakCoordinates['y'][smallestRadiiIndices[peakCInd]]; peakC['z'] = peakCoordinates['z'][smallestRadiiIndices[peakCInd]]; peakC['intensity'] = peakCoordinates['intensity'][smallestRadiiIndices[peakCInd]]
        return peakA, peakB, peakC

    def locate_diffraction_spots(
        self,
        maxNumPeaks_in: int,
        ):
        """
        Calls the maxima finder.
        
        Parameters
        ----------
        maxNumPeaks_in: int
            The number of peaks to return. Noisier data should use a smaller value. For 3D crystals, more than 9 peaks should be sought. 
        Returns
        -------
        peakList: (maxNumPeaks_in) np.ndarray, np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            An array of peak coordinates with a custom datatype.
        """
        peakList = self.get_maxima_3D(np.abs(self.imFFT), maxNumPeaks = maxNumPeaks_in, _ar_FT = self.imFFT)
        return peakList
        
    # Functions adapted from py4DSTEM for 3D peak finding.
    def get_maxima_3D(
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
        Finds the maximal points of a 3D array.

        Parameters
        ----------
        ar: (nx, ny, nz) np.ndarray
            The 3D image with peaks.
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
            Parameter descriptions in filter_3D_maxima.
        _ar_FT: (nx, ny, nz) np.ndarray, complex
            If 'multicorr' is used and this is not None, uses this argument
            as the Fourier transform of `ar`, instead of recomputing it

        Returns
        -------
        maxima: np.ndarray, np.dtype([("x", float), ("y", float), ("z", float), ("intensity", float)])
            A structured array of maxima with fields 'x','y','z','intensity'
        """

        subpixel_modes = ("pixel", "poly", "multicorr")
        er = f"Unrecognized subpixel option {subpixel}. Must be in {subpixel_modes}"
        assert subpixel in subpixel_modes, er

        # gaussian filtering
        ar = ar if sigma <= 0 else gaussian_filter(ar, sigma)

        # local voxelwise maxima
        maxima_bool = (
            (ar >= np.roll(ar, (-1, 0, 0), axis=(0, 1, 2)))
            & (ar > np.roll(ar, (1, 0, 0), axis=(0, 1, 2)))
            & (ar >= np.roll(ar, (0, -1, 0), axis=(0, 1, 2)))
            & (ar > np.roll(ar, (0, 1, 0), axis=(0, 1, 2)))
            & (ar >= np.roll(ar, (-1, -1, 0), axis=(0, 1, 2)))
            & (ar > np.roll(ar, (-1, 1, 0), axis=(0, 1, 2)))
            & (ar >= np.roll(ar, (1, -1, 0), axis=(0, 1, 2)))
            & (ar > np.roll(ar, (1, 1, 0), axis=(0, 1, 2)))
            & (ar >= np.roll(ar, (-1, 0, 1), axis=(0, 1, 2))) ### start z = 1
            & (ar > np.roll(ar, (1, 0, 1), axis=(0, 1, 2)))
            & (ar >= np.roll(ar, (0, -1, 1), axis=(0, 1, 2)))
            & (ar > np.roll(ar, (0, 1, 1), axis=(0, 1, 2)))
            & (ar >= np.roll(ar, (-1, -1, 1), axis=(0, 1, 2)))
            & (ar > np.roll(ar, (-1, 1, 1), axis=(0, 1, 2)))
            & (ar > np.roll(ar, (1, -1, 1), axis=(0, 1, 2))) # change
            & (ar > np.roll(ar, (1, 1, 1), axis=(0, 1, 2)))
            & (ar > np.roll(ar, (0, 0, 1), axis=(0, 1, 2))) # new
            & (ar >= np.roll(ar, (-1, 0, -1), axis=(0, 1, 2)))  ### start z = 1
            & (ar > np.roll(ar, (1, 0, -1), axis=(0, 1, 2)))
            & (ar >= np.roll(ar, (0, -1, -1), axis=(0, 1, 2)))
            & (ar > np.roll(ar, (0, 1, -1), axis=(0, 1, 2)))
            & (ar >= np.roll(ar, (-1, -1, -1), axis=(0, 1, 2)))
            & (ar >= np.roll(ar, (-1, 1, -1), axis=(0, 1, 2))) # change
            & (ar >= np.roll(ar, (1, -1, -1), axis=(0, 1, 2)))
            & (ar > np.roll(ar, (1, 1, -1), axis=(0, 1, 2)))
            & (ar >= np.roll(ar, (0, 0, -1), axis=(0, 1, 2))) # new
        )


        # remove edges
        assert isinstance(edgeBoundary, (int, np.integer))
        if edgeBoundary < 1:
            edgeBoundary = 1
        maxima_bool[:edgeBoundary, :, :] = False
        maxima_bool[-edgeBoundary:, :, :] = False
        maxima_bool[:, :edgeBoundary, :] = False
        maxima_bool[:, -edgeBoundary:, :] = False
        maxima_bool[:, :, :edgeBoundary] = False
        maxima_bool[:, :, -edgeBoundary:] = False

        # get indices
        # sort by intensity
        maxima_x, maxima_y, maxima_z = np.nonzero(maxima_bool)
        dtype = np.dtype([("x", float), ("y", float), ("z", float),("intensity", float)])
        maxima = np.zeros(len(maxima_x), dtype=dtype)
        maxima["x"] = maxima_x
        maxima["y"] = maxima_y
        maxima["z"] = maxima_z
        maxima["intensity"] = ar[maxima_x, maxima_y, maxima_z]
        maxima = np.sort(maxima, order="intensity")[::-1]

        if len(maxima) == 0:
            return maxima

        # filter
        maxima = self.filter_3D_maxima(
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
            Ix1_ = ar[int(maxima["x"][i]) - 1, int(maxima["y"][i]), int(maxima["z"][i])].astype(np.float64)
            Ix0 = ar[int(maxima["x"][i]), int(maxima["y"][i]), int(maxima["z"][i])].astype(np.float64)
            Ix1 = ar[int(maxima["x"][i]) + 1, int(maxima["y"][i]), int(maxima["z"][i])].astype(np.float64)
            
            Iy1_ = ar[int(maxima["x"][i]), int(maxima["y"][i]) - 1, int(maxima["z"][i])].astype(np.float64)
            Iy0 = ar[int(maxima["x"][i]), int(maxima["y"][i]), int(maxima["z"][i])].astype(np.float64)
            Iy1 = ar[int(maxima["x"][i]), int(maxima["y"][i]) + 1, int(maxima["z"][i])].astype(np.float64)
            
            Iz1_ = ar[int(maxima["x"][i]), int(maxima["y"][i]), int(maxima["z"][i]) - 1].astype(np.float64)
            Iz0 = ar[int(maxima["x"][i]), int(maxima["y"][i]), int(maxima["z"][i])].astype(np.float64)
            Iz1 = ar[int(maxima["x"][i]), int(maxima["y"][i]), int(maxima["z"][i]) + 1].astype(np.float64)
            
            deltax = (Ix1 - Ix1_) / (4 * Ix0 - 2 * Ix1 - 2 * Ix1_)
            deltay = (Iy1 - Iy1_) / (4 * Iy0 - 2 * Iy1 - 2 * Iy1_)
            deltaz = (Iz1 - Iz1_) / (4 * Iz0 - 2 * Iz1 - 2 * Iz1_)
            maxima["x"][i] += deltax
            maxima["y"][i] += deltay
            maxima["z"][i] += deltaz
            maxima["intensity"][i] = self.linear_interpolation_3D(
                ar, maxima["x"][i], maxima["y"][i], maxima["z"][i]
            )

        if subpixel == "poly":
            return maxima

        # Fourier upsampling
        if _ar_FT is None:
            _ar_FT = np.fft.fftn(ar)
        for ipeak in range(len(maxima["x"])):
            xyzShift = np.array((maxima["x"][ipeak], maxima["y"][ipeak], maxima["z"][ipeak]))
            # we actually have to lose some precision and go down to half-pixel
            # accuracy for multicorr
            xyzShift[0] = np.round(xyzShift[0] * 2) / 2
            xyzShift[1] = np.round(xyzShift[1] * 2) / 2
            xyzShift[2] = np.round(xyzShift[2] * 2) / 2

            subShift = self.upsampled_correlation_3D(_ar_FT, upsample_factor, xyzShift)
            maxima["x"][ipeak] = subShift[0]
            maxima["y"][ipeak] = subShift[1]
            maxima["z"][ipeak] = subShift[2]

        maxima = np.sort(maxima, order="intensity")[::-1]
        return maxima


    def filter_3D_maxima(
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
            maxima : a numpy structured array with fields 'x', 'y', 'z', 'intensity'
            minAbsoluteIntensity : delete counts with intensity below this value
            minRelativeIntensity : delete counts with intensity below this value times
                the intensity of the i'th peak, where i is given by `relativeToPeak`
            relativeToPeak : see above
            minSpacing : if two peaks are within this euclidean distance from one
                another, delete the less intense of the two
            edgeBoundary : delete peaks within this distance of the image edge
            maxNumPeaks : an integer. defaults to 1

        Returns:
            a numpy structured array with fields 'x', 'y', 'z', 'intensity'
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
                        + (maxima["z"] - maxima["z"][i]) ** 2
                    ) < minSpacing**2
                    tooClose[: i + 1] = False
                    deletemask[tooClose] = True
            maxima = maxima[~deletemask]

        # Remove maxima in excess of maxNumPeaks
        if maxNumPeaks is not None:
            if len(maxima) > maxNumPeaks:
                maxima = maxima[:maxNumPeaks]

        return maxima

    def linear_interpolation_3D(
        self,
        ar,
        x,
        y,
        z,
        ):
        """
        Calculates the 2D linear interpolation of array ar at position x,y using the four
        nearest array elements.
        """
        x0, x1 = int(np.floor(x)), int(np.ceil(x))
        y0, y1 = int(np.floor(y)), int(np.ceil(y))
        z0, z1 = int(np.floor(z)), int(np.ceil(z))
        dx = x - x0
        dy = y - y0
        dz = z - z0
        
        return (
            (1 - dx) * (1 - dy) * (1 - dz) * ar[x0, y0, z0]
            + (1 - dx) * (1 - dy) * dz * ar[x0, y0, z1]
            + (1 - dx) * dy * (1 - dz) * ar[x0, y1, z0]
            + (1 - dx) * dy * dz * ar[x0, y1, z1]
            + dx * (1 - dy) * (1 - dz) * ar[x1, y0, z0]
            + dx * (1 - dy) * dz * ar[x1, y0, z1]
            + dx * dy * (1 - dz) * ar[x1, y1, z0]
            + dx * dy * dz * ar[x1, y1, z1]
        )

    def upsampled_correlation_3D(
        self, 
        imageCorr, 
        upsampleFactor, 
        xyzShift, 
        device="cpu",
        ):
        """
        Refine the correlation peak of imageCorr around xyzShift by DFT upsampling.

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
        upsampled around xyzShift, which MUST be specified to HALF-PIXEL precision
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
            xyzShift:
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

        xyzShift[0] = xp.round(xyzShift[0] * upsampleFactor) / upsampleFactor
        xyzShift[1] = xp.round(xyzShift[1] * upsampleFactor) / upsampleFactor
        xyzShift[2] = xp.round(xyzShift[2] * upsampleFactor) / upsampleFactor

        globalShift = xp.fix(xp.ceil(upsampleFactor * 1.5) / 2)

        upsampleCenter = xp.asarray(globalShift - upsampleFactor * xyzShift)

        imageCorrUpsample = xp.conj(
            self.dftUpsample_3D(xp.conj(imageCorr), upsampleFactor, upsampleCenter, device=device)
        )

        xyzSubShift = xp.asarray(
            xp.unravel_index(imageCorrUpsample.argmax(), imageCorrUpsample.shape)
        )

        # add a subpixel shift via parabolic fitting
        try:
            icc = xp.real(
                imageCorrUpsample[
                    xyzSubShift[0] - 1 : xyzSubShift[0] + 2,
                    xyzSubShift[1] - 1 : xyzSubShift[1] + 2,
                    xyzSubShift[2] - 1 : xyzSubShift[2] + 2,
                ]
            )
            dx = (icc[2, 1, 1] - icc[0, 1, 1]) / (4 * icc[1, 1, 1] - 2 * icc[2, 1, 1] - 2 * icc[0, 1, 1])
            dy = (icc[1, 2, 1] - icc[1, 0, 1]) / (4 * icc[1, 1, 1] - 2 * icc[1, 2, 1] - 2 * icc[1, 0, 1])
            dz = (icc[1, 1, 2] - icc[1, 1, 0]) / (4 * icc[1, 1, 1] - 2 * icc[1, 1, 2] - 2 * icc[1, 1, 0])
        except:
            dx, dy, dz = (
                0,
                0,
                0,
            )  # this is the case when the peak is near the edge and one of the above values does not exist

        xyzSubShift = xyzSubShift - globalShift

        xyzShift = xyzShift + (xyzSubShift + xp.array([dx, dy, dz])) / upsampleFactor

        return xyzShift


    def dftUpsample_3D(
        self,
        imageCorr,
        upsampleFactor,
        xyzShift,
        device="cpu",
        ):
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
            xyzShift (list of 3 floats):
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
        numDep = numRow

        colKern = xp.exp(
            (-1j * 2 * np.pi / (imageSize[1] * upsampleFactor))
            * xp.outer(
                (xp.arange(numCol) - xyzShift[1]),
                (xp.fft.ifftshift((xp.arange(imageSize[1]))) - xp.floor(imageSize[1] / 2)),
            )
        )

        rowKern = xp.exp(
            (-1j * 2 * np.pi / (imageSize[0] * upsampleFactor))
            * xp.outer(
                (xp.arange(numRow) - xyzShift[0]),
                (xp.fft.ifftshift(xp.arange(imageSize[0])) - xp.floor(imageSize[0] / 2)),
            )
        )

        depKern = xp.exp(
            (-1j * 2 * np.pi / (imageSize[2] * upsampleFactor))
            * xp.outer(
                (xp.arange(numDep) - xyzShift[2]),
                (xp.fft.ifftshift(xp.arange(imageSize[2])) - xp.floor(imageSize[2] / 2)),
            )
        )

        imageUpsample = xp.einsum("ax, xyz -> ayz", rowKern, imageCorr)
        imageUpsample = xp.einsum("by, ayz -> abz", colKern, imageUpsample)
        imageUpsample = xp.einsum("cz, abz -> abc", depKern, imageUpsample)
        imageUpsample = xp.real(imageUpsample)
        return imageUpsample