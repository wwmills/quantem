from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import Dataset

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.optimizer_mixin import OptimizerMixin


@dataclass
class DatasetValue:
    """
    Class for storing the forward call for both PixDataset and INRDataset.
    """

    target: torch.Tensor
    tilt_angle: int | float
    pixel_loc: tuple[int, int] | None = None  # Only for INRDataset
    projection_idx: int | None = None  # Only for INRDataset
    pose: tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter] | None = (
        None  # If there is pose optimization.  # Pose is tuple (shifts, z1, z3)
    )


class TomographyDatasetBase(AutoSerialize, OptimizerMixin, nn.Module):
    """
    Base tomography dataset class for all tomography datasets to inherit from.
    """

    _token = object()

    DEFAULT_LRS = {
        "pose_lr": 5e-2,
    }

    def __init__(
        self,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        _token: object | None = None,
    ):
        AutoSerialize.__init__(self)
        OptimizerMixin.__init__(self)
        nn.Module.__init__(self)
        # if _token is not self._token: TODO: Idk why this isn't working.
        #     raise RuntimeError("Use TomographyPixDataset.from_* to instantiate this class.")

        if not (
            tilt_stack.shape[0] < tilt_stack.shape[1] or tilt_stack.shape[0] < tilt_stack.shape[2]
        ):
            raise ValueError(
                "The number of tilt projections should be in the first dimension of the dataset."
            )

        # TODO: Maybe have the validation in here too.
        max_val = np.quantile(tilt_stack, 0.95)
        if type(tilt_stack) is not torch.Tensor:
            tilt_stack = torch.from_numpy(tilt_stack)
        if type(tilt_angles) is not torch.Tensor:
            tilt_angles = torch.from_numpy(tilt_angles)

        # Tilt stack normalization
        tilt_stack = tilt_stack / max_val

        self.tilt_stack = tilt_stack
        self.tilt_angles = tilt_angles
        self.learn_shift = learn_shift
        self.learn_tilt_axis = learn_tilt_axis

        # The reference tilt angle is the one with the smallest absolute tilt angle.
        # I.e, the pose will not be optimized for the reference tilt angle.
        self._reference_tilt_angle_idx = torch.argmin(torch.abs(self.tilt_angles))
        # TODO: Implement AuxParams from old tomography_dataset.py here.

        # TODO: The parameters won't be initialized unless .to(device) is called.
        self._z1_angles = torch.zeros(self.learnable_tilts)
        self._z3_angles = torch.zeros(self.learnable_tilts)
        self._shifts = torch.zeros(self.learnable_tilts, 2)

        # Fixed zeros for reference tilt
        self._z1_ref = torch.zeros(1)
        self._z3_ref = torch.zeros(1)
        self._shifts_ref = torch.zeros(1, 2)

    # --- Class methods ---
    @classmethod
    def from_data(
        cls,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
    ):
        return cls(
            tilt_stack=tilt_stack,
            tilt_angles=tilt_angles,
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
        )

    # --- Optimization Parameters ---
    # @property
    # def params(self):
    #     # TODO: Need to double check if this is correct way, also need to implement get_optimization_parameters @Arthur!!
    #     raise NotImplementedError("This method should be implemented in subclasses.")

    # --- Forward pass ---
    @abstractmethod
    def forward(
        self,
        dummy_input: Any = None,  # Note all nn.Modules require some input.
    ):
        """
        Forward pass should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    # --- Properties ---
    @property
    def tilt_stack(self) -> torch.Tensor:
        return self._tilt_stack

    @tilt_stack.setter
    def tilt_stack(self, tilt_stack: torch.Tensor):
        if type(tilt_stack) is not torch.Tensor:
            # print("Converting tilt stack to torch.Tensor")
            tilt_stack = torch.from_numpy(tilt_stack)

        self._tilt_stack = tilt_stack

    @property
    def tilt_angles(self) -> torch.Tensor:
        return self._tilt_angles

    @tilt_angles.setter
    def tilt_angles(self, tilt_angles: torch.Tensor):
        if type(tilt_angles) is not torch.Tensor:
            # print("Converting tilt angles to torch.Tensor")
            tilt_angles = torch.from_numpy(tilt_angles)

        self._tilt_angles = tilt_angles

    @property
    def learn_shift(self) -> bool:
        return self._learn_shift

    @learn_shift.setter
    def learn_shift(self, learn_shift: bool):
        self._learn_shift = learn_shift

    @property
    def learn_tilt_axis(self) -> bool:
        return self._learn_tilt_axis

    @learn_tilt_axis.setter
    def learn_tilt_axis(self, learn_tilt_axis: bool):
        self._learn_tilt_axis = learn_tilt_axis

    @property
    def reference_tilt_idx(self) -> int:
        return self._reference_tilt_angle_idx

    @reference_tilt_idx.setter
    def reference_tilt_idx(self, reference_tilt_idx: int):
        self._reference_tilt_angle_idx = reference_tilt_idx

    @property
    def learnable_tilts(self) -> int:
        return self.tilt_angles.shape[0] - 1

    @learnable_tilts.setter
    def learnable_tilts(self, learnable_tilts: int):
        self._learnable_tilts = learnable_tilts

    @property
    def params(self) -> dict[str, torch.nn.Parameter]:
        """
        Returns the parameters that should be optimized for this dataset.

        Should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @property
    def z1_params(self) -> torch.nn.Parameter:
        return self._z1_params

    @z1_params.setter
    def z1_params(self, z1_angles: torch.Tensor, device: str):
        self._z1_params = nn.Parameter(z1_angles.to(device))

    @property
    def z3_params(self) -> torch.nn.Parameter:
        return self._z3_params

    @z3_params.setter
    def z3_params(self, z3_angles: torch.Tensor, device: str):
        self._z3_params = nn.Parameter(z3_angles.to(device))

    @property
    def shifts_params(self) -> torch.nn.Parameter:
        return self._shifts_params

    @shifts_params.setter
    def shifts_params(self, shifts: torch.Tensor, device: str):
        self._shifts_params = nn.Parameter(shifts.to(device))

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str):
        self._device = device

    # --- Helper Functions ---
    def to(self, device: str):
        """
        Moves the dataset to the device, and also insantiates the aux params to the device.
        """
        self.tilt_stack = self.tilt_stack.to(device)
        self.tilt_angles = self.tilt_angles.to(device)

        self._z1_params = nn.Parameter(self._z1_angles.to(device))
        self._z3_params = nn.Parameter(self._z3_angles.to(device))
        self._shifts_params = nn.Parameter(self._shifts.to(device))

        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)

        self.device = device

    def get_optimization_parameters(self) -> list[nn.Parameter]:
        """
        Get the parameters that should be optimized for this model.
        """
        return list[nn.Parameter](self.params)


class TomographyPixDataset(TomographyDatasetBase):
    """
    Dataset class for pixel-based tomography, i.e AD, SIRT, WBP, etc...

    TODO:
     - What should the forward pass return? In AD, it should be both the tilt image and the pose.
       In SIRT, it's only the tilt image.
    """

    def __init__(
        self,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        _token: object | None = None,
    ):
        super().__init__(
            tilt_stack=tilt_stack,
            tilt_angles=-tilt_angles,  # TODO: Flip the tilt angles to be negative to match the convention of INR.
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
            _token=_token,
        )

    def forward(
        self,
        proj_idx: int,
    ) -> DatasetValue:
        """
        Forward pass for pixel-based tomography.
        Returns the full tilt image for the given projection index, and the tilt angle.
        """

        return DatasetValue(
            target=self.tilt_stack[proj_idx],
            tilt_angle=self.tilt_angles[proj_idx],
            pixel_loc=None,
        )


class TomographyINRDataset(TomographyDatasetBase, Dataset):
    """
    Dataset class for INR-based tomography.

    The two main methods here are that the `forward` call will return the relative pose parameters,
    while `__getitem__` will actually return the pixel values of the tilt stack.

    TODO: I think TomographyINRDataset shouldn't handle the train/val split and will be handled later? Yea this is handled in setup_dataloader in DDP
    """

    def __init__(
        self,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        seed: int = 42,
        token: object | None = None,
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, token)

    # --- Forward Pass w/ Params Method for OptimizerMixin ---
    def forward(self, dummy_input: Any = None):
        """
        Forward pass for INR-based tomography. In the forward pass, the only parameters that
        are passed will be the shifts, z1 and z3 Euler angles.
        """

        first_half_shifts = self.shifts_params[: self.reference_tilt_idx]
        second_half_shifts = self.shifts_params[self.reference_tilt_idx :]
        shifts = torch.cat([first_half_shifts, self._shifts_ref, second_half_shifts], dim=0)

        first_half_z1 = self.z1_params[: self.reference_tilt_idx]
        second_half_z1 = self.z1_params[self.reference_tilt_idx :]
        z1 = torch.cat([first_half_z1, self._z1_ref, second_half_z1], dim=0)

        first_half_z3 = self.z3_params[: self.reference_tilt_idx]
        second_half_z3 = self.z3_params[self.reference_tilt_idx :]
        z3 = torch.cat([first_half_z3, self._z3_ref, second_half_z3], dim=0)

        if self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3
        elif self.learn_shift:
            return shifts, torch.zeros_like(z1), torch.zeros_like(z3)
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3
        else:
            return torch.zeros_like(shifts), torch.zeros_like(z1), torch.zeros_like(z3)

    @property
    def params(self):
        return self.parameters()

    def get_coords(
        self, batch: dict[str, torch.Tensor], N: int, num_samples_per_ray: int
    ) -> torch.Tensor:
        pixel_i = batch["pixel_i"].float().to(self.device, non_blocking=True)
        pixel_j = batch["pixel_j"].float().to(self.device, non_blocking=True)
        # target_values = batch["target_value"].to(self.device, non_blocking=True)
        phis = batch["phi"].to(self.device, non_blocking=True)
        projection_indices = batch["projection_idx"].to(self.device, non_blocking=True)
        with torch.no_grad():
            batch_ray_coords = self.create_batch_rays(pixel_i, pixel_j, N, num_samples_per_ray)

        shifts, z1_params, z3_params = self.forward(None)
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)

        transformed_rays = self.transform_batch_rays(
            batch_ray_coords,
            z1=batch_z1,
            x=phis,
            z3=batch_z3,
            shifts=batch_shifts,
            N=N,
            sampling_rate=1.0,
        )
        all_coords = transformed_rays.view(-1, 3)

        all_coords = all_coords.to(self.device, dtype=torch.float32, non_blocking=True)
        return all_coords

    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def create_batch_rays(
        pixel_i: torch.Tensor, pixel_j: torch.Tensor, N: int, num_samples_per_ray: int
    ) -> torch.Tensor:
        batch_size = len(pixel_i)
        x_coords = (pixel_j / (N - 1)) * 2 - 1
        y_coords = (pixel_i / (N - 1)) * 2 - 1
        z_coords = torch.linspace(-1, 1, num_samples_per_ray, device=pixel_i.device)

        rays = torch.zeros(batch_size, num_samples_per_ray, 3, device=pixel_i.device)

        rays[:, :, 0] = x_coords.unsqueeze(1)
        rays[:, :, 1] = y_coords.unsqueeze(1)
        rays[:, :, 2] = z_coords.unsqueeze(0)

        return rays

    @staticmethod
    def transform_batch_rays(
        rays: torch.Tensor,
        z1: torch.Tensor,
        x: torch.Tensor,
        z3: torch.Tensor,
        shifts: torch.Tensor,
        N: int,
        sampling_rate: float,
    ) -> torch.Tensor:
        shift_x_norm = (shifts[:, 0:1] * sampling_rate * 2) / (N - 1)
        shift_y_norm = (shifts[:, 1:2] * sampling_rate * 2) / (N - 1)

        rays_x = rays[:, :, 0] - shift_x_norm
        rays_y = rays[:, :, 1] - shift_y_norm
        rays_z = rays[:, :, 2]

        theta = torch.deg2rad(-z3).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot1 = cos_t * rays_x - sin_t * rays_y
        rays_y_rot1 = sin_t * rays_x + cos_t * rays_y
        rays_z_rot1 = rays_z

        theta = torch.deg2rad(x).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot2 = rays_x_rot1
        rays_y_rot2 = cos_t * rays_y_rot1 - sin_t * rays_z_rot1
        rays_z_rot2 = sin_t * rays_y_rot1 + cos_t * rays_z_rot1

        theta = torch.deg2rad(-z1).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_final = cos_t * rays_x_rot2 - sin_t * rays_y_rot2
        rays_y_final = sin_t * rays_x_rot2 + cos_t * rays_y_rot2
        rays_z_final = rays_z_rot2

        transformed_rays = torch.stack([rays_x_final, rays_y_final, rays_z_final], dim=2)

        return transformed_rays

    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def integrate_rays(
        rays: torch.Tensor, num_samples_per_ray: int, target_values_len: int
    ) -> torch.Tensor:
        ray_densities = rays.view(
            target_values_len,
            num_samples_per_ray,
        )
        step_size = 2.0 / (num_samples_per_ray - 1)

        predicted_values = ray_densities.sum(dim=1) * step_size

        return predicted_values

    # --- Torch Dataset Methods ---
    def __getitem__(
        self,
        idx: int,
    ) -> DatasetValue:
        """
        Gets the item for INR i.e, the project index, pixel value at (i, j), and the tilt angle.
        """

        actual_idx = idx

        projection_idx = actual_idx // (self.tilt_stack.shape[1] * self.tilt_stack.shape[2])
        remaining = actual_idx % (self.tilt_stack.shape[1] * self.tilt_stack.shape[2])

        pixel_i = remaining // self.tilt_stack.shape[1]
        pixel_j = remaining % self.tilt_stack.shape[1]

        # return DatasetValue(
        #     target=self.tilt_stack[projection_idx, pixel_i, pixel_j],
        #     tilt_angle=self.tilt_angles[projection_idx],
        #     projection_idx=projection_idx,
        #     pixel_loc=(pixel_i, pixel_j),
        # )
        return {
            "projection_idx": torch.tensor(projection_idx),
            "pixel_i": torch.tensor(pixel_i),
            "pixel_j": torch.tensor(pixel_j),
            "phi": self.tilt_angles[projection_idx],  # tensor
            "target_value": self.tilt_stack[projection_idx, pixel_i, pixel_j],  # tensor
        }

    def __len__(
        self,
    ):
        """
        Returns the number of pixels in the tilt stack.
        """
        N = max(self.tilt_stack.shape)
        return self.tilt_stack.shape[0] * N * N

    def to(self, device: str):
        self._z1_params = nn.Parameter(self._z1_angles.to(device))
        self._z3_params = nn.Parameter(self._z3_angles.to(device))
        self._shifts_params = nn.Parameter(self._shifts.to(device))

        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)

        self.device = device
        self.reconnect_optimizer_to_parameters()


class TomographyINRPretrainDataset(Dataset):
    """
    Dataset class for pretraining INR models.
    """

    def __init__(
        self,
        pretrain_target: torch.Tensor,
    ):
        data = pretrain_target.float()

        total_elements = data.numel()
        if total_elements > 1e6:
            sample_size = min(int(1e6), total_elements)
            flat_data = data.flatten()
            indices = torch.randperm(total_elements)[:sample_size]
            sampled_data = flat_data[indices]
            data_quantile = torch.quantile(sampled_data, 0.95)
        else:
            data_quantile = torch.quantile(data, 0.95)

        data = data / data_quantile
        data = torch.permute(data, (2, 1, 0))
        # data = torch.flip(data, dims=(2,))

        self.volume = data.cpu()
        self.N = pretrain_target.shape[0]  # Assumes cubic volume.
        self.total_samples = pretrain_target.shape[0] ** 3

        coords_1d = torch.linspace(-1, 1, self.N)
        x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
        self.coords = torch.stack([x, y, z], dim=-1).reshape(-1, 3).cpu()
        self.targets = self.volume.reshape(-1).cpu()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return {"coords": self.coords[idx], "target": self.targets[idx]}


class TomographyThroughFocalINRDataset(TomographyINRDataset):
    """
    Dataset class for INR-based tomography that assumes non-parallel illumination condition.
    Inherits from TomographyINRDataset.
    """

    def __init__(
        self,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        ## Through focal parameters:
        convergence_angle: float,
        z_focus: float | torch.Tensor = 0.0,
        num_rays: int = 1,
        random_rays: bool = False,
        learn_z_focus: bool = False,
        ## optional parameters
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        seed: int = 42,
        token: object | None = None,
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, token)
        self.num_rays = int(num_rays)

        self._convergence_angle = convergence_angle
        self._random_rays = random_rays

        # self.print_thing = True

        if not random_rays:
            theta, phi = self.get_theta_phi(
                convergence_angle=convergence_angle,
                num_rays=num_rays,
                random_rays=random_rays,
            )

            self.phis = phi
            self.thetas = theta

            # w = torch.sinc(self.phis / (2 * self.convergence_angle)) ** 2
            # w = w / w.sum()
            # self._ray_weights = w.view(1, 1, -1)
            self._ray_weights = torch.ones(len(self.phis)) / len(self.phis)

        # if not isinstance(z_focus, torch.Tensor):
        #     z_focus = torch.tensor(z_focus, dtype=torch.float32)

        # if learn_z_focus:
        #     self._z_focus = torch.nn.Parameter(z_focus)
        # else:
        #     self._z_focus = z_focus

        # TODO: The parameters won't be initialized unless .to(device) is called.
        # Fixed zeros for reference tilt
        self._z_focuses = torch.zeros(self.learnable_tilts)

        # Fixed zeros for reference tilt
        self._z_focus_ref = torch.zeros(1)

    @property
    def convergence_angle(self) -> float:
        return self._convergence_angle

    @property
    def random_rays(self) -> bool:
        return self._random_rays

    @property
    def z_focus_params(self) -> torch.nn.Parameter:
        return self._z_focus_params

    @z_focus_params.setter
    def z_focus_params(self, z_focus_angles: torch.Tensor, device: str):
        self._z_focus_params = nn.Parameter(z_focus_angles.to(device))

    # --- Forward Pass w/ Params Method for OptimizerMixin ---
    def forward(self, dummy_input: Any = None):
        """
        Forward pass for INR-based through focal tomography. In the forward pass, the only parameters that
        are passed will be the shifts, focal plane, z1 and z3 Euler angles.
        """

        first_half_shifts = self.shifts_params[: self.reference_tilt_idx]
        second_half_shifts = self.shifts_params[self.reference_tilt_idx :]
        shifts = torch.cat([first_half_shifts, self._shifts_ref, second_half_shifts], dim=0)

        first_half_z1 = self.z1_params[: self.reference_tilt_idx]
        second_half_z1 = self.z1_params[self.reference_tilt_idx :]
        z1 = torch.cat([first_half_z1, self._z1_ref, second_half_z1], dim=0)

        first_half_z3 = self.z3_params[: self.reference_tilt_idx]
        second_half_z3 = self.z3_params[self.reference_tilt_idx :]
        z3 = torch.cat([first_half_z3, self._z3_ref, second_half_z3], dim=0)

        first_half_z_focus = self.z_focus_params[: self.reference_tilt_idx]
        second_half_z_focus = self.z_focus_params[self.reference_tilt_idx :]
        z_focus = torch.cat([first_half_z_focus, self._z_focus_ref, second_half_z_focus], dim=0)

        if self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus
        elif self.learn_shift:
            return shifts, torch.zeros_like(z1), torch.zeros_like(z3), z_focus
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3, z_focus
        elif self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus
        else:
            return torch.zeros_like(shifts), torch.zeros_like(z1), torch.zeros_like(z3), z_focus

    def get_coords(
        self,
        batch: dict[str, torch.Tensor],
        N: int,
        num_samples_per_ray: int,
    ) -> torch.Tensor:
        num_rays = self.num_rays
        convergence_angle = self._convergence_angle
        random_rays = self._random_rays
        pixel_i = batch["pixel_i"].float().to(self.device, non_blocking=True)
        pixel_j = batch["pixel_j"].float().to(self.device, non_blocking=True)
        # target_values = batch["target_value"].to(self.device, non_blocking=True)
        phis = batch["phi"].to(self.device, non_blocking=True)
        projection_indices = batch["projection_idx"].to(self.device, non_blocking=True)
        shifts, z1_params, z3_params, z_focus_params = self.forward(None)
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)
        batch_z_focus = torch.index_select(z_focus_params, 0, projection_indices)
        # with torch.no_grad():
        batch_ray_coords = self.create_batch_rays(
            pixel_i,
            pixel_j,
            N,
            num_samples_per_ray,
            num_rays,
            convergence_angle,
            random_rays,
            z_focus=batch_z_focus,
        )

        transformed_rays = self.transform_batch_rays(
            batch_ray_coords,
            z1=batch_z1,
            x=phis,
            z3=batch_z3,
            shifts=batch_shifts,
            N=N,
            sampling_rate=1.0,
        )
        all_coords = transformed_rays.view(-1, 3)

        all_coords = all_coords.to(self.device, dtype=torch.float32, non_blocking=True)
        return all_coords

    def get_theta_phi(
        self,
        convergence_angle: float,
        num_rays: int,
        device: torch.device | str | None = None,
        random_rays: bool = False,
        random_method: str = 's',
    ):
        if device is None:
            device = torch.device("cpu")

        num_rays = int(num_rays)

        # random_rays sampling method
        if random_rays:
            theta = torch.rand(num_rays, device=device) * 2 * torch.pi # this can stay as uniform sampling
            # phi = torch.rand(num_rays, device=device) * convergence_angle # the original uniform sampling of phi
            if random_method.lower() in ['gaussian','g']:
                phi = torch.normal(mean = 0.0, std = convergence_angle, size = (num_rays,), device=device)

            # let's also do the sinc squared, which might be slower?
            # essentially, torch doesn't have a sinc**2 distribution built in, but we can just make a discrete one ourselves
            elif random_method.lower() in ['sinc','s']:
                x = torch.linspace(-0.5, 0.5, 5000, device = device) # in radians
                pdf = torch.sinc(x)**2 # 
                pdf = pdf/pdf.sum() # normalize to 1
                indices = torch.multinomial(pdf, num_rays, replacement=True)
                phi = x[indices]
            else:
                raise ValueError(
                    f"Unsupported random_method={random_method}. "
                    "Supported values are: gaussian, g, sinc, s. .lower() is applied internally."
                )


        else:
            # 1 ray: single central ray
            if num_rays == 1:
                theta = torch.zeros(1, device=device)
                phi = torch.zeros(1, device=device)

            # 2 or 3 rays: single ring
            elif num_rays == 2 or num_rays == 3:
                rand_offset = torch.rand(1, device=device) * 2 * torch.pi
                theta = (
                    torch.linspace(0, 2 * torch.pi, num_rays + 1, device=device)[:-1] + rand_offset
                )
                phi = torch.full(
                    (num_rays,), 0.5 * convergence_angle, device=device
                )  # for the case where there is no center ray, use half of the convergence angle.

            # 4–8 rays: central + ring
            elif 3 < num_rays < 9:
                rand_offset = torch.rand(1, device=device) * 2 * torch.pi
                theta0 = torch.zeros(1, device=device)
                phi0 = torch.zeros(1, device=device)
                theta_ring = (
                    torch.linspace(0, 2 * torch.pi, num_rays, device=device)[:-1] + rand_offset
                )
                phi_ring = torch.full((num_rays - 1,), 1 * convergence_angle, device=device)
                theta = torch.cat((theta0, theta_ring))
                phi = torch.cat((phi0, phi_ring))

            # 9 rays: two rings. Rays at 1/3 and 2/3 * convergence angle
            elif num_rays == 9:
                rand_offset = torch.rand(1, device=device) * 2 * torch.pi
                theta_inner = torch.linspace(0, 2 * torch.pi, 4, device=device)[:-1] + rand_offset
                phi_inner = torch.full((3,), convergence_angle / 3, device=device)
                theta_outer = torch.linspace(0, 2 * torch.pi, 7, device=device)[:-1] + rand_offset
                phi_outer = torch.full((6,), 2 * convergence_angle / 3, device=device)
                theta = torch.cat((theta_inner, theta_outer))
                phi = torch.cat((phi_inner, phi_outer))

            # 10 rays: two rings, one central ray. Rays at 0, 1/2 and 1 * convergence angle
            elif num_rays == 10:
                rand_offset = torch.rand(1, device=device) * 2 * torch.pi
                theta0 = torch.zeros(1, device=device)
                phi0 = torch.zeros(1, device=device)

                theta_inner = torch.linspace(0, 2 * torch.pi, 4, device=device)[:-1] + rand_offset
                phi_inner = torch.full((3,), convergence_angle / 3, device=device)
                theta_outer = torch.linspace(0, 2 * torch.pi, 7, device=device)[:-1] + rand_offset
                phi_outer = torch.full((6,), 2 * convergence_angle / 3, device=device)
                theta = torch.cat((theta0, theta_inner, theta_outer))
                phi = torch.cat((phi0, phi_inner, phi_outer))

            # 19 rays: central + two rings. Rays at 0, 1/2 and 1 * convergence angle
            elif num_rays == 19:
                rand_offset = torch.rand(1, device=device) * 2 * torch.pi
                theta0 = torch.zeros(1, device=device)
                phi0 = torch.zeros(1, device=device)
                theta_inner = torch.linspace(0, 2 * torch.pi, 7, device=device)[:-1] + rand_offset
                phi_inner = torch.full((6,), convergence_angle / 2, device=device)
                theta_outer = torch.linspace(0, 2 * torch.pi, 13, device=device)[:-1] + rand_offset
                phi_outer = torch.full((12,), 1 * convergence_angle, device=device)
                theta = torch.cat((theta0, theta_inner, theta_outer))
                phi = torch.cat((phi0, phi_inner, phi_outer))

            # Unsupported ray counts
            else:
                raise ValueError(
                    f"Unsupported num_rays={num_rays}. "
                    "Supported values are: 1, 2, 3, 4–8, 9, 10, 19."
                )

        return theta, phi

    # @staticmethod
    # @torch.compile(mode="reduce-overhead")
    def create_batch_rays(
        self,
        pixel_i: torch.Tensor,
        pixel_j: torch.Tensor,
        N: int,
        num_samples_per_ray: int,
        num_rays: int,
        convergence_angle: float,
        random_rays: bool,
        z_focus: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(pixel_i)
        x_coords_0 = (pixel_j / (N - 1)) * 2 - 1
        y_coords_0 = (pixel_i / (N - 1)) * 2 - 1
        z_coords = torch.linspace(-1, 1, num_samples_per_ray, device=pixel_i.device)

        if self._random_rays:
            thetas, phis = self.get_theta_phi(
                convergence_angle=convergence_angle,
                num_rays=num_rays,
                device=self.device,
                random_rays=random_rays,
            )
            self.phis = phis
            self.thetas = thetas
        else:
            phis = self.phis
            thetas = self.thetas

        dz = z_coords[None, None, :] - z_focus[:, None, None]
        dx = torch.tan(phis)[:, None] * torch.cos(thetas)[:, None] * dz
        dy = torch.tan(phis)[:, None] * torch.sin(thetas)[:, None] * dz

        x_coords = x_coords_0[:, None, None] + dx[None, :, :]
        y_coords = y_coords_0[:, None, None] + dy[None, :, :]
        z_coords = z_coords[None, None, :].expand(batch_size, num_rays, num_samples_per_ray)

        x_coords = x_coords.reshape(batch_size, num_rays * num_samples_per_ray)
        y_coords = y_coords.reshape(batch_size, num_rays * num_samples_per_ray)
        z_coords = z_coords.reshape(batch_size, num_rays * num_samples_per_ray)

        rays = torch.zeros(
            batch_size,
            num_rays * num_samples_per_ray,
            3,
            device=x_coords.device,
        )

        rays[:, :, 0] = x_coords
        rays[:, :, 1] = y_coords
        rays[:, :, 2] = z_coords

        return rays

    # @staticmethod
    @torch.compile(mode="reduce-overhead")
    def integrate_rays(
        self,
        rays: torch.Tensor,
        num_samples_per_ray: int,
        target_values_len: int,
    ) -> torch.Tensor:
        num_rays = self.num_rays
        ray_densities = rays.view(
            target_values_len,
            num_samples_per_ray,
            num_rays,
        )
        if self._random_rays:
            # original weight generation for uniformly sampled rays
            # w = torch.sinc(self.phis / self.convergence_angle) ** 2
            # w = w / w.sum()
            # self._ray_weights = w.view(1, 1, -1)
            # now we want to use equal weights
            predicted_values_all_rays = ray_densities.view(target_values_len, -1) # this will just be equal weights

        else:
            predicted_values_all_rays = (ray_densities @ self._ray_weights.view(-1, 1)).squeeze(-1)

        # if self.print_thing:
        #     if not self._random_rays:
        #         print(self._ray_weights)
        #     print(self.phis)
        #     self.print_thing = False

        step_size = 2.0 / (num_samples_per_ray - 1)

        predicted_values = predicted_values_all_rays.sum(dim=1) * step_size

        return predicted_values

    @staticmethod
    def transform_batch_rays(
        rays: torch.Tensor,
        z1: torch.Tensor,
        x: torch.Tensor,
        z3: torch.Tensor,
        shifts: torch.Tensor,
        N: int,
        sampling_rate: float,
    ) -> torch.Tensor:
        shift_x_norm = (shifts[:, 0:1] * sampling_rate * 2) / (N - 1)
        shift_y_norm = (shifts[:, 1:2] * sampling_rate * 2) / (N - 1)

        shift_x_norm = shift_x_norm.expand(-1, rays.shape[1])
        shift_y_norm = shift_y_norm.expand(-1, rays.shape[1])

        rays_x = rays[:, :, 0] - shift_x_norm
        rays_y = rays[:, :, 1] - shift_y_norm
        rays_z = rays[:, :, 2]

        theta = torch.deg2rad(-z3).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot1 = cos_t * rays_x - sin_t * rays_y
        rays_y_rot1 = sin_t * rays_x + cos_t * rays_y
        rays_z_rot1 = rays_z

        theta = torch.deg2rad(x).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot2 = rays_x_rot1
        rays_y_rot2 = cos_t * rays_y_rot1 - sin_t * rays_z_rot1
        rays_z_rot2 = sin_t * rays_y_rot1 + cos_t * rays_z_rot1

        theta = torch.deg2rad(-z1).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_final = cos_t * rays_x_rot2 - sin_t * rays_y_rot2
        rays_y_final = sin_t * rays_x_rot2 + cos_t * rays_y_rot2
        rays_z_final = rays_z_rot2

        transformed_rays = torch.stack([rays_x_final, rays_y_final, rays_z_final], dim=2)

        return transformed_rays

    def to(self, device: str):
        self._z1_params = nn.Parameter(self._z1_angles.to(device))
        self._z3_params = nn.Parameter(self._z3_angles.to(device))
        self._shifts_params = nn.Parameter(self._shifts.to(device))
        self._z_focus_params = nn.Parameter(self._z_focuses.to(device))

        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)
        self._z_focus_ref = self._z_focus_ref.to(device)

        if hasattr(self, "_ray_weights"):
            self._ray_weights = self._ray_weights.to(device)
            self.phis = self.phis.to(device)
            self.thetas = self.thetas.to(device)

        self.device = device
        self.reconnect_optimizer_to_parameters()


class TomographyThroughFocalConvergenceINRDataset(TomographyINRDataset):
    """
    Dataset class for INR-based tomography that assumes non-parallel illumination condition.
    Inherits from TomographyINRDataset.
    """

    def __init__(
        self,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        ## Through focal parameters:
        # convergence_angle: float,
        convergence_angle: float,
        num_rays: int = 1,
        random_rays: bool = False,
        ## optional parameters
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        token: object | None = None,
        random_method: str = 'g',
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, token)
        self.num_rays = int(num_rays)

        self._random_rays = random_rays

        # self.print_thing = True

        # if not random_rays:
        #     theta, phi = self.get_theta_phi(
        #         convergence_angle=convergence_angle,
        #         num_rays=num_rays,
        #         random_rays=random_rays,
        #     )

        #     self.phis = phi
        #     self.thetas = theta

        #     w = torch.sinc(self.phis / (2 * self.convergence_angle)) ** 2
        #     w = w / w.sum()
        #     self._ray_weights = w.view(1, 1, -1)

        # TODO: The parameters won't be initialized unless .to(device) is called.
        # Fixed zeros for reference tilt
        self._z_focuses = torch.zeros(self.learnable_tilts+1)

        self._convergence_angle = torch.ones(1) * convergence_angle

        # Fixed zeros for reference tilt
        # self._z_focus_ref = torch.zeros(1)

    @property
    def random_rays(self) -> bool:
        return self._random_rays

    @property
    def convergence_angle_params(self) -> torch.nn.Parameter:
        return self._convergence_angle_params

    @convergence_angle_params.setter
    def convergence_angle_params(self, convergence_angle: torch.Tensor, device: str):
        self._convergence_angle_params = nn.Parameter(convergence_angle.to(device))

    @property
    def z_focus_params(self) -> torch.nn.Parameter:
        return self._z_focus_params

    @z_focus_params.setter
    def z_focus_params(self, z_focus_values: torch.Tensor, device: str):
        self._z_focus_params = nn.Parameter(z_focus_values.to(device))

    # --- Forward Pass w/ Params Method for OptimizerMixin ---
    def forward(self, dummy_input: Any = None):
        """
        Forward pass for INR-based through focal tomography. In the forward pass, the only parameters that
        are passed will be the shifts, focal plane, z1 and z3 Euler angles.
        """

        first_half_shifts = self.shifts_params[: self.reference_tilt_idx]
        second_half_shifts = self.shifts_params[self.reference_tilt_idx :]
        shifts = torch.cat([first_half_shifts, self._shifts_ref, second_half_shifts], dim=0)

        first_half_z1 = self.z1_params[: self.reference_tilt_idx]
        second_half_z1 = self.z1_params[self.reference_tilt_idx :]
        z1 = torch.cat([first_half_z1, self._z1_ref, second_half_z1], dim=0)

        first_half_z3 = self.z3_params[: self.reference_tilt_idx]
        second_half_z3 = self.z3_params[self.reference_tilt_idx :]
        z3 = torch.cat([first_half_z3, self._z3_ref, second_half_z3], dim=0)

        # first_half_z_focus = self.z_focus_params[: self.reference_tilt_idx]
        # second_half_z_focus = self.z_focus_params[self.reference_tilt_idx :]
        # z_focus = torch.cat([first_half_z_focus, self._z_focus_ref, second_half_z_focus], dim=0)
        z_focus = self.z_focus_params

        if self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, self._convergence_angle_params[0]
        elif self.learn_shift:
            return shifts, torch.zeros_like(z1), torch.zeros_like(z3), z_focus, self._convergence_angle_params[0]
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3, z_focus, self._convergence_angle_params[0]
        elif self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, self._convergence_angle_params[0]
        else:
            return torch.zeros_like(shifts), torch.zeros_like(z1), torch.zeros_like(z3), z_focus, self._convergence_angle_params[0]

    def get_coords(
        self,
        batch: dict[str, torch.Tensor],
        N: int,
        num_samples_per_ray: int,
    ) -> torch.Tensor:
        num_rays = self.num_rays
        # convergence_angle = self._convergence_angle
        random_rays = self._random_rays
        pixel_i = batch["pixel_i"].float().to(self.device, non_blocking=True)
        pixel_j = batch["pixel_j"].float().to(self.device, non_blocking=True)
        # target_values = batch["target_value"].to(self.device, non_blocking=True)
        phis = batch["phi"].to(self.device, non_blocking=True)
        projection_indices = batch["projection_idx"].to(self.device, non_blocking=True)
        shifts, z1_params, z3_params, z_focus_params, convergence_angle = self.forward(None)
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)
        batch_z_focus = torch.index_select(z_focus_params, 0, projection_indices)
        # with torch.no_grad():
        batch_ray_coords = self.create_batch_rays(
            pixel_i,
            pixel_j,
            N,
            num_samples_per_ray,
            num_rays,
            convergence_angle,
            random_rays,
            z_focus=batch_z_focus,
        )

        transformed_rays = self.transform_batch_rays(
            batch_ray_coords,
            z1=batch_z1,
            x=phis,
            z3=batch_z3,
            shifts=batch_shifts,
            N=N,
            sampling_rate=1.0,
        )
        all_coords = transformed_rays.view(-1, 3)

        all_coords = all_coords.to(self.device, dtype=torch.float32, non_blocking=True)
        return all_coords

    def get_theta_phi(
        self,
        convergence_angle: torch.Tensor,
        num_rays: int,
        device: torch.device | str | None = None,
        random_rays: bool = False,
        random_method: str = 'g',
    ):
        if device is None:
            device = torch.device("cpu")

        num_rays = int(num_rays)

        # random_rays sampling method
        if random_rays:
            theta = torch.rand(num_rays, device=device) * 2 * torch.pi # this can stay as uniform sampling
            # phi = torch.rand(num_rays, device=device) * convergence_angle # the original uniform sampling of phi
            if random_method.lower() in ['gaussian','g']:
                phi = torch.randn(num_rays, device=device) * convergence_angle

            # let's also do the sinc squared, which might be slower?
            # essentially, torch doesn't have a sinc**2 distribution built in, but we can just make a discrete one ourselves
            elif random_method.lower() in ['sinc','s']:
                x = torch.linspace(-0.5, 0.5, 5000) # in radians
                pdf = torch.sinc(x)**2 # 
                pdf = pdf/pdf.sum() # normalize to 1
                indices = torch.multinomial(pdf, num_rays, replacement=True)
                phi = x[indices]
            elif random_method.lower() in ['probe', 'p']:
                indices = torch.multinomial(self._probe_weights, num_rays, replacement=True)
                kr_s = self._kr_flat[indices]
                kc_s = self._kc_flat[indices]

                k_magnitude = torch.sqrt(kr_s**2 + kc_s**2)
                phi = torch.arctan(k_magnitude * self._wavelength_ang)
                theta = torch.arctan2(kc_s, kr_s)   

            else:
                raise ValueError(
                    f"Unsupported random_method={random_method}. "
                    "Supported values are: gaussian, g, sinc, s. .lower() is applied internally."
                )

        else:
            raise ValueError('Only random rays supported for learning convergence angle right now')

        # else:
        #     # 1 ray: single central ray
        #     if num_rays == 1:
        #         theta = torch.zeros(1, device=device)
        #         phi = torch.zeros(1, device=device)

        #     # 2 or 3 rays: single ring
        #     elif num_rays == 2 or num_rays == 3:
        #         rand_offset = torch.rand(1, device=device) * 2 * torch.pi
        #         theta = (
        #             torch.linspace(0, 2 * torch.pi, num_rays + 1, device=device)[:-1] + rand_offset
        #         )
        #         phi = torch.full(
        #             (num_rays,), 0.5 * convergence_angle, device=device
        #         )  # for the case where there is no center ray, use half of the convergence angle.

        #     # 4–8 rays: central + ring
        #     elif 3 < num_rays < 9:
        #         rand_offset = torch.rand(1, device=device) * 2 * torch.pi
        #         theta0 = torch.zeros(1, device=device)
        #         phi0 = torch.zeros(1, device=device)
        #         theta_ring = (
        #             torch.linspace(0, 2 * torch.pi, num_rays, device=device)[:-1] + rand_offset
        #         )
        #         phi_ring = torch.full((num_rays - 1,), 1 * convergence_angle, device=device)
        #         theta = torch.cat((theta0, theta_ring))
        #         phi = torch.cat((phi0, phi_ring))

        #     # 9 rays: two rings. Rays at 1/3 and 2/3 * convergence angle
        #     elif num_rays == 9:
        #         rand_offset = torch.rand(1, device=device) * 2 * torch.pi
        #         theta_inner = torch.linspace(0, 2 * torch.pi, 4, device=device)[:-1] + rand_offset
        #         phi_inner = torch.full((3,), convergence_angle / 3, device=device)
        #         theta_outer = torch.linspace(0, 2 * torch.pi, 7, device=device)[:-1] + rand_offset
        #         phi_outer = torch.full((6,), 2 * convergence_angle / 3, device=device)
        #         theta = torch.cat((theta_inner, theta_outer))
        #         phi = torch.cat((phi_inner, phi_outer))

        #     # 10 rays: two rings, one central ray. Rays at 0, 1/2 and 1 * convergence angle
        #     elif num_rays == 10:
        #         rand_offset = torch.rand(1, device=device) * 2 * torch.pi
        #         theta0 = torch.zeros(1, device=device)
        #         phi0 = torch.zeros(1, device=device)

        #         theta_inner = torch.linspace(0, 2 * torch.pi, 4, device=device)[:-1] + rand_offset
        #         phi_inner = torch.full((3,), convergence_angle / 3, device=device)
        #         theta_outer = torch.linspace(0, 2 * torch.pi, 7, device=device)[:-1] + rand_offset
        #         phi_outer = torch.full((6,), 2 * convergence_angle / 3, device=device)
        #         theta = torch.cat((theta0, theta_inner, theta_outer))
        #         phi = torch.cat((phi0, phi_inner, phi_outer))

        #     # 19 rays: central + two rings. Rays at 0, 1/2 and 1 * convergence angle
        #     elif num_rays == 19:
        #         rand_offset = torch.rand(1, device=device) * 2 * torch.pi
        #         theta0 = torch.zeros(1, device=device)
        #         phi0 = torch.zeros(1, device=device)
        #         theta_inner = torch.linspace(0, 2 * torch.pi, 7, device=device)[:-1] + rand_offset
        #         phi_inner = torch.full((6,), convergence_angle / 2, device=device)
        #         theta_outer = torch.linspace(0, 2 * torch.pi, 13, device=device)[:-1] + rand_offset
        #         phi_outer = torch.full((12,), 1 * convergence_angle, device=device)
        #         theta = torch.cat((theta0, theta_inner, theta_outer))
        #         phi = torch.cat((phi0, phi_inner, phi_outer))

        #     # Unsupported ray counts
        #     else:
        #         raise ValueError(
        #             f"Unsupported num_rays={num_rays}. "
        #             "Supported values are: 1, 2, 3, 4–8, 9, 10, 19."
        #         )

        return theta, phi

    # @staticmethod
    # @torch.compile(mode="reduce-overhead")
    def create_batch_rays(
        self,
        pixel_i: torch.Tensor,
        pixel_j: torch.Tensor,
        N: int,
        num_samples_per_ray: int,
        num_rays: int,
        convergence_angle: torch.Tensor,
        random_rays: bool,
        z_focus: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(pixel_i)
        x_coords_0 = (pixel_j / (N - 1)) * 2 - 1
        y_coords_0 = (pixel_i / (N - 1)) * 2 - 1
        z_coords = torch.linspace(-1, 1, num_samples_per_ray, device=pixel_i.device)

        if self._random_rays:
            thetas, phis = self.get_theta_phi(
                convergence_angle=convergence_angle,
                num_rays=num_rays,
                device=self.device,
                random_rays=random_rays,
            )
            self.phis = phis
            self.thetas = thetas
        else:
            phis = self.phis
            thetas = self.thetas

        dz = z_coords[None, None, :] - z_focus[:, None, None]
        dx = torch.tan(phis)[:, None] * torch.cos(thetas)[:, None] * dz
        dy = torch.tan(phis)[:, None] * torch.sin(thetas)[:, None] * dz

        x_coords = x_coords_0[:, None, None] + dx[None, :, :]
        y_coords = y_coords_0[:, None, None] + dy[None, :, :]
        z_coords = z_coords[None, None, :].expand(batch_size, num_rays, num_samples_per_ray)

        # if self.print_thing:
        #     print('x_coords before:',x_coords_0)
        #     print('dx:',dx)
        #     print('x_coords after:',x_coords)
        #     self.print_thing = False


        x_coords = x_coords.reshape(batch_size, num_rays * num_samples_per_ray)
        y_coords = y_coords.reshape(batch_size, num_rays * num_samples_per_ray)
        z_coords = z_coords.reshape(batch_size, num_rays * num_samples_per_ray)

        rays = torch.zeros(
            batch_size,
            num_rays * num_samples_per_ray,
            3,
            device=x_coords.device,
        )

        rays[:, :, 0] = x_coords
        rays[:, :, 1] = y_coords
        rays[:, :, 2] = z_coords

        return rays

    # @staticmethod
    @torch.compile(mode="reduce-overhead")
    def integrate_rays(
        self,
        rays: torch.Tensor,
        num_samples_per_ray: int,
        target_values_len: int,
    ) -> torch.Tensor:
        num_rays = self.num_rays
        ray_densities = rays.view(
            target_values_len,
            num_samples_per_ray,
            num_rays,
        )
        if self._random_rays:
            # original weight generation for uniformly sampled rays
            # w = torch.sinc(self.phis / self.convergence_angle) ** 2
            # w = w / w.sum()
            # self._ray_weights = w.view(1, 1, -1)
            # now we want to use equal weights
            predicted_values_all_rays = ray_densities.view(target_values_len, -1) # this will just be equal weights

        else:
            predicted_values_all_rays = (ray_densities @ self._ray_weights.view(-1, 1)).squeeze(-1)

        # if self.print_thing:
        #     if not self._random_rays:
        #         print(self._ray_weights)
        #     print(self.phis)
        #     # self.print_thing = False

        step_size = 2.0 / (num_samples_per_ray - 1)

        predicted_values = predicted_values_all_rays.sum(dim=1) * step_size

        return predicted_values

    @staticmethod
    def transform_batch_rays(
        rays: torch.Tensor,
        z1: torch.Tensor,
        x: torch.Tensor,
        z3: torch.Tensor,
        shifts: torch.Tensor,
        N: int,
        sampling_rate: float,
    ) -> torch.Tensor:
        shift_x_norm = (shifts[:, 0:1] * sampling_rate * 2) / (N - 1)
        shift_y_norm = (shifts[:, 1:2] * sampling_rate * 2) / (N - 1)

        shift_x_norm = shift_x_norm.expand(-1, rays.shape[1])
        shift_y_norm = shift_y_norm.expand(-1, rays.shape[1])

        rays_x = rays[:, :, 0] - shift_x_norm
        rays_y = rays[:, :, 1] - shift_y_norm
        rays_z = rays[:, :, 2]

        theta = torch.deg2rad(-z3).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot1 = cos_t * rays_x - sin_t * rays_y
        rays_y_rot1 = sin_t * rays_x + cos_t * rays_y
        rays_z_rot1 = rays_z

        theta = torch.deg2rad(x).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot2 = rays_x_rot1
        rays_y_rot2 = cos_t * rays_y_rot1 - sin_t * rays_z_rot1
        rays_z_rot2 = sin_t * rays_y_rot1 + cos_t * rays_z_rot1

        theta = torch.deg2rad(-z1).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_final = cos_t * rays_x_rot2 - sin_t * rays_y_rot2
        rays_y_final = sin_t * rays_x_rot2 + cos_t * rays_y_rot2
        rays_z_final = rays_z_rot2

        transformed_rays = torch.stack([rays_x_final, rays_y_final, rays_z_final], dim=2)

        return transformed_rays

    def to(self, device: str):
        self._z1_params = nn.Parameter(self._z1_angles.to(device))
        self._z3_params = nn.Parameter(self._z3_angles.to(device))
        self._shifts_params = nn.Parameter(self._shifts.to(device))
        self._z_focus_params = nn.Parameter(self._z_focuses.to(device))
        self._convergence_angle_params = nn.Parameter(self._convergence_angle.to(device))

        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)
        # self._z_focus_ref = self._z_focus_ref.to(device)
        # self._convergence_angle = self._convergence_angle.to(device)

        if hasattr(self, "_ray_weights"):
            self._ray_weights = self._ray_weights.to(device)
            self.phis = self.phis.to(device)
            self.thetas = self.thetas.to(device)

        self.device = device
        self.reconnect_optimizer_to_parameters()



import torch.nn.functional as F

class TomographyThroughFocalAstigmatismINRDataset(TomographyINRDataset):
    """
    Dataset class for INR-based tomography that assumes non-parallel illumination condition with astigmatism.
    Inherits from TomographyINRDataset.
    """

    def __init__(
        self,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        ## Through focal parameters:
        convergence_angle: float,
        num_rays: int = 1,
        ## optional parameters
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        token: object | None = None,
        random_method: str = 'p',
        wavelength_ang: float = 0.0197,
        stig_2: tuple[float, float] = (0.0, 0.0),
        probe_im_shape: tuple[int, int] = (64, 64),
        pixel_size_ang: float = 0.2,
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, token)
        self.num_rays = int(num_rays)

        self._random_rays = True
        self._random_method = random_method

        self._z_focuses = torch.zeros(self.learnable_tilts+1)

        self._convergence_angle = torch.ones(1) * convergence_angle

        self._stig_2 = torch.tensor(stig_2, dtype = torch.float32)

        self._convergence_angle = torch.ones(1) * convergence_angle
        self._stig_2 = torch.tensor(stig_2, dtype=torch.float32)
        
        # Initialize as parameters immediately (will be moved to device in .to())
        self._convergence_angle_params = nn.Parameter(self._convergence_angle.clone())
        self._stig_2_params = nn.Parameter(self._stig_2.clone())

        if random_method.lower() in ['probe', 'p']:
            self._wavelength_ang = wavelength_ang
            k_max = float(self._convergence_angle.item()) / float(wavelength_ang) # convergence angle should be in radians already
            kr = np.fft.fftfreq(probe_im_shape[0], d=pixel_size_ang)[:, None]
            kc = np.fft.fftfreq(probe_im_shape[1], d=pixel_size_ang)[None, :]
            self._dk = torch.tensor(float(kr[1] - kr[0]), dtype=torch.float32)
            dk = float(kr[1, 0] - kr[0, 0])  # scalar float, not a 1-element array
            k1 = np.sqrt(kr**2 + kc**2)
            self._k1 = torch.tensor(k1, dtype=torch.float32)  # radial k-magnitude, shape (H, W)
            aper = np.clip((k_max - k1) / dk + 0.5, 0.0, 1.0)

            # basis
            self._basis_0 = torch.tensor(
                (np.pi * wavelength_ang) * (kr**2 + kc**2), dtype=torch.float32)
            self._basis_1 = torch.tensor(
                (np.pi * wavelength_ang) * (kr**2 - kc**2), dtype=torch.float32)
            self._basis_2 = torch.tensor(
                (np.pi * wavelength_ang) * (2 * kr * kc),   dtype=torch.float32)
            self._aper = torch.tensor(aper, dtype=torch.float32)
            
            self._gumbel_temp = 10

    @property
    def random_rays(self) -> bool:
        return self._random_rays

    @property
    def convergence_angle_params(self) -> torch.nn.Parameter:
        return self._convergence_angle_params

    @convergence_angle_params.setter
    def convergence_angle_params(self, convergence_angle: torch.Tensor, device: str):
        self._convergence_angle_params = nn.Parameter(convergence_angle.to(device))

    @property
    def z_focus_params(self) -> torch.nn.Parameter:
        return self._z_focus_params

    @z_focus_params.setter
    def z_focus_params(self, z_focus_values: torch.Tensor, device: str):
        self._z_focus_params = nn.Parameter(z_focus_values.to(device))

    @property
    def stig_2_params(self) -> torch.nn.Parameter:
        return self._stig_2_params

    @stig_2_params.setter
    def stig_2_params(self, stig_2_values: torch.Tensor, device: str):
        self._stig_2_params = nn.Parameter(stig_2_values.to(device))

    # --- Forward Pass w/ Params Method for OptimizerMixin ---
    def forward(self, dummy_input: Any = None):
        """
        Forward pass for INR-based through focal tomography. In the forward pass, the only parameters that
        are passed will be the shifts, focal plane, z1 and z3 Euler angles.
        """

        first_half_shifts = self.shifts_params[: self.reference_tilt_idx]
        second_half_shifts = self.shifts_params[self.reference_tilt_idx :]
        shifts = torch.cat([first_half_shifts, self._shifts_ref, second_half_shifts], dim=0)

        first_half_z1 = self.z1_params[: self.reference_tilt_idx]
        second_half_z1 = self.z1_params[self.reference_tilt_idx :]
        z1 = torch.cat([first_half_z1, self._z1_ref, second_half_z1], dim=0)

        first_half_z3 = self.z3_params[: self.reference_tilt_idx]
        second_half_z3 = self.z3_params[self.reference_tilt_idx :]
        z3 = torch.cat([first_half_z3, self._z3_ref, second_half_z3], dim=0)

        z_focus = self.z_focus_params

        if self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, self._convergence_angle_params[0], self._stig_2_params[:]
        elif self.learn_shift:
            return shifts, torch.zeros_like(z1), torch.zeros_like(z3), z_focus, self._convergence_angle_params[0], self._stig_2_params[:]
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3, z_focus, self._convergence_angle_params[0], self._stig_2_params[:]
        elif self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, self._convergence_angle_params[0], self._stig_2_params[:]
        else:
            return torch.zeros_like(shifts), torch.zeros_like(z1), torch.zeros_like(z3), z_focus, self._convergence_angle_params[0], self._stig_2_params[:]

    def differentiable_probe_sample(
        self,
        weights: torch.Tensor,   # (H*W,)
        dx_all: torch.Tensor,    # (H*W,)
        dy_all: torch.Tensor,    # (H*W,)
        num_rays: int,
        temperature: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns dx_norm, dy_norm of shape (num_rays,) directly,
        without materializing the full (num_rays, H*W) soft_samples matrix.
        """
        log_probs = torch.log(weights + 1e-10)  # (H*W,)

        # Process one ray at a time to avoid (num_rays, H*W) allocation
        dx_out = torch.zeros(num_rays, device=weights.device)
        dy_out = torch.zeros(num_rays, device=weights.device)

        for r in range(num_rays):
            gumbel_noise = -torch.log(
                -torch.log(torch.rand(len(weights), device=weights.device) + 1e-10) + 1e-10
            )
            soft = torch.softmax((log_probs + gumbel_noise) / temperature, dim=-1)  # (H*W,)
            dx_out[r] = (soft * dx_all).sum()
            dy_out[r] = (soft * dy_all).sum()

        return dx_out, dy_out



    def get_coords(
        self,
        batch: dict[str, torch.Tensor],
        N: int,
        num_samples_per_ray: int,
    ) -> torch.Tensor:
        num_rays = self.num_rays
        # convergence_angle = self._convergence_angle
        pixel_i = batch["pixel_i"].float().to(self.device, non_blocking=True)
        pixel_j = batch["pixel_j"].float().to(self.device, non_blocking=True)
        # target_values = batch["target_value"].to(self.device, non_blocking=True)
        phis = batch["phi"].to(self.device, non_blocking=True)
        projection_indices = batch["projection_idx"].to(self.device, non_blocking=True)
        shifts, z1_params, z3_params, z_focus_params, convergence_angle, stig_2 = self.forward(None)
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)
        batch_z_focus = torch.index_select(z_focus_params, 0, projection_indices)
        # with torch.no_grad():
        batch_ray_coords = self.create_batch_rays(
            pixel_i,
            pixel_j,
            N,
            num_samples_per_ray,
            num_rays,
            z_focus=batch_z_focus,
            stig_2 = stig_2,
            convergence_angle=convergence_angle,
        )

        transformed_rays = self.transform_batch_rays(
            batch_ray_coords,
            z1=batch_z1,
            x=phis,
            z3=batch_z3,
            shifts=batch_shifts,
            N=N,
            sampling_rate=1.0,
        )
        all_coords = transformed_rays.view(-1, 3)

        all_coords = all_coords.to(self.device, dtype=torch.float32, non_blocking=True)
        return all_coords

    def get_theta_phi(
        self,
        convergence_angle: torch.Tensor,
        num_rays: int,
        device: torch.device | str | None = None,
        random_rays: bool = False,
        random_method: str = 'g',
    ):
        if device is None:
            device = torch.device("cpu")

        num_rays = int(num_rays)

        # random_rays sampling method
        if random_rays:
            theta = torch.rand(num_rays, device=device) * 2 * torch.pi # this can stay as uniform sampling
            # phi = torch.rand(num_rays, device=device) * convergence_angle # the original uniform sampling of phi
            if random_method.lower() in ['gaussian','g']:
                phi = torch.randn(num_rays, device=device) * convergence_angle

            # let's also do the sinc squared, which might be slower?
            # essentially, torch doesn't have a sinc**2 distribution built in, but we can just make a discrete one ourselves
            elif random_method.lower() in ['sinc','s']:
                x = torch.linspace(-0.5, 0.5, 5000) # in radians
                pdf = torch.sinc(x)**2 # 
                pdf = pdf/pdf.sum() # normalize to 1
                indices = torch.multinomial(pdf, num_rays, replacement=True)
                phi = x[indices]
            elif random_method.lower() in ['probe', 'p']:
                indices = torch.multinomial(self._probe_weights, num_rays, replacement=True)
                
                # Convert flat indices back to 2D grid positions
                H, W = self._aper.shape
                row_s = (indices // W).float() - H / 2   # centered row coordinate
                col_s = (indices  % W).float() - W / 2   # centered col coordinate

                # Convert grid position to angle
                # The grid spacing in k-space is dk = 1/(N * pixel_size_ang)
                # but since we auto-set pixel_size, we can recover it from basis_0:
                # basis_0 = pi * lambda * (kr^2 + kc^2), so the k-step is:
                # dk = sqrt(basis_0[0,1] / (pi * lambda))  -- but simpler to just store dk at init
                kr_s = row_s * self._dk
                kc_s = col_s * self._dk

                # k_magnitude = torch.sqrt(kr_s**2 + kc_s**2)
                # phi   = torch.arctan(k_magnitude * self._wavelength_ang)
                # theta = torch.arctan2(kc_s, kr_s)

                return kr_s * self._wavelength_ang, kc_s * self._wavelength_ang

            else:
                raise ValueError(
                    f"Unsupported random_method={random_method}. "
                    "Supported values are: gaussian, g, sinc, s. .lower() is applied internally."
                )

        else:
            raise ValueError('Only random rays supported for this dataset model right now')

        return theta, phi




    def _compute_probe_weights_at_defocus(
        self,
        defocus_ang: torch.Tensor | float,
        stig_2: torch.Tensor | None = None,
        convergence_angle: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if stig_2 is None:
            stig_2 = getattr(self, '_stig_2_params', self._stig_2)
        if convergence_angle is None:
            convergence_angle = getattr(self, '_convergence_angle_params', self._convergence_angle)

        # Keep everything as tensors — no .item() calls
        k_max = convergence_angle / self._wavelength_ang # convergence angle already in radians, not mrad
        aper = torch.clamp(
            (k_max - self._k1) / self._dk + 0.5,
            0.0, 1.0,
        )

        chi = (self._basis_0 * defocus_ang +
            self._basis_1 * stig_2[0] +
            self._basis_2 * stig_2[1])

        Psi = aper * torch.exp(-1j * chi)
        psi = torch.fft.ifft2(Psi)
        psi = torch.fft.fftshift(psi)

        weights = torch.abs(psi) ** 2
        weights = weights.ravel()
        weights = weights / weights.sum()
        return weights  # fully differentiable w.r.t. stig_2, convergence_angle, defocus_ang

    def _compute_probe_weights_at_defocus_vectorized(self, defocus_ang, stig_2=None, convergence_angle=None):
        if stig_2 is None:
            stig_2 = self._stig_2_params

        convergence_angle = F.softplus(self._convergence_angle_params[0]) * 45e-3 + 5e-3
        N = defocus_ang.shape[0]
        
        # DEBUG STEP 1: Check convergence angle
        # print(f"convergence_angle: {convergence_angle}")
        # print(f"wavelength: {self._wavelength_ang}")
        
        k_max = convergence_angle / self._wavelength_ang # convergence angle already in radians, not mrad
        # print(f"k_max: {k_max}")
        
        # DEBUG STEP 2: Check aperture
        # print(f"self._k1 shape: {self._k1.shape}, min: {self._k1.min()}, max: {self._k1.max()}")
        # print(f"self._dk: {self._dk}")
        
        aper = torch.clamp(
            (k_max - self._k1) / self._dk + 0.5,
            0.0, 1.0,
        )
        # print(f"aper min: {aper.min()}, max: {aper.max()}, mean: {aper.mean()}")
        # print(f"aper non-zero fraction: {(aper > 0.01).float().mean()}")
        
        # DEBUG STEP 3: Check phase (chi)
        # print(f"self._basis_0 shape: {self._basis_0.shape}, min: {self._basis_0.min()}, max: {self._basis_0.max()}")
        # print(f"defocus_ang: {defocus_ang}")
        # print(f"stig_2: {stig_2}")
        
        chi = (self._basis_0[None, :, :] * defocus_ang[:, None, None] +
            self._basis_1[None, :, :] * stig_2[0] +
            self._basis_2[None, :, :] * stig_2[1])
        
        # print(f"chi min: {chi.min()}, max: {chi.max()}, range: {chi.max() - chi.min()}")
        # print(f"chi std: {chi.std()}")
        
        # DEBUG STEP 4: Check Psi
        Psi = aper * torch.exp(-1j * chi)
        # print(f"|Psi| min: {torch.abs(Psi).min()}, max: {torch.abs(Psi).max()}, mean: {torch.abs(Psi).mean()}")
        
        # DEBUG STEP 5: Check after FFT
        psi = torch.fft.ifft2(Psi, dim=(-2, -1))
        psi = torch.fft.fftshift(psi, dim=(-2, -1))
        
        # print(f"|psi| min: {torch.abs(psi).min()}, max: {torch.abs(psi).max()}, std: {torch.abs(psi).std()}")
        
        weights = torch.abs(psi) ** 2
        # print(f"weights BEFORE norm - min: {weights.min()}, max: {weights.max()}, std: {weights.std()}")
        

    # After computing psi and weights, but before normalizing:
        # import matplotlib.pyplot as plt
        # import numpy as np
        
        # # Take first sample for visualization
        # Psi_vis = Psi[0].detach().cpu()
        # psi_vis = psi[0].detach().cpu()
        # weights_vis_unnorm = (torch.abs(psi_vis) ** 2).numpy()
        
        # # After normalization
        # weights_vis_norm = weights[0].detach().cpu().reshape(psi_vis.shape).numpy()
        
        # fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # # Row 1: Fourier space
        # im0 = axes[0, 0].imshow(torch.abs(Psi_vis).numpy(), cmap='hot')
        # axes[0, 0].set_title('|Psi| (Fourier space aperture)')
        # plt.colorbar(im0, ax=axes[0, 0])
        
        # im1 = axes[0, 1].imshow(np.angle(Psi_vis.numpy()), cmap='twilight')
        # axes[0, 1].set_title('Phase of Psi (chi)')
        # plt.colorbar(im1, ax=axes[0, 1])
        
        # # Radial profile of aperture
        # center = np.array(Psi_vis.shape) // 2
        # y, x = np.ogrid[:Psi_vis.shape[0], :Psi_vis.shape[1]]
        # r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        # r_int = r.astype(int)
        # aper_radial = np.bincount(r_int.ravel(), torch.abs(Psi_vis).numpy().ravel()) / np.bincount(r_int.ravel())
        # axes[0, 2].plot(aper_radial)
        # axes[0, 2].set_title('Radial aperture profile')
        # axes[0, 2].set_xlabel('Radius (pixels)')
        # axes[0, 2].set_ylabel('Aperture')
        # axes[0, 2].grid(True)
        
        # # Row 2: Real space probe
        # im3 = axes[1, 0].imshow(weights_vis_unnorm, cmap='hot')
        # axes[1, 0].set_title('Probe intensity (before norm)')
        # plt.colorbar(im3, ax=axes[1, 0])
        
        # im4 = axes[1, 1].imshow(weights_vis_norm, cmap='hot')
        # axes[1, 1].set_title('Probe intensity (normalized)')
        # plt.colorbar(im4, ax=axes[1, 1])
        
        # # Log scale
        # im5 = axes[1, 2].imshow(np.log10(weights_vis_norm + 1e-10), cmap='hot', vmin=-10, vmax=0)
        # axes[1, 2].set_title('Log10(Probe intensity)')
        # plt.colorbar(im5, ax=axes[1, 2])
        
        # plt.tight_layout()
        # plt.savefig('debug_probe.png', dpi=150, bbox_inches='tight')
        # print(f"Saved debug_probe.png (convergence_angle={convergence_angle.item() if hasattr(convergence_angle, 'item') else convergence_angle:.6f})")
        # plt.close()
    

        weights = weights.reshape(N, -1)  # (N, H*W)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize each probe
                
        # ADD THIS DEBUG:
        # print(f"Probe weights AFTER normalization - min: {weights.min()}, max: {weights.max()}")
        # print(f"Effective positions (weight > 0.001): {(weights > 0.001).sum(dim=-1).float().mean()}")
        # print(f"Entropy: {-(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()}")
        
        return weights  # (N, n_pixels) where n_pixels = H*W




    def differentiable_probe_sample_vectorized(self, weights, dx_all, dy_all, num_rays, temperature):
        N, n_positions = weights.shape
        
        # ADD THESE CHECKS:
        # print(f"dx_all.requires_grad: {dx_all.requires_grad}")
        # print(f"dy_all.requires_grad: {dy_all.requires_grad}")
        # print(f"dx_all dtype: {dx_all.dtype}, weights dtype: {weights.dtype}")
        
        # Make sure dx_all and dy_all don't require grad (they're just coordinates)
        # but are on the same device and dtype
        if dx_all.dtype != weights.dtype:
            dx_all = dx_all.to(weights.dtype)
            dy_all = dy_all.to(weights.dtype)
        
        gumbel_noise = -torch.log(-torch.log(
            torch.rand(N, num_rays, n_positions, device=weights.device, dtype=weights.dtype) + 1e-10
        ) + 1e-10)
        
        log_weights = torch.log_softmax(weights, dim=-1)
        logits = log_weights[:, None, :] + gumbel_noise
        soft_samples = torch.softmax(logits / temperature, dim=-1)
        
        # ADD DEBUG: Check if soft_samples varies
        # print(f"soft_samples min: {soft_samples.min()}, max: {soft_samples.max()}, mean: {soft_samples.mean()}")
        # print(f"soft_samples entropy: {-(soft_samples * torch.log(soft_samples + 1e-10)).sum(dim=-1).mean()}")
        
        dx_norm = torch.matmul(soft_samples, dx_all)
        dy_norm = torch.matmul(soft_samples, dy_all)
        
        # ADD DEBUG: Check outputs
        # print(f"dx_norm min: {dx_norm.min()}, max: {dx_norm.max()}, mean: {dx_norm.mean()}")
        # print(f"dy_norm min: {dy_norm.min()}, max: {dy_norm.max()}, mean: {dy_norm.mean()}")
        
        return dx_norm, dy_norm

    def create_batch_rays(
        self,
        pixel_i: torch.Tensor,
        pixel_j: torch.Tensor,
        N: int,
        num_samples_per_ray: int,
        num_rays: int,
        z_focus: torch.Tensor,
        stig_2: torch.Tensor,
        convergence_angle: torch.Tensor,
        voxel_size_ang: float = 2.0,   # Å per voxel — swap out for experimental data
    ) -> torch.Tensor:
        batch_size = len(pixel_i)
        x_coords_0 = (pixel_j / (N - 1)) * 2 - 1
        y_coords_0 = (pixel_i / (N - 1)) * 2 - 1

        # z_coords in normalized [-1, 1] volume space
        z_coords_norm = torch.linspace(-1, 1, num_samples_per_ray, device=pixel_i.device)

        # Physical size of the volume half-extent in Å
        # [-1, 1] spans N voxels, so 1 unit = (N * voxel_size_ang / 2) Å
        half_extent_ang = (N * voxel_size_ang) / 2.0

        # Convert z_coords and z_focus from normalized to Å
        # dz per projection: shape (batch_size, num_samples_per_ray)
        z_ang = z_coords_norm[None, :] * half_extent_ang           # (1, num_samples_per_ray)
        z_focus_ang = z_focus[:, None] * half_extent_ang            # (batch_size, 1)
        dz_ang = z_ang - z_focus_ang                                # (batch_size, num_samples_per_ray)

        H, W = self._aper.shape

        # Precompute probe pixel offset grid — fixed, no grad needed
        row_all = torch.arange(H, device=pixel_i.device).float() - H / 2.0
        col_all = torch.arange(W, device=pixel_i.device).float() - W / 2.0
        row_grid, col_grid = torch.meshgrid(row_all, col_all, indexing='ij')
        row_grid = row_grid.ravel()  # (H*W,)
        col_grid = col_grid.ravel()  # (H*W,)

        probe_pixel_size_ang = 1.0 / (H * self._dk)
        dx_all = row_grid * probe_pixel_size_ang / half_extent_ang  # (H*W,)
        dy_all = col_grid * probe_pixel_size_ang / half_extent_ang

        x_offsets_norm = torch.zeros(
            batch_size, num_samples_per_ray, num_rays, device=pixel_i.device)
        y_offsets_norm = torch.zeros(
            batch_size, num_samples_per_ray, num_rays, device=pixel_i.device)

        # Vectorize over both batch and z dimensions at once
        # Reshape to (batch_size * num_samples_per_ray,)
        dz_flat = dz_ang.reshape(-1)
        
        # Compute all weights at once: (batch_size * num_samples_per_ray, n_probe_positions)
        weights_flat = self._compute_probe_weights_at_defocus_vectorized(
            dz_flat, stig_2=stig_2, convergence_angle=convergence_angle
        )
        
        # Sample all probes at once: (batch_size * num_samples_per_ray, num_rays)
        dx_norm_flat, dy_norm_flat = self.differentiable_probe_sample_vectorized(
            weights_flat, dx_all, dy_all, num_rays, temperature=self._gumbel_temp
        )
        
        # Reshape back to (batch_size, num_samples_per_ray, num_rays)
        x_offsets_norm = dx_norm_flat.reshape(batch_size, num_samples_per_ray, num_rays)
        y_offsets_norm = dy_norm_flat.reshape(batch_size, num_samples_per_ray, num_rays)
        
        # Add pixel center coords: (batch_size, 1, 1) + (batch_size, num_samples_per_ray, num_rays)
        x_coords = x_coords_0[:, None, None] + x_offsets_norm
        y_coords = y_coords_0[:, None, None] + y_offsets_norm
        z_coords = z_coords_norm[None, :, None].expand(
            batch_size, num_samples_per_ray, num_rays)

        # Flatten rays dimension: (batch_size, num_samples_per_ray * num_rays, 3)
        x_coords = x_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        y_coords = y_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        z_coords = z_coords.reshape(batch_size, num_samples_per_ray * num_rays)

        rays = torch.stack([x_coords, y_coords, z_coords], dim=2)
        return rays

    # @staticmethod
    @torch.compile(mode="reduce-overhead")
    def integrate_rays(
        self,
        rays: torch.Tensor,
        num_samples_per_ray: int,
        target_values_len: int,
    ) -> torch.Tensor:
        num_rays = self.num_rays
        ray_densities = rays.view(
            target_values_len,
            num_samples_per_ray,
            num_rays,
        )
        if self._random_rays:
            predicted_values_all_rays = ray_densities.view(target_values_len, -1) # equal weights
        else:
            predicted_values_all_rays = (ray_densities @ self._ray_weights.view(-1, 1)).squeeze(-1)
        step_size = 2.0 / (num_samples_per_ray - 1)
        predicted_values = predicted_values_all_rays.sum(dim=1) * step_size

        return predicted_values

    @staticmethod
    def transform_batch_rays(
        rays: torch.Tensor,
        z1: torch.Tensor,
        x: torch.Tensor,
        z3: torch.Tensor,
        shifts: torch.Tensor,
        N: int,
        sampling_rate: float,
    ) -> torch.Tensor:
        shift_x_norm = (shifts[:, 0:1] * sampling_rate * 2) / (N - 1)
        shift_y_norm = (shifts[:, 1:2] * sampling_rate * 2) / (N - 1)

        shift_x_norm = shift_x_norm.expand(-1, rays.shape[1])
        shift_y_norm = shift_y_norm.expand(-1, rays.shape[1])

        rays_x = rays[:, :, 0] - shift_x_norm
        rays_y = rays[:, :, 1] - shift_y_norm
        rays_z = rays[:, :, 2]

        theta = torch.deg2rad(-z3).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot1 = cos_t * rays_x - sin_t * rays_y
        rays_y_rot1 = sin_t * rays_x + cos_t * rays_y
        rays_z_rot1 = rays_z

        theta = torch.deg2rad(x).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot2 = rays_x_rot1
        rays_y_rot2 = cos_t * rays_y_rot1 - sin_t * rays_z_rot1
        rays_z_rot2 = sin_t * rays_y_rot1 + cos_t * rays_z_rot1

        theta = torch.deg2rad(-z1).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_final = cos_t * rays_x_rot2 - sin_t * rays_y_rot2
        rays_y_final = sin_t * rays_x_rot2 + cos_t * rays_y_rot2
        rays_z_final = rays_z_rot2

        transformed_rays = torch.stack([rays_x_final, rays_y_final, rays_z_final], dim=2)

        return transformed_rays

    def to(self, device: str):
        self._z1_params = nn.Parameter(self._z1_angles.to(device))
        self._z3_params = nn.Parameter(self._z3_angles.to(device))
        self._shifts_params = nn.Parameter(self._shifts.to(device))
        self._z_focus_params = nn.Parameter(self._z_focuses.to(device))
        self._convergence_angle_params = nn.Parameter(self._convergence_angle.to(device))
        self._stig_2_params = nn.Parameter(self._stig_2.to(device))

        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)
        # self._z_focus_ref = self._z_focus_ref.to(device)
        # self._convergence_angle = self._convergence_angle.to(device)

        if hasattr(self, "_ray_weights"):
            self._ray_weights = self._ray_weights.to(device)
            self.phis = self.phis.to(device)
            self.thetas = self.thetas.to(device)

        if hasattr(self, "_basis_0"):
            self._basis_0 = self._basis_0.to(device)
            self._basis_1 = self._basis_1.to(device)
            self._basis_2 = self._basis_2.to(device)
            self._aper = self._aper.to(device)
            self._dk = self._dk.to(device)
            # self._probe_weights = self._compute_probe_weights_at_defocus(defocus_ang = 0, stig_2=self._stig_2, convergence_angle=self._convergence_angle)
            self._k1 = self._k1.to(device)

        self.device = device
        self.reconnect_optimizer_to_parameters()

DatasetModelType = TomographyINRDataset | TomographyPixDataset | TomographyThroughFocalINRDataset | TomographyThroughFocalConvergenceINRDataset | TomographyThroughFocalAstigmatismINRDataset
