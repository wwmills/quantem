from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import Dataset

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.constraints import BaseConstraints, Constraints
from quantem.core.ml.optimizer_mixin import OptimizerMixin
from quantem.tomography.utils import tv_loss_1d

# --- Constraints ---


class DatasetConstraintParams:
    """
    Namespace class for dataset constraint parameter dataclasses and parsing utilities.

    Contains constraint definitions for different tomography dataset types and a
    factory method for instantiating the appropriate constraint class from a dict.

    Supported constraint types
    --------------------------
    BaseTomographyDatasetConstraints
        Base soft constraints for z-position and lateral shift regularization.
    ThroughFocalDatasetConstraints
        Inherits base constraints; not yet implemented.

    Examples
    --------
    >>> DatasetConstraintParams.parse_dict({"name": "base_tomography_dataset", "tv_zs": 0.1})
    BaseTomographyDatasetConstraints(tv_zs=0.1, tv_shifts=0.0)
    >>> DatasetConstraintParams.parse_dict({"type": "base_tomography_dataset"})
    BaseTomographyDatasetConstraints(tv_zs=0.0, tv_shifts=0.0)
    """

    @dataclass
    class BaseTomographyDatasetConstraints(Constraints):
        """
        Soft constraints for a base tomography dataset.

        Attributes
        ----------
        tv_zs : float
            Total variation regularization weight for Z1 and Z3 Euler angles.
        tv_shifts : float
            Total variation regularization weight for X and Y shifts.
        soft_constraint_keys : list[str]
            Constraint fields penalized softly during optimization.
        hard_constraint_keys : list[str]
            Constraint fields enforced strictly (none for this class).
        """

        tv_zs: float = 0.0
        tv_shifts: float = 0.0
        _name: str = "base_tomography_dataset"

        soft_constraint_keys = ["tv_zs", "tv_shifts"]
        hard_constraint_keys = []

    @dataclass
    class ThroughFocalDatasetConstraints(BaseTomographyDatasetConstraints):
        """
        Constraints for a through-focal tomography dataset.

        Inherits all constraints from ``BaseTomographyDatasetConstraints``.
        Currently not implemented — instantiation will raise ``NotImplementedError``.
        """

        pass

    @classmethod
    def parse_dict(
        cls, d: dict
    ) -> "DatasetConstraintParams.BaseTomographyDatasetConstraints | DatasetConstraintParams.ThroughFocalDatasetConstraints":
        """
        Instantiate a dataset constraint dataclass from a configuration dictionary.

        The dictionary must contain a ``'name'`` or ``'type'`` key identifying
        which constraint class to construct. All remaining keys are forwarded as
        keyword arguments to the selected dataclass.

        Parameters
        ----------
        d : dict
            Configuration dictionary. Must include ``'name'`` or ``'type'``
            with one of the following values (case-insensitive):

            - ``'base_tomography_dataset'`` → :class:`BaseTomographyDatasetConstraints`
            - ``'through_focal_dataset'`` → :class:`ThroughFocalDatasetConstraints`
              *(not yet implemented)*

            The value may also be a class ``type`` object, in which case its
            ``__name__`` is used after lower-casing.

        Returns
        -------
        BaseTomographyDatasetConstraints or ThroughFocalDatasetConstraints
            An instance of the appropriate constraint dataclass.

        Raises
        ------
        ValueError
            If neither ``'name'`` nor ``'type'`` is present, if the value is not
            a string or type, or if the name does not match any known dataset
            constraint type.
        NotImplementedError
            If ``'through_focal_dataset'`` is requested, as it is not yet implemented.
        """
        d = dict(d)
        name = d.pop("name", None)
        type_ = d.pop("type", None)
        name = name or type_
        if name is None:
            raise ValueError("Must provide either 'name' or 'type' key")
        if isinstance(name, type):
            name = name.__name__.lower()
        elif isinstance(name, str):
            name = name.lower()
        else:
            raise ValueError(f"Unknown dataset constraint type: {name}")
        if name == "base_tomography_dataset":
            return DatasetConstraintParams.BaseTomographyDatasetConstraints(**d)
        elif name == "through_focal_dataset":
            raise NotImplementedError("Through focal dataset constraints are not implemented yet.")
        else:
            raise ValueError(f"Unknown dataset constraint type: {name.lower()}")


DatasetConstraintsType = (
    DatasetConstraintParams.BaseTomographyDatasetConstraints
    | DatasetConstraintParams.ThroughFocalDatasetConstraints
)


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
        if _token is not self._token:
            raise RuntimeError("Use TomographyPixDataset.from_* to instantiate this class.")

        if not (
            tilt_stack.shape[0] < tilt_stack.shape[1] or tilt_stack.shape[0] < tilt_stack.shape[2]
        ):
            raise ValueError(
                "The number of tilt projections should be in the first dimension of the dataset."
            )

        if type(tilt_stack) is not torch.Tensor:
            tilt_stack = torch.from_numpy(tilt_stack)
        if type(tilt_angles) is not torch.Tensor:
            tilt_angles = torch.from_numpy(tilt_angles)
        max_val = torch.quantile(tilt_stack, 0.95)

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
            _token=cls._token,
        )

    # --- Optimization Parameters ---

    def get_optimization_parameters(self) -> dict[str, list[torch.Tensor]]:
        """Single param group keyed by DEFAULT_OPTIMIZER_KEY.

        Hyperparameters are baked by ``set_optimizer``, not here — return only the tensors,
        matching the ``dict[str, list[tensor]]`` contract the object models use.
        """
        return {self.DEFAULT_OPTIMIZER_KEY: list(self.parameters())}

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
        return int(self._reference_tilt_angle_idx)

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
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device | str):
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device

    # --- Helper Functions ---
    @abstractmethod
    def to(self, device: torch.device | str):  # type: ignore
        """
        Moves the dataset to the device, and also insantiates the aux params to the device.
        """

        raise NotImplementedError("This method should be implemented in subclasses.")


class TomographyDatasetConstraints(BaseConstraints, TomographyDatasetBase):
    DEFAULT_CONSTRAINTS = DatasetConstraintParams.BaseTomographyDatasetConstraints()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints: DatasetConstraintParams.BaseTomographyDatasetConstraints = (
            self.DEFAULT_CONSTRAINTS.copy()
        )

    def apply_soft_constraints(self) -> torch.Tensor:
        soft_loss = torch.tensor(0.0, device=self.z1_params.device)
        if self.constraints.tv_zs > 0:
            tv_loss_zs = tv_loss_1d(self.z1_params)
            tv_loss_zs += tv_loss_1d(self.z3_params)
            tv_loss_zs = self.constraints.tv_zs * tv_loss_zs
            soft_loss += tv_loss_zs

        if self.constraints.tv_shifts > 0:
            # Shift params is of shape (N, 2)
            tv_loss_shifts = tv_loss_1d(self.shifts_params[:, 0])
            tv_loss_shifts += tv_loss_1d(self.shifts_params[:, 1])
            tv_loss_shifts = self.constraints.tv_shifts * tv_loss_shifts
            soft_loss += tv_loss_shifts
        return soft_loss

    def apply_hard_constraints(self) -> torch.Tensor:
        """
        No hard constraints have been implemented yet.
        """
        return torch.tensor(0.0)


class TomographyPixDataset(TomographyDatasetConstraints):
    """
    Dataset class for pixel-based tomography, i.e AD, SIRT, WBP, etc...

    These algorithms only require the tilt image in the forward call.
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

    def forward(  # type:ignore
        self,
        proj_idx: int,
    ) -> DatasetValue:
        """
        Forward pass for pixel-based tomography.
        Returns the full tilt image for the given projection index, and the tilt angle.
        """

        return DatasetValue(
            target=self.tilt_stack[proj_idx],
            tilt_angle=self.tilt_angles[proj_idx].item(),
            pixel_loc=None,
        )

    def to(self, device: str | torch.device):
        """
        Moves the tilt stack and tilt_angles to the device, along with other nn.Parameters to the device.
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


class TomographyINRDataset(TomographyDatasetConstraints, Dataset):
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
        _token: object | None = None,
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, _token=_token)

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
    ) -> dict:
        """
        Gets the item for INR i.e, the project index, pixel value at (i, j), and the tilt angle.
        """

        actual_idx = idx

        projection_idx = actual_idx // (self.tilt_stack.shape[1] * self.tilt_stack.shape[2])
        remaining = actual_idx % (self.tilt_stack.shape[1] * self.tilt_stack.shape[2])

        pixel_i = remaining // self.tilt_stack.shape[1]
        pixel_j = remaining % self.tilt_stack.shape[1]

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

    def to(self, device: torch.device | str):
        self._z1_params = nn.Parameter(self._z1_angles.to(device))
        self._z3_params = nn.Parameter(self._z3_angles.to(device))
        self._shifts_params = nn.Parameter(self._shifts.to(device))

        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)

        self.device = device
        self.reconnect_optimizer_to_parameters()

    # --- Save learned parameters ---

    def save_parameters(self, path: str):
        """
        Saves the learned parameters to a file.
        """
        torch.save(
            {
                "z1": self._z1_params.detach().cpu(),
                "z3": self._z3_params.detach().cpu(),
                "shifts": self._shifts_params.detach().cpu(),
            },
            path,
        )

    def load_parameters(self, path: str):
        """
        Loads the learned parameters from a file.
        """
        data = torch.load(path)
        self._z1_params = nn.Parameter(data["z1"]).to(self.device)
        self._z3_params = nn.Parameter(data["z3"]).to(self.device)
        self._shifts_params = nn.Parameter(data["shifts"]).to(self.device)
        if self.optimizer is not None:
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
        data = torch.permute(data, (0, 3, 2, 1))
        # data = torch.flip(data, dims=(2,))

        self.volume = data.cpu()
        self.N = pretrain_target.shape[1]  # Assumes cubic volume.
        self.total_samples = pretrain_target.shape[1] ** 3

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
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, _token = token)
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
            import numpy as np
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
                (torch.pi * wavelength_ang) * (kr**2 + kc**2), dtype=torch.float32)
            self._basis_1 = torch.tensor(
                (torch.pi * wavelength_ang) * (kr**2 - kc**2), dtype=torch.float32)
            self._basis_2 = torch.tensor(
                (torch.pi * wavelength_ang) * (2 * kr * kc),   dtype=torch.float32)
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

    @classmethod
    def from_data(
        cls,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        convergence_angle: float = 10e-3,
        num_rays: int = 5,
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
    ):
        return cls(
            tilt_stack=tilt_stack,
            tilt_angles=tilt_angles,
            convergence_angle=convergence_angle,
            num_rays=num_rays,
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
            token=cls._token,  # <-- changed from _token to token
        )

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

        convergence_angle = F.softplus(self._convergence_angle_params[0]) * 45e-3 + 5e-3


        if self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, convergence_angle, self._stig_2_params[:]
        elif self.learn_shift:
            return shifts, torch.zeros_like(z1), torch.zeros_like(z3), z_focus, convergence_angle, self._stig_2_params[:]
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3, z_focus, convergence_angle, self._stig_2_params[:]
        elif self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, convergence_angle, self._stig_2_params[:]
        else:
            return torch.zeros_like(shifts), torch.zeros_like(z1), torch.zeros_like(z3), z_focus, convergence_angle, self._stig_2_params[:]

    # def get_coords(
    #     self,
    #     batch: dict[str, torch.Tensor],
    #     N: int,
    #     num_samples_per_ray: int,
    # ) -> torch.Tensor:
    #     num_rays = self.num_rays
    #     # convergence_angle = self._convergence_angle
    #     pixel_i = batch["pixel_i"].float().to(self.device, non_blocking=True)
    #     pixel_j = batch["pixel_j"].float().to(self.device, non_blocking=True)
    #     # target_values = batch["target_value"].to(self.device, non_blocking=True)
    #     phis = batch["phi"].to(self.device, non_blocking=True)
    #     projection_indices = batch["projection_idx"].to(self.device, non_blocking=True)
    #     shifts, z1_params, z3_params, z_focus_params, convergence_angle, stig_2 = self.forward(None)
    #     batch_shifts = torch.index_select(shifts, 0, projection_indices)
    #     batch_z1 = torch.index_select(z1_params, 0, projection_indices)
    #     batch_z3 = torch.index_select(z3_params, 0, projection_indices)
    #     batch_z_focus = torch.index_select(z_focus_params, 0, projection_indices)
    #     # with torch.no_grad():
    #     batch_ray_coords = self.create_batch_rays(
    #         pixel_i,
    #         pixel_j,
    #         N,
    #         num_samples_per_ray,
    #         num_rays,
    #         z_focus=batch_z_focus,
    #         stig_2 = stig_2,
    #         convergence_angle=convergence_angle,
    #     )

    #     transformed_rays = self.transform_batch_rays(
    #         batch_ray_coords,
    #         z1=batch_z1,
    #         x=phis,
    #         z3=batch_z3,
    #         shifts=batch_shifts,
    #         N=N,
    #         sampling_rate=1.0,
    #     )
    #     all_coords = transformed_rays.view(-1, 3)

    #     all_coords = all_coords.to(self.device, dtype=torch.float32, non_blocking=True)
    #     return all_coords



    def get_coords(
        self,
        batch: dict[str, torch.Tensor],
        N: int,
        num_samples_per_ray: int,
        use_probe_weighting: bool = True,  # New flag
        ray_pattern: str = 'grid',  # New parameter
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            all_coords: (batch * num_samples * num_rays, 3)
            probe_weights: (batch, num_samples, num_rays) if use_probe_weighting else None
        """
        num_rays = self.num_rays
        pixel_i = batch["pixel_i"].float().to(self.device, non_blocking=True)
        pixel_j = batch["pixel_j"].float().to(self.device, non_blocking=True)
        phis = batch["phi"].to(self.device, non_blocking=True)
        projection_indices = batch["projection_idx"].to(self.device, non_blocking=True)
        
        shifts, z1_params, z3_params, z_focus_params, convergence_angle, stig_2 = self.forward(None)
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)
        batch_z_focus = torch.index_select(z_focus_params, 0, projection_indices)
        
        if use_probe_weighting:
            # New method: uniform/pattern sampling with probe weights
            batch_ray_coords, probe_weights = self.create_batch_rays_uniform(
                pixel_i, pixel_j, N, num_samples_per_ray, num_rays,
                z_focus=batch_z_focus,
                stig_2=stig_2,
                convergence_angle=convergence_angle,
                ray_pattern=ray_pattern,
            )
        else:
            # Old method: Gumbel-softmax sampling
            batch_ray_coords = self.create_batch_rays(
                pixel_i, pixel_j, N, num_samples_per_ray, num_rays,
                z_focus=batch_z_focus,
                stig_2=stig_2,
                convergence_angle=convergence_angle,
            )
            probe_weights = None
        
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
        
        return all_coords, probe_weights



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


    def _compute_probe_weights_at_defocus_vectorized(self, defocus_ang, stig_2=None, convergence_angle=None):
        if stig_2 is None:
            stig_2 = self._stig_2_params

        # convergence_angle = F.softplus(self._convergence_angle_params[0]) * 45e-3 + 5e-3
        N = defocus_ang.shape[0]
                
        k_max = convergence_angle / self._wavelength_ang # convergence angle already in radians, not mrad

        aper = torch.clamp(
            (k_max - self._k1) / self._dk + 0.5,
            0.0, 1.0,
        )
        
        chi = (self._basis_0[None, :, :] * defocus_ang[:, None, None] +
            self._basis_1[None, :, :] * stig_2[0] +
            self._basis_2[None, :, :] * stig_2[1])

        Psi = aper * torch.exp(-1j * chi)

        psi = torch.fft.ifft2(Psi, dim=(-2, -1))
        psi = torch.fft.fftshift(psi, dim=(-2, -1))
        
        weights = torch.abs(psi) ** 2
        
        weights = weights.reshape(N, -1)  # (N, H*W)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize each probe

        return weights  # (N, n_pixels) where n_pixels = H*W


    def differentiable_probe_sample_vectorized(self, weights, dx_all, dy_all, num_rays, temperature):
        N, n_positions = weights.shape
        
        if dx_all.dtype != weights.dtype:
            dx_all = dx_all.to(weights.dtype)
            dy_all = dy_all.to(weights.dtype)
        
        gumbel_noise = -torch.log(-torch.log(
            torch.rand(N, num_rays, n_positions, device=weights.device, dtype=weights.dtype) + 1e-10
        ) + 1e-10)
        
        log_weights = torch.log_softmax(weights, dim=-1)
        logits = log_weights[:, None, :] + gumbel_noise
        soft_samples = torch.softmax(logits / temperature, dim=-1)

        dx_norm = torch.matmul(soft_samples, dx_all)
        dy_norm = torch.matmul(soft_samples, dy_all)
        
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






    def create_batch_rays_uniform(
        self,
        pixel_i: torch.Tensor,
        pixel_j: torch.Tensor,
        N: int,
        num_samples_per_ray: int,
        num_rays: int,
        z_focus: torch.Tensor,
        stig_2: torch.Tensor,
        convergence_angle: torch.Tensor,
        voxel_size_ang: float = 2.0,
        ray_pattern: str = 'grid',  # 'uniform', 'hexagonal', 'grid'
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create rays with uniform/pattern sampling and return both rays and probe weights.
        
        Returns:
            rays: (batch_size, num_samples_per_ray * num_rays, 3)
            probe_weights: (batch_size, num_samples_per_ray, num_rays) - weights for each ray at each slice
        """
        batch_size = len(pixel_i)
        x_coords_0 = (pixel_j / (N - 1)) * 2 - 1
        y_coords_0 = (pixel_i / (N - 1)) * 2 - 1

        # Z coordinates
        z_coords_norm = torch.linspace(-1, 1, num_samples_per_ray, device=pixel_i.device)
        half_extent_ang = (N * voxel_size_ang) / 2.0
        
        z_ang = z_coords_norm[None, :] * half_extent_ang
        z_focus_ang = z_focus[:, None] * half_extent_ang
        dz_ang = z_ang - z_focus_ang  # (batch_size, num_samples_per_ray)

        # Generate ray pattern (uniform or structured)
        if ray_pattern == 'uniform':
            # Uniform random sampling in circular aperture
            # Use sqrt for uniform distribution in 2D circle
            angles = torch.rand(num_rays, device=pixel_i.device) * 2 * torch.pi
            radii = torch.sqrt(torch.rand(num_rays, device=pixel_i.device))
            
            # In angle space (radians)
            alpha_x = radii * torch.cos(angles) * convergence_angle
            alpha_y = radii * torch.sin(angles) * convergence_angle
            
        elif ray_pattern == 'hexagonal':
            # Hexagonal pattern + center
            # num_rays = 7
            if num_rays == 7:
                angles_deg = torch.tensor([0, 60, 120, 180, 240, 300], device=pixel_i.device)
                radius = convergence_angle * 0.7  # 70% of max aperture
                alpha_x = torch.cat([
                    torch.zeros(1, device=pixel_i.device),
                    radius * torch.cos(torch.deg2rad(angles_deg))
                ])
                alpha_y = torch.cat([
                    torch.zeros(1, device=pixel_i.device),
                    radius * torch.sin(torch.deg2rad(angles_deg))
                ])
            else:
                raise ValueError(f"Hexagonal pattern only supports 7 rays, got {num_rays}")
        
        elif ray_pattern == 'grid':
            # Square grid pattern
            grid_size = int(torch.ceil(torch.sqrt(num_rays)))
            x_grid = torch.linspace(-convergence_angle, convergence_angle, grid_size, device=pixel_i.device)
            y_grid = torch.linspace(-convergence_angle, convergence_angle, grid_size, device=pixel_i.device)
            grid_x, grid_y = torch.meshgrid(x_grid, y_grid, indexing='ij')
            alpha_x = grid_x.ravel()[:num_rays]
            alpha_y = grid_y.ravel()[:num_rays]
        
        else:
            raise ValueError(f"Unknown ray_pattern: {ray_pattern}")
        
        # Compute probe weights for each defocus slice
        # Shape: (batch_size, num_samples_per_ray, num_rays)
        probe_weights = self._compute_probe_weights_for_rays(
            dz_ang, alpha_x, alpha_y, stig_2, convergence_angle
        )
        
        # Convert angles to spatial offsets in normalized coordinates
        # alpha (radians) -> physical distance at defocus -> normalized coords
        # For each z-slice, offset = alpha * |dz| / half_extent
        
        # Broadcast: dz_ang is (batch, num_samples), alpha is (num_rays,)
        # Result: (batch, num_samples, num_rays)
        dz_broadcast = dz_ang[:, :, None]  # (batch, num_samples, 1)
        alpha_x_broadcast = alpha_x[None, None, :]  # (1, 1, num_rays)
        alpha_y_broadcast = alpha_y[None, None, :]
        
        # Offset in normalized space
        x_offsets_norm = alpha_x_broadcast * torch.abs(dz_broadcast) / half_extent_ang
        y_offsets_norm = alpha_y_broadcast * torch.abs(dz_broadcast) / half_extent_ang
        
        # Add to pixel center
        x_coords = x_coords_0[:, None, None] + x_offsets_norm  # (batch, num_samples, num_rays)
        y_coords = y_coords_0[:, None, None] + y_offsets_norm
        z_coords = z_coords_norm[None, :, None].expand(batch_size, num_samples_per_ray, num_rays)
        
        # Flatten rays: (batch, num_samples * num_rays, 3)
        x_coords = x_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        y_coords = y_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        z_coords = z_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        
        rays = torch.stack([x_coords, y_coords, z_coords], dim=2)
        
        return rays, probe_weights



    # def _compute_probe_weights_for_rays(
    #     self,
    #     dz_ang: torch.Tensor,  # (batch_size, num_samples_per_ray) - defocus in Angstroms
    #     alpha_x: torch.Tensor,  # (num_rays,) ray angles in radians
    #     alpha_y: torch.Tensor,  # (num_rays,) ray angles in radians
    #     stig_2: torch.Tensor,   # (2,) astigmatism parameters
    #     convergence_angle: torch.Tensor,  # scalar, convergence angle in radians
    # ) -> torch.Tensor:
    #     """
    #     Compute probe intensity at real-space positions corresponding to ray angles.
        
    #     The probe in real space is: psi(r) = IFFT[Aperture(k) * exp(-i*chi(k))]
    #     For a ray at angle alpha, the position at defocus dz is: r = alpha * |dz|
        
    #     We compute psi at these specific r positions using the Fourier transform definition.
        
    #     Returns:
    #         weights: (batch_size, num_samples_per_ray, num_rays)
    #     """
    #     batch_size, num_samples = dz_ang.shape
    #     num_rays = len(alpha_x)
        
    #     # Real-space positions where rays hit at each defocus
    #     # r = alpha * |dz|
    #     dz_broadcast = dz_ang[:, :, None]  # (batch, num_samples, 1)
    #     r_x = alpha_x[None, None, :] * torch.abs(dz_broadcast)  # (batch, num_samples, num_rays) in Angstroms
    #     r_y = alpha_y[None, None, :] * torch.abs(dz_broadcast)
        
    #     # We need the full k-space grid to compute the probe via IFFT
    #     # Get k-space grid (fixed)
    #     H, W = self._aper.shape
    #     kr = torch.fft.fftfreq(H, d=1.0/(H*self._dk), device=dz_ang.device)[:, None]  # (H, 1)
    #     kc = torch.fft.fftfreq(W, d=1.0/(W*self._dk), device=dz_ang.device)[None, :]  # (1, W)
        
    #     # Compute aperture (same for all defocus)
    #     k_max = convergence_angle / self._wavelength_ang
    #     k1 = torch.sqrt(kr**2 + kc**2)
    #     aper = torch.clamp(
    #         (k_max - k1) / self._dk + 0.5,
    #         0.0, 1.0,
    #     )  # (H, W)
        
    #     # Compute basis functions on the grid
    #     basis_0_grid = torch.pi * self._wavelength_ang * (kr**2 + kc**2)  # (H, W)
    #     basis_1_grid = torch.pi * self._wavelength_ang * (kr**2 - kc**2)  # (H, W)
    #     basis_2_grid = torch.pi * self._wavelength_ang * (2 * kr * kc)    # (H, W)
        
    #     # Compute chi for each defocus value
    #     # dz_ang: (batch, num_samples) -> need (batch, num_samples, H, W)
    #     dz_expanded = dz_ang[:, :, None, None]  # (batch, num_samples, 1, 1)
        
    #     chi = (basis_0_grid[None, None, :, :] * dz_expanded +
    #         basis_1_grid[None, None, :, :] * stig_2[0] +
    #         basis_2_grid[None, None, :, :] * stig_2[1])  # (batch, num_samples, H, W)
        
    #     # Compute Psi in k-space
    #     Psi = aper[None, None, :, :] * torch.exp(-1j * chi)  # (batch, num_samples, H, W)
        
    #     # Now we need psi at specific real-space positions r_x, r_y
    #     # psi(r) = Σ_k Psi(k) * exp(2πi k·r) / N
    #     # This is the inverse DFT evaluated at specific points
        
    #     # Flatten k-space for easier computation
    #     kr_flat = kr.ravel()  # (H*W,)
    #     kc_flat = kc.ravel()  # (H*W,)
    #     Psi_flat = Psi.reshape(batch_size, num_samples, -1)  # (batch, num_samples, H*W)
        
    #     # Compute phase: exp(2πi (kr*rx + kc*ry))
    #     # r_x, r_y: (batch, num_samples, num_rays)
    #     # kr_flat, kc_flat: (H*W,)
        
    #     # Reshape for broadcasting:
    #     # (batch, num_samples, num_rays, 1) × (1, 1, 1, H*W)
    #     r_x_expanded = r_x[:, :, :, None]  # (batch, num_samples, num_rays, 1)
    #     r_y_expanded = r_y[:, :, :, None]
    #     kr_expanded = kr_flat[None, None, None, :]  # (1, 1, 1, H*W)
    #     kc_expanded = kc_flat[None, None, None, :]
        
    #     # Phase from real-space position
    #     phase = 2 * torch.pi * (kr_expanded * r_x_expanded + kc_expanded * r_y_expanded)
    #     phase_factor = torch.exp(1j * phase)  # (batch, num_samples, num_rays, H*W)
        
    #     # Psi_flat: (batch, num_samples, 1, H*W)
    #     # phase_factor: (batch, num_samples, num_rays, H*W)
    #     Psi_expanded = Psi_flat[:, :, None, :]  # (batch, num_samples, 1, H*W)
        
    #     # Compute psi at each position
    #     psi_at_rays = (Psi_expanded * phase_factor).sum(dim=-1)  # (batch, num_samples, num_rays)
        
    #     # Intensity is |psi|^2
    #     weights = torch.abs(psi_at_rays) ** 2  # (batch, num_samples, num_rays)
        
    #     # Normalize
    #     weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-10)
        
    #     return weights





    def _compute_probe_weights_for_rays(
            self,
            dz_ang: torch.Tensor,
            alpha_x: torch.Tensor,
            alpha_y: torch.Tensor,
            stig_2: torch.Tensor,
            convergence_angle: torch.Tensor,
        ) -> torch.Tensor:
            batch_size, num_samples = dz_ang.shape
            num_rays = len(alpha_x)
            
            # Real-space positions where rays hit at each defocus
            dz_broadcast = dz_ang[:, :, None]
            r_x = alpha_x[None, None, :] * torch.abs(dz_broadcast)
            r_y = alpha_y[None, None, :] * torch.abs(dz_broadcast)
            
            # Get k-space grid
            H, W = self._aper.shape
            kr_1d = torch.fft.fftfreq(H, d=1.0/(H*self._dk), device=dz_ang.device)
            kc_1d = torch.fft.fftfreq(W, d=1.0/(W*self._dk), device=dz_ang.device)
            
            # Create 2D grids
            kr = kr_1d[:, None]  # (H, 1)
            kc = kc_1d[None, :]  # (1, W)
            
            # Compute aperture
            k_max = convergence_angle / self._wavelength_ang
            k1 = torch.sqrt(kr**2 + kc**2)
            aper = torch.clamp((k_max - k1) / self._dk + 0.5, 0.0, 1.0)
            
            # Compute basis functions
            basis_0_grid = torch.pi * self._wavelength_ang * (kr**2 + kc**2)
            basis_1_grid = torch.pi * self._wavelength_ang * (kr**2 - kc**2)
            basis_2_grid = torch.pi * self._wavelength_ang * (2 * kr * kc)
            
            # Compute chi
            dz_expanded = dz_ang[:, :, None, None]
            chi = (basis_0_grid[None, None, :, :] * dz_expanded +
                basis_1_grid[None, None, :, :] * stig_2[0] +
                basis_2_grid[None, None, :, :] * stig_2[1])
            
            # Compute Psi in k-space
            Psi = aper[None, None, :, :] * torch.exp(-1j * chi)
            
            # Flatten k-space - need meshgrid for proper kr, kc pairs
            kr_grid, kc_grid = torch.meshgrid(kr_1d, kc_1d, indexing='ij')  # Both (H, W)
            kr_flat = kr_grid.ravel()  # (H*W,)
            kc_flat = kc_grid.ravel()  # (H*W,)
            Psi_flat = Psi.reshape(batch_size, num_samples, -1)  # (batch, num_samples, H*W)
            
            # Compute phase
            r_x_expanded = r_x[:, :, :, None]
            r_y_expanded = r_y[:, :, :, None]
            kr_expanded = kr_flat[None, None, None, :]
            kc_expanded = kc_flat[None, None, None, :]
            
            phase = 2 * torch.pi * (kr_expanded * r_x_expanded + kc_expanded * r_y_expanded)
            phase_factor = torch.exp(1j * phase)
            
            Psi_expanded = Psi_flat[:, :, None, :]
            
            # Compute psi at each position
            psi_at_rays = (Psi_expanded * phase_factor).sum(dim=-1)
            
            # Intensity
            weights = torch.abs(psi_at_rays) ** 2
            
            # DEBUG: Print weight statistics
            # if torch.rand(1).item() < 0.01:  # Print 1% of the time
            #     print(f"DEBUG weights - min: {weights.min():.6f}, max: {weights.max():.6f}, "
            #         f"mean: {weights.mean():.6f}, std: {weights.std():.6f}")
            #     print(f"DEBUG convergence_angle: {convergence_angle.item():.6e}")
            #     print(f"DEBUG num_rays: {num_rays}, batch_size: {batch_size}")
            
            # Normalize
            weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-10)
            
            return weights



    @torch.compile(mode="reduce-overhead")
    def integrate_rays_with_probe_weights(
        self,
        rays: torch.Tensor,  # (batch, num_samples * num_rays, 3) -> densities after INR
        probe_weights: torch.Tensor,  # (batch, num_samples, num_rays)
        num_samples_per_ray: int,
        target_values_len: int,
    ) -> torch.Tensor:
        """
        Integrate rays with probe weighting.
        
        For each pixel:
        1. At each z-slice, we have num_rays with their densities
        2. Weight each ray by its probe intensity at that slice
        3. Sum weighted rays at each slice
        4. Integrate over slices
        
        Args:
            rays: INR density values, shape (batch, num_samples * num_rays)
            probe_weights: (batch, num_samples, num_rays)
        """
        num_rays = self.num_rays
        
        # Reshape densities: (batch, num_samples, num_rays)
        ray_densities = rays.view(target_values_len, num_samples_per_ray, num_rays)
        
        # Apply probe weights (element-wise multiplication)
        weighted_densities = ray_densities * probe_weights  # (batch, num_samples, num_rays)
        
        # Sum over rays at each slice
        slice_values = weighted_densities.sum(dim=2)  # (batch, num_samples)
        
        # Integrate over z
        step_size = 2.0 / (num_samples_per_ray - 1)
        predicted_values = slice_values.sum(dim=1) * step_size  # (batch,)
        
        return predicted_values











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









































import torch.nn.functional as F

class TomographyThroughFocalProbeINRDataset(TomographyINRDataset):
    """
    Dataset class for INR-based tomography that assumes non-parallel illumination condition and uses a (more) mathematically accurate probe for ray weighting.
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
        learn_astigmatism: bool = False,
        learn_convergence: bool = False,
        token: object | None = None,
        random_method: str = 'p',
        wavelength_ang: float = 0.0197,
        stig_2: tuple[float, float] = (0.0, 0.0),
        probe_im_shape: tuple[int, int] = (64, 64),
        pixel_size_ang: float = 0.2,
        voxel_size_ang: float = 20.0,
        ray_pattern: str = 'clever',
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, _token = token)
        self.num_rays = int(num_rays)
        self.learn_astigmatism = learn_astigmatism
        self.learn_convergence = learn_convergence


        self._random_rays = True
        self._random_method = random_method

        self._z_focuses = torch.zeros(self.learnable_tilts+1)

        self._convergence_angle = torch.ones(1) * convergence_angle
        self._stig_2 = torch.tensor(stig_2, dtype = torch.float32)


        self.ray_pattern = ray_pattern
        self.voxel_size_ang = voxel_size_ang
        
        self.save_probe = True
        # Initialize as parameters immediately (will be moved to device in .to())
        if self.learn_convergence:
            self._convergence_angle_params = nn.Parameter(self._convergence_angle.clone())
        else:
            self.register_buffer('_convergence_angle_params', self._convergence_angle.clone())



        if self.learn_astigmatism:
            self._stig_2_params = nn.Parameter(self._stig_2.clone())
        else:
            # Register as buffer (non-trainable but part of state_dict)
            self.register_buffer('_stig_2_params', self._stig_2.clone())


        if random_method.lower() in ['probe', 'p']:
            import numpy as np
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
                (torch.pi * wavelength_ang) * (kr**2 + kc**2), dtype=torch.float32)
            self._basis_1 = torch.tensor(
                (torch.pi * wavelength_ang) * (kr**2 - kc**2), dtype=torch.float32)
            self._basis_2 = torch.tensor(
                (torch.pi * wavelength_ang) * (2 * kr * kc),   dtype=torch.float32)
            self._aper = torch.tensor(aper, dtype=torch.float32)
            
            self._gumbel_temp = 10

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

    @classmethod
    def from_data(
        cls,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        convergence_angle: float = 10e-3,
        num_rays: int = 5,
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        learn_astigmatism: bool = False,
        learn_convergence: bool = False,
        wavelength_ang: float = 0.0197,
        stig_2: tuple[float, float] = (0.0, 0.0),
        probe_im_shape: tuple[int, int] = (64, 64),
        pixel_size_ang: float = 0.2,
        random_method: str = 'p',
        ray_pattern: str = 'clever',
        voxel_size_ang: float = 20.0,
    ):
        return cls(
            tilt_stack=tilt_stack,
            tilt_angles=tilt_angles,
            convergence_angle=convergence_angle,
            num_rays=num_rays,
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
            learn_astigmatism=learn_astigmatism,
            learn_convergence=learn_convergence,
            wavelength_ang=wavelength_ang,
            stig_2=stig_2,
            probe_im_shape=probe_im_shape,
            pixel_size_ang=pixel_size_ang,
            voxel_size_ang=voxel_size_ang,
            random_method=random_method,
            token=cls._token,
            ray_pattern=ray_pattern,
        )

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

        # convergence_angle = F.softplus(self._convergence_angle_params[0]) * 45e-3 + 5e-3


        if self.learn_astigmatism:
            stig_2 = self._stig_2_params
        else:
            stig_2 = self._stig_2_params.detach()  # No gradients

        if self.learn_convergence:
            convergence_angle = self._convergence_angle_params[0]
        else:
            convergence_angle = self._convergence_angle_params[0].detach()  # No gradients
        


        if self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, convergence_angle, stig_2
        elif self.learn_shift:
            return shifts, torch.zeros_like(z1), torch.zeros_like(z3), z_focus, convergence_angle, stig_2
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3, z_focus, convergence_angle, stig_2
        elif self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, convergence_angle, stig_2
        else:
            return torch.zeros_like(shifts), torch.zeros_like(z1), torch.zeros_like(z3), z_focus, convergence_angle, stig_2

    def get_coords(
        self,
        batch: dict[str, torch.Tensor],
        N: int,
        num_samples_per_ray: int,
        ray_pattern: None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            all_coords: (batch * num_samples * num_rays, 3)
            probe_weights: (batch, num_samples, num_rays)
        """
        if ray_pattern is None:
            ray_pattern = self.ray_pattern
        num_rays = self.num_rays
        pixel_i = batch["pixel_i"].float().to(self.device, non_blocking=True)
        pixel_j = batch["pixel_j"].float().to(self.device, non_blocking=True)
        phis = batch["phi"].to(self.device, non_blocking=True)
        projection_indices = batch["projection_idx"].to(self.device, non_blocking=True)
        
        shifts, z1_params, z3_params, z_focus_params, convergence_angle, stig_2 = self.forward(None)
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)
        batch_z_focus = torch.index_select(z_focus_params, 0, projection_indices)
        
        batch_ray_coords, probe_weights = self.create_batch_rays(
            pixel_i, pixel_j, N, num_samples_per_ray, num_rays,
            z_focus=batch_z_focus,
            stig_2=stig_2,
            convergence_angle=convergence_angle,
            ray_pattern=ray_pattern,
            voxel_size_ang=self.voxel_size_ang,
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
        
        return all_coords, probe_weights

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

                kr_s = row_s * self._dk
                kc_s = col_s * self._dk

                return kr_s * self._wavelength_ang, kc_s * self._wavelength_ang

            else:
                raise ValueError(
                    f"Unsupported random_method={random_method}. "
                    "Supported values are: gaussian, g, sinc, s. .lower() is applied internally."
                )

        else:
            raise ValueError('Only random rays supported for this dataset model right now')

        return theta, phi

    def _setup_clever_rays_pattern(
        self,
        num_rays: int,
        convergence_angle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the Probe class's optimized ray distribution."""
        
        # Extract scalar convergence angle
        if convergence_angle.dim() > 0:
            conv_angle_scalar = convergence_angle[0].item()
        else:
            conv_angle_scalar = convergence_angle.item()

        # Calculate number of rings
        num_rays_per_ring = 6
        
        if num_rays == 1:
            num_rings = 0
        else:
            a = num_rays_per_ring / 2
            b = num_rays_per_ring / 2
            c = 1 - num_rays
            
            discriminant = b**2 - 4 * a * c
            num_rings = int(np.ceil((-b + np.sqrt(discriminant)) / (2 * a)))
            num_rings = max(1, num_rings)

        wavelength = self._wavelength_ang
        k_max = conv_angle_scalar / wavelength

        # Create temporary Probe instance (uses NumPy, but only for setup)
        temp_probe = Probe(
            k_max=k_max,
            pixel_size=0.2,
            im_shape=(256, 256),
            wavelength=wavelength,
        )
        
        # Define the optimized ray pattern
        temp_probe.define_rays(
            num_rings=num_rings,
            num_rays=num_rays_per_ring,
            offset_mode='maximize_spacing',
            balance_center=True,
            balance_outer=True,
            weight_mode='nearest',
        )
        
        # Get the k-space positions (NumPy array)
        k_positions = temp_probe.k  # Shape: (actual_num_rays, 2)
        
        # Trim or pad to match requested num_rays
        actual_num_rays = k_positions.shape[0]
        if actual_num_rays > num_rays:
            k_positions = k_positions[:num_rays]
        elif actual_num_rays < num_rays:
            padding = np.zeros((num_rays - actual_num_rays, 2))
            k_positions = np.vstack([k_positions, padding])
        
        # Convert k-space positions to angles
        # For small angles: k ≈ alpha / wavelength, so alpha ≈ k * wavelength
        alpha_x = k_positions[:, 0] * wavelength
        alpha_y = k_positions[:, 1] * wavelength
        
        # Convert to torch tensors
        device = convergence_angle.device if torch.is_tensor(convergence_angle) else 'cpu'
        alpha_x = torch.tensor(alpha_x, dtype=torch.float32, device=device)
        alpha_y = torch.tensor(alpha_y, dtype=torch.float32, device=device)
        
        return alpha_x, alpha_y


    def _compute_probe_weights(
        self,
        dz_ang: torch.Tensor,  # (batch_size, num_samples_per_ray)
        alpha_x: torch.Tensor,  # (num_rays,)
        alpha_y: torch.Tensor,  # (num_rays,)
        stig_2: torch.Tensor,  # (2,) - astigmatism parameters
        convergence_angle: torch.Tensor,  # scalar or (batch_size,)
    ) -> torch.Tensor:
        """
        Compute probe intensity weights for each ray at each z-position.
        Uses wave optics with defocus aberration.
        
        Returns:
            weights: (batch_size, num_samples_per_ray, num_rays)
                    Normalized probe intensity at each sampling position
        """
        batch_size, num_samples = dz_ang.shape
        num_rays = len(alpha_x)
        device = dz_ang.device
        
        # Ensure convergence_angle is scalar
        if convergence_angle.dim() > 0:
            convergence_angle = convergence_angle[0]
        
        # Get k-space grid
        H, W = self._aper.shape
        
        # Move tensors to correct device if needed
        aper = self._aper.to(device)
        dk = self._dk.to(device)
        
        # Create k-space coordinates
        kr_1d = torch.fft.fftfreq(H, d=1.0, device=device) * dk * H
        kc_1d = torch.fft.fftfreq(W, d=1.0, device=device) * dk * W
        kr, kc = torch.meshgrid(kr_1d, kc_1d, indexing='ij')
        
        # Compute k magnitude
        k_mag = torch.sqrt(kr**2 + kc**2)
        
        # Aperture function (use pre-computed)
        aperture = aper
        
        # Defocus aberration: chi(k) = pi * lambda * defocus * k^2
        # Shape for broadcasting: (batch_size, num_samples, 1, 1)
        dz_4d = dz_ang[:, :, None, None]
        k2 = k_mag[None, None, :, :] ** 2
        chi = torch.pi * self._wavelength_ang * dz_4d * k2
        # print('dz ang max:', np.max(dz_ang.detach().cpu().numpy()))
        # Add astigmatism if needed (simplified - can be expanded)
        # A1 = pi * lambda * (kr^2 - kc^2) * stig_2[0] + pi * lambda * 2*kr*kc * stig_2[1]
        if torch.any(stig_2 != 0):
            basis_1 = self._basis_1.to(device)
            basis_2 = self._basis_2.to(device)
            chi = chi + basis_1[None, None, :, :] * stig_2[0] + basis_2[None, None, :, :] * stig_2[1]
        
        # Aberrated wave function in k-space
        # Shape: (batch_size, num_samples, H, W)
        # print('chi shape:',chi.shape)
        psi_k = aperture[None, None, :, :] * torch.exp(-1j * chi)
        
        # Transform to real space
        psi_r = torch.fft.ifft2(psi_k)
        intensity = torch.abs(psi_r) ** 2
        if self.save_probe == True:
            self.save_probe = False
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(np.fft.fftshift(intensity[10, 10, :,:].detach().cpu().numpy()))
            plt.savefig('probe_out.png')
        # Calculate real-space positions where rays hit at each z
        # Shape: (batch_size, num_samples, num_rays)
        dz_broadcast = dz_ang[:, :, None]
        r_x = alpha_x[None, None, :] * dz_broadcast
        r_y = alpha_y[None, None, :] * dz_broadcast
        
        # Determine real-space sampling
        dr = self._wavelength_ang / (dk * H)
        r_max = H * dr / 2
        
        # Normalize positions to [-1, 1] for grid_sample
        x_norm = r_y / r_max  # Note: grid_sample expects (H, W) = (y, x)
        y_norm = r_x / r_max
        
        # Clamp to valid range
        x_norm = torch.clamp(x_norm, -1.0, 1.0)
        y_norm = torch.clamp(y_norm, -1.0, 1.0)
        
        # Prepare grid for sampling: (batch_size * num_samples, num_rays, 1, 2)
        grid = torch.stack([x_norm, y_norm], dim=-1)
        grid = grid.reshape(batch_size * num_samples, num_rays, 1, 2)
        
        # Reshape intensity for grid_sample: (batch_size * num_samples, 1, H, W)
        intensity_flat = intensity.reshape(batch_size * num_samples, 1, H, W)
        
        # Sample probe intensity at ray positions
        # Output: (batch_size * num_samples, 1, num_rays, 1)
        sampled_intensity = F.grid_sample(
            intensity_flat,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        


        # Reshape to (batch_size, num_samples, num_rays)
        weights = sampled_intensity.reshape(batch_size, num_samples, num_rays)
        
        # Normalize weights along ray dimension so they sum to 1
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
        
        return weights



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
        voxel_size_ang: float = 2.0,
        ray_pattern: str = 'uniform_angle',  # 'uniform', 'hexagonal', 'grid'
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create rays with uniform/pattern sampling and return both rays and probe weights.
        
        Returns:
            rays: (batch_size, num_samples_per_ray * num_rays, 3)
            probe_weights: (batch_size, num_samples_per_ray, num_rays) - weights for each ray at each slice
        """
        batch_size = len(pixel_i)
        x_coords_0 = (pixel_j / (N - 1)) * 2 - 1
        y_coords_0 = (pixel_i / (N - 1)) * 2 - 1

        # Z coordinates
        z_coords_norm = torch.linspace(-1, 1, num_samples_per_ray, device=pixel_i.device)
        half_extent_ang = (N * voxel_size_ang) / 2.0
        
        z_ang = z_coords_norm[None, :] * half_extent_ang
        z_focus_ang = z_focus[:, None] * half_extent_ang
        dz_ang = z_ang - z_focus_ang  # (batch_size, num_samples_per_ray)
        if self.save_probe is True:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(dz_ang.detach().cpu().numpy())
            plt.plot(z_ang.detach().cpu().numpy())
            plt.plot(z_focus_ang.detach().cpu().numpy())
            plt.savefig('dz_ang.png')
            # print('z_coords_norm:', z_coords_norm)
            # print('half_extent_ang:', half_extent_ang)
            


        device = pixel_i.device

        # Generate ray pattern (uniform or structured)
        if ray_pattern == 'uniform_angle':
            # Uniform sampling in angle space (spherical coordinates)
            # This generates RAYS (same angle throughout z)
            
            # Azimuthal: uniform 0 to 2π
            azimuthal = torch.rand(num_rays, device=device) * 2 * torch.pi
            
            # Altitude: uniform in area (use sqrt)
            altitude = torch.sqrt(torch.rand(num_rays, device=device)) * convergence_angle * 1.2
            
            # Convert to Cartesian angles
            alpha_x = altitude * torch.cos(azimuthal)
            alpha_y = altitude * torch.sin(azimuthal)

        elif ray_pattern == 'uniform_xy':
            # Uniform sampling in x-y plane (Cartesian)
            # This also generates RAYS (same angle throughout z)
            
            # Random angle and radius
            angles = torch.rand(num_rays, device=device) * 2 * torch.pi
            radii = torch.sqrt(torch.rand(num_rays, device=device))
            
            # Convert to angular coordinates
            alpha_x = radii * torch.cos(angles) * convergence_angle
            alpha_y = radii * torch.sin(angles) * convergence_angle

        elif ray_pattern == 'gaussian_angle':
            # Gaussian sampling in angle space
            # Standard deviation = convergence_angle / 2 (so ~95% within aperture)
            sigma = convergence_angle / 2.0
            
            # Sample 2D Gaussian
            alpha_x = torch.randn(num_rays, device=device) * sigma
            alpha_y = torch.randn(num_rays, device=device) * sigma


        elif ray_pattern == 'grid':
            # Regular grid pattern
            grid_size = int(torch.ceil(torch.sqrt(torch.tensor(num_rays, dtype=torch.float32))))
            x_grid = torch.linspace(-convergence_angle, convergence_angle, grid_size, device=device)
            y_grid = torch.linspace(-convergence_angle, convergence_angle, grid_size, device=device)
            grid_x, grid_y = torch.meshgrid(x_grid, y_grid, indexing='ij')
            alpha_x = grid_x.reshape(-1)[:num_rays]
            alpha_y = grid_y.reshape(-1)[:num_rays]
        

        elif ray_pattern == 'hexagonal':
            # Hexagonal pattern (only for 7 rays: 1 center + 6 around)
            if num_rays != 7:
                raise ValueError(f"Hexagonal pattern only supports 7 rays, got {num_rays}")
            
            angles_deg = torch.tensor([0, 60, 120, 180, 240, 300], device=device, dtype=torch.float32)
            radius = convergence_angle * 0.7
            
            alpha_x = torch.cat([
                torch.zeros(1, device=device),
                radius * torch.cos(torch.deg2rad(angles_deg))
            ])
            alpha_y = torch.cat([
                torch.zeros(1, device=device),
                radius * torch.sin(torch.deg2rad(angles_deg))
            ])
        
        elif ray_pattern == 'clever':
            # Use optimized distribution from tomo_rays.py
            alpha_x, alpha_y = self._setup_clever_rays_pattern(num_rays, convergence_angle)
        
        else:
            raise ValueError(f"Unknown ray_pattern: {ray_pattern}")
        
        # Compute probe weights for each defocus slice
        # Shape: (batch_size, num_samples_per_ray, num_rays)
        probe_weights = self._compute_probe_weights(
            dz_ang, alpha_x, alpha_y, stig_2, convergence_angle
        )
        
        # Convert angles to spatial offsets in normalized coordinates
        # alpha (radians) -> physical distance at defocus -> normalized coords
        # For each z-slice, offset = alpha * |dz| / half_extent
        
        # Broadcast: dz_ang is (batch, num_samples), alpha is (num_rays,)
        # Result: (batch, num_samples, num_rays)
        dz_broadcast = dz_ang[:, :, None]  # (batch, num_samples, 1)
        alpha_x_broadcast = alpha_x[None, None, :]  # (1, 1, num_rays)
        alpha_y_broadcast = alpha_y[None, None, :]
        
        # Offset in normalized space
        x_offsets_norm = alpha_x_broadcast * torch.abs(dz_broadcast) / half_extent_ang
        y_offsets_norm = alpha_y_broadcast * torch.abs(dz_broadcast) / half_extent_ang
        
        # Add to pixel center
        x_coords = x_coords_0[:, None, None] + x_offsets_norm  # (batch, num_samples, num_rays)
        y_coords = y_coords_0[:, None, None] + y_offsets_norm
        z_coords = z_coords_norm[None, :, None].expand(batch_size, num_samples_per_ray, num_rays)
        
        # Flatten rays: (batch, num_samples * num_rays, 3)
        x_coords = x_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        y_coords = y_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        z_coords = z_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        
        rays = torch.stack([x_coords, y_coords, z_coords], dim=2)
        
        return rays, probe_weights


    @torch.compile(mode="reduce-overhead")
    def integrate_rays_with_probe_weights(
        self,
        rays: torch.Tensor,  # (batch, num_samples * num_rays, 3) -> densities after INR
        probe_weights: torch.Tensor,  # (batch, num_samples, num_rays)
        num_samples_per_ray: int,
        target_values_len: int,
    ) -> torch.Tensor:
        """
        Integrate rays with probe weighting.
        
        For each pixel:
        1. At each z-slice, we have num_rays with their densities
        2. Weight each ray by its probe intensity at that slice
        3. Sum weighted rays at each slice
        4. Integrate over slices
        
        Args:
            rays: INR density values, shape (batch, num_samples * num_rays)
            probe_weights: (batch, num_samples, num_rays)
        """
        num_rays = self.num_rays
        
        # Reshape densities: (batch, num_samples, num_rays)
        ray_densities = rays.view(target_values_len, num_samples_per_ray, num_rays)
        
        # Apply probe weights (element-wise multiplication)
        weighted_densities = ray_densities * probe_weights  # (batch, num_samples, num_rays)
        
        # Sum over rays at each slice
        slice_values = weighted_densities.sum(dim=2)  # (batch, num_samples)
        
        # Integrate over z
        step_size = 2.0 / (num_samples_per_ray - 1)
        predicted_values = slice_values.sum(dim=1) * step_size  # (batch,)
        
        return predicted_values

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
        if self.learn_astigmatism:
            self._stig_2_params = nn.Parameter(self._stig_2_params.to(device))
        else:
            # Re-register buffer on new device
            self.register_buffer('_stig_2_params', self._stig_2_params.to(device))


        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)

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
            self._k1 = self._k1.to(device)

        self.device = device
        self.reconnect_optimizer_to_parameters()









class TomographyThroughFocalJustStigINRDataset(TomographyINRDataset):
    """
    Dataset class for INR-based tomography that assumes non-parallel illumination condition and uses a (more) mathematically accurate probe for ray weighting.
    Inherits from TomographyINRDataset.
    """

    STIG_SCALE = 50.0 

    def __init__(
        self,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        ## Through focal parameters:
        convergence_angle: float,
        num_rays: int = 1,
        ## optional parameters
        learn_shift: bool = False,
        learn_tilt_axis: bool = False,
        learn_astigmatism: bool = True,
        learn_convergence: bool = False,
        learn_defocus: bool = False,
        token: object | None = None,
        random_method: str = 'p',
        wavelength_ang: float = 0.0197,
        stig_2: tuple[float, float] = (0.0, 0.0),
        probe_im_shape: tuple[int, int] = (64, 64),
        pixel_size_ang: float = 0.2,
        voxel_size_ang: float = 20.0,
        ray_pattern: str = 'clever',
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, _token = token)
        self.num_rays = int(num_rays)
        self.learn_astigmatism = learn_astigmatism
        self.learn_convergence = learn_convergence
        self.learn_defocus = learn_defocus

        self.debug = False

        self._random_rays = True
        self._random_method = random_method

        self._z_focus = torch.zeros(self.learnable_tilts+1) 

        self._convergence_angle = torch.ones(1) * convergence_angle
        self._stig_2 = torch.tensor(stig_2, dtype = torch.float32) / self.STIG_SCALE


        self.ray_pattern = ray_pattern
        self.voxel_size_ang = voxel_size_ang
        
        self.save_probe = True
        # Initialize as parameters immediately (will be moved to device in .to())
        if self.learn_convergence:
            self._convergence_angle_params = nn.Parameter(self._convergence_angle.clone())
        else:
            self.register_buffer('_convergence_angle_params', self._convergence_angle.clone())


        self._clever_k_positions = None
        self._sunflower_positions = None
        self._hexagonal_positions = None
        self._fibonacci_positions = None
    
        
        if self.learn_defocus:
            self._z_focus_params = nn.Parameter(self._z_focus.clone())
        else:
            # Register as buffer (non-trainable but part of state_dict)
            self.register_buffer('_z_focus_params', self._z_focus.clone())

        if self.learn_astigmatism:
            self._stig_2_params = nn.Parameter(self._stig_2.clone())
        else:
            # Register as buffer (non-trainable but part of state_dict)
            self.register_buffer('_stig_2_params', self._stig_2.clone())

        if ray_pattern == 'clever':
            # Pre-compute k-positions ONCE
            k_positions_np = self._precompute_clever_k_positions(
                num_rays=num_rays,
                convergence_angle=convergence_angle,
                wavelength_ang=wavelength_ang,
            )
            # Store as torch tensor (will be moved to device in .to())
            self._clever_k_positions = torch.from_numpy(k_positions_np).float()

        elif ray_pattern == 'sunflower':
            # Sunflower spiral - excellent uniform coverage
            positions_np = self._precompute_sunflower_pattern(
                num_rays=num_rays,
                probe_pixel_count=probe_im_shape[0],
                pixel_size_ang=pixel_size_ang,
            )
            self._sunflower_positions = torch.from_numpy(positions_np).float()
        
        elif ray_pattern == 'hexagonal_grid':
            # Hexagonal close packing - mathematically optimal
            positions_np = self._precompute_hexagonal_grid(
                num_rays=num_rays,
                probe_pixel_count=probe_im_shape[0],
                pixel_size_ang=pixel_size_ang,
            )
            self._hexagonal_positions = torch.from_numpy(positions_np).float()
        
        elif ray_pattern == 'fibonacci':
            # Fibonacci sphere projection
            positions_np = self._precompute_fibonacci_sphere(
                num_rays=num_rays,
                probe_pixel_count=probe_im_shape[0],
                pixel_size_ang=pixel_size_ang,
            )
            self._fibonacci_positions = torch.from_numpy(positions_np).float()




        self._probe_pixel_size_ang = pixel_size_ang
        self._probe_pixel_count = probe_im_shape[0]

        if random_method.lower() in ['probe', 'p']:
            import numpy as np
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
                (torch.pi * wavelength_ang) * (kr**2 + kc**2), dtype=torch.float32)
            self._basis_1 = torch.tensor(
                (torch.pi * wavelength_ang) * (kr**2 - kc**2), dtype=torch.float32)
            self._basis_2 = torch.tensor(
                (torch.pi * wavelength_ang) * (2 * kr * kc),   dtype=torch.float32)
            self._aper = torch.tensor(aper, dtype=torch.float32)
            
            self._gumbel_temp = 10

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

    @classmethod
    def from_data(
        cls,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        convergence_angle: float = 10e-3,
        num_rays: int = 5,
        learn_shift: bool = False,
        learn_tilt_axis: bool = False,
        learn_astigmatism: bool = True,
        learn_convergence: bool = False,
        learn_defocus: bool = False,
        wavelength_ang: float = 0.0197,
        stig_2: tuple[float, float] = (0.0, 0.0),
        probe_im_shape: tuple[int, int] = (64, 64),
        pixel_size_ang: float = 0.2,
        random_method: str = 'p',
        ray_pattern: str = 'clever',
        voxel_size_ang: float = 20.0,
    ):
        return cls(
            tilt_stack=tilt_stack,
            tilt_angles=tilt_angles,
            convergence_angle=convergence_angle,
            num_rays=num_rays,
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
            learn_astigmatism=learn_astigmatism,
            learn_convergence=learn_convergence,
            learn_defocus=learn_defocus,
            wavelength_ang=wavelength_ang,
            stig_2=stig_2,
            probe_im_shape=probe_im_shape,
            pixel_size_ang=pixel_size_ang,
            voxel_size_ang=voxel_size_ang,
            random_method=random_method,
            token=cls._token,
            ray_pattern=ray_pattern,
        )


    def _precompute_sunflower_pattern(self, num_rays, probe_pixel_count, pixel_size_ang):
        """
        Generate optimal sunflower spiral pattern for parallel rays.
        This gives near-optimal packing with uniform density.
        
        Based on Vogel's method using golden angle.
        """
        import numpy as np
        
        # Maximum radius (probe aperture)
        r_max_ang = (probe_pixel_count / 2 - 1) * pixel_size_ang
        
        # Golden angle in radians
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~2.399963 radians
        
        # Generate positions
        indices = np.arange(num_rays)
        
        # Radius grows with sqrt(index) for uniform area density
        radii = np.sqrt(indices / (num_rays - 1 + 1e-8)) * r_max_ang
        
        # Angle increases by golden angle
        theta = indices * golden_angle
        
        # Convert to Cartesian
        r_x = radii * np.cos(theta)
        r_y = radii * np.sin(theta)
        
        positions = np.stack([r_x, r_y], axis=1)  # (num_rays, 2)
        
        return positions


    def _precompute_hexagonal_grid(self, num_rays, probe_pixel_count, pixel_size_ang):
        """
        Generate hexagonal close-packed grid for parallel rays.
        This is mathematically optimal for circle packing.
        """
        import numpy as np
        
        # Maximum radius
        r_max_ang = (probe_pixel_count / 2 - 1) * pixel_size_ang
        
        # Estimate grid size needed
        # Hexagonal packing has area efficiency of π/(2√3) ≈ 0.9069
        est_rings = int(np.ceil(np.sqrt(num_rays / 0.9069) / 2))
        
        positions = []
        
        # Center point
        positions.append([0.0, 0.0])
        
        # Generate hexagonal rings
        for ring in range(1, est_rings + 5):  # Extra rings to ensure enough points
            # Spacing for this ring
            radius = ring * r_max_ang / (est_rings + 1)
            
            # Six-fold symmetry
            num_points_in_ring = 6 * ring
            
            for i in range(num_points_in_ring):
                angle = 2 * np.pi * i / num_points_in_ring
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                # Check if within aperture
                if np.sqrt(x**2 + y**2) <= r_max_ang:
                    positions.append([x, y])
            
            if len(positions) >= num_rays:
                break
        
        positions = np.array(positions[:num_rays])  # Trim to exact number
        
        return positions


    def _precompute_fibonacci_sphere(self, num_rays, probe_pixel_count, pixel_size_ang):
        """
        Fibonacci sphere projection onto disk.
        Another excellent uniform distribution method.
        """
        import numpy as np
        
        r_max_ang = (probe_pixel_count / 2 - 1) * pixel_size_ang
        
        positions = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        for i in range(num_rays):
            # Normalized radius (0 to 1)
            r_norm = np.sqrt(i / (num_rays - 1 + 1e-8))
            
            # Fibonacci angle
            theta = 2 * np.pi * i / phi
            
            # Convert to Cartesian
            r = r_norm * r_max_ang
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            positions.append([x, y])
        
        return np.array(positions)


    def _precompute_clever_k_positions(self, num_rays, convergence_angle, wavelength_ang):
        """Compute k-space ray positions once (slow, but only happens once)."""
        import numpy as np  # Only needed here
        
        k_max = convergence_angle / wavelength_ang
        
        # Create Probe and define rays (SLOW - but only once!)
        temp_probe = Probe(
            k_max=k_max,
            pixel_size=0.2,
            im_shape=(256, 256),
            wavelength=wavelength_ang,
        )
        
        # Calculate num_rings
        num_rays_per_ring = 6
        if num_rays == 1:
            num_rings = 0
        else:
            a = num_rays_per_ring / 2
            b = num_rays_per_ring / 2
            c = 1 - num_rays
            discriminant = b**2 - 4 * a * c
            num_rings = int(np.ceil((-b + np.sqrt(discriminant)) / (2 * a)))
            num_rings = max(1, num_rings)
        
        temp_probe.define_rays(
            num_rings=num_rings,
            num_rays=num_rays_per_ring,
            offset_mode='maximize_spacing',
            balance_center=True,
            balance_outer=True,
            weight_mode='nearest',
        )
        
        # Extract k-positions (numpy)
        k_positions = temp_probe.k  # Shape: (n_rays, 2)
        
        # Trim/pad to exact num_rays
        actual_num_rays = k_positions.shape[0]
        if actual_num_rays > num_rays:
            k_positions = k_positions[:num_rays]
        elif actual_num_rays < num_rays:
            padding = np.zeros((num_rays - actual_num_rays, 2))
            k_positions = np.vstack([k_positions, padding])
        
        return k_positions  # Return as numpy (will be converted in __init__)


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


        # convergence_angle = F.softplus(self._convergence_angle_params[0]) * 45e-3 + 5e-3


        if self.learn_astigmatism:
            # Scale back to physical units (Angstroms)
            stig_2 = self._stig_2_params * self.STIG_SCALE
        else:
            stig_2 = (self._stig_2_params * self.STIG_SCALE).detach()

        if self.learn_convergence:
            convergence_angle = self._convergence_angle_params[0]
        else:
            convergence_angle = self._convergence_angle_params[0].detach()  # No gradients


        if self.learn_defocus:
            z_focus = self.z_focus_params
        else:
            z_focus = self.z_focus_params.detach()  # No gradients
        


        if self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, convergence_angle, stig_2
        elif self.learn_shift:
            return shifts, torch.zeros_like(z1), torch.zeros_like(z3), z_focus, convergence_angle, stig_2
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3, z_focus, convergence_angle, stig_2
        elif self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, convergence_angle, stig_2
        else:
            return torch.zeros_like(shifts), torch.zeros_like(z1), torch.zeros_like(z3), z_focus, convergence_angle, stig_2

    def get_coords(
        self,
        batch: dict[str, torch.Tensor],
        N: int,
        num_samples_per_ray: int,
        ray_pattern: None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            all_coords: (batch * num_samples * num_rays, 3)
            probe_weights: (batch, num_samples, num_rays)
        """
        if ray_pattern is None:
            ray_pattern = self.ray_pattern
        num_rays = self.num_rays
        pixel_i = batch["pixel_i"].float().to(self.device, non_blocking=True)
        pixel_j = batch["pixel_j"].float().to(self.device, non_blocking=True)
        phis = batch["phi"].to(self.device, non_blocking=True)
        projection_indices = batch["projection_idx"].to(self.device, non_blocking=True)
        
        shifts, z1_params, z3_params, z_focus_params, convergence_angle, stig_2 = self.forward(None)
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)
        batch_z_focus = torch.index_select(z_focus_params, 0, projection_indices)

        if self.debug:
            print(f"DEBUG get_coords: stig_2.requires_grad = {stig_2.requires_grad}")

        batch_ray_coords, probe_weights = self.create_batch_rays(
            pixel_i, pixel_j, N, num_samples_per_ray, num_rays,
            z_focus=batch_z_focus,
            stig_2=stig_2,
            convergence_angle=convergence_angle,
            ray_pattern=ray_pattern,
            voxel_size_ang=self.voxel_size_ang,
        )

        if self.debug:
            print(f"DEBUG get_coords: probe_weights.requires_grad = {probe_weights.requires_grad}")

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
        
        return all_coords, probe_weights

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

                kr_s = row_s * self._dk
                kc_s = col_s * self._dk

                return kr_s * self._wavelength_ang, kc_s * self._wavelength_ang

            else:
                raise ValueError(
                    f"Unsupported random_method={random_method}. "
                    "Supported values are: gaussian, g, sinc, s. .lower() is applied internally."
                )

        else:
            raise ValueError('Only random rays supported for this dataset model right now')

        return theta, phi


    def _setup_clever_rays_pattern(
        self,
        num_rays: int,
        convergence_angle: torch.Tensor,
        defocus: torch.Tensor = None,  # NEW: optional
        stig_2: torch.Tensor = None,   # NEW: optional
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fast! Just applies linear transform to pre-computed k-positions.
        """
        device = convergence_angle.device
        
        # Get pre-computed k-positions
        k = torch.from_numpy(self._clever_k_positions).to(device, dtype=torch.float32)
        k_x = k[:, 0]
        k_y = k[:, 1]
        
        # If defocus/stig provided, compute real-space positions and convert back to angles
        if defocus is not None and stig_2 is not None:
            # Apply the affine transform (same as Probe._compute_r)
            C1 = defocus
            A1_x = stig_2[0]
            A1_y = stig_2[1]
            
            # Real-space positions
            r_x = (k_x * (C1 + A1_x) + k_y * A1_y) * self._wavelength_ang
            r_y = (k_y * (C1 - A1_x) + k_x * A1_y) * self._wavelength_ang
            
            # Convert back to effective angles at this defocus
            # alpha = r / |defocus|
            defocus_abs = torch.abs(defocus) + 1e-8  # Avoid div by zero
            alpha_x = r_x / defocus_abs
            alpha_y = r_y / defocus_abs
        else:
            # No aberrations - just convert k to angles
            alpha_x = k_x * self._wavelength_ang
            alpha_y = k_y * self._wavelength_ang
        
        return alpha_x, alpha_y



    def _compute_probe_weights_a(
        self,
        dz_ang: torch.Tensor,  # (batch_size, num_samples_per_ray)
        alpha_x: torch.Tensor,  # (num_rays,)
        alpha_y: torch.Tensor,  # (num_rays,)
        stig_2: torch.Tensor,  # (2,) - astigmatism parameters
        convergence_angle: torch.Tensor,  # scalar or (batch_size,)
    ) -> torch.Tensor:
        """
        Compute probe intensity weights for each ray at each z-position.
        Uses wave optics with defocus aberration.
        
        Returns:
            weights: (batch_size, num_samples_per_ray, num_rays)
                    Normalized probe intensity at each sampling position
        """
        batch_size, num_samples = dz_ang.shape
        num_rays = len(alpha_x)
        device = dz_ang.device
        
        if self.debug:
            print(f"DEBUG _compute_probe_weights: stig_2.requires_grad = {stig_2.requires_grad}")
            print(f"DEBUG _compute_probe_weights: stig_2.data = {stig_2.data}")
        # Ensure convergence_angle is scalar
        if convergence_angle.dim() > 0:
            convergence_angle = convergence_angle[0]
        
        # Get k-space grid
        H, W = self._aper.shape
        
        # Move tensors to correct device if needed
        aper = self._aper.to(device)
        dk = self._dk.to(device)
        
        # Create k-space coordinates
        kr_1d = torch.fft.fftfreq(H, d=1.0, device=device) * dk * H
        kc_1d = torch.fft.fftfreq(W, d=1.0, device=device) * dk * W
        kr, kc = torch.meshgrid(kr_1d, kc_1d, indexing='ij')
        
        # Compute k magnitude
        k_mag = torch.sqrt(kr**2 + kc**2)
        
        # Aperture function (use pre-computed)
        aperture = aper
        
        # Defocus aberration: chi(k) = pi * lambda * defocus * k^2
        # Shape for broadcasting: (batch_size, num_samples, 1, 1)
        dz_4d = dz_ang[:, :, None, None]
        k2 = k_mag[None, None, :, :] ** 2
        chi = torch.pi * self._wavelength_ang * dz_4d * k2
        # print('dz ang max:', np.max(dz_ang.detach().cpu().numpy()))
        # Add astigmatism if needed (simplified - can be expanded)
        # A1 = pi * lambda * (kr^2 - kc^2) * stig_2[0] + pi * lambda * 2*kr*kc * stig_2[1]
    
        basis_1 = self._basis_1.to(device)
        basis_2 = self._basis_2.to(device)
        chi = chi + basis_1[None, None, :, :] * stig_2[0] + basis_2[None, None, :, :] * stig_2[1]
    
        # Aberrated wave function in k-space
        # Shape: (batch_size, num_samples, H, W)
        # print('chi shape:',chi.shape)
        psi_k = aperture[None, None, :, :] * torch.exp(-1j * chi)
        
        # Transform to real space
        psi_r = torch.fft.ifft2(psi_k)
        intensity = torch.abs(psi_r) ** 2
        if self.save_probe == True:
            self.save_probe = False
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(np.fft.fftshift(intensity[10, 10, :,:].detach().cpu().numpy()))
            plt.savefig('probe_out.png')
        # Calculate real-space positions where rays hit at each z
        # Shape: (batch_size, num_samples, num_rays)
        dz_broadcast = dz_ang[:, :, None]
        r_x = alpha_x[None, None, :] * dz_broadcast
        r_y = alpha_y[None, None, :] * dz_broadcast
        
        # Determine real-space sampling
        dr = self._wavelength_ang / (dk * H)
        r_max = H * dr / 2
        
        # Normalize positions to [-1, 1] for grid_sample
        x_norm = r_y / r_max  # Note: grid_sample expects (H, W) = (y, x)
        y_norm = r_x / r_max
        
        # Clamp to valid range
        x_norm = torch.clamp(x_norm, -1.0, 1.0)
        y_norm = torch.clamp(y_norm, -1.0, 1.0)
        
        # Prepare grid for sampling: (batch_size * num_samples, num_rays, 1, 2)
        grid = torch.stack([x_norm, y_norm], dim=-1)
        grid = grid.reshape(batch_size * num_samples, num_rays, 1, 2)
        
        # Reshape intensity for grid_sample: (batch_size * num_samples, 1, H, W)
        intensity_flat = intensity.reshape(batch_size * num_samples, 1, H, W)
        
        # Sample probe intensity at ray positions
        # Output: (batch_size * num_samples, 1, num_rays, 1)
        sampled_intensity = F.grid_sample(
            intensity_flat,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        


        # Reshape to (batch_size, num_samples, num_rays)
        weights = sampled_intensity.reshape(batch_size, num_samples, num_rays)
        
        # Normalize weights along ray dimension so they sum to 1
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)

        if self.debug:
            if self.training and weights.requires_grad:
                # Good - weights can backprop
                pass
            else:
                print(f"WARNING: weights.requires_grad = {weights.requires_grad}")

        return weights

    def _compute_probe_weights_r(
        self,
        dz_ang: torch.Tensor,  # (batch_size, num_samples_per_ray)
        r_x: torch.Tensor,  # (batch_size, num_samples_per_ray, num_rays) - real-space positions
        r_y: torch.Tensor,  # (batch_size, num_samples_per_ray, num_rays) - real-space positions
        stig_2: torch.Tensor,  # (2,)
        convergence_angle: torch.Tensor,  # scalar
    ) -> torch.Tensor:
        """
        Compute probe intensity weights for each ray at each z-position.
        
        Args:
            r_x, r_y: Real-space positions where rays hit the probe (Angstroms)
        
        Returns:
            weights: (batch_size, num_samples_per_ray, num_rays)
        """
        batch_size, num_samples, num_rays = r_x.shape
        device = r_x.device
        
        if self.debug:
            print(f"DEBUG _compute_probe_weights: stig_2.requires_grad = {stig_2.requires_grad}")
            print(f"DEBUG _compute_probe_weights: r_x.shape = {r_x.shape}")
        
        # Ensure convergence_angle is scalar
        if convergence_angle.dim() > 0:
            convergence_angle = convergence_angle[0]
        
        # Get k-space grid
        H, W = self._aper.shape
        
        # Move tensors to correct device if needed
        aper = self._aper.to(device)
        dk = self._dk.to(device)
        
        # Create k-space coordinates
        kr_1d = torch.fft.fftfreq(H, d=1.0, device=device) * dk * H
        kc_1d = torch.fft.fftfreq(W, d=1.0, device=device) * dk * W
        kr, kc = torch.meshgrid(kr_1d, kc_1d, indexing='ij')
        
        k_mag = torch.sqrt(kr**2 + kc**2)
        aperture = aper
        
        # Defocus aberration
        dz_4d = dz_ang[:, :, None, None]
        k2 = k_mag[None, None, :, :] ** 2
        chi = torch.pi * self._wavelength_ang * dz_4d * k2
        
        # Add astigmatism
        basis_1 = self._basis_1.to(device)
        basis_2 = self._basis_2.to(device)
        chi = chi + basis_1[None, None, :, :] * stig_2[0] + basis_2[None, None, :, :] * stig_2[1]
        
        # Aberrated wave function
        psi_k = aperture[None, None, :, :] * torch.exp(-1j * chi)
        psi_r = torch.fft.ifft2(psi_k)
        intensity = torch.abs(psi_r) ** 2
        
        if self.save_probe:
            self.save_probe = False
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(np.fft.fftshift(intensity[10, 10, :, :].detach().cpu().numpy()))
            plt.savefig('probe_out.png')
        
        # Use the provided r_x, r_y positions directly
        # r_x, r_y are already (batch, num_samples, num_rays) in Angstroms
        
        # Determine real-space sampling
        dr = self._probe_pixel_size_ang
        r_max = H * dr / 2
        
        # Normalize positions to [-1, 1] for grid_sample
        x_norm = r_y / r_max  # Note: grid_sample expects (H, W) = (y, x)
        y_norm = r_x / r_max
        
        # Clamp to valid range
        x_norm = torch.clamp(x_norm, -1.0, 1.0)
        y_norm = torch.clamp(y_norm, -1.0, 1.0)
        
        # Prepare grid for sampling
        grid = torch.stack([x_norm, y_norm], dim=-1)
        grid = grid.reshape(batch_size * num_samples, num_rays, 1, 2)
        
        # Reshape intensity
        intensity_flat = intensity.reshape(batch_size * num_samples, 1, H, W)
        
        # Sample probe intensity at ray positions
        sampled_intensity = F.grid_sample(
            intensity_flat,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        # Reshape to (batch_size, num_samples, num_rays)
        weights = sampled_intensity.reshape(batch_size, num_samples, num_rays)
        
        # Normalize weights
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
        
        if self.debug:
            if self.training and weights.requires_grad:
                pass
            else:
                print(f"WARNING: weights.requires_grad = {weights.requires_grad}")
        
        return weights


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
        voxel_size_ang: float = 2.0,
        ray_pattern: str = 'uniform_angle',  # 'uniform', 'hexagonal', 'grid'
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create rays with uniform/pattern sampling and return both rays and probe weights.
        
        Returns:
            rays: (batch_size, num_samples_per_ray * num_rays, 3)
            probe_weights: (batch_size, num_samples_per_ray, num_rays) - weights for each ray at each slice
        """
        batch_size = len(pixel_i)
        x_coords_0 = (pixel_j / (N - 1)) * 2 - 1
        y_coords_0 = (pixel_i / (N - 1)) * 2 - 1

        if self.debug:
            print(f"DEBUG create_batch_rays: stig_2.requires_grad = {stig_2.requires_grad}")

        # Z coordinates
        z_coords_norm = torch.linspace(-1, 1, num_samples_per_ray, device=pixel_i.device)
        half_extent_ang = (N * voxel_size_ang) / 2.0
        
        z_ang = z_coords_norm[None, :] * half_extent_ang
        z_focus_ang = z_focus[:, None] * half_extent_ang
        dz_ang = z_ang - z_focus_ang  # (batch_size, num_samples_per_ray)
        if self.save_probe is True:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(dz_ang.detach().cpu().numpy())
            plt.plot(z_ang.detach().cpu().numpy())
            plt.plot(z_focus_ang.detach().cpu().numpy())
            plt.savefig('dz_ang.png')
            # print('z_coords_norm:', z_coords_norm)
            # print('half_extent_ang:', half_extent_ang)
            


        device = pixel_i.device

        # Generate ray pattern (uniform or structured)
        if ray_pattern == 'uniform_angle':
            # Uniform sampling in angle space (spherical coordinates)
            # This generates RAYS (same angle throughout z)
            
            # Azimuthal: uniform 0 to 2π
            azimuthal = torch.rand(num_rays, device=device) * 2 * torch.pi
            
            # Altitude: uniform in area (use sqrt)
            altitude = torch.sqrt(torch.rand(num_rays, device=device)) * convergence_angle * 2
            
            # Convert to Cartesian angles
            alpha_x = altitude * torch.cos(azimuthal)
            alpha_y = altitude * torch.sin(azimuthal)

        elif ray_pattern == 'uniform_xy':
            # Uniform sampling in x-y plane (Cartesian)
            # This also generates RAYS (same angle throughout z)
            
            # Random angle and radius
            angles = torch.rand(num_rays, device=device) * 2 * torch.pi
            radii = torch.sqrt(torch.rand(num_rays, device=device))
            
            # Convert to angular coordinates
            alpha_x = radii * torch.cos(angles) * convergence_angle
            alpha_y = radii * torch.sin(angles) * convergence_angle


        elif ray_pattern == 'uniform_parallel':
            # Uniform sampling of PARALLEL rays in real space
            # Distributes rays uniformly across the probe aperture
            
            # Maximum radius in Angstroms (probe aperture)
            r_max_ang = (self._probe_pixel_count / 2 - 1) * self._probe_pixel_size_ang
            
            # Random azimuthal angle (uniform 0 to 2π)
            azimuthal = torch.rand(num_rays, device=device) * 2 * torch.pi
            
            # Random radius (uniform in area, so sqrt for uniform spatial distribution)
            radii_ang = torch.sqrt(torch.rand(num_rays, device=device)) * r_max_ang
            
            # Convert to Cartesian real-space positions (in Angstroms)
            r_x_base = radii_ang * torch.cos(azimuthal)  # (num_rays,)
            r_y_base = radii_ang * torch.sin(azimuthal)  # (num_rays,)
            
            # For parallel beams, r is INDEPENDENT of z
            # Expand to all batch samples and z-positions
            r_x = r_x_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)
            r_y = r_y_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)


        elif ray_pattern == 'gaussian_angle':
            # Gaussian sampling in angle space
            # Standard deviation = convergence_angle / 2 (so ~95% within aperture)
            sigma = convergence_angle / 2.0
            
            # Sample 2D Gaussian
            alpha_x = torch.randn(num_rays, device=device) * sigma
            alpha_y = torch.randn(num_rays, device=device) * sigma


        # elif ray_pattern == 'grid':
        #     # Regular grid pattern
        #     grid_size = int(torch.ceil(torch.sqrt(torch.tensor(num_rays, dtype=torch.float32))))
        #     x_grid = torch.linspace(-convergence_angle, convergence_angle, grid_size, device=device)
        #     y_grid = torch.linspace(-convergence_angle, convergence_angle, grid_size, device=device)
        #     grid_x, grid_y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        #     alpha_x = grid_x.reshape(-1)[:num_rays]
        #     alpha_y = grid_y.reshape(-1)[:num_rays]
        
        elif ray_pattern == 'hexagonal':
            # Hexagonal pattern (only for 7 rays: 1 center + 6 around)
            if num_rays != 7:
                raise ValueError(f"Hexagonal pattern only supports 7 rays, got {num_rays}")
            
            angles_deg = torch.tensor([0, 60, 120, 180, 240, 300], device=device, dtype=torch.float32)
            radius = convergence_angle * 0.7
            
            alpha_x = torch.cat([
                torch.zeros(1, device=device),
                radius * torch.cos(torch.deg2rad(angles_deg))
            ])
            alpha_y = torch.cat([
                torch.zeros(1, device=device),
                radius * torch.sin(torch.deg2rad(angles_deg))
            ])
        elif ray_pattern == 'clever':
            # Get fixed k-space positions (already on device from .to())
            k = self._clever_k_positions  # Already torch, already on device
            k_x = k[:, 0]  # (num_rays,)
            k_y = k[:, 1]  # (num_rays,)
            
            # Compute real-space positions using parametric form
            C1 = dz_ang[:, :, None]  # (batch, num_samples, 1) - defocus at each z
            A1_x = stig_2[0]  # scalar
            A1_y = stig_2[1]  # scalar
            
            k_x_broadcast = k_x[None, None, :]  # (1, 1, num_rays)
            k_y_broadcast = k_y[None, None, :]  # (1, 1, num_rays)
            
            # Parametric form: compute real-space positions
            r_x = (k_x_broadcast * (C1 + A1_x) + k_y_broadcast * A1_y) * self._wavelength_ang
            r_y = (k_y_broadcast * (C1 - A1_x) + k_x_broadcast * A1_y) * self._wavelength_ang
            # r_x, r_y are (batch, num_samples, num_rays) in Angstroms

        elif ray_pattern == 'grid':
            # Define a real-space grid (in Angstroms) - PARALLEL beams
            grid_size = int(torch.ceil(torch.sqrt(torch.tensor(num_rays, dtype=torch.float32))))
            grid_spacing_ang = self._probe_pixel_count / grid_size * self._probe_pixel_size_ang * 0.75
            grid_extent = grid_spacing_ang * (grid_size - 1) / 2
            
            # Create grid in real space (Angstroms)
            x_positions = torch.linspace(-grid_extent, grid_extent, grid_size, device=device)
            y_positions = torch.linspace(-grid_extent, grid_extent, grid_size, device=device)
            grid_x, grid_y = torch.meshgrid(x_positions, y_positions, indexing='ij')
            
            # Flatten and take first num_rays
            r_x_base = grid_x.reshape(-1)[:num_rays]  # (num_rays,) in Angstroms
            r_y_base = grid_y.reshape(-1)[:num_rays]
            
            # For parallel beams, r is INDEPENDENT of z
            r_x = r_x_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)
            r_y = r_y_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)


        elif ray_pattern == 'sunflower':
            # Pre-computed sunflower pattern
            positions = self._sunflower_positions.to(device)
            r_x_base = positions[:, 0]  # (num_rays,)
            r_y_base = positions[:, 1]  # (num_rays,)
            
            # Parallel beams - constant with z
            r_x = r_x_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)
            r_y = r_y_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)

        elif ray_pattern == 'hexagonal_grid':
            # Pre-computed hexagonal grid
            positions = self._hexagonal_positions.to(device)
            r_x_base = positions[:, 0]
            r_y_base = positions[:, 1]
            
            r_x = r_x_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)
            r_y = r_y_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)

        elif ray_pattern == 'fibonacci':
            # Pre-computed Fibonacci pattern
            positions = self._fibonacci_positions.to(device)
            r_x_base = positions[:, 0]
            r_y_base = positions[:, 1]
            
            r_x = r_x_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)
            r_y = r_y_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)

        else:
            raise ValueError(f"Unknown ray_pattern: {ray_pattern}")
        
        # Compute probe weights for each defocus slice
        # Shape: (batch_size, num_samples_per_ray, num_rays)

        if ray_pattern == 'uniform_angle' or ray_pattern == 'uniform_xy' or ray_pattern == 'gaussian_angle' or ray_pattern == 'hexagonal':
            probe_weights = self._compute_probe_weights_a(
                dz_ang, alpha_x, alpha_y, stig_2, convergence_angle
            )
            # Broadcast: dz_ang is (batch, num_samples), alpha is (num_rays,)
            # Result: (batch, num_samples, num_rays)
            dz_broadcast = dz_ang[:, :, None]  # (batch, num_samples, 1)
            alpha_x_broadcast = alpha_x[None, None, :]  # (1, 1, num_rays)
            alpha_y_broadcast = alpha_y[None, None, :]
            
            # Offset in normalized space
            x_offsets_norm = alpha_x_broadcast * torch.abs(dz_broadcast) / half_extent_ang
            y_offsets_norm = alpha_y_broadcast * torch.abs(dz_broadcast) / half_extent_ang
        
        elif ray_pattern == 'clever' or ray_pattern == 'grid' or ray_pattern == 'uniform_parallel' or ray_pattern == 'fibonacci' or ray_pattern == 'hexagonal_grid' or ray_pattern == 'sunflower':
            x_offsets_norm = r_x / half_extent_ang
            y_offsets_norm = r_y / half_extent_ang
            probe_weights = self._compute_probe_weights_r(
                dz_ang, r_x, r_y, stig_2, convergence_angle
            )
        else:
            raise ValueError(f"Unknown ray_pattern: {ray_pattern}")
        
        # Add to pixel center
        x_coords = x_coords_0[:, None, None] + x_offsets_norm  # (batch, num_samples, num_rays)
        y_coords = y_coords_0[:, None, None] + y_offsets_norm
        z_coords = z_coords_norm[None, :, None].expand(batch_size, num_samples_per_ray, num_rays)
        
        # Flatten rays: (batch, num_samples * num_rays, 3)
        x_coords = x_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        y_coords = y_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        z_coords = z_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        
        rays = torch.stack([x_coords, y_coords, z_coords], dim=2)

        if self.debug:
            print(f"DEBUG create_batch_rays: probe_weights.requires_grad = {probe_weights.requires_grad}")


        return rays, probe_weights


    @torch.compile(mode="reduce-overhead")
    def integrate_rays_with_probe_weights(
        self,
        rays: torch.Tensor,  # (batch, num_samples * num_rays, 3) -> densities after INR
        probe_weights: torch.Tensor,  # (batch, num_samples, num_rays)
        num_samples_per_ray: int,
        target_values_len: int,
    ) -> torch.Tensor:
        """
        Integrate rays with probe weighting.
        
        For each pixel:
        1. At each z-slice, we have num_rays with their densities
        2. Weight each ray by its probe intensity at that slice
        3. Sum weighted rays at each slice
        4. Integrate over slices
        
        Args:
            rays: INR density values, shape (batch, num_samples * num_rays)
            probe_weights: (batch, num_samples, num_rays)
        """
        num_rays = self.num_rays
        
        # Reshape densities: (batch, num_samples, num_rays)
        ray_densities = rays.view(target_values_len, num_samples_per_ray, num_rays)
        
        # Apply probe weights (element-wise multiplication)
        weighted_densities = ray_densities * probe_weights  # (batch, num_samples, num_rays)
        
        # Sum over rays at each slice
        slice_values = weighted_densities.sum(dim=2)  # (batch, num_samples)
        
        # Integrate over z
        step_size = 2.0 / (num_samples_per_ray - 1)
        predicted_values = slice_values.sum(dim=1) * step_size  # (batch,)
        
        return predicted_values

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

        if self._clever_k_positions is not None:
            self._clever_k_positions = self._clever_k_positions.to(device)

        if self._sunflower_positions is not None:
            self._sunflower_positions = self._sunflower_positions.to(device)
        
        if self._hexagonal_positions is not None:
            self._hexagonal_positions = self._hexagonal_positions.to(device)
        
        if self._fibonacci_positions is not None:
            self._fibonacci_positions = self._fibonacci_positions.to(device)


        if self.learn_convergence:
            self._convergence_angle_params = nn.Parameter(self._convergence_angle_params.to(device))
        else:
            self.register_buffer('_convergence_angle_params', self._convergence_angle_params.to(device))
        
        if self.learn_defocus:
            self._z_focus_params = nn.Parameter(self._z_focus_params.to(device))
        else:
            self.register_buffer('_z_focus_params', self._z_focus_params.to(device))
        
        if self.learn_astigmatism:
            self._stig_2_params = nn.Parameter(self._stig_2_params.to(device))
        else:
            self.register_buffer('_stig_2_params', self._stig_2_params.to(device))

        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)

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
            self._k1 = self._k1.to(device)

        self.device = device
        self.reconnect_optimizer_to_parameters()













import numpy as np
class Probe():
    """Probe model with matched wavefunction and discrete ray pictures."""

    def __init__(
        self,
        k_max=1.0,
        pixel_size=0.2,
        im_shape=np.array([256, 256]),
        wavelength=0.0197,
        device='cpu',
    ):
        self.k_max = k_max
        self.pixel_size = pixel_size
        self.im_shape = im_shape
        self.wavelength = wavelength
        self.device = device

        # Use NumPy for initialization (non-differentiable setup)
        kr_np = np.fft.fftfreq(im_shape[0], pixel_size)[:, None]
        kc_np = np.fft.fftfreq(im_shape[1], pixel_size)[None, :]
        k2_np = kr_np**2 + kc_np**2
        k1_np = np.sqrt(k2_np)
        dk_np = kr_np[1, 0] - kr_np[0, 0]

        aper_np = np.clip((k_max - k1_np) / dk_np + 0.5, 0.0, 1.0)

        # Store as numpy for ray generation (happens once, not in training loop)
        self.kr = kr_np
        self.kc = kc_np
        self.dk = dk_np
        self.aper = aper_np
        
        # Store basis functions for potential future use
        self.basis_C1 = np.pi * wavelength * k2_np
        self.basis_A1x = np.pi * wavelength * (kr_np**2 - kc_np**2)
        self.basis_A1y = np.pi * wavelength * (2 * kr_np * kc_np)

    def get_center_balanced_k_radius_offset(self, num_rings=5, num_rays=6):
        total_rays = 1 + num_rays * num_rings * (num_rings + 1) / 2
        target_first_ring = self.k_max * 2 * np.sqrt(
            np.pi / (total_rays * num_rays * np.tan(np.pi / num_rays))
        )
        base_first_ring = self.k_max * 2 / (2 * num_rings + 1)
        return target_first_ring - base_first_ring

    def get_balanced_k_radii(self, num_rings=5, num_rays=6):
        base_radii = self.k_max * 2 * (np.arange(1, num_rings + 1) / (2 * num_rings + 1))

        if num_rings < 2:
            return base_radii + self.get_center_balanced_k_radius_offset(
                num_rings=num_rings, num_rays=num_rays
            )

        total_rays = 1 + num_rays * num_rings * (num_rings + 1) / 2
        first_radius = self.k_max * 2 * np.sqrt(
            np.pi / (total_rays * num_rays * np.tan(np.pi / num_rays))
        )
        outer_boundary_midpoint = self.k_max * np.sqrt(
            1.0 - (num_rings * num_rays) / total_rays
        )
        slope = (outer_boundary_midpoint - first_radius) / (num_rings - 1.5)
        intercept = first_radius - slope
        return slope * np.arange(1, num_rings + 1) + intercept

    def define_rays(
        self,
        num_rings=4,
        num_rays=6,
        offset_mode='maximize_spacing',
        offset_samples=720,
        balance_center=True,
        balance_outer=True,
        weight_mode='nearest',
        fractional_power=2.0,
    ):
        k = [[0, 0]]
        self.k_ring_offsets = []
        
        if balance_center and balance_outer:
            k_radii = self.get_balanced_k_radii(num_rings=num_rings, num_rays=num_rays)
        elif balance_center:
            radius_offset = self.get_center_balanced_k_radius_offset(
                num_rings=num_rings, num_rays=num_rays
            )
            k_radii = self.k_max * 2 * (
                np.arange(1, num_rings + 1) / (2 * num_rings + 1)
            ) + radius_offset
        else:
            k_radii = self.k_max * 2 * (np.arange(1, num_rings + 1) / (2 * num_rings + 1))

        for a0 in range(num_rings):
            k_radius = k_radii[a0]
            num_ring_rays = (a0 + 1) * num_rays
            phi = 2 * np.pi / num_ring_rays

            if offset_mode == 'alternate':
                dphi = phi * np.mod(a0, 2) / 2
            elif offset_mode == 'golden':
                golden = (np.sqrt(5.0) - 1.0) / 2.0
                dphi = phi * np.mod((a0 + 1) * golden, 1.0)
            elif offset_mode == 'maximize_spacing':
                dphi = self._compute_ring_offset(
                    k_radius=k_radius,
                    num_ring_rays=num_ring_rays,
                    existing_k=np.array(k, dtype=float),
                    offset_samples=offset_samples,
                )
            else:
                raise ValueError(
                    "offset_mode must be 'alternate', 'golden', or 'maximize_spacing'"
                )

            self.k_ring_offsets.append(dphi)

            for a1 in range(num_ring_rays):
                k.append([
                    k_radius * np.cos(phi * a1 + dphi),
                    k_radius * np.sin(phi * a1 + dphi),
                ])
        
        self.k = np.array(k)
        self.k_px = self.k / self.dk + np.array(self.im_shape) // 2
        self.k_weights_raw, self.k_weights = self._compute_k_weights(
            weight_mode=weight_mode, fractional_power=fractional_power
        )

    def _compute_k_weights(self, weight_mode='nearest', fractional_power=2.0):
        weight_mode = weight_mode.lower()
        if weight_mode not in ('nearest', 'fractional'):
            raise ValueError("weight_mode must be 'nearest' or 'fractional'")
        if fractional_power <= 0:
            raise ValueError("fractional_power must be positive")

        grid_shape = tuple(self.im_shape)
        kr_grid = np.broadcast_to(self.kr, grid_shape)
        kc_grid = np.broadcast_to(self.kc, grid_shape)

        k_pixels = np.stack((kr_grid.ravel(), kc_grid.ravel()), axis=1)
        aper_weights = self.aper.ravel()
        valid = aper_weights > 0

        k_pixels = k_pixels[valid]
        aper_weights = aper_weights[valid]

        diff = k_pixels[:, None, :] - self.k[None, :, :]
        dist2 = np.sum(diff**2, axis=2)

        if weight_mode == 'nearest':
            nearest = np.argmin(dist2, axis=1)
            k_weights_raw = np.bincount(
                nearest, weights=aper_weights, minlength=self.k.shape[0]
            ).astype(float)
        else:
            share = np.zeros_like(dist2, dtype=float)
            zero_mask = dist2 <= np.finfo(float).eps
            zero_rows = np.any(zero_mask, axis=1)

            if np.any(zero_rows):
                zero_share = zero_mask[zero_rows].astype(float)
                zero_share /= np.sum(zero_share, axis=1, keepdims=True)
                share[zero_rows] = zero_share

            if np.any(~zero_rows):
                inv_dist = dist2[~zero_rows] ** (-0.5 * fractional_power)
                inv_dist /= np.sum(inv_dist, axis=1, keepdims=True)
                share[~zero_rows] = inv_dist

            k_weights_raw = np.sum(share * aper_weights[:, None], axis=0)

        k_weights = k_weights_raw / np.sum(k_weights_raw)
        return k_weights_raw, k_weights

    def _compute_ring_offset(self, k_radius, num_ring_rays, existing_k, offset_samples=720):
        if offset_samples < 1:
            raise ValueError("offset_samples must be at least 1")
        if existing_k.shape[0] <= 1:
            return 0.0

        phi = 2 * np.pi / num_ring_rays
        candidate_offsets = np.linspace(0.0, phi, offset_samples, endpoint=False)

        best_offset = 0.0
        best_min_dist2 = -np.inf
        best_mean_nearest_dist2 = -np.inf

        for dphi in candidate_offsets:
            angles = phi * np.arange(num_ring_rays) + dphi
            ring_k = np.column_stack((
                k_radius * np.cos(angles),
                k_radius * np.sin(angles),
            ))

            diff = ring_k[:, None, :] - existing_k[None, :, :]
            dist2 = np.sum(diff**2, axis=2)
            nearest_dist2 = np.min(dist2, axis=1)

            min_dist2 = np.min(nearest_dist2)
            mean_nearest_dist2 = np.mean(nearest_dist2)

            better_min = min_dist2 > best_min_dist2
            equal_min = np.isclose(min_dist2, best_min_dist2)
            better_mean = mean_nearest_dist2 > best_mean_nearest_dist2
            
            if better_min or (equal_min and better_mean):
                best_offset = dphi
                best_min_dist2 = min_dist2
                best_mean_nearest_dist2 = mean_nearest_dist2

        return best_offset






















class TomographyThroughFocalINRDataset_0615(TomographyINRDataset):
    """
    Dataset class for INR-based tomography that assumes non-parallel illumination condition and uses a (more) mathematically accurate probe for ray weighting.
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
        learn_shift: bool = False,
        learn_tilt_axis: bool = False,
        learn_astigmatism: bool = True,
        learn_convergence: bool = False,
        learn_defocus: bool = False,
        token: object | None = None,
        random_method: str = 'p',
        wavelength_ang: float = 0.0197,
        stig_2: tuple[float, float] = (0.0, 0.0),
        probe_im_shape: tuple[int, int] = (64, 64),
        pixel_size_ang: float = 0.2,
        voxel_size_ang: float = 20.0,
        ray_pattern: str = 'clever',
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, _token = token)
        self.num_rays = int(num_rays)
        self.learn_astigmatism = learn_astigmatism
        self.learn_convergence = learn_convergence
        self.learn_defocus = learn_defocus

        self.debug = False

        self._random_rays = True
        self._random_method = random_method
        self.compute_probe_weights = False

        self._z_focus = torch.zeros(self.learnable_tilts+1) 

        self._convergence_angle = torch.ones(1) * convergence_angle
        self._stig_2 = torch.tensor(stig_2, dtype = torch.float32)


        self.ray_pattern = ray_pattern
        self.voxel_size_ang = voxel_size_ang
        
        self.save_probe = True
        # Initialize as parameters immediately (will be moved to device in .to())
        if self.learn_convergence:
            self._convergence_angle_params = nn.Parameter(self._convergence_angle.clone())
        else:
            self.register_buffer('_convergence_angle_params', self._convergence_angle.clone())

        self._clever_k_positions = None

        if self.learn_defocus:
            self._z_focus_params = nn.Parameter(self._z_focus.clone())
        else:
            # Register as buffer (non-trainable but part of state_dict)
            self.register_buffer('_z_focus_params', self._z_focus.clone())

        if self.learn_astigmatism:
            self._stig_2_params = nn.Parameter(self._stig_2.clone())
        else:
            # Register as buffer (non-trainable but part of state_dict)
            self.register_buffer('_stig_2_params', self._stig_2.clone())

        if ray_pattern == 'clever':
            # Pre-compute k-positions ONCE
            k_positions_np = self._precompute_clever_k_positions(
                num_rays=num_rays,
                convergence_angle=convergence_angle,
                wavelength_ang=wavelength_ang,
            )
            # Store as torch tensor (will be moved to device in .to())
            self._clever_k_positions = torch.from_numpy(k_positions_np).float()

        self._probe_pixel_size_ang = pixel_size_ang
        self._probe_pixel_count = probe_im_shape[0]

        if random_method.lower() in ['probe', 'p']:
            import numpy as np
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
                (torch.pi * wavelength_ang) * (kr**2 + kc**2), dtype=torch.float32)
            self._basis_1 = torch.tensor(
                (torch.pi * wavelength_ang) * (kr**2 - kc**2), dtype=torch.float32)
            self._basis_2 = torch.tensor(
                (torch.pi * wavelength_ang) * (2 * kr * kc),   dtype=torch.float32)
            self._aper = torch.tensor(aper, dtype=torch.float32)

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

    @classmethod
    def from_data(
        cls,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        convergence_angle: float = 10e-3,
        num_rays: int = 5,
        learn_shift: bool = False,
        learn_tilt_axis: bool = False,
        learn_astigmatism: bool = True,
        learn_convergence: bool = False,
        learn_defocus: bool = False,
        wavelength_ang: float = 0.0197,
        stig_2: tuple[float, float] = (0.0, 0.0),
        probe_im_shape: tuple[int, int] = (64, 64),
        pixel_size_ang: float = 0.2,
        random_method: str = 'p',
        ray_pattern: str = 'clever',
        voxel_size_ang: float = 20.0,
    ):
        return cls(
            tilt_stack=tilt_stack,
            tilt_angles=tilt_angles,
            convergence_angle=convergence_angle,
            num_rays=num_rays,
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
            learn_astigmatism=learn_astigmatism,
            learn_convergence=learn_convergence,
            learn_defocus=learn_defocus,
            wavelength_ang=wavelength_ang,
            stig_2=stig_2,
            probe_im_shape=probe_im_shape,
            pixel_size_ang=pixel_size_ang,
            voxel_size_ang=voxel_size_ang,
            random_method=random_method,
            token=cls._token,
            ray_pattern=ray_pattern,
        )

    def _precompute_clever_k_positions(self, num_rays, convergence_angle, wavelength_ang):
        """Compute k-space ray positions once (slow, but only happens once)."""
        import numpy as np  # Only needed here
        
        k_max = convergence_angle / wavelength_ang
        
        # Create Probe and define rays (SLOW - but only once!)
        temp_probe = Probe(
            k_max=k_max,
            pixel_size=0.2,
            im_shape=(256, 256),
            wavelength=wavelength_ang,
        )
        
        # Calculate num_rings
        num_rays_per_ring = 6
        if num_rays == 1:
            num_rings = 0
        else:
            a = num_rays_per_ring / 2
            b = num_rays_per_ring / 2
            c = 1 - num_rays
            discriminant = b**2 - 4 * a * c
            num_rings = int(np.ceil((-b + np.sqrt(discriminant)) / (2 * a)))
            num_rings = max(1, num_rings)
        
        temp_probe.define_rays(
            num_rings=num_rings,
            num_rays=num_rays_per_ring,
            offset_mode='maximize_spacing',
            balance_center=True,
            balance_outer=True,
            weight_mode='nearest',
        )
        
        # Extract k-positions (numpy)
        k_positions = temp_probe.k  # Shape: (n_rays, 2)
        
        # Trim/pad to exact num_rays
        actual_num_rays = k_positions.shape[0]
        if actual_num_rays > num_rays:
            k_positions = k_positions[:num_rays]
        elif actual_num_rays < num_rays:
            padding = np.zeros((num_rays - actual_num_rays, 2))
            k_positions = np.vstack([k_positions, padding])
        
        return k_positions  # Return as numpy (will be converted in __init__)


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

        if self.learn_astigmatism:
            stig_2 = self._stig_2_params
        else:
            stig_2 = self._stig_2_params.detach()

        if self.learn_convergence:
            convergence_angle = self._convergence_angle_params[0]
        else:
            convergence_angle = self._convergence_angle_params[0].detach()  # No gradients


        if self.learn_defocus:
            z_focus = self.z_focus_params
        else:
            z_focus = self.z_focus_params.detach()  # No gradients
        


        if self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, convergence_angle, stig_2
        elif self.learn_shift:
            return shifts, torch.zeros_like(z1), torch.zeros_like(z3), z_focus, convergence_angle, stig_2
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3, z_focus, convergence_angle, stig_2
        elif self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, convergence_angle, stig_2
        else:
            return torch.zeros_like(shifts), torch.zeros_like(z1), torch.zeros_like(z3), z_focus, convergence_angle, stig_2

    def get_coords(
        self,
        batch: dict[str, torch.Tensor],
        N: int,
        num_samples_per_ray: int,
        ray_pattern: None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            all_coords: (batch * num_samples * num_rays, 3)
            probe_weights: (batch, num_samples, num_rays)
        """
        if ray_pattern is None:
            ray_pattern = self.ray_pattern
        num_rays = self.num_rays
        pixel_i = batch["pixel_i"].float().to(self.device, non_blocking=True)
        pixel_j = batch["pixel_j"].float().to(self.device, non_blocking=True)
        phis = batch["phi"].to(self.device, non_blocking=True)
        projection_indices = batch["projection_idx"].to(self.device, non_blocking=True)
        
        shifts, z1_params, z3_params, z_focus_params, convergence_angle, stig_2 = self.forward(None)
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)
        batch_z_focus = torch.index_select(z_focus_params, 0, projection_indices)

        if self.debug:
            print(f"DEBUG get_coords: stig_2.requires_grad = {stig_2.requires_grad}")

        batch_ray_coords, probe_weights = self.create_batch_rays(
            pixel_i, pixel_j, N, num_samples_per_ray, num_rays,
            z_focus=batch_z_focus,
            stig_2=stig_2,
            convergence_angle=convergence_angle,
            ray_pattern=ray_pattern,
            voxel_size_ang=self.voxel_size_ang,
        )

        if self.debug:
            print(f"DEBUG get_coords: probe_weights.requires_grad = {probe_weights.requires_grad}")

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
        
        return all_coords, probe_weights

    def _compute_probe_weights_r(
        self,
        dz_ang: torch.Tensor,  # (batch_size, num_samples_per_ray)
        r_x: torch.Tensor,  # (batch_size, num_samples_per_ray, num_rays) - real-space positions
        r_y: torch.Tensor,  # (batch_size, num_samples_per_ray, num_rays) - real-space positions
        stig_2: torch.Tensor,  # (2,)
        convergence_angle: torch.Tensor,  # scalar
    ) -> torch.Tensor:

        batch_size, num_samples, num_rays = r_x.shape
        device = r_x.device
        
        if self.debug:
            print(f"DEBUG _compute_probe_weights: stig_2.requires_grad = {stig_2.requires_grad}")
            print(f"DEBUG _compute_probe_weights: r_x.shape = {r_x.shape}")
        
        # Ensure convergence_angle is scalar
        if convergence_angle.dim() > 0:
            convergence_angle = convergence_angle[0]
        
        # Get k-space grid
        H, W = self._aper.shape
        
        # Move tensors to correct device if needed
        aper = self._aper.to(device)
        dk = self._dk.to(device)
        
        # Create k-space coordinates
        kr_1d = torch.fft.fftfreq(H, d=1.0, device=device) * dk * H
        kc_1d = torch.fft.fftfreq(W, d=1.0, device=device) * dk * W
        kr, kc = torch.meshgrid(kr_1d, kc_1d, indexing='ij')
        
        k_mag = torch.sqrt(kr**2 + kc**2)
        aperture = aper
        
        # Defocus aberration
        dz_4d = dz_ang[:, :, None, None]
        k2 = k_mag[None, None, :, :] ** 2
        chi = torch.pi * self._wavelength_ang * dz_4d * k2
        
        # Add astigmatism
        basis_1 = self._basis_1.to(device)
        basis_2 = self._basis_2.to(device)
        chi = chi + basis_1[None, None, :, :] * stig_2[0] + basis_2[None, None, :, :] * stig_2[1]
        
        # Aberrated wave function
        psi_k = aperture[None, None, :, :] * torch.exp(-1j * chi)
        psi_r = torch.fft.ifft2(psi_k)
        intensity = torch.abs(psi_r) ** 2
        
        # if self.save_probe:
            # self.save_probe = False
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(np.fft.fftshift(intensity[10, 10, :, :].detach().cpu().numpy()))
            # plt.savefig('probe_out.png')
        
        # Use the provided r_x, r_y positions directly
        # r_x, r_y are already (batch, num_samples, num_rays) in Angstroms
        
        # Determine real-space sampling
        dr = self._probe_pixel_size_ang
        r_max = H * dr / 2
        
        # Normalize positions to [-1, 1] for grid_sample
        x_norm = r_y / r_max  # Note: grid_sample expects (H, W) = (y, x)
        y_norm = r_x / r_max
        
        # Clamp to valid range
        x_norm = torch.clamp(x_norm, -1.0, 1.0)
        y_norm = torch.clamp(y_norm, -1.0, 1.0)
        
        # Prepare grid for sampling
        grid = torch.stack([x_norm, y_norm], dim=-1)
        grid = grid.reshape(batch_size * num_samples, num_rays, 1, 2)
        
        # Reshape intensity
        intensity_flat = intensity.reshape(batch_size * num_samples, 1, H, W)
        
        # Sample probe intensity at ray positions
        sampled_intensity = F.grid_sample(
            intensity_flat,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        # Reshape to (batch_size, num_samples, num_rays)
        weights = sampled_intensity.reshape(batch_size, num_samples, num_rays)
        
        # Normalize weights
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
        
        if self.debug:
            if self.training and weights.requires_grad:
                pass
            else:
                print(f"WARNING: weights.requires_grad = {weights.requires_grad}")
        
        return weights

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
        voxel_size_ang: float = 2.0,
        ray_pattern: str = 'uniform_angle',  # 'uniform', 'hexagonal', 'grid'
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Create rays with uniform/pattern sampling and return both rays and probe weights.
        
        Returns:
            rays: (batch_size, num_samples_per_ray * num_rays, 3)
            probe_weights: (batch_size, num_samples_per_ray, num_rays) - weights for each ray at each slice
        """
        batch_size = len(pixel_i)
        x_coords_0 = (pixel_j / (N - 1)) * 2 - 1
        y_coords_0 = (pixel_i / (N - 1)) * 2 - 1

        if self.debug:
            print(f"DEBUG create_batch_rays: stig_2.requires_grad = {stig_2.requires_grad}")

        # Z coordinates
        z_coords_norm = torch.linspace(-1, 1, num_samples_per_ray, device=pixel_i.device)
        half_extent_ang = (N * voxel_size_ang) / 2.0
        
        z_ang = z_coords_norm[None, :] * half_extent_ang
        z_focus_ang = z_focus[:, None] * half_extent_ang
        dz_ang = z_ang - z_focus_ang  # (batch_size, num_samples_per_ray)
        # if self.save_probe is True:
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     plt.plot(dz_ang.detach().cpu().numpy())
        #     plt.plot(z_ang.detach().cpu().numpy())
        #     plt.plot(z_focus_ang.detach().cpu().numpy())
        #     plt.savefig('dz_ang.png')
            # print('z_coords_norm:', z_coords_norm)
            # print('half_extent_ang:', half_extent_ang)

        device = pixel_i.device

        if ray_pattern == 'uniform_parallel':
            # Uniform sampling of PARALLEL rays in real space
            # Distributes rays uniformly across the probe aperture
            
            # Maximum radius in Angstroms (probe aperture)
            r_max_ang = (self._probe_pixel_count / 2 - 1) * self._probe_pixel_size_ang
            
            # Random azimuthal angle (uniform 0 to 2π)
            azimuthal = torch.rand(num_rays, device=device) * 2 * torch.pi
            
            # Random radius (uniform in area, so sqrt for uniform spatial distribution)
            radii_ang = torch.sqrt(torch.rand(num_rays, device=device)) * r_max_ang
            
            # Convert to Cartesian real-space positions (in Angstroms)
            r_x_base = radii_ang * torch.cos(azimuthal)  # (num_rays,)
            r_y_base = radii_ang * torch.sin(azimuthal)  # (num_rays,)
            
            # For parallel beams, r is INDEPENDENT of z
            # Expand to all batch samples and z-positions
            r_x = r_x_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)
            r_y = r_y_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)


        elif ray_pattern == 'gaussian_angle':
            # Gaussian sampling in angle space
            # Standard deviation = convergence_angle / 2 (so ~95% within aperture)
            sigma = convergence_angle / 2.0
            
            # Sample 2D Gaussian
            alpha_x = torch.randn(num_rays, device=device) * sigma
            alpha_y = torch.randn(num_rays, device=device) * sigma


        # elif ray_pattern == 'grid':
        #     # Regular grid pattern
        #     grid_size = int(torch.ceil(torch.sqrt(torch.tensor(num_rays, dtype=torch.float32))))
        #     x_grid = torch.linspace(-convergence_angle, convergence_angle, grid_size, device=device)
        #     y_grid = torch.linspace(-convergence_angle, convergence_angle, grid_size, device=device)
        #     grid_x, grid_y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        #     alpha_x = grid_x.reshape(-1)[:num_rays]
        #     alpha_y = grid_y.reshape(-1)[:num_rays]
        
        elif ray_pattern == 'clever':
            # Get fixed k-space positions (already on device from .to())
            k = self._clever_k_positions  # Already torch, already on device
            k_x = k[:, 0]  # (num_rays,)
            k_y = k[:, 1]  # (num_rays,)
            
            # Compute real-space positions using parametric form
            C1 = dz_ang[:, :, None]  # (batch, num_samples, 1) - defocus at each z
            A1_x = stig_2[0]  # scalar
            A1_y = stig_2[1]  # scalar
            
            k_x_broadcast = k_x[None, None, :]  # (1, 1, num_rays)
            k_y_broadcast = k_y[None, None, :]  # (1, 1, num_rays)
            
            # Parametric form: compute real-space positions
            r_x = (k_x_broadcast * (C1 + A1_x) + k_y_broadcast * A1_y) * self._wavelength_ang
            r_y = (k_y_broadcast * (C1 - A1_x) + k_x_broadcast * A1_y) * self._wavelength_ang
            # r_x, r_y are (batch, num_samples, num_rays) in Angstroms

        elif ray_pattern == 'grid':
            # Define a real-space grid (in Angstroms) - PARALLEL beams
            grid_size = int(torch.ceil(torch.sqrt(torch.tensor(num_rays, dtype=torch.float32))))
            grid_spacing_ang = self._probe_pixel_count / grid_size * self._probe_pixel_size_ang * 0.75
            grid_extent = grid_spacing_ang * (grid_size - 1) / 2
            
            # Create grid in real space (Angstroms)
            x_positions = torch.linspace(-grid_extent, grid_extent, grid_size, device=device)
            y_positions = torch.linspace(-grid_extent, grid_extent, grid_size, device=device)
            grid_x, grid_y = torch.meshgrid(x_positions, y_positions, indexing='ij')
            
            # Flatten and take first num_rays
            r_x_base = grid_x.reshape(-1)[:num_rays]  # (num_rays,) in Angstroms
            r_y_base = grid_y.reshape(-1)[:num_rays]
            
            # For parallel beams, r is INDEPENDENT of z
            r_x = r_x_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)
            r_y = r_y_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)

        else:
            raise ValueError(f"Unknown ray_pattern: {ray_pattern}")
        
        # Compute probe weights for each defocus slice
        # Shape: (batch_size, num_samples_per_ray, num_rays)

        if self.compute_probe_weights:
            if ray_pattern == 'uniform_angle' or ray_pattern == 'uniform_xy' or ray_pattern == 'gaussian_angle' or ray_pattern == 'hexagonal':
                probe_weights = self._compute_probe_weights_a(
                    dz_ang, alpha_x, alpha_y, stig_2, convergence_angle
                )
                # Broadcast: dz_ang is (batch, num_samples), alpha is (num_rays,)
                # Result: (batch, num_samples, num_rays)
                dz_broadcast = dz_ang[:, :, None]  # (batch, num_samples, 1)
                alpha_x_broadcast = alpha_x[None, None, :]  # (1, 1, num_rays)
                alpha_y_broadcast = alpha_y[None, None, :]
                
                # Offset in normalized space
                x_offsets_norm = alpha_x_broadcast * torch.abs(dz_broadcast) / half_extent_ang
                y_offsets_norm = alpha_y_broadcast * torch.abs(dz_broadcast) / half_extent_ang
            
            elif ray_pattern == 'clever' or ray_pattern == 'grid' or ray_pattern == 'uniform_parallel' or ray_pattern == 'fibonacci' or ray_pattern == 'hexagonal_grid' or ray_pattern == 'sunflower':
                x_offsets_norm = r_x / half_extent_ang
                y_offsets_norm = r_y / half_extent_ang
                probe_weights = self._compute_probe_weights_r(
                    dz_ang, r_x, r_y, stig_2, convergence_angle
                )
            else:
                raise ValueError(f"Unknown ray_pattern: {ray_pattern}")
        
        else:
            if ray_pattern == 'uniform_angle' or ray_pattern == 'uniform_xy' or ray_pattern == 'gaussian_angle' or ray_pattern == 'hexagonal':
                dz_broadcast = dz_ang[:, :, None]  # (batch, num_samples, 1)
                alpha_x_broadcast = alpha_x[None, None, :]  # (1, 1, num_rays)
                alpha_y_broadcast = alpha_y[None, None, :]
                x_offsets_norm = alpha_x_broadcast * torch.abs(dz_broadcast) / half_extent_ang
                y_offsets_norm = alpha_y_broadcast * torch.abs(dz_broadcast) / half_extent_ang
            
            elif ray_pattern == 'clever' or ray_pattern == 'grid' or ray_pattern == 'uniform_parallel' or ray_pattern == 'fibonacci' or ray_pattern == 'hexagonal_grid' or ray_pattern == 'sunflower':
                x_offsets_norm = r_x / half_extent_ang
                y_offsets_norm = r_y / half_extent_ang
            else:
                raise ValueError(f"Unknown ray_pattern: {ray_pattern}")
        
        # Add to pixel center
        x_coords = x_coords_0[:, None, None] + x_offsets_norm  # (batch, num_samples, num_rays)
        y_coords = y_coords_0[:, None, None] + y_offsets_norm
        z_coords = z_coords_norm[None, :, None].expand(batch_size, num_samples_per_ray, num_rays)
        
        # Flatten rays: (batch, num_samples * num_rays, 3)
        x_coords = x_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        y_coords = y_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        z_coords = z_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        
        rays = torch.stack([x_coords, y_coords, z_coords], dim=2)

        if self.debug:
            print(f"DEBUG create_batch_rays: probe_weights.requires_grad = {probe_weights.requires_grad}")


        if self.compute_probe_weights:
            return rays, probe_weights
        else:
            return rays, None


    @torch.compile(mode="reduce-overhead")
    def integrate_rays_with_probe_weights(
        self,
        rays: torch.Tensor,  # (batch, num_samples * num_rays, 3) -> densities after INR
        probe_weights: torch.Tensor,  # (batch, num_samples, num_rays)
        num_samples_per_ray: int,
        target_values_len: int,
    ) -> torch.Tensor:
        """
        Integrate rays with probe weighting.
        
        For each pixel:
        1. At each z-slice, we have num_rays with their densities
        2. Weight each ray by its probe intensity at that slice
        3. Sum weighted rays at each slice
        4. Integrate over slices
        
        Args:
            rays: INR density values, shape (batch, num_samples * num_rays)
            probe_weights: (batch, num_samples, num_rays)
        """
        num_rays = self.num_rays
        
        # Reshape densities: (batch, num_samples, num_rays)
        ray_densities = rays.view(target_values_len, num_samples_per_ray, num_rays)
        
        # Apply probe weights (element-wise multiplication)
        weighted_densities = ray_densities * probe_weights  # (batch, num_samples, num_rays)
        
        # Sum over rays at each slice
        slice_values = weighted_densities.sum(dim=2)  # (batch, num_samples)
        
        # Integrate over z
        step_size = 2.0 / (num_samples_per_ray - 1)
        predicted_values = slice_values.sum(dim=1) * step_size  # (batch,)
        
        return predicted_values

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

        if self._clever_k_positions is not None:
            self._clever_k_positions = self._clever_k_positions.to(device)
        
        if self.learn_defocus:
            self._z_focus_params = nn.Parameter(self._z_focus_params.to(device))
        else:
            self.register_buffer('_z_focus_params', self._z_focus_params.to(device))
        
        if self.learn_astigmatism:
            self._stig_2_params = nn.Parameter(self._stig_2_params.to(device))
        else:
            self.register_buffer('_stig_2_params', self._stig_2_params.to(device))

        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)

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
            self._k1 = self._k1.to(device)

        self.device = device
        self.reconnect_optimizer_to_parameters()




























# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter

# class Probe():
#     """Probe model with matched wavefunction and discrete ray pictures."""

#     def __init__(
#         self,
#         k_max = 1.0,
#         pixel_size = 0.2,
#         im_shape = (256,256),
#         wavelength = 0.0197,
#     ):
#         self.k_max = np.array(k_max)
#         self.pixel_size = np.array(pixel_size)
#         self.im_shape = np.array(im_shape)
#         self.wavelength = np.array(wavelength)

#         # Coordinates
#         self.kr = np.fft.fftfreq(im_shape[0],pixel_size)[:,None]
#         self.kc = np.fft.fftfreq(im_shape[1],pixel_size)[None,:]
#         k2 = self.kr**2 + self.kc**2
#         k1 = np.sqrt(k2)
#         self.dk = self.kr[1] - self.kr[0]

#         self.aper = np.clip(
#             (k_max - k1)/self.dk + 0.5,
#             0.0,
#             1.0,
#         )

#         self.basis_C1 = np.pi*wavelength*k2
#         self.basis_A1x = np.pi*wavelength*(self.kr**2-self.kc**2)
#         self.basis_A1y = np.pi*wavelength*(2*self.kr*self.kc)



#     def get_center_balanced_k_radius_offset(
#         self,
#         num_rings=5,
#         num_rays=6,
#     ):

#         total_rays = 1 + num_rays * num_rings * (num_rings + 1) / 2
#         target_first_ring = self.k_max * 2 * np.sqrt(
#             np.pi / (
#                 total_rays
#                 * num_rays
#                 * np.tan(np.pi / num_rays)
#             )
#         )
#         base_first_ring = self.k_max * 2 / (2*num_rings + 1)
#         return target_first_ring - base_first_ring


#     def get_balanced_k_radii(
#         self,
#         num_rings=5,
#         num_rays=6,
#     ):

#         base_radii = self.k_max * 2 * (
#             np.arange(1, num_rings + 1) / (2*num_rings + 1)
#         )

#         if num_rings < 2:
#             return (
#                 base_radii
#                 + self.get_center_balanced_k_radius_offset(
#                     num_rings=num_rings,
#                     num_rays=num_rays,
#                 )
#             )

#         total_rays = 1 + num_rays * num_rings * (num_rings + 1) / 2

#         # Match the center-cell area to the mean per-ray pupil area.
#         first_radius = self.k_max * 2 * np.sqrt(
#             np.pi / (
#                 total_rays
#                 * num_rays
#                 * np.tan(np.pi / num_rays)
#             )
#         )

#         # Match the outermost-ring area using the midpoint between the
#         # last two rings as the inner radial boundary of the edge cells.
#         outer_boundary_midpoint = self.k_max * np.sqrt(
#             1.0 - (num_rings * num_rays) / total_rays
#         )

#         slope = (
#             outer_boundary_midpoint - first_radius
#         ) / (num_rings - 1.5)
#         intercept = first_radius - slope

#         return slope*np.arange(1, num_rings + 1) + intercept


#     def define_rays(
#         self,
#         num_rings = 4,
#         num_rays = 6,
#         offset_mode = 'maximize_spacing',
#         offset_samples = 720,
#         balance_center = True,
#         balance_outer = True,
#         weight_mode = 'nearest',
#         fractional_power = 2.0,
#         figsize=(6,6),
#     ):

#         # rays in k space
#         k = [[0,0]]
#         self.k_ring_offsets = []
#         if balance_center and balance_outer:
#             k_radii = self.get_balanced_k_radii(
#                 num_rings=num_rings,
#                 num_rays=num_rays,
#             )
#         elif balance_center:
#             radius_offset = self.get_center_balanced_k_radius_offset(
#                 num_rings=num_rings,
#                 num_rays=num_rays,
#             )
#             k_radii = self.k_max * 2 * (
#                 np.arange(1, num_rings + 1) / (2*num_rings + 1)
#             ) + radius_offset
#         else:
#             k_radii = self.k_max * 2 * (
#                 np.arange(1, num_rings + 1) / (2*num_rings + 1)
#             )
#         for a0 in range(num_rings):
#             k_radius = k_radii[a0]
#             num_ring_rays = (a0+1) * num_rays
#             phi = 2*np.pi / num_ring_rays

#             if offset_mode == 'alternate':
#                 dphi = phi * np.mod(a0,2) / 2
#             elif offset_mode == 'golden':
#                 golden = (np.sqrt(5.0) - 1.0) / 2.0
#                 dphi = phi * np.mod((a0 + 1) * golden, 1.0)
#             elif offset_mode == 'maximize_spacing':
#                 dphi = self._compute_ring_offset(
#                     k_radius=k_radius,
#                     num_ring_rays=num_ring_rays,
#                     existing_k=np.array(k, dtype=float),
#                     offset_samples=offset_samples,
#                 )
#             else:
#                 raise ValueError(
#                     "offset_mode must be 'alternate', 'golden', "
#                     "or 'maximize_spacing'"
#                 )

#             self.k_ring_offsets.append(dphi)

#             for a1 in range(num_ring_rays):
#                 k.append([
#                     k_radius*np.cos(phi*a1 + dphi),
#                     k_radius*np.sin(phi*a1 + dphi),
#                 ])
#         self.k = np.array(k)

#         # k in pixel space
#         self.k_px = self.k / self.dk + self.im_shape//2


#         # weights of rays
#         self.k_weights_raw, self.k_weights = self._compute_k_weights(
#             weight_mode=weight_mode,
#             fractional_power=fractional_power,
#         )

#     def make_probe(
#         self,
#         C1,
#         A1,

#     ):
#         C1 = np.array(C1)
#         A1 = np.array(A1)
#         psi = None
#         r, r_px = self._compute_r(
#             C1=C1,
#             A1=A1,
#         )
#         return r, r_px


#     def _compute_wavefunction(
#         self,
#         C1,
#         A1,
#     ):

#         chi = self.basis_C1*C1 \
#             + self.basis_A1x*A1[0] \
#             + self.basis_A1y*A1[1]
#         Psi = np.exp(-1j*chi) * self.aper
#         return np.fft.ifft2(Psi)


#     def _compute_r(
#         self,
#         C1,
#         A1,
#     ):

#         r = np.array((
#             self.k[:,0]*(C1 + A1[0]) + self.k[:,1]*A1[1],
#             self.k[:,1]*(C1 - A1[0]) + self.k[:,0]*A1[1],
#         )).T * self.wavelength
#         r_px = r / self.pixel_size + self.im_shape[None,:]//2
#         return r, r_px


#     def _estimate_ray_spacing_px(
#         self,
#         r_px,
#     ):

#         if r_px.shape[0] <= 1:
#             return 0.0

#         diff = r_px[:, None, :] - r_px[None, :, :]
#         dist2 = np.sum(diff**2, axis=2)
#         np.fill_diagonal(dist2, np.inf)
#         nearest_dist = np.sqrt(np.min(dist2, axis=1))

#         if hasattr(self, 'k_weights') and self.k_weights is not None:
#             return np.sum(self.k_weights * nearest_dist)
#         return np.mean(nearest_dist)


#     def _compute_k_weights(
#         self,
#         weight_mode='nearest',
#         fractional_power=2.0,
#     ):

#         weight_mode = weight_mode.lower()
#         if weight_mode not in ('nearest', 'fractional'):
#             raise ValueError(
#                 "weight_mode must be 'nearest' or 'fractional'"
#             )

#         if fractional_power <= 0:
#             raise ValueError("fractional_power must be positive")

#         grid_shape = tuple(self.im_shape)
#         kr_grid = np.broadcast_to(self.kr, grid_shape)
#         kc_grid = np.broadcast_to(self.kc, grid_shape)

#         k_pixels = np.stack(
#             (kr_grid.ravel(), kc_grid.ravel()),
#             axis=1,
#         )
#         aper_weights = self.aper.ravel()
#         valid = aper_weights > 0

#         k_pixels = k_pixels[valid]
#         aper_weights = aper_weights[valid]

#         diff = k_pixels[:, None, :] - self.k[None, :, :]
#         dist2 = np.sum(diff**2, axis=2)

#         if weight_mode == 'nearest':
#             nearest = np.argmin(dist2, axis=1)
#             k_weights_raw = np.bincount(
#                 nearest,
#                 weights=aper_weights,
#                 minlength=self.k.shape[0],
#             ).astype(float)
#         else:
#             share = np.zeros_like(dist2, dtype=float)
#             zero_mask = dist2 <= np.finfo(float).eps
#             zero_rows = np.any(zero_mask, axis=1)

#             if np.any(zero_rows):
#                 zero_share = zero_mask[zero_rows].astype(float)
#                 zero_share /= np.sum(
#                     zero_share,
#                     axis=1,
#                     keepdims=True,
#                 )
#                 share[zero_rows] = zero_share

#             if np.any(~zero_rows):
#                 inv_dist = dist2[~zero_rows]**(-0.5 * fractional_power)
#                 inv_dist /= np.sum(
#                     inv_dist,
#                     axis=1,
#                     keepdims=True,
#                 )
#                 share[~zero_rows] = inv_dist

#             k_weights_raw = np.sum(
#                 share * aper_weights[:, None],
#                 axis=0,
#             )

#         k_weights = k_weights_raw / np.sum(k_weights_raw)
#         return k_weights_raw, k_weights


#     def _compute_ring_offset(
#         self,
#         k_radius,
#         num_ring_rays,
#         existing_k,
#         offset_samples=720,
#     ):

#         if offset_samples < 1:
#             raise ValueError("offset_samples must be at least 1")

#         if existing_k.shape[0] <= 1:
#             return 0.0

#         phi = 2*np.pi / num_ring_rays
#         candidate_offsets = np.linspace(
#             0.0,
#             phi,
#             offset_samples,
#             endpoint=False,
#         )

#         best_offset = 0.0
#         best_min_dist2 = -np.inf
#         best_mean_nearest_dist2 = -np.inf

#         for dphi in candidate_offsets:
#             angles = phi*np.arange(num_ring_rays) + dphi
#             ring_k = np.column_stack((
#                 k_radius*np.cos(angles),
#                 k_radius*np.sin(angles),
#             ))

#             diff = ring_k[:, None, :] - existing_k[None, :, :]
#             dist2 = np.sum(diff**2, axis=2)
#             nearest_dist2 = np.min(dist2, axis=1)

#             min_dist2 = np.min(nearest_dist2)
#             mean_nearest_dist2 = np.mean(nearest_dist2)

#             better_min = min_dist2 > best_min_dist2
#             equal_min = np.isclose(min_dist2, best_min_dist2)
#             better_mean = mean_nearest_dist2 > best_mean_nearest_dist2
#             if better_min or (equal_min and better_mean):
#                 best_offset = dphi
#                 best_min_dist2 = min_dist2
#                 best_mean_nearest_dist2 = mean_nearest_dist2

#         return best_offset







DatasetModelType = TomographyINRDataset | TomographyPixDataset | TomographyThroughFocalINRDataset | TomographyThroughFocalConvergenceINRDataset | TomographyThroughFocalAstigmatismINRDataset | TomographyThroughFocalProbeINRDataset | TomographyThroughFocalJustStigINRDataset | TomographyThroughFocalINRDataset_0615






    # def create_batch_rays(
    #     self,
    #     pixel_i: torch.Tensor,
    #     pixel_j: torch.Tensor,
    #     N: int,
    #     num_samples_per_ray: int,
    #     num_rays: int,
    #     z_focus: torch.Tensor,
    #     stig_2: torch.Tensor,
    #     convergence_angle: torch.Tensor,
    #     voxel_size_ang: float = 2.0,   # Å per voxel — swap out for experimental data
    # ) -> torch.Tensor:
    #     batch_size = len(pixel_i)
    #     x_coords_0 = (pixel_j / (N - 1)) * 2 - 1
    #     y_coords_0 = (pixel_i / (N - 1)) * 2 - 1

    #     # z_coords in normalized [-1, 1] volume space
    #     z_coords_norm = torch.linspace(-1, 1, num_samples_per_ray, device=pixel_i.device)

    #     # Physical size of the volume half-extent in Å
    #     # [-1, 1] spans N voxels, so 1 unit = (N * voxel_size_ang / 2) Å
    #     half_extent_ang = (N * voxel_size_ang) / 2.0

    #     # Convert z_coords and z_focus from normalized to Å
    #     # dz per projection: shape (batch_size, num_samples_per_ray)
    #     z_ang = z_coords_norm[None, :] * half_extent_ang           # (1, num_samples_per_ray)
    #     z_focus_ang = z_focus[:, None] * half_extent_ang            # (batch_size, 1)
    #     dz_ang = z_ang - z_focus_ang                                # (batch_size, num_samples_per_ray)

    #     H, W = self._aper.shape

    #     # Precompute probe pixel offset grid — fixed, no grad needed
    #     row_all = torch.arange(H, device=pixel_i.device).float() - H / 2.0
    #     col_all = torch.arange(W, device=pixel_i.device).float() - W / 2.0
    #     row_grid, col_grid = torch.meshgrid(row_all, col_all, indexing='ij')
    #     row_grid = row_grid.ravel()  # (H*W,)
    #     col_grid = col_grid.ravel()  # (H*W,)

    #     probe_pixel_size_ang = 1.0 / (H * self._dk)
    #     dx_all = row_grid * probe_pixel_size_ang / half_extent_ang  # (H*W,)
    #     dy_all = col_grid * probe_pixel_size_ang / half_extent_ang

    #     x_offsets_norm = torch.zeros(
    #         batch_size, num_samples_per_ray, num_rays, device=pixel_i.device)
    #     y_offsets_norm = torch.zeros(
    #         batch_size, num_samples_per_ray, num_rays, device=pixel_i.device)

    #     # Vectorize over both batch and z dimensions at once
    #     # Reshape to (batch_size * num_samples_per_ray,)
    #     dz_flat = dz_ang.reshape(-1)
        
    #     # Compute all weights at once: (batch_size * num_samples_per_ray, n_probe_positions)
    #     weights_flat = self._compute_probe_weights_at_defocus_vectorized(
    #         dz_flat, stig_2=stig_2, convergence_angle=convergence_angle
    #     )
        
    #     # Sample all probes at once: (batch_size * num_samples_per_ray, num_rays)
    #     dx_norm_flat, dy_norm_flat = self.differentiable_probe_sample_vectorized(
    #         weights_flat, dx_all, dy_all, num_rays, temperature=self._gumbel_temp
    #     )
        
    #     # Reshape back to (batch_size, num_samples_per_ray, num_rays)
    #     x_offsets_norm = dx_norm_flat.reshape(batch_size, num_samples_per_ray, num_rays)
    #     y_offsets_norm = dy_norm_flat.reshape(batch_size, num_samples_per_ray, num_rays)
        
    #     # Add pixel center coords: (batch_size, 1, 1) + (batch_size, num_samples_per_ray, num_rays)
    #     x_coords = x_coords_0[:, None, None] + x_offsets_norm
    #     y_coords = y_coords_0[:, None, None] + y_offsets_norm
    #     z_coords = z_coords_norm[None, :, None].expand(
    #         batch_size, num_samples_per_ray, num_rays)

    #     # Flatten rays dimension: (batch_size, num_samples_per_ray * num_rays, 3)
    #     x_coords = x_coords.reshape(batch_size, num_samples_per_ray * num_rays)
    #     y_coords = y_coords.reshape(batch_size, num_samples_per_ray * num_rays)
    #     z_coords = z_coords.reshape(batch_size, num_samples_per_ray * num_rays)

    #     rays = torch.stack([x_coords, y_coords, z_coords], dim=2)
    #     return rays











































# 20260412_2000
# import torch.nn.functional as F

# class TomographyThroughFocalAstigmatismINRDataset(TomographyINRDataset):
#     """
#     Dataset class for INR-based tomography that assumes non-parallel illumination condition with astigmatism.
#     Inherits from TomographyINRDataset.
#     """

#     def __init__(
#         self,
#         tilt_stack: Dataset3d | NDArray | torch.Tensor,
#         tilt_angles: NDArray | torch.Tensor,
#         ## Through focal parameters:
#         convergence_angle: float,
#         num_rays: int = 1,
#         ## optional parameters
#         learn_shift: bool = True,
#         learn_tilt_axis: bool = True,
#         token: object | None = None,
#         random_method: str = 'p',
#         wavelength_ang: float = 0.0197,
#         stig_2: tuple[float, float] = (0.0, 0.0),
#         probe_im_shape: tuple[int, int] = (64, 64),
#         pixel_size_ang: float = 0.2,
#     ):
#         super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, token)
#         self.num_rays = int(num_rays)

#         self._random_rays = True
#         self._random_method = random_method

#         self._z_focuses = torch.zeros(self.learnable_tilts+1)

#         self._convergence_angle = torch.ones(1) * convergence_angle

#         self._stig_2 = torch.tensor(stig_2, dtype = torch.float32)

#         self._convergence_angle = torch.ones(1) * convergence_angle
#         self._stig_2 = torch.tensor(stig_2, dtype=torch.float32)
        
#         # Initialize as parameters immediately (will be moved to device in .to())
#         self._convergence_angle_params = nn.Parameter(self._convergence_angle.clone())
#         self._stig_2_params = nn.Parameter(self._stig_2.clone())

#         if random_method.lower() in ['probe', 'p']:
#             import numpy as np
#             self._wavelength_ang = wavelength_ang
#             k_max = float(self._convergence_angle.item()) / float(wavelength_ang) # convergence angle should be in radians already
#             kr = np.fft.fftfreq(probe_im_shape[0], d=pixel_size_ang)[:, None]
#             kc = np.fft.fftfreq(probe_im_shape[1], d=pixel_size_ang)[None, :]
#             self._dk = torch.tensor(float(kr[1] - kr[0]), dtype=torch.float32)
#             dk = float(kr[1, 0] - kr[0, 0])  # scalar float, not a 1-element array
#             k1 = np.sqrt(kr**2 + kc**2)
#             self._k1 = torch.tensor(k1, dtype=torch.float32)  # radial k-magnitude, shape (H, W)
#             aper = np.clip((k_max - k1) / dk + 0.5, 0.0, 1.0)

#             # basis
#             self._basis_0 = torch.tensor(
#                 (torch.pi * wavelength_ang) * (kr**2 + kc**2), dtype=torch.float32)
#             self._basis_1 = torch.tensor(
#                 (torch.pi * wavelength_ang) * (kr**2 - kc**2), dtype=torch.float32)
#             self._basis_2 = torch.tensor(
#                 (torch.pi * wavelength_ang) * (2 * kr * kc),   dtype=torch.float32)
#             self._aper = torch.tensor(aper, dtype=torch.float32)
            
#             self._gumbel_temp = 10

#     @property
#     def random_rays(self) -> bool:
#         return self._random_rays

#     @property
#     def convergence_angle_params(self) -> torch.nn.Parameter:
#         return self._convergence_angle_params

#     @convergence_angle_params.setter
#     def convergence_angle_params(self, convergence_angle: torch.Tensor, device: str):
#         self._convergence_angle_params = nn.Parameter(convergence_angle.to(device))

#     @property
#     def z_focus_params(self) -> torch.nn.Parameter:
#         return self._z_focus_params

#     @z_focus_params.setter
#     def z_focus_params(self, z_focus_values: torch.Tensor, device: str):
#         self._z_focus_params = nn.Parameter(z_focus_values.to(device))

#     @property
#     def stig_2_params(self) -> torch.nn.Parameter:
#         return self._stig_2_params

#     @stig_2_params.setter
#     def stig_2_params(self, stig_2_values: torch.Tensor, device: str):
#         self._stig_2_params = nn.Parameter(stig_2_values.to(device))

#     # --- Forward Pass w/ Params Method for OptimizerMixin ---
#     def forward(self, dummy_input: Any = None):
#         """
#         Forward pass for INR-based through focal tomography. In the forward pass, the only parameters that
#         are passed will be the shifts, focal plane, z1 and z3 Euler angles.
#         """

#         first_half_shifts = self.shifts_params[: self.reference_tilt_idx]
#         second_half_shifts = self.shifts_params[self.reference_tilt_idx :]
#         shifts = torch.cat([first_half_shifts, self._shifts_ref, second_half_shifts], dim=0)

#         first_half_z1 = self.z1_params[: self.reference_tilt_idx]
#         second_half_z1 = self.z1_params[self.reference_tilt_idx :]
#         z1 = torch.cat([first_half_z1, self._z1_ref, second_half_z1], dim=0)

#         first_half_z3 = self.z3_params[: self.reference_tilt_idx]
#         second_half_z3 = self.z3_params[self.reference_tilt_idx :]
#         z3 = torch.cat([first_half_z3, self._z3_ref, second_half_z3], dim=0)

#         z_focus = self.z_focus_params

#         if self.learn_shift and self.learn_tilt_axis:
#             return shifts, z1, z3, z_focus, self._convergence_angle_params[0], self._stig_2_params[:]
#         elif self.learn_shift:
#             return shifts, torch.zeros_like(z1), torch.zeros_like(z3), z_focus, self._convergence_angle_params[0], self._stig_2_params[:]
#         elif self.learn_tilt_axis:
#             return torch.zeros_like(shifts), z1, z3, z_focus, self._convergence_angle_params[0], self._stig_2_params[:]
#         elif self.learn_shift and self.learn_tilt_axis:
#             return shifts, z1, z3, z_focus, self._convergence_angle_params[0], self._stig_2_params[:]
#         else:
#             return torch.zeros_like(shifts), torch.zeros_like(z1), torch.zeros_like(z3), z_focus, self._convergence_angle_params[0], self._stig_2_params[:]

#     def differentiable_probe_sample(
#         self,
#         weights: torch.Tensor,   # (H*W,)
#         dx_all: torch.Tensor,    # (H*W,)
#         dy_all: torch.Tensor,    # (H*W,)
#         num_rays: int,
#         temperature: float = 0.1,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Returns dx_norm, dy_norm of shape (num_rays,) directly,
#         without materializing the full (num_rays, H*W) soft_samples matrix.
#         """
#         log_probs = torch.log(weights + 1e-10)  # (H*W,)

#         # Process one ray at a time to avoid (num_rays, H*W) allocation
#         dx_out = torch.zeros(num_rays, device=weights.device)
#         dy_out = torch.zeros(num_rays, device=weights.device)

#         for r in range(num_rays):
#             gumbel_noise = -torch.log(
#                 -torch.log(torch.rand(len(weights), device=weights.device) + 1e-10) + 1e-10
#             )
#             soft = torch.softmax((log_probs + gumbel_noise) / temperature, dim=-1)  # (H*W,)
#             dx_out[r] = (soft * dx_all).sum()
#             dy_out[r] = (soft * dy_all).sum()

#         return dx_out, dy_out



#     def get_coords(
#         self,
#         batch: dict[str, torch.Tensor],
#         N: int,
#         num_samples_per_ray: int,
#     ) -> torch.Tensor:
#         num_rays = self.num_rays
#         # convergence_angle = self._convergence_angle
#         pixel_i = batch["pixel_i"].float().to(self.device, non_blocking=True)
#         pixel_j = batch["pixel_j"].float().to(self.device, non_blocking=True)
#         # target_values = batch["target_value"].to(self.device, non_blocking=True)
#         phis = batch["phi"].to(self.device, non_blocking=True)
#         projection_indices = batch["projection_idx"].to(self.device, non_blocking=True)
#         shifts, z1_params, z3_params, z_focus_params, convergence_angle, stig_2 = self.forward(None)
#         batch_shifts = torch.index_select(shifts, 0, projection_indices)
#         batch_z1 = torch.index_select(z1_params, 0, projection_indices)
#         batch_z3 = torch.index_select(z3_params, 0, projection_indices)
#         batch_z_focus = torch.index_select(z_focus_params, 0, projection_indices)
#         # with torch.no_grad():
#         batch_ray_coords = self.create_batch_rays(
#             pixel_i,
#             pixel_j,
#             N,
#             num_samples_per_ray,
#             num_rays,
#             z_focus=batch_z_focus,
#             stig_2 = stig_2,
#             convergence_angle=convergence_angle,
#         )

#         transformed_rays = self.transform_batch_rays(
#             batch_ray_coords,
#             z1=batch_z1,
#             x=phis,
#             z3=batch_z3,
#             shifts=batch_shifts,
#             N=N,
#             sampling_rate=1.0,
#         )
#         all_coords = transformed_rays.view(-1, 3)

#         all_coords = all_coords.to(self.device, dtype=torch.float32, non_blocking=True)
#         return all_coords

#     def get_theta_phi(
#         self,
#         convergence_angle: torch.Tensor,
#         num_rays: int,
#         device: torch.device | str | None = None,
#         random_rays: bool = False,
#         random_method: str = 'g',
#     ):
#         if device is None:
#             device = torch.device("cpu")

#         num_rays = int(num_rays)

#         # random_rays sampling method
#         if random_rays:
#             theta = torch.rand(num_rays, device=device) * 2 * torch.pi # this can stay as uniform sampling
#             # phi = torch.rand(num_rays, device=device) * convergence_angle # the original uniform sampling of phi
#             if random_method.lower() in ['gaussian','g']:
#                 phi = torch.randn(num_rays, device=device) * convergence_angle

#             # let's also do the sinc squared, which might be slower?
#             # essentially, torch doesn't have a sinc**2 distribution built in, but we can just make a discrete one ourselves
#             elif random_method.lower() in ['sinc','s']:
#                 x = torch.linspace(-0.5, 0.5, 5000) # in radians
#                 pdf = torch.sinc(x)**2 # 
#                 pdf = pdf/pdf.sum() # normalize to 1
#                 indices = torch.multinomial(pdf, num_rays, replacement=True)
#                 phi = x[indices]
#             elif random_method.lower() in ['probe', 'p']:
#                 indices = torch.multinomial(self._probe_weights, num_rays, replacement=True)
                
#                 # Convert flat indices back to 2D grid positions
#                 H, W = self._aper.shape
#                 row_s = (indices // W).float() - H / 2   # centered row coordinate
#                 col_s = (indices  % W).float() - W / 2   # centered col coordinate

#                 # Convert grid position to angle
#                 # The grid spacing in k-space is dk = 1/(N * pixel_size_ang)
#                 # but since we auto-set pixel_size, we can recover it from basis_0:
#                 # basis_0 = pi * lambda * (kr^2 + kc^2), so the k-step is:
#                 # dk = sqrt(basis_0[0,1] / (pi * lambda))  -- but simpler to just store dk at init
#                 kr_s = row_s * self._dk
#                 kc_s = col_s * self._dk

#                 # k_magnitude = torch.sqrt(kr_s**2 + kc_s**2)
#                 # phi   = torch.arctan(k_magnitude * self._wavelength_ang)
#                 # theta = torch.arctan2(kc_s, kr_s)

#                 return kr_s * self._wavelength_ang, kc_s * self._wavelength_ang

#             else:
#                 raise ValueError(
#                     f"Unsupported random_method={random_method}. "
#                     "Supported values are: gaussian, g, sinc, s. .lower() is applied internally."
#                 )

#         else:
#             raise ValueError('Only random rays supported for this dataset model right now')

#         return theta, phi




#     def _compute_probe_weights_at_defocus(
#         self,
#         defocus_ang: torch.Tensor | float,
#         stig_2: torch.Tensor | None = None,
#         convergence_angle: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         if stig_2 is None:
#             stig_2 = getattr(self, '_stig_2_params', self._stig_2)
#         if convergence_angle is None:
#             convergence_angle = getattr(self, '_convergence_angle_params', self._convergence_angle)

#         # Keep everything as tensors — no .item() calls
#         k_max = convergence_angle / self._wavelength_ang # convergence angle already in radians, not mrad
#         aper = torch.clamp(
#             (k_max - self._k1) / self._dk + 0.5,
#             0.0, 1.0,
#         )

#         chi = (self._basis_0 * defocus_ang +
#             self._basis_1 * stig_2[0] +
#             self._basis_2 * stig_2[1])

#         Psi = aper * torch.exp(-1j * chi)
#         psi = torch.fft.ifft2(Psi)
#         psi = torch.fft.fftshift(psi)

#         weights = torch.abs(psi) ** 2
#         weights = weights.ravel()
#         weights = weights / weights.sum()
#         return weights  # fully differentiable w.r.t. stig_2, convergence_angle, defocus_ang

#     def _compute_probe_weights_at_defocus_vectorized(self, defocus_ang, stig_2=None, convergence_angle=None):
#         if stig_2 is None:
#             stig_2 = self._stig_2_params

#         convergence_angle = F.softplus(self._convergence_angle_params[0]) * 45e-3 + 5e-3
#         N = defocus_ang.shape[0]
        
#         # DEBUG STEP 1: Check convergence angle
#         # print(f"convergence_angle: {convergence_angle}")
#         # print(f"wavelength: {self._wavelength_ang}")
        
#         k_max = convergence_angle / self._wavelength_ang # convergence angle already in radians, not mrad
#         # print(f"k_max: {k_max}")
        
#         # DEBUG STEP 2: Check aperture
#         # print(f"self._k1 shape: {self._k1.shape}, min: {self._k1.min()}, max: {self._k1.max()}")
#         # print(f"self._dk: {self._dk}")
        
#         aper = torch.clamp(
#             (k_max - self._k1) / self._dk + 0.5,
#             0.0, 1.0,
#         )
#         # print(f"aper min: {aper.min()}, max: {aper.max()}, mean: {aper.mean()}")
#         # print(f"aper non-zero fraction: {(aper > 0.01).float().mean()}")
        
#         # DEBUG STEP 3: Check phase (chi)
#         # print(f"self._basis_0 shape: {self._basis_0.shape}, min: {self._basis_0.min()}, max: {self._basis_0.max()}")
#         # print(f"defocus_ang: {defocus_ang}")
#         # print(f"stig_2: {stig_2}")
        
#         chi = (self._basis_0[None, :, :] * defocus_ang[:, None, None] +
#             self._basis_1[None, :, :] * stig_2[0] +
#             self._basis_2[None, :, :] * stig_2[1])
        
#         # print(f"chi min: {chi.min()}, max: {chi.max()}, range: {chi.max() - chi.min()}")
#         # print(f"chi std: {chi.std()}")
        
#         # DEBUG STEP 4: Check Psi
#         Psi = aper * torch.exp(-1j * chi)
#         # print(f"|Psi| min: {torch.abs(Psi).min()}, max: {torch.abs(Psi).max()}, mean: {torch.abs(Psi).mean()}")
        
#         # DEBUG STEP 5: Check after FFT
#         psi = torch.fft.ifft2(Psi, dim=(-2, -1))
#         psi = torch.fft.fftshift(psi, dim=(-2, -1))
        
#         # print(f"|psi| min: {torch.abs(psi).min()}, max: {torch.abs(psi).max()}, std: {torch.abs(psi).std()}")
        
#         weights = torch.abs(psi) ** 2
#         # print(f"weights BEFORE norm - min: {weights.min()}, max: {weights.max()}, std: {weights.std()}")
        

#     # After computing psi and weights, but before normalizing:
#         # import matplotlib.pyplot as plt
#         # import numpy as np
        
#         # # Take first sample for visualization
#         # Psi_vis = Psi[0].detach().cpu()
#         # psi_vis = psi[0].detach().cpu()
#         # weights_vis_unnorm = (torch.abs(psi_vis) ** 2).numpy()
        
#         # # After normalization
#         # weights_vis_norm = weights[0].detach().cpu().reshape(psi_vis.shape).numpy()
        
#         # fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
#         # # Row 1: Fourier space
#         # im0 = axes[0, 0].imshow(torch.abs(Psi_vis).numpy(), cmap='hot')
#         # axes[0, 0].set_title('|Psi| (Fourier space aperture)')
#         # plt.colorbar(im0, ax=axes[0, 0])
        
#         # im1 = axes[0, 1].imshow(np.angle(Psi_vis.numpy()), cmap='twilight')
#         # axes[0, 1].set_title('Phase of Psi (chi)')
#         # plt.colorbar(im1, ax=axes[0, 1])
        
#         # # Radial profile of aperture
#         # center = np.array(Psi_vis.shape) // 2
#         # y, x = np.ogrid[:Psi_vis.shape[0], :Psi_vis.shape[1]]
#         # r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
#         # r_int = r.astype(int)
#         # aper_radial = np.bincount(r_int.ravel(), torch.abs(Psi_vis).numpy().ravel()) / np.bincount(r_int.ravel())
#         # axes[0, 2].plot(aper_radial)
#         # axes[0, 2].set_title('Radial aperture profile')
#         # axes[0, 2].set_xlabel('Radius (pixels)')
#         # axes[0, 2].set_ylabel('Aperture')
#         # axes[0, 2].grid(True)
        
#         # # Row 2: Real space probe
#         # im3 = axes[1, 0].imshow(weights_vis_unnorm, cmap='hot')
#         # axes[1, 0].set_title('Probe intensity (before norm)')
#         # plt.colorbar(im3, ax=axes[1, 0])
        
#         # im4 = axes[1, 1].imshow(weights_vis_norm, cmap='hot')
#         # axes[1, 1].set_title('Probe intensity (normalized)')
#         # plt.colorbar(im4, ax=axes[1, 1])
        
#         # # Log scale
#         # im5 = axes[1, 2].imshow(np.log10(weights_vis_norm + 1e-10), cmap='hot', vmin=-10, vmax=0)
#         # axes[1, 2].set_title('Log10(Probe intensity)')
#         # plt.colorbar(im5, ax=axes[1, 2])
        
#         # plt.tight_layout()
#         # plt.savefig('debug_probe.png', dpi=150, bbox_inches='tight')
#         # print(f"Saved debug_probe.png (convergence_angle={convergence_angle.item() if hasattr(convergence_angle, 'item') else convergence_angle:.6f})")
#         # plt.close()
    

#         weights = weights.reshape(N, -1)  # (N, H*W)
#         weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize each probe
                
#         # ADD THIS DEBUG:
#         # print(f"Probe weights AFTER normalization - min: {weights.min()}, max: {weights.max()}")
#         # print(f"Effective positions (weight > 0.001): {(weights > 0.001).sum(dim=-1).float().mean()}")
#         # print(f"Entropy: {-(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()}")
        
#         return weights  # (N, n_pixels) where n_pixels = H*W




#     def differentiable_probe_sample_vectorized(self, weights, dx_all, dy_all, num_rays, temperature):
#         N, n_positions = weights.shape
        
#         # ADD THESE CHECKS:
#         # print(f"dx_all.requires_grad: {dx_all.requires_grad}")
#         # print(f"dy_all.requires_grad: {dy_all.requires_grad}")
#         # print(f"dx_all dtype: {dx_all.dtype}, weights dtype: {weights.dtype}")
        
#         # Make sure dx_all and dy_all don't require grad (they're just coordinates)
#         # but are on the same device and dtype
#         if dx_all.dtype != weights.dtype:
#             dx_all = dx_all.to(weights.dtype)
#             dy_all = dy_all.to(weights.dtype)
        
#         gumbel_noise = -torch.log(-torch.log(
#             torch.rand(N, num_rays, n_positions, device=weights.device, dtype=weights.dtype) + 1e-10
#         ) + 1e-10)
        
#         log_weights = torch.log_softmax(weights, dim=-1)
#         logits = log_weights[:, None, :] + gumbel_noise
#         soft_samples = torch.softmax(logits / temperature, dim=-1)
        
#         # ADD DEBUG: Check if soft_samples varies
#         # print(f"soft_samples min: {soft_samples.min()}, max: {soft_samples.max()}, mean: {soft_samples.mean()}")
#         # print(f"soft_samples entropy: {-(soft_samples * torch.log(soft_samples + 1e-10)).sum(dim=-1).mean()}")
        
#         dx_norm = torch.matmul(soft_samples, dx_all)
#         dy_norm = torch.matmul(soft_samples, dy_all)
        
#         # ADD DEBUG: Check outputs
#         # print(f"dx_norm min: {dx_norm.min()}, max: {dx_norm.max()}, mean: {dx_norm.mean()}")
#         # print(f"dy_norm min: {dy_norm.min()}, max: {dy_norm.max()}, mean: {dy_norm.mean()}")
        
#         return dx_norm, dy_norm

#     def create_batch_rays(
#         self,
#         pixel_i: torch.Tensor,
#         pixel_j: torch.Tensor,
#         N: int,
#         num_samples_per_ray: int,
#         num_rays: int,
#         z_focus: torch.Tensor,
#         stig_2: torch.Tensor,
#         convergence_angle: torch.Tensor,
#         voxel_size_ang: float = 2.0,   # Å per voxel — swap out for experimental data
#     ) -> torch.Tensor:
#         batch_size = len(pixel_i)
#         x_coords_0 = (pixel_j / (N - 1)) * 2 - 1
#         y_coords_0 = (pixel_i / (N - 1)) * 2 - 1

#         # z_coords in normalized [-1, 1] volume space
#         z_coords_norm = torch.linspace(-1, 1, num_samples_per_ray, device=pixel_i.device)

#         # Physical size of the volume half-extent in Å
#         # [-1, 1] spans N voxels, so 1 unit = (N * voxel_size_ang / 2) Å
#         half_extent_ang = (N * voxel_size_ang) / 2.0

#         # Convert z_coords and z_focus from normalized to Å
#         # dz per projection: shape (batch_size, num_samples_per_ray)
#         z_ang = z_coords_norm[None, :] * half_extent_ang           # (1, num_samples_per_ray)
#         z_focus_ang = z_focus[:, None] * half_extent_ang            # (batch_size, 1)
#         dz_ang = z_ang - z_focus_ang                                # (batch_size, num_samples_per_ray)

#         H, W = self._aper.shape

#         # Precompute probe pixel offset grid — fixed, no grad needed
#         row_all = torch.arange(H, device=pixel_i.device).float() - H / 2.0
#         col_all = torch.arange(W, device=pixel_i.device).float() - W / 2.0
#         row_grid, col_grid = torch.meshgrid(row_all, col_all, indexing='ij')
#         row_grid = row_grid.ravel()  # (H*W,)
#         col_grid = col_grid.ravel()  # (H*W,)

#         probe_pixel_size_ang = 1.0 / (H * self._dk)
#         dx_all = row_grid * probe_pixel_size_ang / half_extent_ang  # (H*W,)
#         dy_all = col_grid * probe_pixel_size_ang / half_extent_ang

#         x_offsets_norm = torch.zeros(
#             batch_size, num_samples_per_ray, num_rays, device=pixel_i.device)
#         y_offsets_norm = torch.zeros(
#             batch_size, num_samples_per_ray, num_rays, device=pixel_i.device)

#         # Vectorize over both batch and z dimensions at once
#         # Reshape to (batch_size * num_samples_per_ray,)
#         dz_flat = dz_ang.reshape(-1)
        
#         # Compute all weights at once: (batch_size * num_samples_per_ray, n_probe_positions)
#         weights_flat = self._compute_probe_weights_at_defocus_vectorized(
#             dz_flat, stig_2=stig_2, convergence_angle=convergence_angle
#         )
        
#         # Sample all probes at once: (batch_size * num_samples_per_ray, num_rays)
#         dx_norm_flat, dy_norm_flat = self.differentiable_probe_sample_vectorized(
#             weights_flat, dx_all, dy_all, num_rays, temperature=self._gumbel_temp
#         )
        
#         # Reshape back to (batch_size, num_samples_per_ray, num_rays)
#         x_offsets_norm = dx_norm_flat.reshape(batch_size, num_samples_per_ray, num_rays)
#         y_offsets_norm = dy_norm_flat.reshape(batch_size, num_samples_per_ray, num_rays)
        
#         # Add pixel center coords: (batch_size, 1, 1) + (batch_size, num_samples_per_ray, num_rays)
#         x_coords = x_coords_0[:, None, None] + x_offsets_norm
#         y_coords = y_coords_0[:, None, None] + y_offsets_norm
#         z_coords = z_coords_norm[None, :, None].expand(
#             batch_size, num_samples_per_ray, num_rays)

#         # Flatten rays dimension: (batch_size, num_samples_per_ray * num_rays, 3)
#         x_coords = x_coords.reshape(batch_size, num_samples_per_ray * num_rays)
#         y_coords = y_coords.reshape(batch_size, num_samples_per_ray * num_rays)
#         z_coords = z_coords.reshape(batch_size, num_samples_per_ray * num_rays)

#         rays = torch.stack([x_coords, y_coords, z_coords], dim=2)
#         return rays

#     # @staticmethod
#     @torch.compile(mode="reduce-overhead")
#     def integrate_rays(
#         self,
#         rays: torch.Tensor,
#         num_samples_per_ray: int,
#         target_values_len: int,
#     ) -> torch.Tensor:
#         num_rays = self.num_rays
#         ray_densities = rays.view(
#             target_values_len,
#             num_samples_per_ray,
#             num_rays,
#         )
#         if self._random_rays:
#             predicted_values_all_rays = ray_densities.view(target_values_len, -1) # equal weights
#         else:
#             predicted_values_all_rays = (ray_densities @ self._ray_weights.view(-1, 1)).squeeze(-1)
#         step_size = 2.0 / (num_samples_per_ray - 1)
#         predicted_values = predicted_values_all_rays.sum(dim=1) * step_size

#         return predicted_values

#     @staticmethod
#     def transform_batch_rays(
#         rays: torch.Tensor,
#         z1: torch.Tensor,
#         x: torch.Tensor,
#         z3: torch.Tensor,
#         shifts: torch.Tensor,
#         N: int,
#         sampling_rate: float,
#     ) -> torch.Tensor:
#         shift_x_norm = (shifts[:, 0:1] * sampling_rate * 2) / (N - 1)
#         shift_y_norm = (shifts[:, 1:2] * sampling_rate * 2) / (N - 1)

#         shift_x_norm = shift_x_norm.expand(-1, rays.shape[1])
#         shift_y_norm = shift_y_norm.expand(-1, rays.shape[1])

#         rays_x = rays[:, :, 0] - shift_x_norm
#         rays_y = rays[:, :, 1] - shift_y_norm
#         rays_z = rays[:, :, 2]

#         theta = torch.deg2rad(-z3).view(-1, 1)
#         cos_t = torch.cos(theta)
#         sin_t = torch.sin(theta)

#         rays_x_rot1 = cos_t * rays_x - sin_t * rays_y
#         rays_y_rot1 = sin_t * rays_x + cos_t * rays_y
#         rays_z_rot1 = rays_z

#         theta = torch.deg2rad(x).view(-1, 1)
#         cos_t = torch.cos(theta)
#         sin_t = torch.sin(theta)

#         rays_x_rot2 = rays_x_rot1
#         rays_y_rot2 = cos_t * rays_y_rot1 - sin_t * rays_z_rot1
#         rays_z_rot2 = sin_t * rays_y_rot1 + cos_t * rays_z_rot1

#         theta = torch.deg2rad(-z1).view(-1, 1)
#         cos_t = torch.cos(theta)
#         sin_t = torch.sin(theta)

#         rays_x_final = cos_t * rays_x_rot2 - sin_t * rays_y_rot2
#         rays_y_final = sin_t * rays_x_rot2 + cos_t * rays_y_rot2
#         rays_z_final = rays_z_rot2

#         transformed_rays = torch.stack([rays_x_final, rays_y_final, rays_z_final], dim=2)

#         return transformed_rays

#     def to(self, device: str):
#         self._z1_params = nn.Parameter(self._z1_angles.to(device))
#         self._z3_params = nn.Parameter(self._z3_angles.to(device))
#         self._shifts_params = nn.Parameter(self._shifts.to(device))
#         self._z_focus_params = nn.Parameter(self._z_focuses.to(device))
#         self._convergence_angle_params = nn.Parameter(self._convergence_angle.to(device))
#         self._stig_2_params = nn.Parameter(self._stig_2.to(device))

#         self._z1_ref = self._z1_ref.to(device)
#         self._z3_ref = self._z3_ref.to(device)
#         self._shifts_ref = self._shifts_ref.to(device)
#         # self._z_focus_ref = self._z_focus_ref.to(device)
#         # self._convergence_angle = self._convergence_angle.to(device)

#         if hasattr(self, "_ray_weights"):
#             self._ray_weights = self._ray_weights.to(device)
#             self.phis = self.phis.to(device)
#             self.thetas = self.thetas.to(device)

#         if hasattr(self, "_basis_0"):
#             self._basis_0 = self._basis_0.to(device)
#             self._basis_1 = self._basis_1.to(device)
#             self._basis_2 = self._basis_2.to(device)
#             self._aper = self._aper.to(device)
#             self._dk = self._dk.to(device)
#             # self._probe_weights = self._compute_probe_weights_at_defocus(defocus_ang = 0, stig_2=self._stig_2, convergence_angle=self._convergence_angle)
#             self._k1 = self._k1.to(device)

#         self.device = device
#         self.reconnect_optimizer_to_parameters()

# DatasetModelType = TomographyINRDataset | TomographyPixDataset | TomographyThroughFocalINRDataset | TomographyThroughFocalConvergenceINRDataset | TomographyThroughFocalAstigmatismINRDataset
