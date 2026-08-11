import math
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import Dataset

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.constraints import BaseConstraints, Constraints
from quantem.core.ml.optimizer_mixin import OptimizerMixin
from quantem.tomography.utils import tv_loss_1d

# --- Ray patterns ---
#
# The two families of ray_pattern accepted by TomographyFocalINRDataset. They
# differ in what the generator produces and therefore in which probe-weight
# routine applies:
#
#   ANGLE  -- per-ray angles (num_rays,). Rays fan out from the pixel centre, so
#             the lateral offset scales with |dz| and the weights come from
#             _compute_probe_weights_a.
#   RADIAL -- real-space offsets in Angstroms, already resolved per z-sample as
#             (batch, num_samples, num_rays). Weights come from
#             _compute_probe_weights_r.
#
# Every name here must have a branch in create_batch_rays; the sets and that
# dispatch are what keep the two in step.
ANGLE_RAY_PATTERNS = frozenset(
    {"uniform_angle", "uniform_xy", "gaussian_angle", "hexagonal", "theta_phi", "clever_probe"}
)
RADIAL_RAY_PATTERNS = frozenset(
    {"clever", "grid", "uniform_parallel", "sunflower", "hexagonal_grid", "fibonacci"}
)
# Patterns handled by create_batch_rays_uniform rather than by the branch chain
# in create_batch_rays. They carry their own generator AND their own weighting
# (_compute_probe_weights_for_rays), including the Gumbel-softmax position
# sampler, so they are a third family rather than more ANGLE entries.
PROBE_UNIFORM_RAY_PATTERNS = frozenset({"uniform_probe", "hexagonal_probe", "grid_probe"})

# Handled by create_batch_rays_gumbel. Its "layout" IS the sampling: ray
# positions are drawn from the probe intensity through a Gumbel-softmax, so the
# placement carries gradient. Takes no sub-pattern and returns no separate
# weight vector -- the weighting is already in the positions.
GUMBEL_RAY_PATTERNS = frozenset({"probe_gumbel"})

RAY_PATTERNS = (
    ANGLE_RAY_PATTERNS | RADIAL_RAY_PATTERNS | PROBE_UNIFORM_RAY_PATTERNS | GUMBEL_RAY_PATTERNS
)

# Angle patterns whose lateral offset uses SIGNED dz rather than |dz|.
#
# A straight ray crossing its focus has offset alpha*(z - z_focus), which changes
# sign past the waist. The other angle patterns use |dz|, which reflects each ray
# at the focus instead. For an azimuthally symmetric ray set the two agree as a
# SET -- reflecting a ray onto the position of its opposite-azimuth partner -- so
# the difference is invisible there and only shows up for asymmetric sets: an odd
# ray count, a single off-axis ray, or a random draw. `theta_phi` is ported from
# TomographyThroughFocalINRDataset, which used the signed form, so it keeps it.
# Do not "fix" the others to match without checking what it does to the existing
# results; every published run used |dz| with a symmetric set.
SIGNED_DZ_RAY_PATTERNS = frozenset({"theta_phi"})

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
        norm_quantile: bool = True,
        _token: object | None = None,
    ):
        AutoSerialize.__init__(self)
        OptimizerMixin.__init__(self)
        nn.Module.__init__(self)
        if _token is not self._token:
            raise RuntimeError("Use TomographyPixDataset.from_* to instantiate this class.")

        if not (tilt_stack.shape[0] == tilt_angles.shape[0]):
            raise ValueError(
                "The number of tilt projections should be in the first dimension of the dataset."
            )

        if type(tilt_stack) is not torch.Tensor:
            tilt_stack = torch.from_numpy(tilt_stack)
        if type(tilt_angles) is not torch.Tensor:
            tilt_angles = torch.from_numpy(tilt_angles)
        if norm_quantile:
            max_val = torch.quantile(tilt_stack, 0.95)
            if max_val == 0:
                max_val = torch.max(tilt_stack)
                if max_val == 0:
                    print("The maximum value of the tilt series is zero.")
                    max_val = 1
        else:
            max_val = torch.max(tilt_stack)
            if max_val == 0:
                print("The maximum value of the tilt series is zero.")
                max_val = 1

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
        norm_quantile: bool = True,
    ):
        return cls(
            tilt_stack=tilt_stack,
            tilt_angles=tilt_angles,
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
            norm_quantile=norm_quantile,
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
        norm_quantile: bool = True,
        _token: object | None = None,
    ):
        super().__init__(
            tilt_stack=tilt_stack,
            tilt_angles=-tilt_angles,  # TODO: Flip the tilt angles to be negative to match the convention of INR.
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
            norm_quantile=norm_quantile,
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
        norm_quantile: bool = True,
        seed: int = 42,
        _token: object | None = None,
    ):
        super().__init__(
            tilt_stack,
            tilt_angles,
            learn_shift,
            learn_tilt_axis,
            norm_quantile,
            _token=_token,
        )

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
        # tilt image's own per-axis pixel count -- see create_batch_rays for why this
        # must be decoupled from the volume-derived N.
        Nx_img = self.tilt_stack.shape[2]
        Ny_img = self.tilt_stack.shape[1]
        with torch.no_grad():
            batch_ray_coords = self.create_batch_rays(
                pixel_i, pixel_j, N, num_samples_per_ray, Nx=Nx_img, Ny=Ny_img
            )

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
            Nx=Nx_img,
            Ny=Ny_img,
        )
        all_coords = transformed_rays.view(-1, 3)

        all_coords = all_coords.to(self.device, dtype=torch.float32, non_blocking=True)
        return all_coords

    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def create_batch_rays(
        pixel_i: torch.Tensor,
        pixel_j: torch.Tensor,
        N: int,
        num_samples_per_ray: int,
        Nx: int | None = None,
        Ny: int | None = None,
    ) -> torch.Tensor:
        # Lateral pixel-position normalization uses the tilt image's OWN per-axis
        # pixel count (Nx, Ny), not the volume-derived N -- these only coincided for
        # square-image/cubic-volume data. Nx/Ny default to N for backwards
        # compatibility with any caller that hasn't been updated to pass them.
        Nx = N if Nx is None else Nx
        Ny = N if Ny is None else Ny
        batch_size = len(pixel_i)
        x_coords = (pixel_j / (Nx - 1)) * 2 - 1
        y_coords = (pixel_i / (Ny - 1)) * 2 - 1
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
        Nx: int | None = None,
        Ny: int | None = None,
    ) -> torch.Tensor:
        # Shift correction is a sub-pixel pose offset in the same lateral units
        # as x_coords/y_coords (create_batch_rays), so it must normalize by the
        # tilt image's own per-axis pixel count (Nx, Ny), not the volume-derived N.
        Nx = N if Nx is None else Nx
        Ny = N if Ny is None else Ny
        shift_x_norm = (shifts[:, 0:1] * sampling_rate * 2) / (Nx - 1)
        shift_y_norm = (shifts[:, 1:2] * sampling_rate * 2) / (Ny - 1)

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

        pixel_i = remaining // self.tilt_stack.shape[2]
        pixel_j = remaining % self.tilt_stack.shape[2]

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
        return self.tilt_stack.shape[0] * self.tilt_stack.shape[1] * self.tilt_stack.shape[2]

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
        random_method: str = "s",
    ):
        if device is None:
            device = torch.device("cpu")

        num_rays = int(num_rays)

        # random_rays sampling method
        if random_rays:
            theta = (
                torch.rand(num_rays, device=device) * 2 * torch.pi
            )  # this can stay as uniform sampling
            # phi = torch.rand(num_rays, device=device) * convergence_angle # the original uniform sampling of phi
            if random_method.lower() in ["gaussian", "g"]:
                phi = torch.normal(
                    mean=0.0, std=convergence_angle, size=(num_rays,), device=device
                )

            # let's also do the sinc squared, which might be slower?
            # essentially, torch doesn't have a sinc**2 distribution built in, but we can just make a discrete one ourselves
            elif random_method.lower() in ["sinc", "s"]:
                x = torch.linspace(-0.5, 0.5, 5000, device=device)  # in radians
                pdf = torch.sinc(x) ** 2  #
                pdf = pdf / pdf.sum()  # normalize to 1
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
            predicted_values_all_rays = ray_densities.view(
                target_values_len, -1
            )  # this will just be equal weights

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
        random_method: str = "g",
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
        self._z_focuses = torch.zeros(self.learnable_tilts + 1)

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
            return (
                shifts,
                torch.zeros_like(z1),
                torch.zeros_like(z3),
                z_focus,
                self._convergence_angle_params[0],
            )
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3, z_focus, self._convergence_angle_params[0]
        elif self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, self._convergence_angle_params[0]
        else:
            return (
                torch.zeros_like(shifts),
                torch.zeros_like(z1),
                torch.zeros_like(z3),
                z_focus,
                self._convergence_angle_params[0],
            )

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
        random_method: str = "g",
    ):
        if device is None:
            device = torch.device("cpu")

        num_rays = int(num_rays)

        # random_rays sampling method
        if random_rays:
            theta = (
                torch.rand(num_rays, device=device) * 2 * torch.pi
            )  # this can stay as uniform sampling
            # phi = torch.rand(num_rays, device=device) * convergence_angle # the original uniform sampling of phi
            if random_method.lower() in ["gaussian", "g"]:
                phi = torch.randn(num_rays, device=device) * convergence_angle

            # let's also do the sinc squared, which might be slower?
            # essentially, torch doesn't have a sinc**2 distribution built in, but we can just make a discrete one ourselves
            elif random_method.lower() in ["sinc", "s"]:
                x = torch.linspace(-0.5, 0.5, 5000)  # in radians
                pdf = torch.sinc(x) ** 2  #
                pdf = pdf / pdf.sum()  # normalize to 1
                indices = torch.multinomial(pdf, num_rays, replacement=True)
                phi = x[indices]
            elif random_method.lower() in ["probe", "p"]:
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
            raise ValueError("Only random rays supported for learning convergence angle right now")

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
            predicted_values_all_rays = ray_densities.view(
                target_values_len, -1
            )  # this will just be equal weights

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
        random_method: str = "p",
        wavelength_ang: float = 0.0197,
        stig_2: tuple[float, float] = (0.0, 0.0),
        probe_im_shape: tuple[int, int] = (64, 64),
        pixel_size_ang: float = 0.2,
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, _token=token)
        self.num_rays = int(num_rays)

        self._random_rays = True
        self._random_method = random_method

        self._z_focuses = torch.zeros(self.learnable_tilts + 1)

        self._convergence_angle = torch.ones(1) * convergence_angle

        self._stig_2 = torch.tensor(stig_2, dtype=torch.float32)

        self._convergence_angle = torch.ones(1) * convergence_angle
        self._stig_2 = torch.tensor(stig_2, dtype=torch.float32)

        # Initialize as parameters immediately (will be moved to device in .to())
        self._convergence_angle_params = nn.Parameter(self._convergence_angle.clone())
        self._stig_2_params = nn.Parameter(self._stig_2.clone())

        if random_method.lower() in ["probe", "p"]:
            import numpy as np

            self._wavelength_ang = wavelength_ang
            k_max = float(self._convergence_angle.item()) / float(
                wavelength_ang
            )  # convergence angle should be in radians already
            kr = np.fft.fftfreq(probe_im_shape[0], d=pixel_size_ang)[:, None]
            kc = np.fft.fftfreq(probe_im_shape[1], d=pixel_size_ang)[None, :]
            # kr is (H, 1) so kr[1] is a 1-ELEMENT 1-D array, not a scalar.
            # float() on that was deprecated in numpy 2.2 and RAISES in 2.4
            # ("only 0-dimensional arrays can be converted to Python scalars"),
            # so this worked in the conda env (numpy 2.2.6) and died under
            # `ml pytorch/2.11.0` (numpy 2.4.3). Index both axes, exactly as the
            # local `dk` below already did -- identical value, no behaviour change.
            self._dk = torch.tensor(float(kr[1, 0] - kr[0, 0]), dtype=torch.float32)
            dk = float(kr[1, 0] - kr[0, 0])  # scalar float, not a 1-element array
            k1 = np.sqrt(kr**2 + kc**2)
            self._k1 = torch.tensor(k1, dtype=torch.float32)  # radial k-magnitude, shape (H, W)
            aper = np.clip((k_max - k1) / dk + 0.5, 0.0, 1.0)

            # basis
            self._basis_0 = torch.tensor(
                (torch.pi * wavelength_ang) * (kr**2 + kc**2), dtype=torch.float32
            )
            self._basis_1 = torch.tensor(
                (torch.pi * wavelength_ang) * (kr**2 - kc**2), dtype=torch.float32
            )
            self._basis_2 = torch.tensor(
                (torch.pi * wavelength_ang) * (2 * kr * kc), dtype=torch.float32
            )
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
            return (
                shifts,
                torch.zeros_like(z1),
                torch.zeros_like(z3),
                z_focus,
                convergence_angle,
                self._stig_2_params[:],
            )
        elif self.learn_tilt_axis:
            return (
                torch.zeros_like(shifts),
                z1,
                z3,
                z_focus,
                convergence_angle,
                self._stig_2_params[:],
            )
        elif self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, convergence_angle, self._stig_2_params[:]
        else:
            return (
                torch.zeros_like(shifts),
                torch.zeros_like(z1),
                torch.zeros_like(z3),
                z_focus,
                convergence_angle,
                self._stig_2_params[:],
            )

    def get_coords(
        self,
        batch: dict[str, torch.Tensor],
        N: int,
        num_samples_per_ray: int,
        use_probe_weighting: bool = True,  # New flag
        ray_pattern: str = "grid",  # New parameter
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

        shifts, z1_params, z3_params, z_focus_params, convergence_angle, stig_2 = self.forward(
            None
        )
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)
        batch_z_focus = torch.index_select(z_focus_params, 0, projection_indices)

        if use_probe_weighting:
            # New method: uniform/pattern sampling with probe weights
            batch_ray_coords, probe_weights = self.create_batch_rays_uniform(
                pixel_i,
                pixel_j,
                N,
                num_samples_per_ray,
                num_rays,
                z_focus=batch_z_focus,
                stig_2=stig_2,
                convergence_angle=convergence_angle,
                ray_pattern=ray_pattern,
            )
        else:
            # Old method: Gumbel-softmax sampling
            batch_ray_coords = self.create_batch_rays(
                pixel_i,
                pixel_j,
                N,
                num_samples_per_ray,
                num_rays,
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
        random_method: str = "g",
    ):
        if device is None:
            device = torch.device("cpu")

        num_rays = int(num_rays)

        # random_rays sampling method
        if random_rays:
            theta = (
                torch.rand(num_rays, device=device) * 2 * torch.pi
            )  # this can stay as uniform sampling
            # phi = torch.rand(num_rays, device=device) * convergence_angle # the original uniform sampling of phi
            if random_method.lower() in ["gaussian", "g"]:
                phi = torch.randn(num_rays, device=device) * convergence_angle

            # let's also do the sinc squared, which might be slower?
            # essentially, torch doesn't have a sinc**2 distribution built in, but we can just make a discrete one ourselves
            elif random_method.lower() in ["sinc", "s"]:
                x = torch.linspace(-0.5, 0.5, 5000)  # in radians
                pdf = torch.sinc(x) ** 2  #
                pdf = pdf / pdf.sum()  # normalize to 1
                indices = torch.multinomial(pdf, num_rays, replacement=True)
                phi = x[indices]
            elif random_method.lower() in ["probe", "p"]:
                indices = torch.multinomial(self._probe_weights, num_rays, replacement=True)

                # Convert flat indices back to 2D grid positions
                H, W = self._aper.shape
                row_s = (indices // W).float() - H / 2  # centered row coordinate
                col_s = (indices % W).float() - W / 2  # centered col coordinate

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
            raise ValueError("Only random rays supported for this dataset model right now")

        return theta, phi

    def _compute_probe_weights_at_defocus_vectorized(
        self, defocus_ang, stig_2=None, convergence_angle=None
    ):
        if stig_2 is None:
            stig_2 = self._stig_2_params

        # convergence_angle = F.softplus(self._convergence_angle_params[0]) * 45e-3 + 5e-3
        N = defocus_ang.shape[0]

        k_max = (
            convergence_angle / self._wavelength_ang
        )  # convergence angle already in radians, not mrad

        aper = torch.clamp(
            (k_max - self._k1) / self._dk + 0.5,
            0.0,
            1.0,
        )

        chi = (
            self._basis_0[None, :, :] * defocus_ang[:, None, None]
            + self._basis_1[None, :, :] * stig_2[0]
            + self._basis_2[None, :, :] * stig_2[1]
        )

        Psi = aper * torch.exp(-1j * chi)

        psi = torch.fft.ifft2(Psi, dim=(-2, -1))
        psi = torch.fft.fftshift(psi, dim=(-2, -1))

        weights = torch.abs(psi) ** 2

        weights = weights.reshape(N, -1)  # (N, H*W)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize each probe

        return weights  # (N, n_pixels) where n_pixels = H*W

    def differentiable_probe_sample_vectorized(
        self, weights, dx_all, dy_all, num_rays, temperature
    ):
        N, n_positions = weights.shape

        if dx_all.dtype != weights.dtype:
            dx_all = dx_all.to(weights.dtype)
            dy_all = dy_all.to(weights.dtype)

        gumbel_noise = -torch.log(
            -torch.log(
                torch.rand(N, num_rays, n_positions, device=weights.device, dtype=weights.dtype)
                + 1e-10
            )
            + 1e-10
        )

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
        voxel_size_ang: float = 2.0,  # Å per voxel — swap out for experimental data
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
        z_ang = z_coords_norm[None, :] * half_extent_ang  # (1, num_samples_per_ray)
        z_focus_ang = z_focus[:, None] * half_extent_ang  # (batch_size, 1)
        dz_ang = z_ang - z_focus_ang  # (batch_size, num_samples_per_ray)

        H, W = self._aper.shape

        # Precompute probe pixel offset grid — fixed, no grad needed
        row_all = torch.arange(H, device=pixel_i.device).float() - H / 2.0
        col_all = torch.arange(W, device=pixel_i.device).float() - W / 2.0
        row_grid, col_grid = torch.meshgrid(row_all, col_all, indexing="ij")
        row_grid = row_grid.ravel()  # (H*W,)
        col_grid = col_grid.ravel()  # (H*W,)

        probe_pixel_size_ang = 1.0 / (H * self._dk)
        dx_all = row_grid * probe_pixel_size_ang / half_extent_ang  # (H*W,)
        dy_all = col_grid * probe_pixel_size_ang / half_extent_ang

        x_offsets_norm = torch.zeros(
            batch_size, num_samples_per_ray, num_rays, device=pixel_i.device
        )
        y_offsets_norm = torch.zeros(
            batch_size, num_samples_per_ray, num_rays, device=pixel_i.device
        )

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
        z_coords = z_coords_norm[None, :, None].expand(batch_size, num_samples_per_ray, num_rays)

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
        ray_pattern: str = "grid",  # 'uniform', 'hexagonal', 'grid'
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
        if ray_pattern == "uniform":
            # Uniform random sampling in circular aperture
            # Use sqrt for uniform distribution in 2D circle
            angles = torch.rand(num_rays, device=pixel_i.device) * 2 * torch.pi
            radii = torch.sqrt(torch.rand(num_rays, device=pixel_i.device))

            # In angle space (radians)
            alpha_x = radii * torch.cos(angles) * convergence_angle
            alpha_y = radii * torch.sin(angles) * convergence_angle

        elif ray_pattern == "hexagonal":
            # Hexagonal pattern + center
            # num_rays = 7
            if num_rays == 7:
                angles_deg = torch.tensor([0, 60, 120, 180, 240, 300], device=pixel_i.device)
                radius = convergence_angle * 0.7  # 70% of max aperture
                alpha_x = torch.cat(
                    [
                        torch.zeros(1, device=pixel_i.device),
                        radius * torch.cos(torch.deg2rad(angles_deg)),
                    ]
                )
                alpha_y = torch.cat(
                    [
                        torch.zeros(1, device=pixel_i.device),
                        radius * torch.sin(torch.deg2rad(angles_deg)),
                    ]
                )
            else:
                raise ValueError(f"Hexagonal pattern only supports 7 rays, got {num_rays}")

        elif ray_pattern == "grid":
            # Square grid pattern
            grid_size = int(torch.ceil(torch.sqrt(num_rays)))
            x_grid = torch.linspace(
                -convergence_angle, convergence_angle, grid_size, device=pixel_i.device
            )
            y_grid = torch.linspace(
                -convergence_angle, convergence_angle, grid_size, device=pixel_i.device
            )
            grid_x, grid_y = torch.meshgrid(x_grid, y_grid, indexing="ij")
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
        kr_1d = torch.fft.fftfreq(H, d=1.0 / (H * self._dk), device=dz_ang.device)
        kc_1d = torch.fft.fftfreq(W, d=1.0 / (W * self._dk), device=dz_ang.device)

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
        chi = (
            basis_0_grid[None, None, :, :] * dz_expanded
            + basis_1_grid[None, None, :, :] * stig_2[0]
            + basis_2_grid[None, None, :, :] * stig_2[1]
        )

        # Compute Psi in k-space
        Psi = aper[None, None, :, :] * torch.exp(-1j * chi)

        # Flatten k-space - need meshgrid for proper kr, kc pairs
        kr_grid, kc_grid = torch.meshgrid(kr_1d, kc_1d, indexing="ij")  # Both (H, W)
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
            predicted_values_all_rays = ray_densities.view(target_values_len, -1)  # equal weights
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
        random_method: str = "p",
        wavelength_ang: float = 0.0197,
        stig_2: tuple[float, float] = (0.0, 0.0),
        probe_im_shape: tuple[int, int] = (64, 64),
        pixel_size_ang: float = 0.2,
        voxel_size_ang: float = 20.0,
        ray_pattern: str = "clever",
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, _token=token)
        self.num_rays = int(num_rays)
        self.learn_astigmatism = learn_astigmatism
        self.learn_convergence = learn_convergence

        self._random_rays = True
        self._random_method = random_method

        self._z_focuses = torch.zeros(self.learnable_tilts + 1)

        self._convergence_angle = torch.ones(1) * convergence_angle
        self._stig_2 = torch.tensor(stig_2, dtype=torch.float32)

        self.ray_pattern = ray_pattern
        self.voxel_size_ang = voxel_size_ang

        self.save_probe = True
        # Initialize as parameters immediately (will be moved to device in .to())
        if self.learn_convergence:
            self._convergence_angle_params = nn.Parameter(self._convergence_angle.clone())
        else:
            self.register_buffer("_convergence_angle_params", self._convergence_angle.clone())

        if self.learn_astigmatism:
            self._stig_2_params = nn.Parameter(self._stig_2.clone())
        else:
            # Register as buffer (non-trainable but part of state_dict)
            self.register_buffer("_stig_2_params", self._stig_2.clone())

        if random_method.lower() in ["probe", "p"]:
            import numpy as np

            self._wavelength_ang = wavelength_ang
            k_max = float(self._convergence_angle.item()) / float(
                wavelength_ang
            )  # convergence angle should be in radians already
            kr = np.fft.fftfreq(probe_im_shape[0], d=pixel_size_ang)[:, None]
            kc = np.fft.fftfreq(probe_im_shape[1], d=pixel_size_ang)[None, :]
            # kr is (H, 1) so kr[1] is a 1-ELEMENT 1-D array, not a scalar.
            # float() on that was deprecated in numpy 2.2 and RAISES in 2.4
            # ("only 0-dimensional arrays can be converted to Python scalars"),
            # so this worked in the conda env (numpy 2.2.6) and died under
            # `ml pytorch/2.11.0` (numpy 2.4.3). Index both axes, exactly as the
            # local `dk` below already did -- identical value, no behaviour change.
            self._dk = torch.tensor(float(kr[1, 0] - kr[0, 0]), dtype=torch.float32)
            dk = float(kr[1, 0] - kr[0, 0])  # scalar float, not a 1-element array
            k1 = np.sqrt(kr**2 + kc**2)
            self._k1 = torch.tensor(k1, dtype=torch.float32)  # radial k-magnitude, shape (H, W)
            aper = np.clip((k_max - k1) / dk + 0.5, 0.0, 1.0)

            # basis
            self._basis_0 = torch.tensor(
                (torch.pi * wavelength_ang) * (kr**2 + kc**2), dtype=torch.float32
            )
            self._basis_1 = torch.tensor(
                (torch.pi * wavelength_ang) * (kr**2 - kc**2), dtype=torch.float32
            )
            self._basis_2 = torch.tensor(
                (torch.pi * wavelength_ang) * (2 * kr * kc), dtype=torch.float32
            )
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
        random_method: str = "p",
        ray_pattern: str = "clever",
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
            return (
                shifts,
                torch.zeros_like(z1),
                torch.zeros_like(z3),
                z_focus,
                convergence_angle,
                stig_2,
            )
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3, z_focus, convergence_angle, stig_2
        elif self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, convergence_angle, stig_2
        else:
            return (
                torch.zeros_like(shifts),
                torch.zeros_like(z1),
                torch.zeros_like(z3),
                z_focus,
                convergence_angle,
                stig_2,
            )

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

        shifts, z1_params, z3_params, z_focus_params, convergence_angle, stig_2 = self.forward(
            None
        )
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)
        batch_z_focus = torch.index_select(z_focus_params, 0, projection_indices)

        batch_ray_coords, probe_weights = self.create_batch_rays(
            pixel_i,
            pixel_j,
            N,
            num_samples_per_ray,
            num_rays,
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
        random_method: str = "g",
    ):
        if device is None:
            device = torch.device("cpu")

        num_rays = int(num_rays)

        # random_rays sampling method
        if random_rays:
            theta = (
                torch.rand(num_rays, device=device) * 2 * torch.pi
            )  # this can stay as uniform sampling
            # phi = torch.rand(num_rays, device=device) * convergence_angle # the original uniform sampling of phi
            if random_method.lower() in ["gaussian", "g"]:
                phi = torch.randn(num_rays, device=device) * convergence_angle

            # let's also do the sinc squared, which might be slower?
            # essentially, torch doesn't have a sinc**2 distribution built in, but we can just make a discrete one ourselves
            elif random_method.lower() in ["sinc", "s"]:
                x = torch.linspace(-0.5, 0.5, 5000)  # in radians
                pdf = torch.sinc(x) ** 2  #
                pdf = pdf / pdf.sum()  # normalize to 1
                indices = torch.multinomial(pdf, num_rays, replacement=True)
                phi = x[indices]
            elif random_method.lower() in ["probe", "p"]:
                indices = torch.multinomial(self._probe_weights, num_rays, replacement=True)

                # Convert flat indices back to 2D grid positions
                H, W = self._aper.shape
                row_s = (indices // W).float() - H / 2  # centered row coordinate
                col_s = (indices % W).float() - W / 2  # centered col coordinate

                kr_s = row_s * self._dk
                kc_s = col_s * self._dk

                return kr_s * self._wavelength_ang, kc_s * self._wavelength_ang

            else:
                raise ValueError(
                    f"Unsupported random_method={random_method}. "
                    "Supported values are: gaussian, g, sinc, s. .lower() is applied internally."
                )

        else:
            raise ValueError("Only random rays supported for this dataset model right now")

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
            offset_mode="maximize_spacing",
            balance_center=True,
            balance_outer=True,
            weight_mode="nearest",
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
        device = convergence_angle.device if torch.is_tensor(convergence_angle) else "cpu"
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
        kr, kc = torch.meshgrid(kr_1d, kc_1d, indexing="ij")

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
            chi = (
                chi + basis_1[None, None, :, :] * stig_2[0] + basis_2[None, None, :, :] * stig_2[1]
            )

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
            plt.imshow(np.fft.fftshift(intensity[10, 10, :, :].detach().cpu().numpy()))
            plt.savefig("probe_out.png")
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
            intensity_flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
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
        ray_pattern: str = "uniform_angle",  # 'uniform', 'hexagonal', 'grid'
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
            plt.savefig("dz_ang.png")
            # print('z_coords_norm:', z_coords_norm)
            # print('half_extent_ang:', half_extent_ang)

        device = pixel_i.device

        # Generate ray pattern (uniform or structured)
        if ray_pattern == "uniform_angle":
            # Uniform sampling in angle space (spherical coordinates)
            # This generates RAYS (same angle throughout z)

            # Azimuthal: uniform 0 to 2π
            azimuthal = torch.rand(num_rays, device=device) * 2 * torch.pi

            # Altitude: uniform in area (use sqrt)
            altitude = torch.sqrt(torch.rand(num_rays, device=device)) * convergence_angle * 1.2

            # Convert to Cartesian angles
            alpha_x = altitude * torch.cos(azimuthal)
            alpha_y = altitude * torch.sin(azimuthal)

        elif ray_pattern == "uniform_xy":
            # Uniform sampling in x-y plane (Cartesian)
            # This also generates RAYS (same angle throughout z)

            # Random angle and radius
            angles = torch.rand(num_rays, device=device) * 2 * torch.pi
            radii = torch.sqrt(torch.rand(num_rays, device=device))

            # Convert to angular coordinates
            alpha_x = radii * torch.cos(angles) * convergence_angle
            alpha_y = radii * torch.sin(angles) * convergence_angle

        elif ray_pattern == "gaussian_angle":
            # Gaussian sampling in angle space
            # Standard deviation = convergence_angle / 2 (so ~95% within aperture)
            sigma = convergence_angle / 2.0

            # Sample 2D Gaussian
            alpha_x = torch.randn(num_rays, device=device) * sigma
            alpha_y = torch.randn(num_rays, device=device) * sigma

        elif ray_pattern == "grid":
            # Regular grid pattern
            grid_size = int(torch.ceil(torch.sqrt(torch.tensor(num_rays, dtype=torch.float32))))
            x_grid = torch.linspace(
                -convergence_angle, convergence_angle, grid_size, device=device
            )
            y_grid = torch.linspace(
                -convergence_angle, convergence_angle, grid_size, device=device
            )
            grid_x, grid_y = torch.meshgrid(x_grid, y_grid, indexing="ij")
            alpha_x = grid_x.reshape(-1)[:num_rays]
            alpha_y = grid_y.reshape(-1)[:num_rays]

        elif ray_pattern == "hexagonal":
            # Hexagonal pattern (only for 7 rays: 1 center + 6 around)
            if num_rays != 7:
                raise ValueError(f"Hexagonal pattern only supports 7 rays, got {num_rays}")

            angles_deg = torch.tensor(
                [0, 60, 120, 180, 240, 300], device=device, dtype=torch.float32
            )
            radius = convergence_angle * 0.7

            alpha_x = torch.cat(
                [torch.zeros(1, device=device), radius * torch.cos(torch.deg2rad(angles_deg))]
            )
            alpha_y = torch.cat(
                [torch.zeros(1, device=device), radius * torch.sin(torch.deg2rad(angles_deg))]
            )

        elif ray_pattern == "clever":
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
            predicted_values_all_rays = ray_densities.view(target_values_len, -1)  # equal weights
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
            self.register_buffer("_stig_2_params", self._stig_2_params.to(device))

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
        random_method: str = "p",
        wavelength_ang: float = 0.0197,
        stig_2: tuple[float, float] = (0.0, 0.0),
        probe_im_shape: tuple[int, int] = (64, 64),
        pixel_size_ang: float = 0.2,
        voxel_size_ang: float = 20.0,
        ray_pattern: str = "clever",
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, _token=token)
        self.num_rays = int(num_rays)
        self.learn_astigmatism = learn_astigmatism
        self.learn_convergence = learn_convergence
        self.learn_defocus = learn_defocus

        self._random_rays = True
        self._random_method = random_method

        self._z_focus = torch.zeros(self.learnable_tilts + 1)

        self._convergence_angle = torch.ones(1) * convergence_angle
        self._stig_2 = torch.tensor(stig_2, dtype=torch.float32) / self.STIG_SCALE

        self.ray_pattern = ray_pattern
        self.voxel_size_ang = voxel_size_ang

        self.save_probe = True
        # Initialize as parameters immediately (will be moved to device in .to())
        if self.learn_convergence:
            self._convergence_angle_params = nn.Parameter(self._convergence_angle.clone())
        else:
            self.register_buffer("_convergence_angle_params", self._convergence_angle.clone())

        self._clever_k_positions = None
        self._sunflower_positions = None
        self._hexagonal_positions = None
        self._fibonacci_positions = None

        if self.learn_defocus:
            self._z_focus_params = nn.Parameter(self._z_focus.clone())
        else:
            # Register as buffer (non-trainable but part of state_dict)
            self.register_buffer("_z_focus_params", self._z_focus.clone())

        if self.learn_astigmatism:
            self._stig_2_params = nn.Parameter(self._stig_2.clone())
        else:
            # Register as buffer (non-trainable but part of state_dict)
            self.register_buffer("_stig_2_params", self._stig_2.clone())

        if ray_pattern == "clever":
            # Pre-compute k-positions ONCE
            k_positions_np = self._precompute_clever_k_positions(
                num_rays=num_rays,
                convergence_angle=convergence_angle,
                wavelength_ang=wavelength_ang,
            )
            # Store as torch tensor (will be moved to device in .to())
            self._clever_k_positions = torch.from_numpy(k_positions_np).float()

        elif ray_pattern == "sunflower":
            # Sunflower spiral - excellent uniform coverage
            positions_np = self._precompute_sunflower_pattern(
                num_rays=num_rays,
                probe_pixel_count=probe_im_shape[0],
                pixel_size_ang=pixel_size_ang,
            )
            self._sunflower_positions = torch.from_numpy(positions_np).float()

        elif ray_pattern == "hexagonal_grid":
            # Hexagonal close packing - mathematically optimal
            positions_np = self._precompute_hexagonal_grid(
                num_rays=num_rays,
                probe_pixel_count=probe_im_shape[0],
                pixel_size_ang=pixel_size_ang,
            )
            self._hexagonal_positions = torch.from_numpy(positions_np).float()

        elif ray_pattern == "fibonacci":
            # Fibonacci sphere projection
            positions_np = self._precompute_fibonacci_sphere(
                num_rays=num_rays,
                probe_pixel_count=probe_im_shape[0],
                pixel_size_ang=pixel_size_ang,
            )
            self._fibonacci_positions = torch.from_numpy(positions_np).float()

        self._probe_pixel_size_ang = pixel_size_ang
        self._probe_pixel_count = probe_im_shape[0]

        if random_method.lower() in ["probe", "p"]:
            import numpy as np

            self._wavelength_ang = wavelength_ang
            k_max = float(self._convergence_angle.item()) / float(
                wavelength_ang
            )  # convergence angle should be in radians already
            kr = np.fft.fftfreq(probe_im_shape[0], d=pixel_size_ang)[:, None]
            kc = np.fft.fftfreq(probe_im_shape[1], d=pixel_size_ang)[None, :]
            # kr is (H, 1) so kr[1] is a 1-ELEMENT 1-D array, not a scalar.
            # float() on that was deprecated in numpy 2.2 and RAISES in 2.4
            # ("only 0-dimensional arrays can be converted to Python scalars"),
            # so this worked in the conda env (numpy 2.2.6) and died under
            # `ml pytorch/2.11.0` (numpy 2.4.3). Index both axes, exactly as the
            # local `dk` below already did -- identical value, no behaviour change.
            self._dk = torch.tensor(float(kr[1, 0] - kr[0, 0]), dtype=torch.float32)
            dk = float(kr[1, 0] - kr[0, 0])  # scalar float, not a 1-element array
            k1 = np.sqrt(kr**2 + kc**2)
            self._k1 = torch.tensor(k1, dtype=torch.float32)  # radial k-magnitude, shape (H, W)
            aper = np.clip((k_max - k1) / dk + 0.5, 0.0, 1.0)

            # basis
            self._basis_0 = torch.tensor(
                (torch.pi * wavelength_ang) * (kr**2 + kc**2), dtype=torch.float32
            )
            self._basis_1 = torch.tensor(
                (torch.pi * wavelength_ang) * (kr**2 - kc**2), dtype=torch.float32
            )
            self._basis_2 = torch.tensor(
                (torch.pi * wavelength_ang) * (2 * kr * kc), dtype=torch.float32
            )
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
        random_method: str = "p",
        ray_pattern: str = "clever",
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
            offset_mode="maximize_spacing",
            balance_center=True,
            balance_outer=True,
            weight_mode="nearest",
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
            return (
                shifts,
                torch.zeros_like(z1),
                torch.zeros_like(z3),
                z_focus,
                convergence_angle,
                stig_2,
            )
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3, z_focus, convergence_angle, stig_2
        elif self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, convergence_angle, stig_2
        else:
            return (
                torch.zeros_like(shifts),
                torch.zeros_like(z1),
                torch.zeros_like(z3),
                z_focus,
                convergence_angle,
                stig_2,
            )

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

        shifts, z1_params, z3_params, z_focus_params, convergence_angle, stig_2 = self.forward(
            None
        )
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)
        batch_z_focus = torch.index_select(z_focus_params, 0, projection_indices)

        batch_ray_coords, probe_weights = self.create_batch_rays(
            pixel_i,
            pixel_j,
            N,
            num_samples_per_ray,
            num_rays,
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
        random_method: str = "g",
    ):
        if device is None:
            device = torch.device("cpu")

        num_rays = int(num_rays)

        # random_rays sampling method
        if random_rays:
            theta = (
                torch.rand(num_rays, device=device) * 2 * torch.pi
            )  # this can stay as uniform sampling
            # phi = torch.rand(num_rays, device=device) * convergence_angle # the original uniform sampling of phi
            if random_method.lower() in ["gaussian", "g"]:
                phi = torch.randn(num_rays, device=device) * convergence_angle

            # let's also do the sinc squared, which might be slower?
            # essentially, torch doesn't have a sinc**2 distribution built in, but we can just make a discrete one ourselves
            elif random_method.lower() in ["sinc", "s"]:
                x = torch.linspace(-0.5, 0.5, 5000)  # in radians
                pdf = torch.sinc(x) ** 2  #
                pdf = pdf / pdf.sum()  # normalize to 1
                indices = torch.multinomial(pdf, num_rays, replacement=True)
                phi = x[indices]
            elif random_method.lower() in ["probe", "p"]:
                indices = torch.multinomial(self._probe_weights, num_rays, replacement=True)

                # Convert flat indices back to 2D grid positions
                H, W = self._aper.shape
                row_s = (indices // W).float() - H / 2  # centered row coordinate
                col_s = (indices % W).float() - W / 2  # centered col coordinate

                kr_s = row_s * self._dk
                kc_s = col_s * self._dk

                return kr_s * self._wavelength_ang, kc_s * self._wavelength_ang

            else:
                raise ValueError(
                    f"Unsupported random_method={random_method}. "
                    "Supported values are: gaussian, g, sinc, s. .lower() is applied internally."
                )

        else:
            raise ValueError("Only random rays supported for this dataset model right now")

        return theta, phi

    def _setup_clever_rays_pattern(
        self,
        num_rays: int,
        convergence_angle: torch.Tensor,
        defocus: torch.Tensor = None,  # NEW: optional
        stig_2: torch.Tensor = None,  # NEW: optional
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
        kr, kc = torch.meshgrid(kr_1d, kc_1d, indexing="ij")

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
            plt.imshow(np.fft.fftshift(intensity[10, 10, :, :].detach().cpu().numpy()))
            plt.savefig("probe_out.png")
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
            intensity_flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        # Reshape to (batch_size, num_samples, num_rays)
        weights = sampled_intensity.reshape(batch_size, num_samples, num_rays)

        # Normalize weights along ray dimension so they sum to 1
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)

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
        kr, kc = torch.meshgrid(kr_1d, kc_1d, indexing="ij")

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
            plt.savefig("probe_out.png")

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
            intensity_flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        # Reshape to (batch_size, num_samples, num_rays)
        weights = sampled_intensity.reshape(batch_size, num_samples, num_rays)

        # Normalize weights
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
        ray_pattern: str = "uniform_angle",  # 'uniform', 'hexagonal', 'grid'
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
            plt.savefig("dz_ang.png")
            # print('z_coords_norm:', z_coords_norm)
            # print('half_extent_ang:', half_extent_ang)

        device = pixel_i.device

        # Generate ray pattern (uniform or structured)
        if ray_pattern == "uniform_angle":
            # Uniform sampling in angle space (spherical coordinates)
            # This generates RAYS (same angle throughout z)

            # Azimuthal: uniform 0 to 2π
            azimuthal = torch.rand(num_rays, device=device) * 2 * torch.pi

            # Altitude: uniform in area (use sqrt)
            altitude = torch.sqrt(torch.rand(num_rays, device=device)) * convergence_angle * 2

            # Convert to Cartesian angles
            alpha_x = altitude * torch.cos(azimuthal)
            alpha_y = altitude * torch.sin(azimuthal)

        elif ray_pattern == "uniform_xy":
            # Uniform sampling in x-y plane (Cartesian)
            # This also generates RAYS (same angle throughout z)

            # Random angle and radius
            angles = torch.rand(num_rays, device=device) * 2 * torch.pi
            radii = torch.sqrt(torch.rand(num_rays, device=device))

            # Convert to angular coordinates
            alpha_x = radii * torch.cos(angles) * convergence_angle
            alpha_y = radii * torch.sin(angles) * convergence_angle

        elif ray_pattern == "uniform_parallel":
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

        elif ray_pattern == "gaussian_angle":
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

        elif ray_pattern == "hexagonal":
            # Hexagonal pattern (only for 7 rays: 1 center + 6 around)
            if num_rays != 7:
                raise ValueError(f"Hexagonal pattern only supports 7 rays, got {num_rays}")

            angles_deg = torch.tensor(
                [0, 60, 120, 180, 240, 300], device=device, dtype=torch.float32
            )
            radius = convergence_angle * 0.7

            alpha_x = torch.cat(
                [torch.zeros(1, device=device), radius * torch.cos(torch.deg2rad(angles_deg))]
            )
            alpha_y = torch.cat(
                [torch.zeros(1, device=device), radius * torch.sin(torch.deg2rad(angles_deg))]
            )
        elif ray_pattern == "clever":
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

        elif ray_pattern == "grid":
            # Define a real-space grid (in Angstroms) - PARALLEL beams
            grid_size = int(torch.ceil(torch.sqrt(torch.tensor(num_rays, dtype=torch.float32))))
            grid_spacing_ang = (
                self._probe_pixel_count / grid_size * self._probe_pixel_size_ang * 0.75
            )
            grid_extent = grid_spacing_ang * (grid_size - 1) / 2

            # Create grid in real space (Angstroms)
            x_positions = torch.linspace(-grid_extent, grid_extent, grid_size, device=device)
            y_positions = torch.linspace(-grid_extent, grid_extent, grid_size, device=device)
            grid_x, grid_y = torch.meshgrid(x_positions, y_positions, indexing="ij")

            # Flatten and take first num_rays
            r_x_base = grid_x.reshape(-1)[:num_rays]  # (num_rays,) in Angstroms
            r_y_base = grid_y.reshape(-1)[:num_rays]

            # For parallel beams, r is INDEPENDENT of z
            r_x = r_x_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)
            r_y = r_y_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)

        elif ray_pattern == "sunflower":
            # Pre-computed sunflower pattern
            positions = self._sunflower_positions.to(device)
            r_x_base = positions[:, 0]  # (num_rays,)
            r_y_base = positions[:, 1]  # (num_rays,)

            # Parallel beams - constant with z
            r_x = r_x_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)
            r_y = r_y_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)

        elif ray_pattern == "hexagonal_grid":
            # Pre-computed hexagonal grid
            positions = self._hexagonal_positions.to(device)
            r_x_base = positions[:, 0]
            r_y_base = positions[:, 1]

            r_x = r_x_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)
            r_y = r_y_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)

        elif ray_pattern == "fibonacci":
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

        if (
            ray_pattern == "uniform_angle"
            or ray_pattern == "uniform_xy"
            or ray_pattern == "gaussian_angle"
            or ray_pattern == "hexagonal"
        ):
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

        elif (
            ray_pattern == "clever"
            or ray_pattern == "grid"
            or ray_pattern == "uniform_parallel"
            or ray_pattern == "fibonacci"
            or ray_pattern == "hexagonal_grid"
            or ray_pattern == "sunflower"
        ):
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
            predicted_values_all_rays = ray_densities.view(target_values_len, -1)  # equal weights
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
            self._convergence_angle_params = nn.Parameter(
                self._convergence_angle_params.to(device)
            )
        else:
            self.register_buffer(
                "_convergence_angle_params", self._convergence_angle_params.to(device)
            )

        if self.learn_defocus:
            self._z_focus_params = nn.Parameter(self._z_focus_params.to(device))
        else:
            self.register_buffer("_z_focus_params", self._z_focus_params.to(device))

        if self.learn_astigmatism:
            self._stig_2_params = nn.Parameter(self._stig_2_params.to(device))
        else:
            self.register_buffer("_stig_2_params", self._stig_2_params.to(device))

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


class Probe:
    """Probe model with matched wavefunction and discrete ray pictures."""

    def __init__(
        self,
        k_max=1.0,
        pixel_size=0.2,
        im_shape=np.array([256, 256]),
        wavelength=0.0197,
        device="cpu",
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
        target_first_ring = (
            self.k_max * 2 * np.sqrt(np.pi / (total_rays * num_rays * np.tan(np.pi / num_rays)))
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
        first_radius = (
            self.k_max * 2 * np.sqrt(np.pi / (total_rays * num_rays * np.tan(np.pi / num_rays)))
        )
        outer_boundary_midpoint = self.k_max * np.sqrt(1.0 - (num_rings * num_rays) / total_rays)
        slope = (outer_boundary_midpoint - first_radius) / (num_rings - 1.5)
        intercept = first_radius - slope
        return slope * np.arange(1, num_rings + 1) + intercept

    def define_rays(
        self,
        num_rings=4,
        num_rays=6,
        offset_mode="maximize_spacing",
        offset_samples=720,
        balance_center=True,
        balance_outer=True,
        weight_mode="nearest",
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
            k_radii = (
                self.k_max * 2 * (np.arange(1, num_rings + 1) / (2 * num_rings + 1))
                + radius_offset
            )
        else:
            k_radii = self.k_max * 2 * (np.arange(1, num_rings + 1) / (2 * num_rings + 1))

        for a0 in range(num_rings):
            k_radius = k_radii[a0]
            num_ring_rays = (a0 + 1) * num_rays
            phi = 2 * np.pi / num_ring_rays

            if offset_mode == "alternate":
                dphi = phi * np.mod(a0, 2) / 2
            elif offset_mode == "golden":
                golden = (np.sqrt(5.0) - 1.0) / 2.0
                dphi = phi * np.mod((a0 + 1) * golden, 1.0)
            elif offset_mode == "maximize_spacing":
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
                k.append(
                    [
                        k_radius * np.cos(phi * a1 + dphi),
                        k_radius * np.sin(phi * a1 + dphi),
                    ]
                )

        self.k = np.array(k)
        self.k_px = self.k / self.dk + np.array(self.im_shape) // 2
        self.k_weights_raw, self.k_weights = self._compute_k_weights(
            weight_mode=weight_mode, fractional_power=fractional_power
        )

    def _compute_k_weights(self, weight_mode="nearest", fractional_power=2.0):
        weight_mode = weight_mode.lower()
        if weight_mode not in ("nearest", "fractional"):
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

        if weight_mode == "nearest":
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
            ring_k = np.column_stack(
                (
                    k_radius * np.cos(angles),
                    k_radius * np.sin(angles),
                )
            )

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


# =============================================================================
# Learnable thickness dropoff
#
# Moved here from quantem/tomography/dropoff.py on 2026-08-10 so the package
# does not carry a separate module for it. Only TomographyFocalINRDataset uses
# these; the rest of this file is unaffected.
#
#
# The linear forward model assumes collected HAADF intensity is proportional to the
# line integral of the object. It is not: as the beam progresses, electrons already
# scattered are no longer available to scatter again, and the signal saturates.
#
# WHY A FREE-FORM MONOTONE SPLINE AND NOT A BEER-LAMBERT EXPONENTIAL.
# Van den Broek et al., Ultramicroscopy 116 (2012) 8-12, Eq. (4), give
#
#     I = I0 (1 - exp(-mu t))
#
# which is a one-parameter saturating form. Multislice measurement of an 80 nm Au
# slab (300 kV, 20 mrad, 50-100 mrad detector; see
# execution_py/20260806_haadf_depletion_slab.py) shows the real curve is not of
# that family. Signal per unit thickness RISES before it falls:
#
#     I/t at 1, 3, 6, 11, 30, 80 nm = 6.22  7.13  8.00  7.49  5.48  3.06 (1e-4/A)
#
# Beer-Lambert requires I/t to decrease monotonically from t = 0. Fitting it anyway
# invents a thin-limit tangent of 8.7e-4 /A where the measurement at 10 A is 6.2e-4
# and a compensating alpha ~ 3.9, and the resulting curve is refuted by the
# au_disc_12nm data. What IS true is that I is monotone increasing in thickness
# even though I/t is not -- so monotonicity is the correct and only shape
# constraint, and the spline supplies the rest.
#
# THE GAUGE. object -> c*object with g(p) -> g(p/c) is an EXACT symmetry of the
# forward model. No regulariser breaks it: smoothness is equally happy with either.
# Pinning g'(0) = 1 is correct in theory and vacuous in practice -- the optimiser
# satisfies it on a first knot interval that no ray samples and then decouples
# (three trials landed the object 1.7x, 18x and 39x off). The scale must be
# anchored on the OBJECT, against an independently known material response; see
# ``anchor_loss``. In simulation that comes from a slab calibration, in experiment
# from a known-composition region or a calibrated incident intensity.
#
# Initialised to the identity, so a model that switches this on starts exactly at
# its current linear behaviour.
#
# PARAMETRIC ALTERNATIVES, ADDED 2026-08-07. The free spline has a known failure
# (README_dropoff.md sec 6): it recovers g well above ~20 nm and badly below,
# because the consistency loss is dominated by high-signal pixels and the thin end
# carries almost no weight. A low-parameter form attacks that directly -- it ties
# the thin end to the well-measured thick end through the functional form, so the
# thin limit is determined even where the data has no say. The cost is bias if the
# family is wrong, so the family was chosen by fitting the slab measurement
# (execution_py/20260807_fit_dropoff_families.py, six concordant runs averaged,
# relative residuals):
#
#     family                np   rms%   rms% t<=10nm   g'(0) 1e-4/A
#     Beer-Lambert           2   5.93       14.74          7.68   <- impossible
#     logistic-through-0     3   4.35       10.50          6.86   <- impossible
#     Gompertz               3   4.09        9.69          6.68   <- impossible
#     Richards               4   4.11        9.76          6.70   <- impossible
#     Weibull                3   2.56        5.65          1.99   <- g'(0)=0
#     Hill                   3   1.16        2.84          0.80   <- g'(0)=0
#     buildup x depletion    4   1.22        2.89          4.08   <- admissible
#
# TWO CONSTRAINTS ELIMINATE EVERYTHING BUT ONE FAMILY.
#
# (a) g'(0) is finite and NONZERO. In the thin limit atoms scatter independently
# and HAADF intensity is strictly proportional to thickness. Hill and Weibull --
# the textbook sigmoids -- have g'(0) = 0 identically. They fit the curve best
# precisely BECAUSE they are free to send the thin-limit slope to zero, which is
# the wrong direction in exactly the regime sec 6 says is already broken.
#
# (b) g'(0) <= the first measured chord. For a function convex on [0, h] the
# average slope over that interval bounds g'(0) from above, and the first chord is
# 5.63e-4 /A. Beer-Lambert, the logistic, Gompertz and Richards all fit g'(0) ABOVE
# it (+37%, +22%, +19%, +19%) -- not a poor fit but a geometric impossibility, the
# same defect sec 2 already caught in the Beer-Lambert fit.
#
# That bound assumes I is convex on [0, 10 A], which is below the measurement's
# sampling and so cannot be read off directly. Two things support it: the chord
# sequence 5.63, 6.42, 7.44, 8.11, 8.36 is convex over the whole measured range up
# to 50 A, and at t -> 0 there is no depletion yet, so the only first-order effect
# is the cross-section buildup, which raises dI/dt. If I were instead concave there
# the bound would flip, but no mechanism produces a FALLING dI/dt at zero
# thickness. Worth revisiting if the amorphous-slab test (sec 9) is ever run with
# finer exit-plane spacing, since that is what would measure it directly.
#
# Related, and not resolved here: the sec 3 anchor mu = 6.216e-4 /A comes from
# au_slab_80nm.npz, which sits 4.7% above the six concordant runs at 800 A and
# 10.5% above them at 10 A. That is self-consistent in simulation, because the
# phantoms are built from the same file, but it is a 10% shift in the gauge for
# real data.
#
# So a plain logistic is the wrong S-curve even though the curve IS S-shaped: the
# data wants a sharp rise over ~4 nm against a 30+ nm tail, and a logistic is
# symmetric about its inflection. ``BuildupDepletionDropoff`` gives the two ends
# independent length scales and is the only family that fits to the measurement
# noise floor while staying physically admissible.
#
# ``SigmoidDropoff`` is kept anyway, as the honest test of the literal
# sigmoid hypothesis and as an ablation.
# =============================================================================


class MonotoneDropoff(nn.Module):
    """Monotone piecewise-linear g mapping a line integral to collected signal.

    Parameters
    ----------
    p_max : float
        Upper knot. Should cover the largest line integral the data produces;
        beyond it the curve extrapolates linearly with the final slope rather
        than clamping, so gradients survive if the object overshoots early in
        training.
    n_knots : int, optional
        Number of knots, by default 32. More knots resolve more structure but
        need more thickness coverage in the data to constrain; pair a large
        value with ``curvature_penalty``.
    """

    def __init__(self, p_max: float, n_knots: int = 32):
        super().__init__()
        if not p_max > 0:
            raise ValueError(f"p_max must be positive, got {p_max}")
        if n_knots < 3:
            raise ValueError(f"n_knots must be at least 3, got {n_knots}")
        self.n_knots = int(n_knots)
        self.register_buffer("p_max", torch.tensor(float(p_max)))
        # softplus(raw) == 1 at raw = log(e - 1), so the curve starts as identity.
        init = torch.log(torch.expm1(torch.ones(1))).item()
        self.raw_slopes = nn.Parameter(torch.full((self.n_knots - 1,), init))

    @property
    def dk(self) -> torch.Tensor:
        return self.p_max / (self.n_knots - 1)

    def slopes(self) -> torch.Tensor:
        """Positive segment slopes -- positivity is what makes g monotone."""
        return F.softplus(self.raw_slopes)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        s = self.slopes()
        dk = self.dk
        # Cumulative knot values, g(0) = 0.
        vals = torch.cat([torch.zeros(1, device=p.device, dtype=p.dtype), torch.cumsum(s * dk, 0)])
        idx = torch.clamp((p / dk).floor().long(), 0, self.n_knots - 2)
        frac = p - idx.to(p.dtype) * dk
        return vals[idx] + frac * s[idx]

    def curvature_penalty(self) -> torch.Tensor:
        """Sum of squared second differences. Discourages knot-to-knot ringing
        where the data is sparse; it does NOT fix the scale gauge."""
        s = self.slopes()
        return (s[1:] - s[:-1]).pow(2).sum()

    @staticmethod
    def anchor_loss(object_plateau: torch.Tensor, known_response: float) -> torch.Tensor:
        """Relative squared error pinning the object's scale.

        ``object_plateau`` is a differentiable estimate of the reconstructed
        value inside known-composition material (e.g. the mean over voxels above
        half the object maximum). ``known_response`` is that material's
        independently measured thin-limit response per voxel -- for Au at 300 kV
        and 50-100 mrad, mu = 6.216e-4 /A times the voxel size.

        This term, not any property of g, is what makes the problem well posed.
        """
        return ((object_plateau - known_response) / known_response) ** 2

    def extra_repr(self) -> str:
        return f"n_knots={self.n_knots}, p_max={float(self.p_max):.4g}"


# WHY THE PARAMETRIC FORMS BELOW ARE PARAMETERISED IN LOG SPACE, AND NOT AS
# softplus(raw) THE WAY MonotoneDropoff'S SLOPES ARE.
#
# The spline's raw parameters are all slopes near 1, so one learning rate suits
# all of them. A parametric curve's are not: it carries an amplitude, a rate and
# two lengths, and under softplus their raw values span 1e-4 to 1e5 for the same
# curve. Adam takes the same absolute step in every direction, so a length
# parameter initialised at 1e3 * p_max needs O(1e5) steps at lr 3e-3 to reach its
# target while the amplitude has long since converged. Measured: fitting the
# measured slab curve with DIRECT supervision -- no tomography, no degeneracy,
# just Adam against the known answer -- reached 27.5 % rms for both forms where
# scipy's curve_fit reaches 1.2 % and 4.2 %. That is not the model failing, it is
# the parameterisation, and inside the recon it would have looked exactly like
# the model failing.
#
# Writing every positive parameter as ``scale * exp(raw)`` makes a fixed step in
# raw a fixed RELATIVE change, so one learning rate is appropriate for all of
# them, and expressing the two lengths in units of p_max makes the init
# independent of the object's scale.

# exp range e^+-8, i.e. ~3000x either side of the init. Generous -- every
# physically fitted value on the slab curve sits within e^+-6 -- and deliberately
# not wider.
_LOG_CLAMP = 8.0

# BuildupDepletionDropoff's closed form is a difference of two terms that both
# scale like mu0*r*Lam while their difference scales like mu0*Lam, so the
# cancellation is a factor of r. When tau >> Lam the two terms additionally
# converge in shape and the loss is worst. Measured over random parameter draws,
# the largest relative dip below monotone scales directly with the cap on r:
#
#     cap on r-1     e^8      e^5      e^4      e^3      e^2
#     worst dip     6e-4     3e-5     1e-5     4e-6     1e-6
#
# The dip COUNT stays at ~4 % of draws for every cap -- that part is 1-ulp
# float32 noise on the flat tail and is unavoidable -- but the magnitude is
# controllable, so r is capped separately and more tightly than everything else.
# e^4 allows r up to 56, which is 24x the value fitted to the slab curve, and
# holds the worst dip at 1e-5 relative. Rewriting the closed form to split off
# the exactly-monotone mu0 term was tried and moved the count only 232 -> 198.
_LOG_CLAMP_R = 4.0


def _exp(raw: torch.Tensor, hi: float = _LOG_CLAMP) -> torch.Tensor:
    """exp with the exponent clamped, so a runaway step cannot produce inf."""
    return torch.exp(torch.clamp(raw, -_LOG_CLAMP, hi))


class SigmoidDropoff(nn.Module):
    """Logistic in the line integral, shifted to pass through the origin.

    ``g(p) = A [s(k(p - p0)) - s(-k p0)] / [1 - s(-k p0)]``

    which reduces exactly to the numerically stable, manifestly-zero-at-origin

    ``g(p) = A (1 - e^{-k p}) s(k(p - p0))``

    -- a Beer-Lambert saturation times a logistic gate that suppresses the thin
    end, which is what makes the S. Written this way it is unconditionally
    monotone (a product of two positive increasing factors) instead of only
    monotone where float32 does not cancel: the literal shifted-logistic form
    divides by ``s(k p0)``, which underflows to zero and produces inf for large
    negative ``k p0`` (161 of 2000 random parameter draws).

    g(0) = 0 and g'(0) = A k s(-k p0), finite and nonzero, which the
    Hill/Weibull/Gompertz sigmoids are not. Inflection near p = p0 when p0 > 0.

    This is the literal "the dropoff should be a sigmoid" hypothesis. On the slab
    measurement it fits to 4.35 % rms and puts g'(0) 22 % ABOVE the first measured
    chord, which is geometrically impossible for a curve that is convex at the
    origin -- see the module docstring. Provided as an ablation against
    ``BuildupDepletionDropoff``; prefer that one for production runs.

    Parameters
    ----------
    p_max : float
        Largest line integral the data is expected to produce. Sets the units the
        parameters are expressed in, not a hard range: the form extrapolates
        smoothly beyond it.

    Notes
    -----
    Init is a mildly saturating curve, NOT the identity -- see the note on
    ``BuildupDepletionDropoff``, which applies here for the same reason.
    """

    def __init__(self, p_max: float):
        super().__init__()
        if not p_max > 0:
            raise ValueError(f"p_max must be positive, got {p_max}")
        self.register_buffer("p_max", torch.tensor(float(p_max)))
        # k in units of 1/p_max, A in units of p_max, p0 in units of p_max, so
        # every raw parameter is dimensionless and O(1).
        # With p0 = 0 the curve is g(p) = A(1-e^{-kp}) s(kp), slope A k / 2 at the
        # origin, so kappa * a = 2 keeps g'(0) = 1. kappa = 1 puts the knee at
        # p_max, giving an 8 % sag by p_max -- saturating enough to be a log-unit
        # or two from a real curve rather than five.

        kappa0 = 1.0
        self.log_k = nn.Parameter(torch.tensor(math.log(kappa0)))
        self.log_A = nn.Parameter(torch.tensor(math.log(2.0 / kappa0)))
        self.p0_rel = nn.Parameter(torch.tensor(0.0))

    def params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(A, k, p0) in the same units as the line integral."""
        pm = self.p_max
        return _exp(self.log_A) * pm, _exp(self.log_k) / pm, self.p0_rel * pm

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        A, k, p0 = self.params()
        return A * (-torch.expm1(-k * p)) * torch.sigmoid(k * (p - p0))

    def curvature_penalty(self) -> torch.Tensor:
        """Zero: a three-parameter form has no knot-to-knot ringing to damp.
        Present so parametric and spline dropoffs are interchangeable."""
        return self.log_k.new_zeros(())

    anchor_loss = staticmethod(MonotoneDropoff.anchor_loss)

    def extra_repr(self) -> str:
        A, k, p0 = (x.detach() for x in self.params())
        return f"A={float(A):.4g}, k={float(k):.4g}, p0={float(p0):.4g}"


class BuildupDepletionDropoff(nn.Module):
    """Cross-section buildup times flux depletion. The recommended form.

    ``g'(p) = mu0 [r - (r-1) e^{-p/tau}] e^{-p/Lam}``

    The bracket rises from ``mu0`` to ``r*mu0`` over ``tau`` -- the effective
    cross-section growing as the beam broadens and multiple elastic scattering
    builds up -- and the exponential depletes the unscattered flux over ``Lam``.
    Their product rises then falls, which is the measured shape, and because the
    two ends carry independent length scales it captures a sharp ~4 nm rise
    against a 30+ nm tail. A logistic, symmetric about its inflection, cannot.

    Integrates in closed form::

        g(p) = mu0 r Lam (1 - e^{-p/Lam}) - mu0 (r-1) L2 (1 - e^{-p/L2})
        L2 = Lam tau / (Lam + tau)

    Monotone unconditionally in exact arithmetic: the bracket is bounded below by
    ``r - (r-1) = 1 > 0`` for any ``r >= 1``. In float32 the closed form can dip
    by up to ~1e-5 relative on the flat tail, from the cancellation described at
    ``_LOG_CLAMP_R``; that is ulp-level and far below anything the recon
    resolves, but it is not exactly zero. And ``g'(0) = mu0 > 0`` exactly, so
    the thin limit is a fitted parameter with a direct physical meaning rather
    than an accident of the tail -- it IS the single-scattering response the
    sec 3 anchor is quoted in.

    Fits the slab measurement to 1.22 % rms (2.89 % below 10 nm), which is the
    run-to-run noise floor, with a physically admissible g'(0).

    Parameters
    ----------
    p_max : float
        Largest line integral the data is expected to produce. Sets the units the
        two lengths are expressed in, not a hard range: the form extrapolates
        smoothly beyond it.

    Notes
    -----
    INIT IS NOT THE IDENTITY, unlike ``MonotoneDropoff``, and deliberately so.
    Identity needs Lam -> inf, and an identity init is not merely far from the
    answer, it is unreachable: measured on the slab curve with DIRECT supervision
    (no tomography, no degeneracy, just Adam against the known answer), an init
    at Lam = 1e3 p_max converged to 27.5 % rms and one at 50 p_max to 4.5 %,
    against scipy's 1.6 %, because the depletion length must cross ~5 log units
    along a narrow curved valley. The identity is the right init for a spline,
    whose parameters are all local slopes near 1; it is a trap for a parametric
    form, whose parameters are global.

    So the init is instead a mildly saturating curve of the RIGHT QUALITATIVE
    SHAPE -- g'(0) = 1, slope rising ~50 % over a short tau, then decaying over
    Lam = p_max -- which sits ~1 log unit from a real depletion curve and still
    only ~8 % below the identity across [0, p_max], so switching the dropoff on
    does not meaningfully disturb a converged linear run.
    """

    def __init__(self, p_max: float):
        super().__init__()
        if not p_max > 0:
            raise ValueError(f"p_max must be positive, got {p_max}")
        self.register_buffer("p_max", torch.tensor(float(p_max)))
        # Lengths in units of p_max; mu0 and r-1 dimensionless.

        self.log_mu0 = nn.Parameter(torch.tensor(0.0))  # mu0 = 1
        self.log_rm1 = nn.Parameter(torch.tensor(math.log(0.5)))  # r = 1.5
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.05)))  # tau = 0.05 p_max
        self.log_Lam = nn.Parameter(torch.tensor(math.log(1.0)))  # Lam = p_max

    def params(self) -> tuple[torch.Tensor, ...]:
        """(mu0, r, tau, Lam), all constrained to the monotone region."""
        pm = self.p_max
        return (
            _exp(self.log_mu0),
            1.0 + _exp(self.log_rm1, hi=_LOG_CLAMP_R),
            _exp(self.log_tau) * pm,
            _exp(self.log_Lam) * pm,
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        mu0, r, tau, Lam = self.params()
        L2 = Lam * tau / (Lam + tau)
        # expm1 keeps Lam*(1 - e^{-p/Lam}) -> p accurate when Lam >> p_max, where
        # the naive difference would cancel to float32 noise.
        return mu0 * r * Lam * (-torch.expm1(-p / Lam)) - mu0 * (r - 1.0) * L2 * (
            -torch.expm1(-p / L2)
        )

    def slope(self, p: torch.Tensor) -> torch.Tensor:
        """g'(p). ``slope(0)`` is mu0, the single-scattering response."""
        mu0, r, tau, Lam = self.params()
        return mu0 * (r - (r - 1.0) * torch.exp(-p / tau)) * torch.exp(-p / Lam)

    def curvature_penalty(self) -> torch.Tensor:
        """Zero: a four-parameter form has no knot-to-knot ringing to damp.
        Present so parametric and spline dropoffs are interchangeable."""
        return self.log_mu0.new_zeros(())

    anchor_loss = staticmethod(MonotoneDropoff.anchor_loss)

    def extra_repr(self) -> str:
        mu0, r, tau, Lam = (x.detach() for x in self.params())
        return (
            f"mu0={float(mu0):.4g}, r={float(r):.4g}, tau={float(tau):.4g}, Lam={float(Lam):.4g}"
        )


class TomographyFocalINRDataset(TomographyINRDataset):
    """
    Dataset class for INR-based tomography that assumes non-parallel illumination condition and uses a (more) mathematically accurate probe for ray weighting.
    Inherits from TomographyINRDataset.

    Previously named ``TomographyThroughFocalINRDataset_0615``; that name is kept
    as an alias at the bottom of this module, so existing scripts and every
    checkpoint written under the old name still load.
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
        random_method: str = "p",
        wavelength_ang: float = 0.0197,
        stig_2: tuple[float, float] = (0.0, 0.0),
        probe_im_shape: tuple[int, int] = (64, 64),
        pixel_size_ang: float = 0.2,
        voxel_size_ang: float = 20.0,
        ray_pattern: str = "clever",
        random_rays: bool = True,
        gumbel_temp: float = 10.0,
        lr_defocus: float | None = None,
        lr_astigmatism: float | None = None,
        norm_quantile: bool = False,
        compute_probe_weights: bool = False,
        learn_dropoff: bool = False,
        dropoff_p_max: float | None = None,
        dropoff_n_knots: int = 32,
    ):
        if ray_pattern not in RAY_PATTERNS:
            raise ValueError(
                f"Unknown ray_pattern {ray_pattern!r}. Expected one of {sorted(RAY_PATTERNS)}."
            )
        # theta_phi draws phi through get_theta_phi, which implements only the
        # gaussian and sinc distributions. The class default random_method="p"
        # (probe) belongs to the probe-weighted patterns and has no meaning here,
        # so reject it now rather than several hundred lines into the first batch.
        if ray_pattern == "theta_phi" and random_method.lower() not in (
            "gaussian",
            "g",
            "sinc",
            "s",
        ):
            raise ValueError(
                f"ray_pattern='theta_phi' needs random_method in "
                f"('gaussian', 'g', 'sinc', 's'), got {random_method!r}. "
                f"The default 'p' (probe) applies to the probe-weighted patterns only."
            )
        super().__init__(
            tilt_stack,
            tilt_angles,
            learn_shift,
            learn_tilt_axis,
            norm_quantile=norm_quantile,
            _token=token,
        )
        self.num_rays = int(num_rays)
        self.learn_astigmatism = learn_astigmatism
        self.learn_convergence = learn_convergence
        self.learn_defocus = learn_defocus
        self.lr_defocus = lr_defocus
        self.lr_astigmatism = lr_astigmatism

        # random_rays=False draws ONE ray set at construction and reuses it every
        # batch, with fixed per-ray weights in integrate_rays. Only meaningful for
        # ray_pattern='theta_phi'; the other patterns are either deterministic
        # already or redraw per batch by construction.
        self._random_rays = bool(random_rays)
        # Gumbel-softmax temperature for ray_pattern="probe_gumbel". 10.0 is the
        # value hardcoded in the class this was ported from; exposed because it
        # trades sampling sharpness against gradient quality and there is no
        # reason to believe one number suits every probe.
        self._gumbel_temp = float(gumbel_temp)
        self._random_method = random_method
        # Kept assignable after construction -- that is how the existing scripts
        # turn weighting on -- but exposed as an argument so a run is described
        # entirely by its constructor call.
        self.compute_probe_weights = compute_probe_weights

        self._z_focus = torch.zeros(self.learnable_tilts + 1)

        self._convergence_angle = torch.ones(1) * convergence_angle
        self._stig_2 = torch.tensor(stig_2, dtype=torch.float32)

        self.ray_pattern = ray_pattern
        self.voxel_size_ang = voxel_size_ang

        self.save_probe = True
        # Initialize as parameters immediately (will be moved to device in .to())
        if self.learn_convergence:
            self._convergence_angle_params = nn.Parameter(self._convergence_angle.clone())
        else:
            self.register_buffer("_convergence_angle_params", self._convergence_angle.clone())

        self._clever_k_positions = None
        self._sunflower_positions = None
        self._hexagonal_positions = None
        self._fibonacci_positions = None

        if self.learn_defocus:
            self._z_focus_params = nn.Parameter(self._z_focus.clone())
        else:
            # Register as buffer (non-trainable but part of state_dict)
            self.register_buffer("_z_focus_params", self._z_focus.clone())

        if self.learn_astigmatism:
            self._stig_2_params = nn.Parameter(self._stig_2.clone())
        else:
            # Register as buffer (non-trainable but part of state_dict)
            self.register_buffer("_stig_2_params", self._stig_2.clone())
        self._probe_pixel_size_ang = pixel_size_ang
        self._probe_pixel_count = probe_im_shape[0]

        if ray_pattern == "clever":
            # Pre-compute k-positions ONCE
            k_positions_np = self._precompute_clever_k_positions(
                num_rays=num_rays,
                convergence_angle=convergence_angle,
                wavelength_ang=wavelength_ang,
            )
            # Store as torch tensor (will be moved to device in .to())
            self._clever_k_positions = torch.from_numpy(k_positions_np).float()
            # _precompute_clever_k_positions never truncates a complete ring, so the
            # actual count can exceed what was requested -- num_rays must track the
            # real array length, or every shape reliant on it elsewhere
            # (create_batch_rays, integrate_rays, ...) mismatches.
            self.num_rays = self._clever_k_positions.shape[0]

        elif ray_pattern == "sunflower":
            # Sunflower spiral - excellent uniform coverage
            self._sunflower_positions = torch.from_numpy(
                self._precompute_sunflower_pattern(
                    num_rays=num_rays,
                    probe_pixel_count=probe_im_shape[0],
                    pixel_size_ang=pixel_size_ang,
                )
            ).float()

        elif ray_pattern == "hexagonal_grid":
            # Hexagonal close packing - mathematically optimal
            self._hexagonal_positions = torch.from_numpy(
                self._precompute_hexagonal_grid(
                    num_rays=num_rays,
                    probe_pixel_count=probe_im_shape[0],
                    pixel_size_ang=pixel_size_ang,
                )
            ).float()

        elif ray_pattern == "fibonacci":
            # Fibonacci sphere projection
            self._fibonacci_positions = torch.from_numpy(
                self._precompute_fibonacci_sphere(
                    num_rays=num_rays,
                    probe_pixel_count=probe_im_shape[0],
                    pixel_size_ang=pixel_size_ang,
                )
            ).float()

        # Probe basis: always built, cheap (a probe_im_shape grid, 64x64 by
        # default). It used to be gated on random_method in ('probe','p'), which
        # silently coupled the RAY PATTERN to the WEIGHTING MODE: theta_phi needs
        # random_method gaussian/sinc, so it could never have probe weights, and
        # _compute_probe_weights_a died on a missing self._aper. The two axes are
        # independent now. Existing probe configs are unaffected -- they took this
        # branch already.
        import numpy as np

        self._wavelength_ang = wavelength_ang
        k_max = float(self._convergence_angle.item()) / float(
            wavelength_ang
        )  # convergence angle should be in radians already
        kr = np.fft.fftfreq(probe_im_shape[0], d=pixel_size_ang)[:, None]
        kc = np.fft.fftfreq(probe_im_shape[1], d=pixel_size_ang)[None, :]
        # kr is (H, 1) so kr[1] is a 1-ELEMENT 1-D array, not a scalar.
        # float() on that was deprecated in numpy 2.2 and RAISES in 2.4
        # ("only 0-dimensional arrays can be converted to Python scalars"),
        # so this worked in the conda env (numpy 2.2.6) and died under
        # `ml pytorch/2.11.0` (numpy 2.4.3). Index both axes, exactly as the
        # local `dk` below already did -- identical value, no behaviour change.
        self._dk = torch.tensor(float(kr[1, 0] - kr[0, 0]), dtype=torch.float32)
        dk = float(kr[1, 0] - kr[0, 0])  # scalar float, not a 1-element array
        k1 = np.sqrt(kr**2 + kc**2)
        self._k1 = torch.tensor(k1, dtype=torch.float32)  # radial k-magnitude, shape (H, W)
        aper = np.clip((k_max - k1) / dk + 0.5, 0.0, 1.0)

        # basis
        self._basis_0 = torch.tensor(
            (torch.pi * wavelength_ang) * (kr**2 + kc**2), dtype=torch.float32
        )
        self._basis_1 = torch.tensor(
            (torch.pi * wavelength_ang) * (kr**2 - kc**2), dtype=torch.float32
        )
        self._basis_2 = torch.tensor(
            (torch.pi * wavelength_ang) * (2 * kr * kc), dtype=torch.float32
        )
        self._aper = torch.tensor(aper, dtype=torch.float32)

        # Learnable thickness dropoff, OFF by default: while this is None every
        # code path is bit-identical to the linear forward model, so existing
        # runs are unaffected. Either pass learn_dropoff=True, or attach any of
        # the curves defined above directly:
        #   dset.dropoff = BuildupDepletionDropoff(p_max=...)
        #
        # A PLAIN ATTRIBUTE, not a property. nn.Module.__setattr__ intercepts any
        # Module-valued assignment before Python's descriptor machinery runs, so
        # a `dropoff` property would silently register the module under
        # _modules["dropoff"] while its getter kept returning the never-updated
        # backing field. Letting nn.Module own the attribute registers it
        # properly and gets state_dict/.to()/.parameters() for free.

        if not self._random_rays:
            # Draw the ray set once and keep it, with equal weights. Focal's
            # integrate_rays has always had an `else` branch reading
            # self._ray_weights, but nothing ever set it -- that branch was
            # unreachable (AttributeError) until this landed.
            if ray_pattern != "theta_phi":
                raise ValueError(
                    f"random_rays=False is only implemented for ray_pattern='theta_phi', "
                    f"got {ray_pattern!r}. The other patterns are either already "
                    f"deterministic or redraw per batch by construction."
                )
            thetas, phis = self.get_theta_phi(
                convergence_angle=convergence_angle,
                num_rays=num_rays,
                random_rays=False,
                random_method=random_method,
            )
            self.thetas, self.phis = thetas, phis
            self._ray_weights = torch.ones(len(phis)) / len(phis)

        self.dropoff: nn.Module | None = None

        # Scale anchor for the dropoff, as (target_plateau, weight). Required
        # whenever a dropoff is active: object -> c*object with g(p) -> g(p/c) is
        # an EXACT symmetry, so without external information the two are only
        # jointly determined and the optimiser is free to wander along that
        # direction. Set with set_dropoff_anchor(); see anchor_value_from_mu()
        # for turning a known material mu into the right target.
        self.dropoff_anchor: tuple[float, float] | None = None

        if learn_dropoff:
            if dropoff_p_max is None:
                raise ValueError(
                    "learn_dropoff=True requires dropoff_p_max: the curve's knots have to span "
                    "the largest line integral the data produces, and nothing here can infer "
                    "that from the tilt series alone."
                )
            self.dropoff = MonotoneDropoff(p_max=dropoff_p_max, n_knots=dropoff_n_knots)
            # Not anchored here: the anchor needs a known material mu and the
            # ray length, which the caller has and this constructor does not.
            # apply_dropoff_anchor warns once if it is still unset at train time.

    @property
    def random_rays(self) -> bool:
        """False means one fixed ray set, drawn at construction."""
        return self._random_rays

    @property
    def convergence_angle(self) -> float:
        """Nominal convergence semi-angle in radians, as passed to __init__."""
        return float(self._convergence_angle.item())

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
        random_method: str = "p",
        ray_pattern: str = "clever",
        random_rays: bool = True,
        gumbel_temp: float = 10.0,
        lr_defocus: float | None = None,
        lr_astigmatism: float | None = None,
        voxel_size_ang: float = 20.0,
        norm_quantile: bool = False,
        compute_probe_weights: bool = False,
        learn_dropoff: bool = False,
        dropoff_p_max: float | None = None,
        dropoff_n_knots: int = 32,
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
            random_rays=random_rays,
            gumbel_temp=gumbel_temp,
            lr_defocus=lr_defocus,
            lr_astigmatism=lr_astigmatism,
            norm_quantile=norm_quantile,
            compute_probe_weights=compute_probe_weights,
            learn_dropoff=learn_dropoff,
            dropoff_p_max=dropoff_p_max,
            dropoff_n_knots=dropoff_n_knots,
        )

    def get_optimization_parameters(self) -> dict[str, list[torch.nn.Parameter]]:
        params = {}

        pose_params = []
        if self.learn_shift:
            pose_params.append(self.shifts_params)
        if self.learn_tilt_axis:
            pose_params.append(self.z1_params)
            pose_params.append(self.z3_params)
        if self.learn_convergence:
            pose_params.append(self._convergence_angle_params)
        if pose_params:
            params["pose"] = pose_params

        if self.learn_defocus:
            params["defocus"] = [self._z_focus_params]

        if self.learn_astigmatism:
            params["astigmatism"] = [self._stig_2_params]

        # This override enumerates named groups rather than sweeping
        # self.parameters(), so anything not listed above is silently never
        # optimized. The dropoff is a submodule and would fall through exactly
        # that way -- the curve would sit at its initialisation for the whole
        # run while the fit still converged, which is close to invisible. It
        # gets its own group for the same reason the base class gives it one:
        # a handful of spline slopes shared across every ray want a much
        # smaller learning rate than the per-tilt pose parameters.
        drop_params = list(self.dropoff.parameters()) if self.dropoff is not None else []
        if drop_params:
            params["dropoff"] = drop_params

        if not params:
            return {self.DEFAULT_OPTIMIZER_KEY: list(self.parameters())}

        return params

    # --- Learnable dropoff ---
    # Lives on this class, not on TomographyDatasetBase: it is experimental and
    # only this dataset supports it. Tomography.reconstruct calls apply_dropoff
    # and apply_dropoff_anchor through hasattr, so the other dataset types are
    # unaffected by their absence.
    def set_dropoff_anchor(self, target_plateau: float, weight: float = 1e2):
        """Pin the object's scale so the dropoff carries only the curve shape."""
        if not target_plateau > 0:
            raise ValueError(f"target_plateau must be positive, got {target_plateau}")
        self.dropoff_anchor = (float(target_plateau), float(weight))

    @staticmethod
    def anchor_value_from_mu(mu_per_angstrom: float, ray_length_angstrom: float) -> float:
        """Object plateau value corresponding to a known material response.

        Rays are parameterised over normalised z in [-1, 1] and integrated as
        ``densities.sum() * 2 / (n_samples - 1)``, so a ray passing entirely
        through uniform material of value rho yields a line integral of 2*rho,
        independent of the sample count. Matching that to the physical
        ``mu * L`` gives ``rho = mu * L / 2``.

        ``ray_length_angstrom`` is the physical length spanned by the full
        normalised [-1, 1] extent, i.e. the object's depth along the ray, not the
        thickness of the sample feature.

        For Au at 300 kV with a 50-100 mrad detector, mu = 6.216e-4 /A, measured
        on an 80 nm slab by multislice (execution_py/20260806_haadf_depletion_slab.py).
        """
        return mu_per_angstrom * ray_length_angstrom / 2.0

    def apply_dropoff_anchor(self, densities: torch.Tensor) -> torch.Tensor:
        """Soft constraint pulling the object's material plateau to a known value.

        Zero unless both a dropoff and an anchor are configured, so it is safe to
        call unconditionally.

        The plateau is estimated from the current batch as the mean of every
        sampled density above half the batch's 99th percentile. The threshold is
        detached, so gradient flows only through the selected values and not
        through the choice of which voxels count as material. Batches that
        contain too little material to estimate a plateau contribute nothing
        rather than a noisy pull -- this matters for sparse phantoms such as
        five_np_ring_small, which is 99% vacuum.
        """
        if self.dropoff is None:
            return torch.zeros((), device=densities.device, dtype=densities.dtype)
        if self.dropoff_anchor is None:
            # Loud, because the failure is silent otherwise: the fit still
            # converges and the object still looks plausible, but object and
            # curve drift along the degenerate direction together. Trials without
            # an anchor ended up 1.7x, 18x and 39x off in scale while fitting the
            # data perfectly well.
            if not getattr(self, "_warned_no_anchor", False):
                warnings.warn(
                    "A learnable dropoff is attached but no scale anchor is set. "
                    "object -> c*object with g(p) -> g(p/c) is an exact symmetry of the "
                    "forward model, so the two are only jointly determined and the "
                    "recovered curve will be wrong by an arbitrary factor. Call "
                    "dset.set_dropoff_anchor(...); anchor_value_from_mu() converts a "
                    "known material mu into the target.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_no_anchor = True
            return torch.zeros((), device=densities.device, dtype=densities.dtype)
        target, weight = self.dropoff_anchor
        flat = densities.reshape(-1)
        with torch.no_grad():
            ref = torch.quantile(flat.float(), 0.99)
            mask = flat > 0.5 * ref
            n_material = int(mask.sum())
        if n_material < 32:
            return torch.zeros((), device=densities.device, dtype=densities.dtype)
        plateau = flat[mask].mean()
        return weight * ((plateau - target) / target) ** 2

    def apply_dropoff(self, integrated: torch.Tensor) -> torch.Tensor:
        """Map a line integral through the learnable dropoff.

        Identity when no dropoff is attached, so this is safe to call
        unconditionally.

        SINGLE RAY ONLY, and deliberately so. The depletion a ray experiences
        depends on that ray's own path, so the correct multi-ray model is
        ``sum_r w_r g(p_r)``. But ``integrate_rays`` and
        ``integrate_rays_with_probe_weights`` both collapse the ray axis before
        returning, so all this method ever receives is ``sum_r w_r p_r`` -- and
        ``g(sum_r w_r p_r) != sum_r w_r g(p_r)`` for any nonlinear g. Rather than
        apply a silently wrong correction, refuse. Implementing the exact form
        means integrating per ray and applying g before the ray combination,
        inside those methods.
        """
        if self.dropoff is None:
            return integrated
        # Lazily co-locate with the data. The dataset's to() implementations move
        # a hand-written list of tensors and set self.device; they never call
        # nn.Module.to, so a registered submodule is left behind on CPU no matter
        # what order the caller attaches it in. Checking one parameter's device
        # per call is cheap and immune to that.
        if next(self.dropoff.parameters()).device != integrated.device:
            self.dropoff.to(integrated.device)
        num_rays = int(getattr(self, "num_rays", 1) or 1)
        if num_rays > 1:
            raise NotImplementedError(
                f"Learnable dropoff is only implemented for a single ray, but num_rays="
                f"{num_rays}. g(sum_r w_r p_r) != sum_r w_r g(p_r), so applying it here "
                f"would be wrong. Use one ray, or extend integrate_rays to apply the "
                f"dropoff per ray before combining them."
            )
        return self.dropoff(integrated)

    # --- theta/phi ray sampling ------------------------------------------
    # Ported from TomographyThroughFocalINRDataset on 2026-08-10, verbatim.
    # Samples rays in spherical angle rather than as alpha_x/alpha_y offsets,
    # which is what the `theta_phi` ray_pattern uses. `random_method` selects
    # the phi distribution: gaussian/g, sinc/s, uniform, or probe/p.

    def get_theta_phi(
        self,
        convergence_angle: float,
        num_rays: int,
        device: torch.device | str | None = None,
        random_rays: bool = False,
        random_method: str = "s",
    ):
        if device is None:
            device = torch.device("cpu")

        num_rays = int(num_rays)

        # random_rays sampling method
        if random_rays:
            theta = (
                torch.rand(num_rays, device=device) * 2 * torch.pi
            )  # this can stay as uniform sampling
            # phi = torch.rand(num_rays, device=device) * convergence_angle # the original uniform sampling of phi
            if random_method.lower() in ["gaussian", "g"]:
                phi = torch.normal(
                    mean=0.0, std=convergence_angle, size=(num_rays,), device=device
                )

            # let's also do the sinc squared, which might be slower?
            # essentially, torch doesn't have a sinc**2 distribution built in, but we can just make a discrete one ourselves
            elif random_method.lower() in ["sinc", "s"]:
                x = torch.linspace(-0.5, 0.5, 5000, device=device)  # in radians
                pdf = torch.sinc(x) ** 2  #
                pdf = pdf / pdf.sum()  # normalize to 1
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

    # --- Probe-class ray layout and wavefunction weighting ------------------
    # Ported verbatim from TomographyThroughFocalProbeINRDataset on 2026-08-10.
    # _setup_clever_rays_pattern builds the `clever_probe` ray set in ANGLE
    # space via the Probe class (the plain `clever` pattern uses the cached
    # k-positions and is RADIAL -- they are different layouts, not two spellings
    # of one). _compute_probe_weights is the third weighting mode: it propagates
    # the actual probe wavefunction instead of the analytic forms in
    # _compute_probe_weights_a / _r. Select it with probe_weight_mode=
    # 'wavefunction'.

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
            offset_mode="maximize_spacing",
            balance_center=True,
            balance_outer=True,
            weight_mode="nearest",
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
        device = convergence_angle.device if torch.is_tensor(convergence_angle) else "cpu"
        alpha_x = torch.tensor(alpha_x, dtype=torch.float32, device=device)
        alpha_y = torch.tensor(alpha_y, dtype=torch.float32, device=device)

        return alpha_x, alpha_y

    # --- Uniform/hex/grid probe-sampled rays --------------------------------
    # Ported from TomographyThroughFocalAstigmatismINRDataset on 2026-08-10.
    # A second ray-generation entry point: create_batch_rays delegates here for
    # the three *_probe patterns. They were called 'uniform'/'hexagonal'/'grid'
    # in the source class, but 'hexagonal' and 'grid' already name DIFFERENT
    # layouts here, so they are suffixed rather than merged.
    #
    # differentiable_probe_sample_vectorized is the reason this path exists:
    # it Gumbel-softmax samples ray POSITIONS from the probe intensity, making
    # ray placement differentiable. Nothing else in this class does that.

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
        ray_pattern: str = "grid_probe",  # 'uniform', 'hexagonal', 'grid'
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
        if ray_pattern == "uniform_probe":
            # Uniform random sampling in circular aperture
            # Use sqrt for uniform distribution in 2D circle
            angles = torch.rand(num_rays, device=pixel_i.device) * 2 * torch.pi
            radii = torch.sqrt(torch.rand(num_rays, device=pixel_i.device))

            # In angle space (radians)
            alpha_x = radii * torch.cos(angles) * convergence_angle
            alpha_y = radii * torch.sin(angles) * convergence_angle

        elif ray_pattern == "hexagonal_probe":
            # Hexagonal pattern + center
            # num_rays = 7
            if num_rays == 7:
                angles_deg = torch.tensor([0, 60, 120, 180, 240, 300], device=pixel_i.device)
                radius = convergence_angle * 0.7  # 70% of max aperture
                alpha_x = torch.cat(
                    [
                        torch.zeros(1, device=pixel_i.device),
                        radius * torch.cos(torch.deg2rad(angles_deg)),
                    ]
                )
                alpha_y = torch.cat(
                    [
                        torch.zeros(1, device=pixel_i.device),
                        radius * torch.sin(torch.deg2rad(angles_deg)),
                    ]
                )
            else:
                raise ValueError(f"Hexagonal pattern only supports 7 rays, got {num_rays}")

        elif ray_pattern == "grid_probe":
            # Square grid pattern
            # torch.sqrt on a plain int raises TypeError; the source class had this
            # bug too, which is how its 'grid' branch went unnoticed -- it could never
            # have run. Same idiom as every other grid branch in this file.
            grid_size = int(torch.ceil(torch.sqrt(torch.tensor(num_rays, dtype=torch.float32))))
            x_grid = torch.linspace(
                -convergence_angle, convergence_angle, grid_size, device=pixel_i.device
            )
            y_grid = torch.linspace(
                -convergence_angle, convergence_angle, grid_size, device=pixel_i.device
            )
            grid_x, grid_y = torch.meshgrid(x_grid, y_grid, indexing="ij")
            alpha_x = grid_x.ravel()[:num_rays]
            alpha_y = grid_y.ravel()[:num_rays]

        else:
            raise ValueError(f"Unknown ray_pattern: {ray_pattern}")

        # Compute probe weights for each defocus slice
        # Shape: (batch_size, num_samples_per_ray, num_rays)
        # Gated, unlike the source class which always computed them: the target
        # returns None when compute_probe_weights is off, and integrate_rays vs
        # integrate_rays_with_probe_weights is selected on that.
        if self.compute_probe_weights:
            probe_weights = self._compute_probe_weights_for_rays(
                dz_ang, alpha_x, alpha_y, stig_2, convergence_angle
            )
        else:
            probe_weights = None

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
        kr_1d = torch.fft.fftfreq(H, d=1.0 / (H * self._dk), device=dz_ang.device)
        kc_1d = torch.fft.fftfreq(W, d=1.0 / (W * self._dk), device=dz_ang.device)

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
        chi = (
            basis_0_grid[None, None, :, :] * dz_expanded
            + basis_1_grid[None, None, :, :] * stig_2[0]
            + basis_2_grid[None, None, :, :] * stig_2[1]
        )

        # Compute Psi in k-space
        Psi = aper[None, None, :, :] * torch.exp(-1j * chi)

        # Flatten k-space - need meshgrid for proper kr, kc pairs
        kr_grid, kc_grid = torch.meshgrid(kr_1d, kc_1d, indexing="ij")  # Both (H, W)
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

        # Normalize
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-10)

        return weights

    def _compute_probe_weights_at_defocus_vectorized(
        self, defocus_ang, stig_2=None, convergence_angle=None
    ):
        if stig_2 is None:
            stig_2 = self._stig_2_params

        # convergence_angle = F.softplus(self._convergence_angle_params[0]) * 45e-3 + 5e-3
        N = defocus_ang.shape[0]

        k_max = (
            convergence_angle / self._wavelength_ang
        )  # convergence angle already in radians, not mrad

        aper = torch.clamp(
            (k_max - self._k1) / self._dk + 0.5,
            0.0,
            1.0,
        )

        chi = (
            self._basis_0[None, :, :] * defocus_ang[:, None, None]
            + self._basis_1[None, :, :] * stig_2[0]
            + self._basis_2[None, :, :] * stig_2[1]
        )

        Psi = aper * torch.exp(-1j * chi)

        psi = torch.fft.ifft2(Psi, dim=(-2, -1))
        psi = torch.fft.fftshift(psi, dim=(-2, -1))

        weights = torch.abs(psi) ** 2

        weights = weights.reshape(N, -1)  # (N, H*W)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize each probe

        return weights  # (N, n_pixels) where n_pixels = H*W

    def differentiable_probe_sample_vectorized(
        self, weights, dx_all, dy_all, num_rays, temperature
    ):
        N, n_positions = weights.shape

        if dx_all.dtype != weights.dtype:
            dx_all = dx_all.to(weights.dtype)
            dy_all = dy_all.to(weights.dtype)

        gumbel_noise = -torch.log(
            -torch.log(
                torch.rand(N, num_rays, n_positions, device=weights.device, dtype=weights.dtype)
                + 1e-10
            )
            + 1e-10
        )

        log_weights = torch.log_softmax(weights, dim=-1)
        logits = log_weights[:, None, :] + gumbel_noise
        soft_samples = torch.softmax(logits / temperature, dim=-1)

        dx_norm = torch.matmul(soft_samples, dx_all)
        dy_norm = torch.matmul(soft_samples, dy_all)

        return dx_norm, dy_norm

    # --- Gumbel-sampled probe rays ------------------------------------------
    # Ported from TomographyThroughFocalAstigmatismINRDataset.create_batch_rays
    # on 2026-08-10, renamed because the target already has a create_batch_rays.
    # This is the caller of _compute_probe_weights_at_defocus_vectorized and
    # differentiable_probe_sample_vectorized: rather than weighting fixed rays,
    # it SAMPLES ray positions from the probe intensity through a Gumbel-softmax,
    # so the placement itself carries gradient. Exposed as ray_pattern=
    # 'probe_gumbel'. It takes no sub-pattern -- the layout is the sampling.

    def create_batch_rays_gumbel(
        self,
        pixel_i: torch.Tensor,
        pixel_j: torch.Tensor,
        N: int,
        num_samples_per_ray: int,
        num_rays: int,
        z_focus: torch.Tensor,
        stig_2: torch.Tensor,
        convergence_angle: torch.Tensor,
        voxel_size_ang: float = 2.0,  # Å per voxel — swap out for experimental data
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        z_ang = z_coords_norm[None, :] * half_extent_ang  # (1, num_samples_per_ray)
        z_focus_ang = z_focus[:, None] * half_extent_ang  # (batch_size, 1)
        dz_ang = z_ang - z_focus_ang  # (batch_size, num_samples_per_ray)

        H, W = self._aper.shape

        # Precompute probe pixel offset grid — fixed, no grad needed
        row_all = torch.arange(H, device=pixel_i.device).float() - H / 2.0
        col_all = torch.arange(W, device=pixel_i.device).float() - W / 2.0
        row_grid, col_grid = torch.meshgrid(row_all, col_all, indexing="ij")
        row_grid = row_grid.ravel()  # (H*W,)
        col_grid = col_grid.ravel()  # (H*W,)

        probe_pixel_size_ang = 1.0 / (H * self._dk)
        dx_all = row_grid * probe_pixel_size_ang / half_extent_ang  # (H*W,)
        dy_all = col_grid * probe_pixel_size_ang / half_extent_ang

        x_offsets_norm = torch.zeros(
            batch_size, num_samples_per_ray, num_rays, device=pixel_i.device
        )
        y_offsets_norm = torch.zeros(
            batch_size, num_samples_per_ray, num_rays, device=pixel_i.device
        )

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
        z_coords = z_coords_norm[None, :, None].expand(batch_size, num_samples_per_ray, num_rays)

        # Flatten rays dimension: (batch_size, num_samples_per_ray * num_rays, 3)
        x_coords = x_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        y_coords = y_coords.reshape(batch_size, num_samples_per_ray * num_rays)
        z_coords = z_coords.reshape(batch_size, num_samples_per_ray * num_rays)

        rays = torch.stack([x_coords, y_coords, z_coords], dim=2)
        # The source returned bare rays: its probe weighting is baked into the ray
        # POSITIONS by the Gumbel sampler above, not applied as a separate weight
        # vector. None keeps the (rays, probe_weights) contract and routes the
        # caller to plain integrate_rays, which is the correct combiner here.
        return rays, None

    def _precompute_clever_k_positions(self, num_rays, convergence_angle, wavelength_ang):
        """Compute k-space ray positions once (slow, but only happens once)."""
        import numpy as np  # Only needed here

        k_max = convergence_angle / wavelength_ang

        # Create Probe and define rays (SLOW - but only once!)
        temp_probe = Probe(
            k_max=k_max,
            pixel_size=self._probe_pixel_size_ang,
            im_shape=(self._probe_pixel_count, self._probe_pixel_count),
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
            offset_mode="maximize_spacing",
            balance_center=True,
            balance_outer=True,
            weight_mode="nearest",
        )

        # Extract k-positions (numpy)
        k_positions = temp_probe.k  # Shape: (n_rays, 2)

        # NEVER truncate a complete ring: ray generation order is a fixed angular
        # sweep, so slicing k_positions[:num_rays] doesn't uniformly subsample the
        # outer ring -- it chops off one contiguous arc of it, turning a symmetric
        # hexagonal illumination cone into an asymmetric one with a bite taken out
        # of one side. The ring packing (1 center + 6*num_rings*(num_rings+1)/2 per
        # full hexagonal shell) only lands exactly on the requested num_rays for the
        # "hexagonal numbers" (1, 7, 19, 37, 61, ...); for anything else we keep the
        # full ring (using slightly MORE rays than requested) rather than drop any.
        # Undershoot (fewer positions than requested) is a separate, genuine edge
        # case caused by aperture-radius clipping in Probe.define_rays, not ring
        # rounding -- that case still gets padded with dummy on-axis rays, and is
        # flagged loudly since it also changes the effective ray count.
        actual_num_rays = k_positions.shape[0]
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if actual_num_rays == num_rays:
                print(
                    f"  [clever rays] requested={num_rays} -> num_rings={num_rings}, exact match"
                )
            elif actual_num_rays > num_rays:
                print(
                    f"  [clever rays] requested={num_rays} -> num_rings={num_rings} gives "
                    f"a complete ring of {actual_num_rays} positions -- using all "
                    f"{actual_num_rays} (kept whole, not truncated, to stay symmetric)"
                )
            else:
                print(
                    f"  [clever rays] requested={num_rays} -> num_rings={num_rings} gives "
                    f"only {actual_num_rays} positions after aperture clipping, padded with "
                    f"{num_rays - actual_num_rays} dummy on-axis (k=0,0) rays"
                )
        if actual_num_rays < num_rays:
            padding = np.zeros((num_rays - actual_num_rays, 2))
            k_positions = np.vstack([k_positions, padding])

        return k_positions  # Return as numpy (will be converted in __init__); caller
        # must set num_rays = len(k_positions), NOT the originally requested value,
        # since that length is now the actual (possibly larger) ray count.

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
            return (
                shifts,
                torch.zeros_like(z1),
                torch.zeros_like(z3),
                z_focus,
                convergence_angle,
                stig_2,
            )
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3, z_focus, convergence_angle, stig_2
        elif self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3, z_focus, convergence_angle, stig_2
        else:
            return (
                torch.zeros_like(shifts),
                torch.zeros_like(z1),
                torch.zeros_like(z3),
                z_focus,
                convergence_angle,
                stig_2,
            )

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

        shifts, z1_params, z3_params, z_focus_params, convergence_angle, stig_2 = self.forward(
            None
        )
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)
        batch_z_focus = torch.index_select(z_focus_params, 0, projection_indices)

        batch_ray_coords, probe_weights = self.create_batch_rays(
            pixel_i,
            pixel_j,
            N,
            num_samples_per_ray,
            num_rays,
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
            Nx=self.tilt_stack.shape[2],
            Ny=self.tilt_stack.shape[1],
        )

        all_coords = transformed_rays.view(-1, 3)

        return all_coords, probe_weights

    # --- Ray-pattern generators -------------------------------------------
    # Ported verbatim from TomographyThroughFocalJustStigINRDataset on
    # 2026-08-10. create_batch_rays and the probe-weight dispatch below
    # already named these patterns; only the generators were missing, so the
    # branches were unreachable and self._compute_probe_weights_a raised
    # AttributeError if it was ever reached.

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
        kr, kc = torch.meshgrid(kr_1d, kc_1d, indexing="ij")

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
        # Short-circuit when there is no astigmatism, adopted from the duplicate
        # implementation in TomographyThroughFocalProbeINRDataset (removed
        # 2026-08-10 -- the two were the same algorithm and this guard was their
        # only difference). Bit-identical: it only skips adding zeros.
        if torch.any(stig_2 != 0):
            chi = (
                chi + basis_1[None, None, :, :] * stig_2[0] + basis_2[None, None, :, :] * stig_2[1]
            )

        # Aberrated wave function in k-space
        # Shape: (batch_size, num_samples, H, W)
        # print('chi shape:',chi.shape)
        psi_k = aperture[None, None, :, :] * torch.exp(-1j * chi)

        # Transform to real space
        psi_r = torch.fft.ifft2(psi_k)
        intensity = torch.abs(psi_r) ** 2
        # (A one-shot matplotlib dump of the probe intensity lived here in the
        # class this was ported from. Dropped 2026-08-10: it wrote probe_out.png
        # into the run's working directory, imported matplotlib inside a
        # numerical method, and indexed intensity[10, 10], which raises for any
        # batch or sample count below 11.)
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
            intensity_flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        # Reshape to (batch_size, num_samples, num_rays)
        weights = sampled_intensity.reshape(batch_size, num_samples, num_rays)

        # Normalize weights along ray dimension so they sum to 1
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)

        return weights

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
        kr, kc = torch.meshgrid(kr_1d, kc_1d, indexing="ij")

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
            intensity_flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        # Reshape to (batch_size, num_samples, num_rays)
        weights = sampled_intensity.reshape(batch_size, num_samples, num_rays)

        # Normalize weights
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
        ray_pattern: str = "uniform_angle",  # 'uniform', 'hexagonal', 'grid'
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Create rays with uniform/pattern sampling and return both rays and probe weights.

        Returns:
            rays: (batch_size, num_samples_per_ray * num_rays, 3)
            probe_weights: (batch_size, num_samples_per_ray, num_rays) - weights for each ray at each slice
        """
        if ray_pattern in GUMBEL_RAY_PATTERNS:
            return self.create_batch_rays_gumbel(
                pixel_i,
                pixel_j,
                N,
                num_samples_per_ray,
                num_rays,
                z_focus=z_focus,
                stig_2=stig_2,
                convergence_angle=convergence_angle,
                voxel_size_ang=voxel_size_ang,
            )

        if ray_pattern in PROBE_UNIFORM_RAY_PATTERNS:
            # Separate generator + weighting; same (rays, probe_weights) contract
            # and the same (batch, num_samples, num_rays) ordering as below.
            return self.create_batch_rays_uniform(
                pixel_i,
                pixel_j,
                N,
                num_samples_per_ray,
                num_rays,
                z_focus=z_focus,
                stig_2=stig_2,
                convergence_angle=convergence_angle,
                voxel_size_ang=voxel_size_ang,
                ray_pattern=ray_pattern,
            )

        batch_size = len(pixel_i)
        # Lateral pixel-position normalization uses the tilt image's OWN per-axis
        # pixel count, not the (volume-derived, depth-only-meaningful) N -- these
        # only coincided for the old square-image/cubic-volume phantom.
        Nx_img = self.tilt_stack.shape[2]  # pixel_j axis (x / rotation axis)
        Ny_img = self.tilt_stack.shape[1]  # pixel_i axis (y axis)
        x_coords_0 = (pixel_j / (Nx_img - 1)) * 2 - 1
        y_coords_0 = (pixel_i / (Ny_img - 1)) * 2 - 1

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

        if ray_pattern == "uniform_angle":
            # Uniform sampling in angle space (spherical coordinates)
            # This generates RAYS (same angle throughout z)

            # Azimuthal: uniform 0 to 2π
            azimuthal = torch.rand(num_rays, device=device) * 2 * torch.pi

            # Altitude: uniform in area (use sqrt)
            altitude = torch.sqrt(torch.rand(num_rays, device=device)) * convergence_angle * 2

            # Convert to Cartesian angles
            alpha_x = altitude * torch.cos(azimuthal)
            alpha_y = altitude * torch.sin(azimuthal)

        elif ray_pattern == "uniform_xy":
            # Uniform sampling in x-y plane (Cartesian)
            # This also generates RAYS (same angle throughout z)

            # Random angle and radius
            angles = torch.rand(num_rays, device=device) * 2 * torch.pi
            radii = torch.sqrt(torch.rand(num_rays, device=device))

            # Convert to angular coordinates
            alpha_x = radii * torch.cos(angles) * convergence_angle
            alpha_y = radii * torch.sin(angles) * convergence_angle

        elif ray_pattern == "uniform_parallel":
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

        elif ray_pattern == "gaussian_angle":
            # Gaussian sampling in angle space
            # Standard deviation = convergence_angle / 2 (so ~95% within aperture)
            sigma = convergence_angle / 2.0

            # Sample 2D Gaussian
            alpha_x = torch.randn(num_rays, device=device) * sigma
            alpha_y = torch.randn(num_rays, device=device) * sigma

        elif ray_pattern == "clever_probe":
            # Probe-class ray layout, in angle space. Distinct from "clever",
            # which reads the cached k-positions and is a RADIAL pattern.
            alpha_x, alpha_y = self._setup_clever_rays_pattern(num_rays, convergence_angle)

        elif ray_pattern == "theta_phi":
            # Spherical-angle sampling. With random_rays=False the set is drawn
            # once in __init__ and reused every batch, which is what makes the
            # fixed _ray_weights path in integrate_rays meaningful; with
            # random_rays=True a fresh set is drawn per batch, as the source did.
            if self._random_rays:
                thetas, phis = self.get_theta_phi(
                    convergence_angle=convergence_angle,
                    num_rays=num_rays,
                    device=device,
                    random_rays=True,
                    random_method=self._random_method,
                )
                self.thetas, self.phis = thetas, phis
            else:
                thetas, phis = self.thetas.to(device), self.phis.to(device)
            alpha_x = torch.tan(phis) * torch.cos(thetas)
            alpha_y = torch.tan(phis) * torch.sin(thetas)

        elif ray_pattern == "hexagonal":
            # Hexagonal pattern (only for 7 rays: 1 center + 6 around)
            if num_rays != 7:
                raise ValueError(f"Hexagonal pattern only supports 7 rays, got {num_rays}")

            angles_deg = torch.tensor(
                [0, 60, 120, 180, 240, 300], device=device, dtype=torch.float32
            )
            radius = convergence_angle * 0.7

            alpha_x = torch.cat(
                [torch.zeros(1, device=device), radius * torch.cos(torch.deg2rad(angles_deg))]
            )
            alpha_y = torch.cat(
                [torch.zeros(1, device=device), radius * torch.sin(torch.deg2rad(angles_deg))]
            )

        elif ray_pattern == "clever":
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

        elif ray_pattern == "grid":
            # Define a real-space grid (in Angstroms) - PARALLEL beams
            grid_size = int(torch.ceil(torch.sqrt(torch.tensor(num_rays, dtype=torch.float32))))
            grid_spacing_ang = (
                self._probe_pixel_count / grid_size * self._probe_pixel_size_ang * 0.75
            )
            grid_extent = grid_spacing_ang * (grid_size - 1) / 2

            # Create grid in real space (Angstroms)
            x_positions = torch.linspace(-grid_extent, grid_extent, grid_size, device=device)
            y_positions = torch.linspace(-grid_extent, grid_extent, grid_size, device=device)
            grid_x, grid_y = torch.meshgrid(x_positions, y_positions, indexing="ij")

            # Flatten and take first num_rays
            r_x_base = grid_x.reshape(-1)[:num_rays]  # (num_rays,) in Angstroms
            r_y_base = grid_y.reshape(-1)[:num_rays]

            # For parallel beams, r is INDEPENDENT of z
            r_x = r_x_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)
            r_y = r_y_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)

        elif ray_pattern in ("sunflower", "hexagonal_grid", "fibonacci"):
            # Pre-computed parallel-beam patterns, laid out in __init__. All three
            # are (num_rays, 2) real-space offsets in Angstroms and are constant
            # with z, so they share one branch.
            positions = {
                "sunflower": self._sunflower_positions,
                "hexagonal_grid": self._hexagonal_positions,
                "fibonacci": self._fibonacci_positions,
            }[ray_pattern]
            if positions is None:
                raise RuntimeError(
                    f"ray_pattern={ray_pattern!r} was requested but its positions were never "
                    f"precomputed. They are built in __init__ from the ray_pattern argument, so "
                    f"pass ray_pattern={ray_pattern!r} to the constructor rather than only here."
                )
            positions = positions.to(device)
            r_x_base = positions[:, 0]  # (num_rays,)
            r_y_base = positions[:, 1]  # (num_rays,)

            # Parallel beams - constant with z
            r_x = r_x_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)
            r_y = r_y_base[None, None, :].expand(batch_size, num_samples_per_ray, num_rays)

        else:
            raise ValueError(f"Unknown ray_pattern: {ray_pattern}")

        # Lateral offsets, and the probe weights if they were asked for.
        #
        # The two families differ in what the ray generator above produced:
        # ANGLE patterns give per-ray angles that fan out with |dz|, RADIAL ones
        # give real-space offsets that are already resolved per z-sample. The
        # membership tests used to be four hand-written or-chains that had
        # already drifted apart from what create_batch_rays actually implements;
        # they are single constants now so the two cannot disagree again.
        if ray_pattern in ANGLE_RAY_PATTERNS:
            # Broadcast: dz_ang is (batch, num_samples), alpha is (num_rays,)
            # Result: (batch, num_samples, num_rays)
            dz_broadcast = dz_ang[:, :, None]  # (batch, num_samples, 1)
            alpha_x_broadcast = alpha_x[None, None, :]  # (1, 1, num_rays)
            alpha_y_broadcast = alpha_y[None, None, :]

            # Offset in normalized space. Signed dz for the patterns listed in
            # SIGNED_DZ_RAY_PATTERNS (rays cross the focus), |dz| for the rest
            # (rays reflect at it) -- see that constant for why both exist.
            _dz = (
                dz_broadcast if ray_pattern in SIGNED_DZ_RAY_PATTERNS else torch.abs(dz_broadcast)
            )
            x_offsets_norm = alpha_x_broadcast * _dz / half_extent_ang
            y_offsets_norm = alpha_y_broadcast * _dz / half_extent_ang

            if self.compute_probe_weights:
                probe_weights = self._compute_probe_weights_a(
                    dz_ang, alpha_x, alpha_y, stig_2, convergence_angle
                )

        elif ray_pattern in RADIAL_RAY_PATTERNS:
            x_offsets_norm = r_x / half_extent_ang
            y_offsets_norm = r_y / half_extent_ang

            if self.compute_probe_weights:
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
            predicted_values_all_rays = ray_densities.view(target_values_len, -1)  # equal weights
        else:
            predicted_values_all_rays = (ray_densities @ self._ray_weights.view(-1, 1)).squeeze(-1)
        step_size = 2.0 / (num_samples_per_ray - 1)
        predicted_values = predicted_values_all_rays.sum(dim=1) * step_size / num_rays

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
        Nx: int | None = None,
        Ny: int | None = None,
    ) -> torch.Tensor:
        # Shift correction is a sub-pixel pose offset in the same lateral units
        # as x_coords_0/y_coords_0 (create_batch_rays), so it must normalize by
        # the tilt image's own per-axis pixel count (Nx, Ny), not the
        # volume-derived N -- see create_batch_rays for the full rationale.
        # Nx/Ny default to N for backwards compatibility with any caller that
        # hasn't been updated to pass them.
        Nx = N if Nx is None else Nx
        Ny = N if Ny is None else Ny
        shift_x_norm = (shifts[:, 0:1] * sampling_rate * 2) / (Nx - 1)
        shift_y_norm = (shifts[:, 1:2] * sampling_rate * 2) / (Ny - 1)

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
            self.register_buffer("_z_focus_params", self._z_focus_params.to(device))

        if self.learn_astigmatism:
            self._stig_2_params = nn.Parameter(self._stig_2_params.to(device))
        else:
            self.register_buffer("_stig_2_params", self._stig_2_params.to(device))

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


# Backwards-compatible alias. TomographyFocalINRDataset was called
# TomographyThroughFocalINRDataset_0615 until 2026-08-10, and is imported under
# that name by the existing recon scripts. AutoSerialize resolves a checkpoint's
# stored class through getattr(module, class_name) (serialize.py:507), so this
# alias is also what keeps every checkpoint written before the rename loadable.
# Both names refer to the same class object; prefer the new one in new code.
TomographyThroughFocalINRDataset_0615 = TomographyFocalINRDataset

# Second alias, and an unhappy one. quantem is installed EDITABLE against this
# checkout, so the sweep running on 2026-08-10 picked up the rename mid-flight.
# Between roughly 14:00 and 15:30 the class was briefly called
# TomographyFocalSeriesINRDataset, and job 56577581's `aa_so3lr1e-3_e28` cell
# saved under that name at 15:23. Without this line those two zips
# (final_state.zip, tomo.zip) do not load at all. Keep it until that cell is
# rescored or rerun, then it can go.
TomographyFocalSeriesINRDataset = TomographyFocalINRDataset

DatasetModelType = (
    TomographyINRDataset
    | TomographyPixDataset
    | TomographyThroughFocalINRDataset
    | TomographyThroughFocalConvergenceINRDataset
    | TomographyThroughFocalAstigmatismINRDataset
    | TomographyThroughFocalProbeINRDataset
    | TomographyThroughFocalJustStigINRDataset
    | TomographyFocalINRDataset
)
