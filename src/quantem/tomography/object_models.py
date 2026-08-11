import os
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm.auto import tqdm

from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.constraints import BaseConstraints, Constraints
from quantem.core.ml.ddp import DDPMixin
from quantem.core.ml.loss_functions import get_loss_module
from quantem.core.ml.models.model_base import PlanarDecompositionModel
from quantem.core.ml.optimizer_mixin import OptimizerMixin
from quantem.core.utils.rng import RNGMixin
from quantem.tomography.dataset_models import TomographyINRPretrainDataset
from quantem.tomography.tomography_context import ReconstructionContext


class ObjConstraintParams:
    """
    Namespace class for object reconstruction constraint dataclasses and parsing utilities.

    Contains constraint definitions for pixelated and implicit neural representation
    (INR) object types, along with a factory method for instantiating the appropriate
    class from a configuration dictionary.

    Supported constraint types
    --------------------------
    ObjPixelatedConstraints
        Constraints for a voxel-grid (pixelated) object representation.
    ObjINRConstraints
        Constraints for a network-parameterized (INR) object representation;
        adds a sparsity term not present in the pixelated variant.

    Examples
    --------
    >>> ObjConstraintParams.parse_dict({"name": "obj_pixelated", "tv_vol": 0.01})
    ObjPixelatedConstraints(positivity=True, shrinkage=0.0, tv_vol=0.01)
    >>> ObjConstraintParams.parse_dict({"type": "obj_inr", "sparsity": 0.05})
    ObjINRConstraints(positivity=True, shrinkage=0.0, tv_vol=0.0, sparsity=0.05)
    """

    @dataclass
    class ObjPixelatedConstraints(Constraints):
        """
        Constraints for a pixelated (voxel-grid) object representation.

        Attributes
        ----------
        positivity : bool
            If ``True``, enforces non-negative values in the reconstruction.
        shrinkage : float
            Shrinkage regularization strength; pushes values toward zero.
        tv_vol : float
            Total variation regularization weight for the 3-D volume.
        soft_constraint_keys : list[str]
            Constraint fields penalized softly during optimization.
        hard_constraint_keys : list[str]
            Constraint fields enforced strictly during optimization.
        """

        positivity: bool = True
        shrinkage: float = 0.0
        tv_vol: float = 0.0
        _name: str = "obj_pixelated"

        soft_constraint_keys = ["tv_vol"]
        hard_constraint_keys = ["positivity", "shrinkage"]

    @dataclass
    class ObjINRConstraints(Constraints):
        """
        Constraints for an implicit neural representation (INR) object.

        Extends pixelated constraints with an additional sparsity term suited
        to the continuous, network-parameterized object representation.

        Attributes
        ----------
        positivity : bool
            If ``True``, enforces non-negative values in the reconstruction.
        shrinkage : float
            Shrinkage regularization strength; pushes values toward zero.
        tv_vol : float
            Total variation regularization weight for the 3-D volume.
        sparsity : float
            Sparsity regularization weight; encourages near-zero activations.
        soft_constraint_keys : list[str]
            Constraint fields penalized softly during optimization.
        hard_constraint_keys : list[str]
            Constraint fields enforced strictly during optimization.
        """

        positivity: bool = True
        shrinkage: float = 0.0
        tv_vol: float = 0.0
        sparsity: float = 0.0
        _name: str = "obj_inr"

        soft_constraint_keys = ["tv_vol", "sparsity"]
        hard_constraint_keys = ["positivity", "shrinkage"]

    @dataclass
    class ObjTensorDecompConstraints(Constraints):
        """
        Constraints for a tensor decomposition object representation.

        Attributes
        ----------
        positivity : bool
            If ``True``, enforces non-negative values in the reconstruction.
        shrinkage : float
            Shrinkage regularization strength; pushes values toward zero.
        tv_vol : float
            Total variation regularization weight for the 3-D volume.
        soft_constraint_keys : list[str]
            Constraint fields penalized softly during optimization.
        hard_constraint_keys : list[str]
            Constraint fields enforced strictly during optimization.
        """

        positivity: bool = True
        shrinkage: float = 0.0
        tv_vol: float = 0.0
        tv_plane: float = 0.0
        sparsity: float = 0.0
        _name: str = "obj_tensor_decomp"

        soft_constraint_keys = ["tv_vol", "tv_plane", "sparsity"]
        hard_constraint_keys = ["positivity", "shrinkage"]

    @classmethod
    def parse_dict(
        cls, d: dict
    ) -> "ObjConstraintParams.ObjPixelatedConstraints | ObjConstraintParams.ObjINRConstraints | ObjConstraintParams.ObjTensorDecompConstraints":
        """
        Instantiate an object constraint dataclass from a configuration dictionary.

        The dictionary must contain a ``'name'`` or ``'type'`` key identifying
        which constraint class to construct. All remaining keys are forwarded as
        keyword arguments to the selected dataclass.

        Parameters
        ----------
        d : dict
            Configuration dictionary. Must include ``'name'`` or ``'type'``
            with one of the following values (case-insensitive):

            - ``'obj_pixelated'`` → :class:`ObjPixelatedConstraints`
            - ``'obj_inr'`` → :class:`ObjINRConstraints`

            The value may also be a class ``type`` object, in which case its
            ``__name__`` is used after lower-casing.

        Returns
        -------
        ObjPixelatedConstraints or ObjINRConstraints
            An instance of the appropriate object constraint dataclass.

        Raises
        ------
        ValueError
            If neither ``'name'`` nor ``'type'`` is present, if the value is not
            a string or type, or if the name does not match any known object
            constraint type.
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
            raise ValueError(f"Unknown object constraint type: {name}")
        if name == "obj_pixelated":
            return ObjConstraintParams.ObjPixelatedConstraints(**d)
        elif name == "obj_inr":
            return ObjConstraintParams.ObjINRConstraints(**d)
        elif name == "obj_tensor_decomp":
            return ObjConstraintParams.ObjTensorDecompConstraints(**d)
        else:
            raise ValueError(f"Unknown object constraint type: {name.lower()}")


ObjConstraintsType = (
    ObjConstraintParams.ObjPixelatedConstraints
    | ObjConstraintParams.ObjINRConstraints
    | ObjConstraintParams.ObjTensorDecompConstraints
)


def _unwrap(model: nn.Module | nn.parallel.DistributedDataParallel) -> PlanarDecompositionModel:
    """Unwrap a DistributedDataParallel model to get the underlying module ONLY for tensor decomposition models."""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return cast(PlanarDecompositionModel, model.module)
    return cast(PlanarDecompositionModel, model)


class ObjectBase(AutoSerialize, nn.Module, RNGMixin, OptimizerMixin):
    DEFAULT_LRS = {
        "object": 8e-6,
    }
    _token = object()
    """
    Base class for all ObjectModels to inherit from.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],  # pyright: ignore[reportRedeclaration]
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use a factory method to instantiate this class.")

        self._shape = shape

        # Initialize dependencies
        nn.Module.__init__(self)
        RNGMixin.__init__(self, rng=rng, device=device)
        OptimizerMixin.__init__(self)

        # --- Instantiation ----

    # --- Properties ---
    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Shape of the object (x, y, z).
        """
        return self._shape

    @shape.setter
    def shape(self, new_shape: tuple[int, int, int]):
        self._shape = new_shape

    @property
    def obj(self) -> torch.Tensor:
        """
        Returns the object, should be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def model(self) -> nn.Module:
        """
        Returns the model, should be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the object.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass, should be implemented in subclasses. Note for any nn.Module this is
        a required method.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the object, should be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def params(self) -> Generator[torch.nn.Parameter, None, None]:
        """
        Get the parameters that should be optimized for this model.

        Should be implemented in subclasses.
        """
        raise NotImplementedError

    # --- Helper Functions ---
    def get_optimization_parameters(self) -> "dict[str, list[torch.Tensor]]":
        """Default: a single param group keyed by DEFAULT_OPTIMIZER_KEY.

        Hyperparameters are baked by ``set_optimizer``, not here — return only the tensors.
        """
        return {self.DEFAULT_OPTIMIZER_KEY: list(self.params)}

    @abstractmethod  # Each subclass should implement this.
    def to(self, device: str | torch.device):
        """
        Move the object to a device
        """

        raise NotImplementedError


class ObjectConstraints(BaseConstraints, ObjectBase):  # TODO: Ask Arthur why we still need this
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_tv_loss(self, ctx: ReconstructionContext) -> torch.Tensor:
        """
        Get the TV loss for the object model. Must be implemented in each subclass.
        """
        raise NotImplementedError


class ObjectPixelated(ObjectConstraints):
    """
    Object model for pixelated objects.

    Supports: Conventional algorithms (SIRT, FBP), and AD-based reconstructions.
    """

    DEFAULT_CONSTRAINTS = ObjConstraintParams.ObjPixelatedConstraints()

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        super().__init__(
            shape=shape,
            device=device,
            rng=rng,
            _token=self._token,
        )
        self.constraints: ObjConstraintsType = self.DEFAULT_CONSTRAINTS.copy()

    # --- Instantiation ----
    @classmethod
    def from_uniform(
        cls,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        # Initialize a torch.zeros volume with the given shape
        obj = torch.zeros(shape, device=device, dtype=torch.float32)
        obj_model = cls(shape=shape, device=device, rng=rng)
        obj_model._obj = obj
        return obj_model

    @classmethod
    def from_array(
        cls,
        initial_obj: torch.Tensor | np.ndarray,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        obj_model = cls(shape=initial_obj.shape, device=device, rng=rng)
        if isinstance(initial_obj, np.ndarray):
            initial_obj = torch.tensor(initial_obj, dtype=torch.float32)
        else:
            initial_obj = initial_obj.clone()
        obj_model._obj = initial_obj.to(device)
        return obj_model

    # --- Properties ----
    @property
    def obj(self) -> torch.Tensor:
        return self.apply_hard_constraints(
            self._obj
        )  # TODO: Normalization factor to ensure object agrees with INR.

    @obj.setter
    def obj(self, obj: torch.Tensor):
        self._obj = obj

    @property
    def obj_view(self) -> np.ndarray:
        return self.obj.cpu().unsqueeze(0).numpy()

    # @property
    # def soft_loss(self) -> torch.Tensor:
    #     return self.apply_soft_constraints(self._obj)

    @property
    def name(self) -> str:
        return "obj_pixelated"

    @property
    def obj_type(self) -> str:
        return "pixelated"

    @property
    def dtype(self) -> torch.dtype:
        return self._obj.dtype

    def apply_hard_constraints(
        self,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply hard constraints to the object model.

        Only hard constraint here is the positivity and shrinkage. TODO: Add the other hard constraints.
        """
        obj2 = pred.clone()
        if self.constraints.positivity:
            obj2 = torch.clamp(obj2, min=0.0, max=None)
        if self.constraints.shrinkage:
            obj2 = torch.max(obj2 - self.constraints.shrinkage, torch.zeros_like(obj2))

        # TODO: Need to implement the other hard constraints: Fourier Filter and Circular Mask.
        return obj2

    def apply_soft_constraints(self, ctx: ReconstructionContext) -> torch.Tensor:
        assert ctx.obj is not None, "ObjectPixelated requires ctx.obj to be set"
        soft_loss = torch.tensor(
            0.0, device=ctx.obj.device, dtype=ctx.obj.dtype, requires_grad=True
        )
        if self.constraints.tv_vol > 0:
            tv_loss = self.get_tv_loss(ctx)
            soft_loss += tv_loss
        return soft_loss

    # --- Forward method ---
    def forward(self, coords=None) -> torch.Tensor:
        return self.obj

    # --- Defining the TV loss ---
    def get_tv_loss(self, ctx: ReconstructionContext) -> torch.Tensor:
        assert ctx.obj is not None, "ObjectPixelated requires ctx.obj to be set"
        # TV over the three trailing spatial dims, leaving any leading channel/batch axes
        # intact. Works for a 3-D volume, obj_view's [1, D, H, W], and a multimodal
        # [C, D, H, W] (channels = elemental compositions), matching the INR / tensor-decomp
        # convention where the object carries a leading channel dimension.
        tv_d = torch.pow(ctx.obj[..., 1:, :, :] - ctx.obj[..., :-1, :, :], 2).sum()
        tv_h = torch.pow(ctx.obj[..., :, 1:, :] - ctx.obj[..., :, :-1, :], 2).sum()
        tv_w = torch.pow(ctx.obj[..., :, :, 1:] - ctx.obj[..., :, :, :-1], 2).sum()
        tv_loss = tv_d + tv_h + tv_w

        return tv_loss * self.constraints.tv_vol / ctx.obj.numel()

    # --- Helper Functions ---
    def to(self, device: str | torch.device):
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        self._obj = self._obj.to(device)
        self.reconnect_optimizer_to_parameters()
        return self


class ObjectINR(ObjectConstraints, DDPMixin):
    DEFAULT_CONSTRAINTS = ObjConstraintParams.ObjINRConstraints()

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        model: nn.Module | None = None,
        _token: object | None = None,
    ):
        super().__init__(
            shape=shape,
            device=device,
            rng=rng,
            _token=self._token,
        )
        self._pretrain_losses = []
        self._pretrain_lrs = []
        self.constraints: ObjConstraintParams.ObjINRConstraints = self.DEFAULT_CONSTRAINTS.copy()
        # Register the network submodule (important: real nn.Module attribute)
        if model is not None:
            self.setup_distributed(device=device)
            self._model = self.distribute_model(model)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        obj_model = cls(
            shape=shape,
            device=device,
            rng=rng,
            model=model,  # ✅ build/register in __init__
        )

        obj_model.setup_distributed(device=device)
        # Use obj_model.device (resolved by setup_distributed just above), not the
        # raw `device` param -- under torchrun, setup_distributed ignores/overrides
        # a "cpu" (or any) default and sets the real per-rank cuda:N device. Passing
        # the stale literal `device` string here used to be harmless only because
        # ObjectINR.to()'s world_size>1 branch was a no-op when already DDP-wrapped
        # (see the bugfix in .to() above); now that .to() actually re-places the
        # model, calling it with the wrong device would move params back to CPU
        # while distribute_model's DDP(..., device_ids=[local_rank]) still expects
        # them on that rank's GPU.
        obj_model.to(obj_model.device)
        return obj_model

    # --- Properties ---

    @property
    def model(self) -> nn.Module | nn.parallel.DistributedDataParallel:
        """
        Returns the INR model.
        """
        return self._model

    # @model.setter
    # def model(self, model: "nn.Module"):
    #     """
    #     This doesn't work -- can't have setters for torch sub modules
    #     https://github.com/pytorch/pytorch/issues/52664

    #     For now, upon initialization private variable `._model` is set to the built model.
    #     """
    #     raise RuntimeError("\n\n\nsetting model, this shouldn't be reachable???\n\n\n")

    @property
    def obj(self) -> torch.Tensor:
        return self._obj

    @obj.setter
    def obj(self, obj: torch.Tensor):
        self._obj = obj

    @property
    def obj_view(self) -> np.ndarray:
        """
        Returns the object as a view of the x, y, z axes.

        Matches the axes of conventionally reconstructed objects, this is the object that will be saved.
        """
        self.create_volume()
        return self._obj.cpu().numpy().transpose(0, 1, 3, 2)

    def apply_soft_constraints(
        self,
        ctx: ReconstructionContext,
    ) -> torch.Tensor:
        soft_loss = torch.tensor(0.0, device=ctx.coords.device)
        if self.constraints.tv_vol > 0:
            assert ctx.coords is not None, (
                "coords must be provided for INR object model to compute the TV loss"
            )
            soft_loss += self.get_tv_loss(ctx)

        if (
            isinstance(self.constraints, ObjConstraintParams.ObjINRConstraints)
            and self.constraints.sparsity > 0
        ):  # NOTE: For the linter, I must make this :)
            assert ctx.pred is not None, (
                "pred must be provided for INR object model to compute the sparsity loss"
            )
            sparsity_loss = self.constraints.sparsity * torch.norm(ctx.pred, p=1)
            soft_loss += sparsity_loss

        return soft_loss

    def apply_hard_constraints(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Apply hard constraints to the predicted values of the INR model.
        """

        if self.constraints.positivity:
            pred = torch.clamp(pred, min=0.0, max=None)
        if self.constraints.shrinkage:
            pred = torch.max(pred - self.constraints.shrinkage, torch.zeros_like(pred))

        return pred

    # --- Define get_tv_loss ---

    def get_tv_loss(self, ctx: ReconstructionContext) -> torch.Tensor:
        """
        Compute the total variation loss for the INR model.
        """
        assert ctx.coords is not None, "coords must be provided for INR object model"
        num_tv_samples = min(10_000, ctx.coords.shape[0])
        tv_indices = torch.randperm(ctx.coords.shape[0], device=ctx.coords.device)[:num_tv_samples]

        tv_coords = ctx.coords[tv_indices].detach().requires_grad_(True)
        tv_densities_recomputed = self.model(tv_coords)
        if isinstance(tv_densities_recomputed, tuple):
            tv_densities_recomputed = tv_densities_recomputed[0]

        # Ensure shape is [num_samples, num_channels]
        if tv_densities_recomputed.dim() == 1:
            tv_densities_recomputed = tv_densities_recomputed.unsqueeze(-1)

        # Compute gradients for each channel
        grad_outputs = torch.autograd.grad(
            outputs=tv_densities_recomputed,
            inputs=tv_coords,
            grad_outputs=torch.ones_like(tv_densities_recomputed),
            create_graph=True,
        )[0]  # Shape: [num_samples, coord_dim]

        # Compute TV loss - gradient magnitude per sample
        grad_norm = torch.norm(grad_outputs, dim=1)  # Shape: [num_samples]
        return self.constraints.tv_vol * grad_norm.mean()

    # --- Optimization Parameters ---
    @property
    def params(self) -> Generator[torch.nn.Parameter, None, None]:
        return self.model.parameters()  # type: ignore[attr-defined]

    # Pretraining
    @property
    def pretrained_weights(self) -> dict[str, torch.Tensor]:
        """get the pretrained weights of the INR model"""
        return self._pretrained_weights

    def _set_pretrained_weights(self, model: "torch.nn.Module"):
        """set the pretrained weights of the INR model"""
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Pretrained model must be a torch.nn.Module, got {type(model)}")
        self._pretrained_weights = deepcopy(model.state_dict())

    @property
    def pretrain_target(self) -> TomographyINRPretrainDataset:
        """get the pretrain target"""
        return self._pretrain_target

    @pretrain_target.setter
    def pretrain_target(self, target: TomographyINRPretrainDataset):
        """set the pretrain target"""
        self._pretrain_target = target

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the object.
        """
        # TODO: This is a temporary solution to get the dtype of the object.
        return torch.float32

    # --- Helper Functions ---
    def rebuild_model(self):
        self._model = self.distribute_model(self._model)

    # Reset method that goes back to the pretrained weights.
    def reset(self):
        """reset the model to the pretrained weights"""
        self.model.load_state_dict(self._pretrained_weights.copy())
        self._model = self.distribute_model(
            self.model
        )  # Maybe add a check to see if distributed or not, but not very computationally expensive to do this.

    # --- Forward Method ---

    def forward(self, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """forward pass for the INR model"""
        assert coords is not None, "ObjectINR.forward requires coords"

        all_densities = self.model(coords)

        if all_densities.dim() > 1:
            all_densities = all_densities.squeeze(-1)
        valid_mask = (
            (coords[:, 0] >= -1) & (coords[:, 0] <= 1) & (coords[:, 1] >= -1) & (coords[:, 1] <= 1)
        ).float()

        if all_densities.dim() > 1:
            valid_mask = valid_mask.unsqueeze(-1)
        # Multi-dimensional mask
        all_densities = all_densities * valid_mask

        all_densities = self.apply_hard_constraints(all_densities)

        return all_densities

    # Pretrain Loop

    def pretrain(
        self,
        pretrain_dataset: TomographyINRPretrainDataset,
        batch_size: int,
        reset: bool = False,
        num_iters: int = 10,
        num_workers: int = 0,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        loss_fn: Callable | str = "l1",
        verbose: bool = True,
    ):
        """
        Pretrain the INR model to fit target volume.
        """

        if (
            pretrain_dataset is not None
        ):  # Need to make a check if there's already a pretrain dataset to not go through with the setup again.
            self.pretrain_dataset = pretrain_dataset
            (
                self.pretraining_dataloader,
                self.pretraining_sampler,
                self.pretraining_val_dataloader,
                self.pretraining_val_sampler,
            ) = self.setup_dataloader(pretrain_dataset, batch_size, num_workers=num_workers)

        if optimizer_params is not None:
            self.set_optimizer(optimizer_params)
        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_iters)

        if reset:
            self.reset()

        loss_fn = get_loss_module(loss_fn, self.dtype)

        self._pretrain(
            num_iters=num_iters,
            loss_fn=loss_fn,
            verbose=verbose,
        )

    def _pretrain(
        self,
        num_iters: int,
        loss_fn: Callable,
        verbose: bool,
    ):
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")
        if self.scheduler is None:
            raise RuntimeError("Scheduler not set. Call set_scheduler() first.")

        self.model.train()
        optimizer = self.optimizer
        scheduler = self.scheduler

        pbar = tqdm(range(num_iters), desc="Pretraining", disable=not verbose)
        for a0 in pbar:
            epoch_loss = 0
            for batch_idx, batch in enumerate[Any](self.pretraining_dataloader):
                coords = batch["coords"].to(self.device, non_blocking=True)
                target = batch["target"].to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16, enabled=True
                ):
                    outputs = self.forward(coords)
                    loss = loss_fn(outputs, target)

                loss.backward()
                epoch_loss += loss.item()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()

            self._pretrain_losses.append(epoch_loss / len(self.pretraining_dataloader))
            print(
                f"Epoch {a0 + 1}/{num_iters}, Pretrain Loss: {epoch_loss / len(self.pretraining_dataloader):.4f}"
            )
            self._pretrain_lrs.append(optimizer.param_groups[0]["lr"])

    def create_volume(self, return_vol: bool = False):
        # Generalized to a possibly-anisotropic (Nx, Ny, Nz) shape -- for the
        # existing cubic case (Nx == Ny == Nz) this is numerically identical
        # to the old hardcoded-cube behavior.
        Nx, Ny, Nz = self._shape
        with torch.no_grad():
            x_1d = torch.linspace(-1, 1, Nx)
            y_1d = torch.linspace(-1, 1, Ny)
            z_1d = torch.linspace(-1, 1, Nz)
            x, y, z = torch.meshgrid(x_1d, y_1d, z_1d, indexing="ij")
            inputs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
            # Unwrap DDP too, not just DataParallel -- create_volume calls the raw
            # model directly (per-rank, on an uneven grid subset, then all_gathers),
            # which is incompatible with DDP's forward hooks (expects every rank to
            # call forward in lockstep with a matching batch); calling the DDP
            # wrapper itself here manifests as a device-mismatch error inside the
            # model's own layers.
            model = (
                self.model.module
                if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
                else self.model
            )

            inference_batch_size = 5 * Ny * Nz  # ~5 x-slices per batch, same intent as before
            total_samples = Nx * Ny * Nz
            samples_per_gpu = total_samples // self.world_size
            remainder = total_samples % self.world_size

            if self.global_rank < remainder:
                start_idx = self.global_rank * (samples_per_gpu + 1)
                end_idx = start_idx + samples_per_gpu + 1
            else:
                start_idx = self.global_rank * samples_per_gpu + remainder
                end_idx = start_idx + samples_per_gpu

            inputs_subset = inputs[start_idx:end_idx]
            num_samples = inputs_subset.shape[0]

            outputs_list = []
            for batch_start in range(0, num_samples, inference_batch_size):
                batch_end = min(batch_start + inference_batch_size, num_samples)
                batch_coords = inputs_subset[batch_start:batch_end].to(
                    self.device, non_blocking=True
                )

                batch_outputs = model(batch_coords)  # (B, C) or (B,) etc.

                if isinstance(batch_outputs, tuple):
                    batch_outputs = batch_outputs[0]
                batch_outputs = self.apply_hard_constraints(batch_outputs)

                # Ensure shape is (B, C)
                if batch_outputs.dim() == 1:
                    batch_outputs = batch_outputs.unsqueeze(-1)  # (B, 1)

                outputs_list.append(batch_outputs.cpu())

            outputs = torch.cat(outputs_list, dim=0)  # (local_B, C)
            C = outputs.shape[-1]  # e.g. 5

            if self.world_size > 1:
                # gather variable-sized first dimension (local_B) while keeping channels
                local_B = outputs.shape[0]
                output_size = torch.tensor(local_B, device=self.device, dtype=torch.long)
                all_sizes = [
                    torch.zeros(1, device=self.device, dtype=torch.long)
                    for _ in range(self.world_size)
                ]
                dist.all_gather(all_sizes, output_size)
                max_size = max(size.item() for size in all_sizes)

                outputs_dev = outputs.to(self.device)  # (local_B, C)
                if local_B < max_size:
                    pad = torch.zeros(
                        (max_size - local_B, C),  # type: ignore
                        device=self.device,
                        dtype=outputs_dev.dtype,
                    )
                    outputs_padded = torch.cat([outputs_dev, pad], dim=0)  # (max_size, C)
                else:
                    outputs_padded = outputs_dev

                gathered_outputs = [
                    torch.empty((max_size, C), device=self.device, dtype=outputs_dev.dtype)  # type: ignore
                    for _ in range(self.world_size)
                ]
                dist.all_gather(gathered_outputs, outputs_padded.contiguous())

                trimmed_outputs = []
                for rank, size in enumerate(all_sizes):
                    trimmed_outputs.append(gathered_outputs[rank][: size.item(), :])

                pred_full = torch.cat(trimmed_outputs, dim=0).reshape(C, Nx, Ny, Nz).float()
            else:
                pred_full = outputs.reshape(C, Nx, Ny, Nz).float()

            if return_vol:
                return pred_full.detach().cpu()

            self._obj = pred_full.detach().cpu()

    def to(self, device: str | torch.device):  # pyright: ignore[reportIncompatibleMethodOverride] -> better to do this device change
        if isinstance(device, str):
            device = torch.device(device)

        # Tomography.reconstruct() calls obj_model.to(self.device) unconditionally on
        # every single call (once per warmup sub-phase, once per Phase-2 chunk, ...).
        # Without this guard, the world_size>1 branch below tears down and rebuilds the
        # DistributedDataParallel wrapper (incl. a full parameter broadcast) every time,
        # even when the device hasn't changed and the model is already correctly
        # wrapped -- pure wasted work on every reconstruct() call.
        already_wrapped = isinstance(self._model, torch.nn.parallel.DistributedDataParallel)
        # Checking the cached self._device attribute alone is not enough: after
        # Tomography.from_file() deserializes a checkpoint, torch.load(...,
        # map_location="cpu") forces the actual parameter tensors onto CPU
        # regardless of what self._device says (that plain Python attribute
        # round-trips through pickling unchanged, so it still reads the
        # pre-save device). Without also checking real parameter placement,
        # this guard wrongly short-circuits and skips the .to(device) call
        # below, leaving weights stranded on CPU while inputs are on GPU --
        # confirmed via a reproducible "Expected all tensors to be on the same
        # device" crash on the very next reconstruct() call after a reload.
        actual_device = next(self._model.parameters()).device
        already_correct = (
            getattr(self, "_device", None) == device
            and actual_device == device
            and (already_wrapped if self.world_size > 1 else not already_wrapped)
        )
        if already_correct:
            return

        self._device = device
        if self.world_size == 1:
            self._model = self._model.to(device)
        else:
            # Bug fix: previously, if self._model was ALREADY DistributedDataParallel
            # (e.g. after Tomography.from_file() deserializes a checkpoint that was
            # saved DDP-wrapped), this branch was a no-op -- it never moved the
            # underlying parameters to `device`, so a resumed model's weights stayed
            # on CPU (deserialization defaults to CPU) while later code assumed they
            # were on the correct GPU, surfacing as a device-mismatch error deep in
            # the model's own layers. Also, the old code discarded distribute_model's
            # return value, so even the "not yet wrapped" case never re-assigned
            # self._model. Unwrap (if needed), then always re-move+re-wrap via
            # distribute_model, which internally does model.to(self.device) before
            # wrapping.
            raw_model = (
                self._model.module
                if isinstance(self._model, torch.nn.parallel.DistributedDataParallel)
                else self._model
            )
            self._model = self.distribute_model(raw_model)
        self.reconnect_optimizer_to_parameters()


class ObjectTensorDecomp(ObjectINR):
    DEFAULT_CONSTRAINTS = ObjConstraintParams.ObjTensorDecompConstraints()

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        model: nn.Module | None = None,
        _token: object | None = None,
    ):
        super().__init__(
            shape=shape,
            device=device,
            rng=rng,
            _token=self._token,
        )
        self._pretrain_losses = []
        self._pretrain_lrs = []
        self.constraints: ObjConstraintParams.ObjTensorDecompConstraints = (
            self.DEFAULT_CONSTRAINTS.copy()
        )
        # Register the network submodule (important: real nn.Module attribute)
        if model is not None:
            self.setup_distributed(device=device)
            self._model = self.distribute_model(model)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        obj_model = cls(
            shape=shape,
            device=device,
            rng=rng,
            model=model,  # ✅ build/register in __init__
        )

        obj_model.setup_distributed(device=device)
        # Use obj_model.device (resolved by setup_distributed just above), not the
        # raw `device` param -- under torchrun, setup_distributed ignores/overrides
        # a "cpu" (or any) default and sets the real per-rank cuda:N device. Passing
        # the stale literal `device` string here used to be harmless only because
        # ObjectINR.to()'s world_size>1 branch was a no-op when already DDP-wrapped
        # (see the bugfix in .to() above); now that .to() actually re-places the
        # model, calling it with the wrong device would move params back to CPU
        # while distribute_model's DDP(..., device_ids=[local_rank]) still expects
        # them on that rank's GPU.
        obj_model.to(obj_model.device)
        return obj_model

    # --- Constraints ---

    def apply_soft_constraints(self, ctx: ReconstructionContext) -> torch.Tensor:
        soft_loss = torch.tensor(
            0.0, device=ctx.pred.device if ctx.pred is not None else self.device
        )
        # Gate on EITHER coefficient. Previously this tested only tv_vol, so a
        # run with tv_plane > 0 and tv_vol == 0 -- the natural setting, since
        # volume TV penalizes density gradients and fights sharp atomic peaks --
        # silently applied no plane regularization whatsoever.
        if self.constraints.tv_vol > 0 or self.constraints.tv_plane > 0:
            assert ctx.coords is not None, "Coordinates must be provided for TV loss"
            assert ctx.pred is not None, "Prediction must be provided for TV loss"
            soft_loss += self.get_tv_loss(ctx)

        if self.constraints.sparsity > 0:  # NOTE: For the linter, I must make this :)
            assert ctx.all_densities is not None, (
                "All densities must be provided for sparsity loss"
            )
            sparsity_loss = self.constraints.sparsity * ctx.all_densities.abs().mean()
            soft_loss += sparsity_loss

        return soft_loss

    # TV Losses

    def get_tv_loss(self, ctx: ReconstructionContext) -> torch.Tensor:
        """
        Gets the summed total variational loss for the tensor decomposition model.

        _get_plane_tv_loss: Total-variation across the planes.
        _get_volume_tv_loss: Isotropic volume TV
        """
        assert ctx.coords is not None, "Coordinates must be provided for TV loss"
        assert ctx.pred is not None, "Prediction must be provided for TV loss"
        tv_loss = torch.tensor(0.0, device=ctx.pred.device)
        # Each term is skipped when its own coefficient is zero. Beyond being
        # correct, this matters for cost: get_volume_tv_loss runs three extra
        # forward passes per call (one per axis, plus the base), so evaluating
        # it at tv_vol == 0 would roughly quadruple the object-model cost to
        # multiply the result by zero.
        if self.constraints.tv_plane > 0:
            tv_loss += self._get_plane_tv_loss()
        if self.constraints.tv_vol > 0:
            tv_loss += self.get_volume_tv_loss(ctx.coords)
        return tv_loss

    def _get_plane_tv_loss(self) -> torch.Tensor:
        """
        Gets the total-variation across the planes.
        """
        # Read `tilted` and `T` from the UNWRAPPED module. DistributedDataParallel
        # does not proxy attribute access, so `self.model.tilted` raised
        # AttributeError at world_size>1 -- i.e. any multi-GPU run with
        # tv_plane > 0 crashed on its first backward. Harmless at world_size==1
        # because distribute_model does not wrap there (core/ml/ddp.py:158),
        # which is why single-GPU runs never hit it.
        model = _unwrap(self.model)
        is_tilted = model.tilted
        per_level = []

        # QUANTEM_TV_PLANE_NORM -- per-scale weighting, added 2026-08-08.
        #
        #   "mean"    (default) historical behaviour, every prior run reproduces
        #   "percell" equalise the gradient reaching one PLANE CELL across scales
        #
        # Why "percell" exists: the .mean(dim=(1,2,3)) below normalises each
        # scale by its own C*H*W, so with multiscale [0.25,0.5,1.0] a cell of the
        # 341-cell plane receives (341/85)^2 = 16x LESS gradient than a cell of
        # the 85-cell plane. The scale that carries the Nyquist-rate content --
        # and that README section 13.1 measures as the noisiest -- is the one the
        # penalty barely touches. Weighting each level by its area relative to
        # the FINEST level makes the per-cell gradient equal while leaving the
        # finest level's magnitude unchanged, so the section 8 tv_plane
        # calibration (5.666e-7 ~ 10 % of the data loss) stays approximately
        # valid instead of needing a ~1e6 rescale as a bare .sum() would.
        _norm = os.environ.get("QUANTEM_TV_PLANE_NORM", "mean")
        if _norm not in ("mean", "percell"):
            raise ValueError(f"QUANTEM_TV_PLANE_NORM must be mean/percell, got {_norm}")
        _area_max = max(int(g.shape[2]) * int(g.shape[3]) for g in model.grids)

        # With an anisotropic resolution the three planes of a scale differ in
        # shape and cannot share a tensor, so kplanes.py stores them as three
        # separate parameters per scale rather than one packed (3*T, C, H, W).
        # getattr: checkpoints pickled before that fix have no such attribute
        # and were necessarily packed.
        _packed = getattr(model, "planes_packed", True)
        _grids = list(model.grids)
        _stride = 1 if _packed else 3

        for s in range(0, len(_grids), _stride):
            entries = _grids[s : s + _stride]
            # packed:   one p of (3*T, C, H, W) for TILTED, (3, C, H, W) for KPlanes
            # unpacked: three p of (T, C, H, W)     "     , (1, C, H, W)     "
            per_entry = []
            for p in entries:
                dh = (p[:, :, 1:, :] - p[:, :, :-1, :]).pow(2).mean(dim=(1, 2, 3))
                dw = (p[:, :, :, 1:] - p[:, :, :, :-1]).pow(2).mean(dim=(1, 2, 3))
                per_entry.append(dh + dw)

            if _packed:
                per_plane = per_entry[0]  # (3*T,) or (3,)
                if is_tilted:
                    # sum the 3 planes of each rotation, then average rotations
                    level_tv = per_plane.view(model.T, 3).sum(dim=1).mean()
                else:
                    level_tv = per_plane.sum()
            else:
                # per_entry[p] is (T,) or (1,) -- summing across the list is the
                # same "sum 3 planes per rotation" reduction as the view above.
                per_rotation = torch.stack(per_entry, dim=0).sum(dim=0)
                level_tv = per_rotation.mean() if is_tilted else per_rotation.sum()

            if _norm == "percell":
                # The three planes of an anisotropic scale have different areas;
                # weight the scale by its largest, which is the packed value.
                _area = max(int(p.shape[2]) * int(p.shape[3]) for p in entries)
                level_tv = level_tv * (_area / float(_area_max))

            per_level.append(level_tv)

        return self.constraints.tv_plane * torch.stack(per_level).sum()

    def get_volume_tv_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Isotropic volume TV via finite differences. Same form as the autograd
        version (L1 of gradient L2-norm) but avoids double-backward, so it
        works for KPlanesTILTED, CPTilted, and anything else.
        """
        num_tv_samples = min(10_000, coords.shape[0])
        tv_indices = torch.randperm(coords.shape[0], device=coords.device)[:num_tv_samples]
        tv_coords = coords[tv_indices]  # (N, 3)

        model = _unwrap(self.model)
        h = 2.0 / min(model.resolution)

        pred = model(tv_coords)
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)  # (N, 1)

        grads = []
        for axis in range(3):
            offset = torch.zeros(3, device=tv_coords.device)
            offset[axis] = h
            shifted_pred = self.model(tv_coords + offset)
            if isinstance(shifted_pred, tuple):
                shifted_pred = shifted_pred[0]
            if shifted_pred.dim() == 1:
                shifted_pred = shifted_pred.unsqueeze(-1)
            grads.append((shifted_pred - pred) / h)  # (N, 1)

        grad_stack = torch.stack(grads, dim=-1)  # (N, C, 3)
        grad_norm = torch.norm(grad_stack, dim=-1)  # (N, C)

        return self.constraints.tv_vol * grad_norm.mean()

    def apply_hard_constraints(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Apply hard constraints to the predicted values of the INR model.
        """

        if self.constraints.positivity:
            pred = torch.clamp(pred, min=0.0, max=None)
        if self.constraints.shrinkage:
            pred = torch.max(pred - self.constraints.shrinkage, torch.zeros_like(pred))

        return pred

    # --- Optimization Parameters ---
    @property
    def params(self) -> Generator[torch.nn.Parameter, None, None]:
        """
        Returns the optimization parameters, here we also check if PPLR is used and return the appropriate parameters.
        """

        return self.model.parameters()  # type: ignore[attr-defined]

    def get_optimization_parameters(self) -> "dict[str, list[torch.Tensor]]":
        """PPLR: per-key param groups (hyperparameters are baked by set_optimizer)."""
        model = _unwrap(self.model)
        return {key: list(model.get_params()[key]) for key in model.param_keys}

    def _normalize_optimizer_params(self, params):
        """ObjectTensorDecomp requires a dict matching model.param_keys."""
        if not isinstance(params, dict) or self._is_single_optimizer_dict(params):
            raise TypeError(
                f"ObjectTensorDecomp requires dict[str, OptimizerParamsType] keyed by "
                f"param_keys; got {type(params)}"
            )
        model = _unwrap(self.model)
        expected = set(model.param_keys)
        got = set(params.keys())
        if got != expected:
            raise ValueError(
                f"optimizer_params keys must match model.param_keys: "
                f"got {got}, expected {expected}"
            )
        return super()._normalize_optimizer_params(params)

    def pretrain(self) -> None:
        raise NotImplementedError(
            "Tensor decomposition pretraining is not usually required, and for TILTED there is a two-phase warmup approach."
        )


ObjectModelType = ObjectPixelated | ObjectINR | ObjectTensorDecomp
