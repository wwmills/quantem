from collections.abc import Mapping

import torch

from quantem.core.ml.optimizer_mixin import (
    OptimizerParams,
    OptimizerParamsType,
    SchedulerParamsType,
)
from quantem.tomography.tomography_base import TomographyBase


class TomographyOpt(TomographyBase):
    """
    Class for handling all the optimizers and schedulers for the tomography reconstruction.
    """

    OPTIMIZABLE_VALS = ["object", "pose"]
    DEFAULT_OPTIMIZER_TYPE: OptimizerParamsType = OptimizerParams.Adam()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_default_lr(self, key: str) -> float:
        """Get default learning rate for a given optimization key."""
        if key == "object":
            return self.obj_model.DEFAULT_LRS.get("object", 1e-5)
        elif key == "pose":
            return self.dset.DEFAULT_LRS.get("pose", 5e-2)
        else:
            raise ValueError(f"Unknown optimization key: {key}")

    @property
    def optimizer_params(self) -> dict[str, OptimizerParamsType | dict[str, OptimizerParamsType]]:
        return {
            key: params
            for key, params in [
                ("object", self.obj_model.optimizer_params),
                ("pose", self.dset.optimizer_params),
            ]
            if params
        }

    @optimizer_params.setter
    def optimizer_params(self, d: dict[str, OptimizerParamsType] | dict[str, dict]):
        """Set the optimizer parameters."""
        if isinstance(d, (tuple, list)):
            d = {k: {} for k in d}

        dset_params = {k: v for k, v in d.items() if k != "object"}
        if "object" in d:
            self.obj_model.optimizer_params = d["object"]
        if dset_params:
            self.dset.optimizer_params = dset_params

    @property
    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        optimizers = {}

        if self.obj_model.optimizer is not None:
            optimizers["object"] = self.obj_model.optimizer
        if self.dset.optimizer is not None:
            optimizers["pose"] = self.dset.optimizer

        return optimizers

    def set_optimizers(self):
        if self.obj_model.has_optimizer() or self.obj_model.optimizer_params:
            self.obj_model.set_optimizer()
        if self.dset.has_optimizer() or self.dset.optimizer_params:
            self.dset.set_optimizer()

    def get_current_lrs(self) -> dict[str, float]:
        lrs: dict[str, float] = {}

        lrs["object"] = self.obj_model.get_current_lr() if self.obj_model.has_optimizer() else 0.0

        if self.dset.has_optimizer():
            dset_opt = self.dset.optimizer
            dset_keys = list(self.dset.optimizer_params.keys())
            for i, pg in enumerate(dset_opt.param_groups):
                label = dset_keys[i] if i < len(dset_keys) else f"dset_{i}"
                lrs[label] = pg["lr"]
        return lrs

    def remove_optimizer(self, key: str):
        if key == "object":
            self.obj_model.remove_optimizer()
        elif key == "pose":
            self.dset.remove_optimizer()
        else:
            raise ValueError(f"Unknown optimization key: {key}")

    @property
    def scheduler_params(self) -> dict[str, SchedulerParamsType]:
        """Returns the parameters used to set the schedulers."""
        return {
            "object": self.obj_model.scheduler_params,
            "pose": self.dset.scheduler_params,
        }

    @scheduler_params.setter
    def scheduler_params(self, d: dict):
        """Set the scheduler parameters."""
        d = dict(d) if d else {}
        self._scheduler_params = d.copy()

        if "object" in d:
            self.obj_model.scheduler_params = d["object"]

        dset_sched = {k: v for k, v in d.items() if k != "object"}
        if dset_sched:
            # dset has one scheduler shared across its param groups; use the first entry
            first_key = next(iter(dset_sched))
            self.dset.scheduler_params = dset_sched[first_key]

    @property
    def schedulers(self) -> dict[str, torch.optim.lr_scheduler._LRScheduler]:
        schedulers = {}

        if self.obj_model.scheduler is not None:
            schedulers["object"] = self.obj_model.scheduler
        if self.dset.scheduler is not None:
            schedulers["pose"] = self.dset.scheduler

        return schedulers

    def set_schedulers(
        self, params: Mapping[str, SchedulerParamsType | dict], num_iter: int | None = None
    ):
        if "object" in params:
            self.obj_model.set_scheduler(params["object"], num_iter=num_iter)
        dset_sched = {k: v for k, v in params.items() if k != "object"}
        if dset_sched:
            first_key = next(iter(dset_sched))
            self.dset.set_scheduler(dset_sched[first_key], num_iter=num_iter)

    def step_optimizers(self):
        if self.obj_model.has_optimizer():
            self.obj_model.step_optimizer()
        if self.dset.has_optimizer():
            self.dset.step_optimizer()

    def zero_grad_all(self):
        if self.obj_model.has_optimizer():
            self.obj_model.zero_optimizer_grad()
        if self.dset.has_optimizer():
            self.dset.zero_optimizer_grad()

    def step_schedulers(self, loss: float | None = None):
        if self.obj_model.scheduler is not None:
            self.obj_model.step_scheduler(loss)
        if self.dset.scheduler is not None:
            self.dset.step_scheduler(loss)
