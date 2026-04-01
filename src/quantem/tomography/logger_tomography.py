import matplotlib.pyplot as plt
import torch

from quantem.core.ml.logger import LoggerBase
from quantem.tomography.dataset_models import DatasetModelType
from quantem.tomography.object_models import ObjectModelType
import numpy as np
from skimage.metrics import structural_similarity as ssim

class LoggerTomography(LoggerBase):
    """
    Logger for ML-based tomography reconstructions.
    """

    def __init__(
        self,
        log_dir: str,
        run_prefix: str,
        run_suffix: str = None,
        log_images_every: int = 10,
    ):
        super().__init__(log_dir, run_prefix, run_suffix, log_images_every)

    def log_epoch(self, epoch: int, loss: float, tilt_series_loss: float, soft_loss: float):
        self.log_scalar("loss/total", loss, epoch)
        self.log_scalar("loss/tilt_series", tilt_series_loss, epoch)
        self.log_scalar("loss/soft", soft_loss, epoch)

    def log_iter(
        self,
        object_model: ObjectModelType,
        iter: int,
        consistency_loss: float,
        total_loss: float,
        learning_rates: dict[str, float],
        num_samples_per_ray: int,
        val_loss: float | None = None,
    ):
        self.log_scalar("loss/consistency", consistency_loss, iter)
        self.log_scalar("loss/total", total_loss, iter)
        self.log_scalar("loss/soft", object_model._soft_constraint_losses[-1], iter)
        self.log_scalar("num_samples_per_ray", num_samples_per_ray, iter)
        for param_name, lr_value in learning_rates.items():
            self.log_scalar(f"learning_rate/{param_name}", float(lr_value), iter)
        if val_loss is not None:
            self.log_scalar("loss/val", val_loss, iter)

    def log_iter_images(
        self,
        pred_volume: torch.Tensor,
        dataset_model: DatasetModelType,
        iter: int,
        logger_cmap: str = "turbo",
        gt_z_focus: torch.Tensor | np.ndarray | None = None,
    ):
        with torch.no_grad():
            z1_vals = dataset_model.z1_params.detach().cpu().numpy()
            z3_vals = dataset_model.z3_params.detach().cpu().numpy()
            shifts_vals = dataset_model.shifts_params.detach().cpu().numpy()

        if hasattr(dataset_model, "z_focus_params"):
            z_focus_vals = dataset_model.z_focus_params.detach().cpu().numpy()
        else:
            z_focus_vals = None

        # Plotting z-focus (defocus)
        if z_focus_vals is not None:
            fig, ax = plt.subplots()
            ax.plot(z_focus_vals, label="Learned Z-Focus")

            if gt_z_focus is not None:
                if isinstance(gt_z_focus, torch.Tensor):
                    gt_vals = gt_z_focus.detach().cpu().numpy()
                else:
                    gt_vals = gt_z_focus
                ax.plot(gt_vals, "--", label="GT Z-Focus")

            ax.legend()
            ax.set_title("Z-Focus / Defocus")
            ax.set_xlabel("Tilt Image")
            ax.set_ylabel("Defocus (units)")
            self.log_figure("z_focus", fig, iter)
            plt.close(fig)

        if z_focus_vals is not None:
            self.log_scalar("z_focus/mean", float(np.mean(z_focus_vals)), iter)
            self.log_scalar("z_focus/std", float(np.std(z_focus_vals)), iter)

            if gt_z_focus is not None:
                error = np.mean((z_focus_vals - gt_z_focus) ** 2)
                self.log_scalar("z_focus/mse_to_gt", float(error), iter)



        # print("Logging volume...")
        self.log_image("volume/sum_z", pred_volume.sum(axis=0), iter, logger_cmap)
        self.log_image("volume/sum_y", pred_volume.sum(axis=1), iter, logger_cmap)
        self.log_image("volume/sum_x", pred_volume.sum(axis=2), iter, logger_cmap)

        # Plotting z1 and z3 vals
        print("Plotting z1 and z3 angles...")
        fig, ax = plt.subplots()
        ax.plot(z1_vals, label="Z1")
        ax.plot(z3_vals, label="Z3")
        ax.legend()
        ax.set_title("Z1 and Z3 Angles")
        ax.set_xlabel("Tilt Image")
        ax.set_ylabel("Degree")
        self.log_figure("z1_z3_angles", fig, iter)
        plt.close(fig)

        # Plotting shifts
        print("Plotting shifts...")
        fig, ax = plt.subplots()
        ax.plot(shifts_vals[:, 0], label="Shifts X")
        ax.plot(shifts_vals[:, 1], label="Shifts Y")
        ax.legend()
        ax.set_title("Shifts")
        ax.set_xlabel("Tilt Image")
        ax.set_ylabel("Pixel")
        self.log_figure("shifts", fig, iter)
        plt.close(fig)

    @staticmethod
    def _normalize(vol: torch.Tensor) -> torch.Tensor:
        vmin = vol.min()
        vmax = vol.max()
        return (vol - vmin) / (vmax - vmin + 1e-8)

    def log_defocus(
        self,
        dataset_model:DatasetModelType,
        gt_defocus,
    ):
        """
        Accepts GT as torch.Tensor or numpy array.
        """

        if gt_defocus is not None:
            with torch.no_grad():
                if isinstance(gt_defocus, torch.Tensor):
                    gt = gt_defocus.detach().cpu().numpy()
                else:
                    gt = gt_defocus

        assert dataset_model.hasattr('_z_focus_params')
        found_defocus =dataset_model._z_focus_params

        mean_ssim = float(np.mean(ssim_vals))
        self.log_scalar("metrics/ssim", mean_ssim, step)



    def log_ssim(
        self,
        pred_volume: torch.Tensor,
        gt_volume,
        step: int,
    ):
        """
        Accepts GT as torch.Tensor or numpy array.
        """

        with torch.no_grad():
            pred = self._normalize(pred_volume).detach().cpu().numpy()
            if isinstance(gt_volume, torch.Tensor):
                gt = self._normalize(gt_volume).detach().cpu().numpy()
            else:
                gt = gt_volume
                gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)

        assert pred.shape == gt.shape

        gt_t = gt.transpose(2,0,1)
        pred_t = pred.transpose(0, 2, 1)

        ssim_vals = [
            ssim(gt_t[i], pred_t[i], data_range=1.0)
            for i in range(pred_t.shape[0])
        ]

        mean_ssim = float(np.mean(ssim_vals))
        self.log_scalar("metrics/ssim", mean_ssim, step)
