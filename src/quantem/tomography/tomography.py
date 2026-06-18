import os
from pathlib import Path
from typing import Literal, Self, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from quantem.core.io.serialize import load as autoserialize_load
from quantem.core.ml.loss_functions import get_loss_module
from quantem.core.utils.filter import gaussian_filter_2d_stack, gaussian_kernel_1d
from quantem.core.utils.tomography_utils import torch_phase_cross_correlation
from quantem.tomography.dataset_models import (
    DatasetConstraintParams,
    DatasetConstraintsType,
    DatasetModelType,
    TomographyINRDataset,
    TomographyPixDataset,
)
from quantem.tomography.logger_tomography import LoggerTomography
from quantem.tomography.object_models import (
    ObjConstraintParams,
    ObjConstraintsType,
    ObjectINR,
    ObjectPixelated,
)
from quantem.tomography.radon.radon import iradon_torch, radon_torch
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.tomography_opt import TomographyOpt

import time

class Tomography(TomographyOpt, TomographyBase):
    """
    Class for handling all ML tomography reconstruction methods.
    Automatic handling between AD and INR-based tomography.
    """

    @classmethod
    def from_models(
        cls,
        dset: DatasetModelType,
        obj_model: ObjectINR,
        logger: LoggerTomography | None = None,
        device: str = "cuda",
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        return cls(
            dset=dset,
            obj_model=obj_model,
            logger=logger,
            device=device,
            rng=rng,
            verbose=verbose,
            _token=cls._token,
        )

    def reconstruct(
        self,
        num_iter: int = 10,
        batch_size: int = 1024,
        num_workers: int = 32,
        reset: bool = False,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        obj_constraints: dict | ObjConstraintsType | None = None,
        dset_constraints: dict | DatasetConstraintsType | None = None,
        num_samples_per_ray: int | list[tuple[int, int]] | None = None,
        profiling_mode: bool = False,
        val_fraction: float = 0.0,
        loss_type: Literal[
            "l2",
            "l1",
            "smooth_l1",
            "charbonnier",
            "llmse",
            "mse_log_mse",
        ] = "l2",
        loss_func_kwargs: dict = {},
        reset_dset: DatasetModelType | None = None,
        show_metrics: bool = False,
        gt_volume: torch.Tensor | np.ndarray | None = None,
        gt_defocus: torch.Tensor | np.ndarray | None = None,
        gt_stig: torch.Tensor | np.ndarray | None = None,
        ):
        """
        This function should be able to handle both AD and INR-based tomography reconstruction methods.
        I.e, auto-detection through the obj model type, while both share the same pose optimization.
        """

        # Check device consistency
        self.obj_model.to(self.device)

        # Saving batch size, num workers, and val fraction for reloading
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction

        if profiling_mode:
            if self.global_rank == 0:
                print("Profiling mode enabled.")

        if reset:
            raise NotImplementedError("Reset is not implemented yet.")

        new_scheduler = reset
        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
            self.set_optimizers()
            new_scheduler = True

        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
            new_scheduler = True

        if new_scheduler:
            self.set_schedulers(self.scheduler_params, num_iter=num_iter)

        if obj_constraints is not None:
            if isinstance(obj_constraints, dict):
                obj_constraints = ObjConstraintParams.parse_dict(obj_constraints)

            self.obj_model.constraints = obj_constraints

        if dset_constraints is not None:
            if isinstance(dset_constraints, dict):
                dset_constraints = DatasetConstraintParams.parse_dict(dset_constraints)

            self.dset.constraints = dset_constraints
        # Setting up DDP
        if not hasattr(self, "dataloader") or reset_dset is not None:
            if reset_dset is not None:
                print("Resetting Dataloader")
                print("Putting in params from previous dataset.")

                self.dset = reset_dset
                self.dset.to(self.device)

                if optimizer_params is not None:
                    self.optimizer_params = optimizer_params
                    self.set_optimizers()
                if scheduler_params is not None:
                    self.scheduler_params = scheduler_params
                    self.set_schedulers(self.scheduler_params)

            self.dataloader, self.sampler, self.val_dataloader, self.val_sampler = (
                self.setup_dataloader(
                    self.dset,
                    batch_size,
                    num_workers=num_workers,
                    val_fraction=val_fraction,
                )
            )

        # Type check for INR-based reconstruction
        if not isinstance(self.dset, TomographyINRDataset):
            raise NotImplementedError(
                "Only TomographyINRDataset is supported for this reconstruction method."
            )

        N = max(self.obj_model.shape)

        if num_samples_per_ray is None:
            num_samples_per_ray = max(self.obj_model.shape)
        else:
            if isinstance(num_samples_per_ray, int):
                num_samples_per_ray = num_samples_per_ray
            else:
                if len(num_samples_per_ray) != num_iter:
                    raise ValueError(
                        "num_samples_per_ray schedule must have the same length as num_iter"
                    )
                if self.global_rank == 0:
                    print("num_samples_per_ray schedule provided.")

        loss_func = get_loss_module(name=loss_type, dtype=self.obj_model.dtype, **loss_func_kwargs)


        pbar = tqdm(range(num_iter), disable=not self.verbose)
        for a0 in pbar:
            consistency_loss = torch.tensor(0.0, device=self.device)
            total_loss = torch.tensor(0.0, device=self.device)
            epoch_soft_constraint_loss = torch.tensor(0.0, device=self.device)
            if isinstance(self.obj_model, ObjectINR):
                self.obj_model.model.train()
            else:
                raise NotImplementedError(
                    "AD Pixelated reconstruction is not yet implemented. Use ObjectINR instead."
                )
            self.dset.train()
            # self._reset_iter_constraints()

            if self.sampler is not None:
                self.sampler.set_epoch(a0)

            if isinstance(num_samples_per_ray, list):
                curr_num_samples_per_ray = num_samples_per_ray[a0][1]
            else:
                curr_num_samples_per_ray = num_samples_per_ray

            for batch_idx, batch in enumerate(self.dataloader):
                self.zero_grad_all()
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=True,
                ):
                    # all_coords = self.dset.get_coords(batch, N, curr_num_samples_per_ray)

                    # all_densities = self.obj_model.forward(all_coords)

                    # integrated_densities = self.dset.integrate_rays(
                    #     all_densities,
                    #     curr_num_samples_per_ray,
                    #     len(batch["target_value"]),
                    # )
                    # In tomography.py reconstruction loop:
                    # all_coords = self.dset.get_coords(
                    if hasattr(self.dset, 'ray_pattern'):
                        all_coords, probe_weights = self.dset.get_coords(
                            batch, N, curr_num_samples_per_ray,
                            ray_pattern = self.dset.ray_pattern
                        )
                    else:
                        all_coords = self.dset.get_coords(
                            batch, N, curr_num_samples_per_ray,
                        )
                        probe_weights = None

                    all_densities = self.obj_model.forward(all_coords)

                    if probe_weights is not None:
                        integrated_densities = self.dset.integrate_rays_with_probe_weights(
                            all_densities,
                            probe_weights,
                            curr_num_samples_per_ray,
                            len(batch["target_value"]),
                        )
                    else:
                        integrated_densities = self.dset.integrate_rays(
                            all_densities,
                            curr_num_samples_per_ray,
                            len(batch["target_value"]),
                        )

                pred = integrated_densities.float()
                soft_constraints_loss = 0.0
                if self.num_epochs > 0:
                    soft_constraints_loss = self.obj_model.apply_soft_constraints(all_coords, pred)

                target = batch["target_value"].to(self.device, non_blocking=True).float()

                batch_consistency_loss = loss_func(pred, target)

                soft_constraints_loss += self.dset.apply_soft_constraints()

                epoch_soft_constraint_loss += soft_constraints_loss.detach()

                batch_loss = batch_consistency_loss.float() + soft_constraints_loss.float()

                batch_loss.backward()

                # if batch_idx == 0 and a0 == 0 and self.global_rank == 0:
                #     if hasattr(self.dset, '_stig_2_params') and self.dset._stig_2_params.grad is not None:
                #         print(f"\n=== Stig_2 Gradient Check ===")
                #         print(f"stig_2 value: {self.dset._stig_2_params.data}")
                #         print(f"stig_2 grad: {self.dset._stig_2_params.grad}")
                #         print(f"stig_2 grad norm: {self.dset._stig_2_params.grad.norm().item()}")
                #     else:
                #         print(f"\n=== WARNING: No gradient on stig_2_params! ===")
                


                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.obj_model.model.parameters(), max_norm=1.0)
                self.step_optimizers()
                total_loss += batch_loss.detach()
                consistency_loss += batch_consistency_loss.detach()

            if self.world_size > 1:
                dist.all_reduce(total_loss, dist.ReduceOp.AVG)
                dist.all_reduce(consistency_loss, dist.ReduceOp.AVG)
                dist.all_reduce(epoch_soft_constraint_loss, dist.ReduceOp.AVG)

            total_loss = total_loss.item() / len(self.dataloader)
            consistency_loss = consistency_loss.item() / len(self.dataloader)
            epoch_soft_constraint_loss = epoch_soft_constraint_loss.item() / len(self.dataloader)

            self.step_schedulers(loss=total_loss)
            # TODO: Maybe reorganize the losses so that the order makes sense lol.


            avg_val_loss = None
            if self.val_dataloader is not None:
                print("Validating...")
                self.obj_model.model.eval()
                self.dset.eval()
                with torch.no_grad():
                    val_loss = torch.tensor(0.0, device=self.device)

                    for batch in self.val_dataloader:
                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=torch.bfloat16,
                            enabled=True,
                        ):
                            # Handle probe weights in validation too
                            if hasattr(self.dset, 'ray_pattern'):
                                all_coords, probe_weights = self.dset.get_coords(
                                    batch, N, curr_num_samples_per_ray,
                                    ray_pattern=self.dset.ray_pattern
                                )
                            else:
                                all_coords = self.dset.get_coords(
                                    batch, N, curr_num_samples_per_ray,
                                )
                                probe_weights = None

                            all_densities = self.obj_model.forward(all_coords)
                            
                            # Use probe weights if available
                            if probe_weights is not None:
                                integrated_densities = self.dset.integrate_rays_with_probe_weights(
                                    all_densities,
                                    probe_weights,
                                    curr_num_samples_per_ray,
                                    len(batch["target_value"]),
                                )
                            else:
                                integrated_densities = self.dset.integrate_rays(
                                    all_densities,
                                    curr_num_samples_per_ray,
                                    len(batch["target_value"]),
                                )

                            target = (
                                batch["target_value"].to(self.device, non_blocking=True).float()
                            )

                            batch_val_loss = torch.nn.functional.mse_loss(
                                integrated_densities, target
                            )

                            val_loss += batch_val_loss.detach()

                    avg_val_loss = val_loss.item() / len(self.val_dataloader)






            # avg_val_loss = None
            # if self.val_dataloader is not None:
            #     print("Validating...")
            #     self.obj_model.model.eval()
            #     self.dset.eval()
            #     with torch.no_grad():
            #         val_loss = torch.tensor(0.0, device=self.device)

            #         for batch in self.val_dataloader:
            #             with torch.autocast(
            #                 device_type=self.device.type,
            #                 dtype=torch.bfloat16,
            #                 enabled=True,
            #             ):
            #                 all_coords = self.dset.get_coords(batch, N, curr_num_samples_per_ray)

            #                 all_densities = self.obj_model.forward(all_coords)

            #                 integrated_densities = self.dset.integrate_rays(
            #                     all_densities,
            #                     curr_num_samples_per_ray,
            #                     len(batch["target_value"]),
            #                 )

            #                 target = (
            #                     batch["target_value"].to(self.device, non_blocking=True).float()
            #                 )

            #                 batch_val_loss = torch.nn.functional.mse_loss(
            #                     integrated_densities, target
            #                 )

            #                 val_loss += batch_val_loss.detach()

            #         avg_val_loss = val_loss.item() / len(self.val_dataloader)

            metrics = torch.tensor(
                [total_loss, consistency_loss, epoch_soft_constraint_loss], device=self.device
            )

            if self.world_size > 1:
                dist.all_reduce(metrics, dist.ReduceOp.AVG)

            total_loss, consistency_loss, epoch_soft_constraint_loss = metrics.tolist()

            pbar.set_description(
                f"Reconstruction | Loss: {total_loss:.5e}, Consistency Loss: {consistency_loss:.5e}, Soft Constraint Loss: {epoch_soft_constraint_loss:.5e}"
            )

            self._epoch_losses.append(total_loss)
            self._consistency_losses.append(consistency_loss)
            self.append_learning_rates(self.get_current_lrs())
            self.obj_model._soft_constraint_losses.append(epoch_soft_constraint_loss)
            if avg_val_loss is not None:
                self._val_losses.append(avg_val_loss)

            if self.logger is not None:

                convergence_angle = None
                stig_2 = None
                
                if hasattr(self.dset, '_convergence_angle_params'):
                    convergence_angle = self.dset._convergence_angle_params[0]
                
                if hasattr(self.dset, '_stig_2_params'):
                    stig_2_tensor = self.dset._stig_2_params  # Scaled values
                    # Scale back to physical units for logging
                    scale = self.dset.STIG_SCALE if hasattr(self.dset, 'STIG_SCALE') else 1.0
                    stig_2 = (stig_2_tensor[0].item() * scale, stig_2_tensor[1].item() * scale)
                    
                if (
                    self.logger.log_images_every > 0
                    and self.num_epochs % self.logger.log_images_every == 0
                ):
                    pred_full = self.obj_model.obj_view

                    if self.global_rank == 0:
                        self.logger.log_iter_images(
                            pred_volume=pred_full,
                            dataset_model=self.dset,
                            iter=self.num_epochs,
                            gt_z_focus = gt_defocus
                        )
                    pbar.set_description(
                        f"Reconstruction | Loss: {total_loss:.5e}, Consistency Loss: {consistency_loss:.5e}, Soft Constraint Loss: {epoch_soft_constraint_loss:.5e} | Images Logged"
                                )
                    # if hasattr(self.dset, '_z_focus_params'):
                    #     print("Logging defocus...")
                    #     self.logger.log_defocus(
                    #         dataset_model=self.dset,
                    #         gt_defocus=gt_defocus,
                    #     )
                    if gt_volume is not None:
                        print("Logging SSIM...")
                        self.logger.log_ssim(
                            pred_volume=pred_full,
                            gt_volume=gt_volume,
                            step=self.num_epochs,
                        )

                if self.global_rank == 0:
                    self.logger.log_iter(
                        object_model=self.obj_model,
                        iter=self.num_epochs,
                        consistency_loss=consistency_loss,
                        total_loss=total_loss,
                        learning_rates=self.get_current_lrs(),
                        num_samples_per_ray=curr_num_samples_per_ray,
                        val_loss=avg_val_loss if self.val_dataloader is not None else None,
                        convergence_angle=convergence_angle,
                        stig_2=stig_2,
                        gt_stig=gt_stig,
                    )

                self.logger.flush()
            if not self.verbose:
                if self.global_rank == 0:
                    print(
                        f"Reconstruction Epoch {self.num_epochs} | Loss: {total_loss:.5e}, Consistency Loss: {consistency_loss:.5e}, Soft Constraint Loss: {epoch_soft_constraint_loss:.5e}"
                    )
        if show_metrics and self.world_size == 1:
            self.plot_losses()

    def generate_forward_projections(
        self,
        phantom_vol: np.ndarray | torch.Tensor,
        true_focus: np.ndarray | torch.Tensor,
        batch_size: int = 128,
        num_workers: int = 32,
        num_samples_per_ray: int = 100,
        save_path: str | None = None,
        drop_last: bool = False,
    ) -> np.ndarray | None:
        """
        Generate forward projections using the ray-based model (like reconstruct but single pass).
        Uses the dataloader infrastructure for proper batching and DDP.
        
        Parameters:
        -----------
        phantom_vol : ndarray or tensor
            Ground truth volume (nz, ny, nx)
        true_focus : ndarray or tensor
            True focus values (normalized -1 to 1) for each tilt
        batch_size : int
            Pixels per batch
        num_workers : int
            Number of dataloader workers
        num_samples_per_ray : int
            Number of samples along each ray
        save_path : str, optional
            Path to save the generated tilt series (rank 0 only)
            
        Returns:
        --------
        tilt_series : ndarray or None
            Generated projections (n_tilts, ny, nx) on rank 0, None on other ranks
        """
        import torch.nn.functional as F
        from tqdm import tqdm
        
        # Convert to torch if needed
        if isinstance(phantom_vol, np.ndarray):
            phantom_vol = torch.from_numpy(phantom_vol).float()
        if isinstance(true_focus, np.ndarray):
            true_focus = torch.from_numpy(true_focus).float()
        
        phantom_vol = phantom_vol.to(self.device)
        true_focus = true_focus.to(self.device)
        
        N = phantom_vol.shape[0]
        ny, nx = phantom_vol.shape[1], phantom_vol.shape[2]
        n_tilts = len(self.dset.tilt_angles)
        
        if self.global_rank == 0:
            print(f"Generating forward projections with {self.dset.num_rays} rays, {num_samples_per_ray} samples/ray")
            print(f"Using {self.world_size} GPUs, batch size {batch_size}")
        
        # Sample phantom using grid_sample (efficient trilinear interpolation)
        def sample_phantom(coords):
            """Sample phantom at given coordinates."""
            n_points = coords.shape[0]
            grid = coords.view(1, n_points, 1, 1, 3)
            phantom_5d = phantom_vol.unsqueeze(0).unsqueeze(0)
            sampled = F.grid_sample(
                phantom_5d, 
                grid, 
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            return sampled.view(n_points)
        
        # Override focus in dataset with ground truth
        self.dset._z_focus_params = torch.nn.Parameter(true_focus)
        
        # Set up dataloader if not already done
        if not hasattr(self, "dataloader"):
            self.dataloader, self.sampler, _, _ = self.setup_dataloader(
                self.dset,
                batch_size,
                num_workers=num_workers,
                val_fraction=0.0,
                drop_last = False,
            )
        
        pixel_coverage = torch.zeros((n_tilts, ny, nx), dtype=torch.int32, device=self.device)


        # Initialize storage for projections
        # Each GPU will accumulate its batches into full projections
        projections = torch.zeros((n_tilts, ny, nx), device=self.device)
        
        if self.global_rank == 0:
            print("Processing batches...")
        
        # Single forward pass through all data (like one epoch of reconstruct)
        self.dset.eval()  # Not training, just forward pass
        
        with torch.no_grad():  # No gradients needed
            pbar = tqdm(self.dataloader, disable=(self.global_rank != 0), desc="Forward projection")
            

            batch_count = 0
            pixel_count = 0
            for batch_idx, batch in enumerate(pbar):
                batch_count += 1
                pixel_count += len(batch["pixel_i"])
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=True,
                ):
                    # Get ray coordinates and probe weights
                    all_coords, probe_weights = self.dset.get_coords(
                        batch, N, num_samples_per_ray,
                        ray_pattern=self.dset.ray_pattern
                    )
                    
                    # Sample phantom at ray coordinates
                    all_densities = sample_phantom(all_coords)
                    
                    # Integrate rays
                    if probe_weights is not None:
                        integrated_densities = self.dset.integrate_rays_with_probe_weights(
                            all_densities,
                            probe_weights,
                            num_samples_per_ray,
                            len(batch["target_value"]),
                        )
                    else:
                        integrated_densities = self.dset.integrate_rays(
                            all_densities,
                            num_samples_per_ray,
                            len(batch["target_value"]),
                        )
                    
                    # Place predictions back into full projections
                    # batch contains pixel_i, pixel_j, projection_idx
                    pixel_i = batch["pixel_i"].to(self.device, non_blocking=True)
                    pixel_j = batch["pixel_j"].to(self.device, non_blocking=True)
                    proj_idx = batch["projection_idx"].to(self.device, non_blocking=True)
                    
                    # Scatter predictions to correct positions
                    for k in range(len(integrated_densities)):
                        projections[proj_idx[k], pixel_i[k], pixel_j[k]] = integrated_densities[k].float()
                        pixel_coverage[proj_idx[k], pixel_i[k], pixel_j[k]] += 1  # Track coverage

        if self.global_rank == 0:
            print(f"Rank 0 processed {pixel_count} pixels in {batch_count} batches")
    

        # Gather all projections to rank 0
        if self.world_size > 1:
            if self.global_rank == 0:
                print("Gathering projections from all GPUs...")
            dist.all_reduce(pixel_coverage, op=dist.ReduceOp.SUM)
            dist.all_reduce(projections, op=dist.ReduceOp.SUM)
            dist.barrier()
            
            # All-reduce to combine partial results from all GPUs
            dist.all_reduce(projections, op=dist.ReduceOp.SUM)
            
            # Barrier to ensure completion
            dist.barrier()
        
        # Convert to numpy and save on rank 0
        if self.global_rank == 0:

            dead_pixels = (pixel_coverage == 0).sum().item()
            duplicate_pixels = (pixel_coverage > 1).sum().item()
            total_pixels = n_tilts * ny * nx
            
            print(f"\nPixel Coverage Diagnostics:")
            print(f"  Total pixels: {total_pixels}")
            print(f"  Dead pixels (coverage=0): {dead_pixels} ({100*dead_pixels/total_pixels:.2f}%)")
            print(f"  Duplicate pixels (coverage>1): {duplicate_pixels} ({100*duplicate_pixels/total_pixels:.2f}%)")
            print(f"  Coverage min/max/mean: {pixel_coverage.min().item()}/{pixel_coverage.max().item()}/{pixel_coverage.float().mean().item():.2f}")
            
            # Fix dead pixels by averaging neighbors (simple inpainting)
            if dead_pixels > 0:
                print("Fixing dead pixels with neighbor interpolation...")
                projections_np = projections.cpu().numpy()
                
                for t in range(n_tilts):
                    dead_mask = (pixel_coverage[t] == 0).cpu().numpy()
                    if dead_mask.any():
                        # Simple average of 4-neighbors
                        from scipy.ndimage import binary_dilation, convolve
                        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4
                        
                        for _ in range(3):  # Iterate to fill isolated pixels
                            neighbor_avg = convolve(projections_np[t], kernel, mode='constant')
                            projections_np[t][dead_mask] = neighbor_avg[dead_mask]
                            dead_mask = (projections_np[t] == 0) & dead_mask  # Update mask
            else:
                projections_np = projections.cpu().numpy()
            
            # projections_np = projections.cpu().numpy()
            # Handle duplicate pixels (average them)
            if duplicate_pixels > 0:
                print("Averaging duplicate pixel contributions...")
                coverage_np = pixel_coverage.cpu().numpy()
                coverage_np[coverage_np == 0] = 1  # Avoid division by zero
                projections_np = projections_np / coverage_np



            if save_path is not None:
                np.save(save_path, projections_np)
                print(f"Saved forward projections to {save_path}")

            return projections_np
        else:
            return None




    # --- Helper Functions ---

    def save_volume(self, path: str = "recon_volume.npz", overwrite: bool = False):
        """
        Saves volume to a numpy array file. Does not save the full Tomography object.
        """
        if self.global_rank == 0:
            if not overwrite and os.path.exists(path):
                raise FileExistsError(
                    f"File {path} already exists. Use overwrite=True to overwrite."
                )
            print(f"Saving volume to {path}")
            np.savez(path, volume=self.obj_model.obj_view)

        if torch.distributed.is_initialized():
            print("Barrier")
            torch.distributed.barrier()

    # Loading and Saving
    @classmethod
    def _recursive_load_from_path(cls, path: str):
        return autoserialize_load(path)

    @classmethod
    def from_file(
        cls,
        path: str,
        device: str = "cpu",
    ) -> Self:
        tomography = cls._recursive_load_from_path(path)
        tomography.to(device)
        tomography._rebuild_dataloader(
            batch_size=tomography.batch_size,
            num_workers=tomography.num_workers,
            val_fraction=tomography.val_fraction,
        )
        return tomography

    def _rebuild_dataloader(self, batch_size: int, num_workers: int, val_fraction: float):
        """
        Rebuilds the dataloader due to persistent workers error when reloading the object.
        """
        self.dataloader, self.sampler, self.val_dataloader, self.val_sampler = (
            self.setup_dataloader(
                self.dset,
                batch_size,
                num_workers=num_workers,
                val_fraction=val_fraction,
            )
        )

    def save(
        self,
        path: str | Path,
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: str | type | Sequence[str | type] = ["dataloader"],
        compression_level: int | None = 4,
    ) -> None:
        super(Tomography, self).save(
            path=path,
            mode=mode,
            store=store,
            skip=skip,
            compression_level=compression_level,
        )

    def plot_losses(self):
        fig, ax = plt.subplots(figsize=(10, 4), ncols=2)

        ax[0].plot(self._epoch_losses, label="Total Training Loss")
        if len(self._val_losses) > 0:
            ax[0].plot(self._val_losses, label="Validation Loss")
        ax[0].legend()

        for key, value in self._lrs.items():
            ax[1].plot(value, label=key)

        ax[1].legend()
        ax[0].legend()
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax[0].set_xlabel("Epoch")
        ax[1].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[1].set_ylabel("Learning Rate")


class TomographyConventional(TomographyBase):
    """
    Class for handling all conventional tomography reconstruction methods.
    Will also handle choosing the appropriate dataset model to use.
    """

    @classmethod
    def from_models(
        cls,
        dset: TomographyPixDataset,
        obj_model: ObjectPixelated,
        logger: LoggerTomography | None = None,
        device: str = "cuda",
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        return cls(
            dset=dset,
            obj_model=obj_model,
            logger=logger,
            device=device,
            rng=rng,
            verbose=verbose,
            _token=cls._token,
        )

    def reconstruct(
        self,
        num_iter: int = 10,
        obj_constraints: dict | ObjConstraintsType | None = None,
        mode: Literal["sirt", "fbp"] = "sirt",
        relaxation: float = 0.25,
        reset: bool = False,
        inline_alignment: bool = False,
        smoothing_sigma: float | None = None,
        show_metrics: bool = False,
    ):
        if obj_constraints is not None:
            if isinstance(obj_constraints, dict):
                obj_constraints = ObjConstraintParams.parse_dict(obj_constraints)

            self.obj_model.constraints = obj_constraints

        pbar = tqdm(
            range(num_iter),
            desc=f"{mode} Reconstruction | Loss: {0:.4f}",
            disable=not self.verbose,
        )
        if mode == "sirt" or mode == "fbp":
            proj_forward = torch.zeros_like(self.dset.tilt_stack).permute(2, 0, 1)
        else:
            proj_forward = torch.zeros_like(self.dset.tilt_stack)

        if smoothing_sigma is not None:
            gaussian_kernel = gaussian_kernel_1d(smoothing_sigma).to(self.device)
        else:
            gaussian_kernel = None

        patience = self.dset.tilt_angles.max() // 10
        for iter in pbar:
            proj_forward, loss = self._reconstruction_epoch(
                inline_alignment=inline_alignment,
                mode=mode,
                proj_forward=proj_forward,
                gaussian_kernel=gaussian_kernel,
                relaxation=relaxation,
            )

            pbar.set_description(f"{mode} Reconstruction | Loss: {loss.item():.4f}")

            self._epoch_losses.append(loss.item())

            # Change relaxation parameter if loss greater than last epoch
            if len(self._epoch_losses) > 1 and self._epoch_losses[-1] > self._epoch_losses[-2]:
                if patience == 0:
                    relaxation *= 0.85
                    print(f"Relaxation parameter changed to: {relaxation}")
                    patience = 10
                else:
                    patience -= 1

            if mode == "fbp":
                break

        if show_metrics:
            self.plot_losses()

    # --- Conventional reconstruction method ---
    def _adaptive_relaxation(self, n_power_iter: int = 10) -> float:
        raise NotImplementedError(
            "Adaptive relaxation hasn't been implemented, please input a valid relaxation parameter."
        )

    def _reconstruction_epoch(
        self,
        inline_alignment: bool,
        mode: Literal["sirt", "fbp"],
        proj_forward: torch.Tensor,
        relaxation: float,
        gaussian_kernel: torch.Tensor | None = None,
    ):
        loss = 0

        if relaxation == 0.0:
            relaxation = self._adaptive_relaxation()
            print(f"Adaptive relaxation: {relaxation}")
        if inline_alignment:
            for ind in range(len(self.dset.tilt_angles)):
                im_proj = proj_forward[:, ind, :]
                im_meas = self.dset.forward(ind).target  # type: ignore
                shift = torch_phase_cross_correlation(im_proj, im_meas)
                if torch.linalg.norm(shift) <= 32:
                    shifted = torch.fft.ifft2(
                        torch.fft.fft2(im_meas)
                        * torch.exp(
                            -2j
                            * np.pi
                            * (
                                shift[0]
                                * torch.fft.fftfreq(
                                    im_meas.shape[0], device=im_meas.device
                                ).unsqueeze(1)
                                + shift[1]
                                * torch.fft.fftfreq(im_meas.shape[1], device=im_meas.device)
                            )
                        )
                    ).real

                    proj_forward[:, ind, :] = shifted

        if mode == "sirt" or mode == "fbp":
            proj_forward = radon_torch(
                self.obj_model.obj,
                theta=self.dset.tilt_angles,
                device=self.device,
            )

            error = self.dset.tilt_stack.permute(2, 0, 1) - proj_forward

            correction = iradon_torch(
                error,
                theta=self.dset.tilt_angles,
                device=self.device,
                filter_name="ramp",
                circle=True,
            )

            normalization = iradon_torch(
                torch.ones_like(error),
                theta=self.dset.tilt_angles,
                device=self.device,
                circle=True,
                filter_name=None,
            )

            normalization[normalization == 0] = 1e-6

            correction /= normalization

            self.obj_model.obj += correction * relaxation

            if gaussian_kernel is not None:
                self.obj_model.obj = gaussian_filter_2d_stack(self.obj_model.obj, gaussian_kernel)

        loss = torch.mean(torch.abs(error))

        return proj_forward, loss

    # --- Helper Functions ---

    def plot_losses(self):
        fig, ax = plt.subplots()
        ax.plot(self._epoch_losses)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Reconstruction Loss")
        ax.set_yscale("log")
        plt.show()
