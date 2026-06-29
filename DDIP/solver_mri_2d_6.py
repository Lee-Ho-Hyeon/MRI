import os
import logging
import time
import glob
import json
import sys

from tensorboard import summary
import odl
import functools

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import torch.utils.data as data

from datasets import get_dataset

import torchvision.utils as tvu
import lpips

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, classifier_defaults, args_to_dict
from guided_diffusion.utils import get_alpha_schedule
import random

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.linalg import orth
from scipy import ndimage
from pathlib import Path

from physics.mri import MulticoilMRI, SinglecoilMRI_comp
from time import time
from utils import shrink, CG, clear, batchfy, _Dz, _DzT, get_mask, real_to_nchw_comp, comp_to_nchw_real, PSNR, SSIM

# adaptation
from lora.lora import adapt_model, LoraInjectedConv1d, LoraInjectedConv2d, LoraInjectedLinear
from lora.adaptation import adapt_loss_fn



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def make_support_mask_from_aty(aty, threshold_ratio=0.05, dilation_iters=6):
    """
    Make an image-domain foreground/support mask from A^H y.

    Important:
    - This is NOT the k-space undersampling mask.
    - The undersampling mask is already used inside A and A^H.
    - This function uses the measured-data initial reconstruction A^H y
      to estimate the head/brain support region in image domain.

    Args:
        aty: complex tensor, shape [B, 1, H, W]
        threshold_ratio: threshold relative to slice-wise maximum magnitude
        dilation_iters: expand the mask boundary to avoid cutting anatomy

    Returns:
        support_mask: real float tensor, shape [B, 1, H, W], values 0 or 1
    """
    mag = torch.abs(aty).detach().cpu().numpy()
    support = np.zeros_like(mag, dtype=np.float32)

    for b in range(mag.shape[0]):
        for c in range(mag.shape[1]):
            m = mag[b, c]
            max_val = float(np.max(m))
            if max_val <= 1e-12:
                continue

            mask = m > (threshold_ratio * max_val)

            # Keep only the largest connected component to suppress isolated noise.
            labeled, num = ndimage.label(mask)
            if num > 0:
                sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
                largest = int(np.argmax(sizes)) + 1
                mask = labeled == largest

            # Fill small holes inside the head region and slightly dilate the boundary.
            mask = ndimage.binary_fill_holes(mask)
            if dilation_iters is not None and int(dilation_iters) > 0:
                mask = ndimage.binary_dilation(mask, iterations=int(dilation_iters))

            support[b, c] = mask.astype(np.float32)

    return torch.from_numpy(support).to(device=aty.device, dtype=torch.float32)


def apply_support_mask_complex(x, support_mask):
    """Apply a real-valued support mask to a complex image tensor."""
    return x * support_mask.to(device=x.device, dtype=x.real.dtype)


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        # [MODIFIED 3] CUDA-only execution.
        # This project is intended to run on GPU. Keeping all tensors on the
        # same CUDA device prevents CPU/CUDA tensor mismatch during CG-SENSE.
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this MRI reconstruction code.")
            device = torch.device("cuda")
        self.device = torch.device(device)

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    # [MODIFIED 4] Utilities for slice-wise LoRA reset.
    # LoRA is adapted within one slice/refinement process, but reset when a new
    # slice starts. This avoids information leakage from slice 0 to slice 1.
    def _clone_trainable_state(self, model):
        return {
            name: param.detach().clone().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def _restore_trainable_state(self, model, state):
        if state is None:
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in state:
                    param.copy_(state[name].to(param.device))

    def sample(self):
        cls_fn = None
        config_dict = vars(self.config.model)
        model = create_model(**config_dict)
        ckpt = self.config.model.model_ckpt
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"Model ckpt loaded from {ckpt}")
        model.to(self.device)
        model.convert_to_fp32()
        model.dtype = torch.float32
        model.eval()
        # Augment with adaptation parameters
        self.initial_trainable_state = None
        if self.args.adaptation:
            adapt_kwargs = {'r': int(self.args.lora_rank)}
            adapt_model(model, adapt_kwargs=adapt_kwargs)

            # [MODIFIED 4] Save the initial trainable parameters right after
            # LoRA injection. These values are restored at the start of each slice.
            self.initial_trainable_state = self._clone_trainable_state(model)

        self.adaptation = True if self.args.adaptation else False
        print('Run DDS 2D for MRI reconstruction.',
            f'{self.args.T_sampling} sampling steps. ',
            f'Task: {self.args.deg}. '
            f'Adaptation?: {self.adaptation}'
            )
        self.simplified_ddnm_plus(model)
            
            
    def simplified_ddnm_plus(self, model):
        args, config = self.args, self.config
        img_size = config.data.image_size

        root_list = []
        if config.data.vol_name != "all":
            root = Path(config.data.root) / f"{config.data.vol_name}"
            root_list.append(root)
        else:
            root = Path(config.data.root)
            vol_names = os.listdir(root)
            for vol in vol_names:
                root_list.append((root / vol))
            
        print(f"Retrieving test data: {config.data.dataset}")
        
        # Iterate over all vols
        for root in root_list:
            vol = str(root).split('/')[-1]
            print(f"root: {root}")
            print(f"vol: {vol}")
        
            # Specify save directory for saving generated samples
            save_root = Path(f'{self.args.save_root}/{self.config.data.dataset}/{vol}/{self.args.mask_type}_acc{self.args.acc_factor}/cg_gamma{self.args.gamma}')
            save_root = save_root / f"adapt_{self.adaptation}" / f"lr{self.args.lr}_{self.args.num_steps}" / f"{self.args.start_t}_{self.args.end_t}_every{self.args.adapt_every_k}"
            save_root.mkdir(parents=True, exist_ok=True)

            irl_types = ['input', 'recon', 'label', 'progress', 'support_mask', 'undersampling_mask']
            for t in irl_types:
                save_root_f = save_root / t
                save_root_f.mkdir(parents=True, exist_ok=True)
            
            # read all data: TODO
            print("Loading all data")
            root_img = root / "slice"
            root_mps = root / "mps"
            fname_list = sorted(os.listdir(root_img))
            x_orig = []
            mps_orig = []
            for fname in fname_list:
                img = torch.from_numpy(np.load(os.path.join(root_img, fname)))
                mps = torch.from_numpy(np.load(os.path.join(root_mps, fname)))
                h, w = img.shape
                c, h, w = mps.shape
                img = img.view(1, 1, h, w)
                mps = mps.view(1, c, h, w)
                x_orig.append(img)
                mps_orig.append(mps)
            x_orig = torch.cat(x_orig, dim=0)
            mps_orig = torch.cat(mps_orig, dim=0)
            print(f"Data loaded shape - img: {x_orig.shape}")
            print(f"                    mps: {mps_orig.shape}")
            
            img_shape = (x_orig.shape[0], config.data.channels, img_size, img_size)
            
            # MRI forward operator
            mask = get_mask(
                torch.zeros([1, 1, img_size, img_size]), 
                img_size,
                1,
                type=self.args.mask_type,
                acc_factor=self.args.acc_factor, 
                center_fraction=self.args.center_fraction,
            ).to(self.device)

            # [UNDERSAMPLING MASK VISUALIZATION]
            # Save the exact k-space sampling mask used by the forward model A.
            # This is different from the image-domain support_mask.
            # - undersampling_mask: k-space sampling pattern
            # - support_mask: image-domain foreground/background mask
            mask_save_dir = save_root / "undersampling_mask"
            mask_save_dir.mkdir(parents=True, exist_ok=True)

            mask_for_vis = mask.detach().cpu()
            if torch.is_complex(mask_for_vis):
                mask_for_vis = torch.abs(mask_for_vis)

            mask_np = mask_for_vis.squeeze().numpy()
            if mask_np.ndim > 2:
                # Robust fallback for unexpected extra singleton/batch dimensions.
                mask_np = np.squeeze(mask_np)
                while mask_np.ndim > 2:
                    mask_np = mask_np[0]

            mask_np = np.asarray(mask_np, dtype=np.float32)
            mask_binary = (np.abs(mask_np) > 0).astype(np.float32)

            plt.imsave(
                str(mask_save_dir / "kspace_mask.png"),
                mask_binary,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            np.save(str(mask_save_dir / "kspace_mask.npy"), mask_binary)

            total_points = int(mask_binary.size)
            sampled_points = int(mask_binary.sum())
            actual_sample_ratio = float(sampled_points / total_points) if total_points > 0 else 0.0
            actual_acceleration = float(1.0 / actual_sample_ratio) if actual_sample_ratio > 0 else float("inf")

            with open(str(mask_save_dir / "kspace_mask_info.json"), "w") as f:
                json.dump(
                    {
                        "mask_type": str(self.args.mask_type),
                        "acc_factor_arg": int(self.args.acc_factor),
                        "center_fraction": float(self.args.center_fraction),
                        "mask_shape": list(mask_binary.shape),
                        "sampled_points": sampled_points,
                        "total_points": total_points,
                        "actual_sample_ratio": actual_sample_ratio,
                        "actual_acceleration": actual_acceleration,
                        "note": (
                            "This is the k-space undersampling mask used inside A/A^H. "
                            "It is not the image-domain support mask used for background suppression."
                        ),
                    },
                    f,
                    indent=2,
                )

            print(
                f"[Mask] Saved k-space undersampling mask: {mask_save_dir / 'kspace_mask.png'} "
                f"(sample ratio={actual_sample_ratio:.6f}, acceleration={actual_acceleration:.3f})"
            )

            A_funcs = MulticoilMRI(mask=mask)
            
            # Alias
            A = lambda z, mps: A_funcs._A(z, mps)
            AT = lambda z, mps: A_funcs._AT(z, mps)
            Ap = lambda z, mps: A_funcs._Adagger(z, mps)
            
            def Acg(x, mps, gamma):
                return x + gamma * A_funcs._AT(A_funcs._A(x, mps), mps)
            
            # [MODIFIED 1, 3] Keep measurement tensors on CUDA and add noise
            # directly to each measured k-space y_idx.
            y = torch.zeros_like(mps_orig, device=self.device)
            ATy = torch.zeros_like(x_orig, device=self.device)
            for idx in range(x_orig.shape[0]):
                x_idx = x_orig[idx:idx+1, ...].to(self.device)
                mps_idx = mps_orig[idx:idx+1, ...].to(self.device)

                y_idx = A(x_idx, mps_idx)

                # Measurement noise should be added to the current k-space,
                # not to the whole preallocated y tensor.
                if self.args.sigma_y > 0:
                    y_idx = y_idx + torch.randn_like(y_idx) * self.args.sigma_y

                ATy_idx = AT(y_idx, mps_idx)

                y[idx:idx+1, ...] = y_idx
                ATy[idx:idx+1, ...] = ATy_idx

                input = np.abs(clear(ATy_idx))
                label = np.abs(clear(x_idx))
                plt.imsave(str(save_root / "input" / f"{str(idx).zfill(3)}.png"), input, cmap='gray')
                plt.imsave(str(save_root / "label" / f"{str(idx).zfill(3)}.png"), label, cmap='gray')
                
            """
            Actual inference running...
            Self-refinement version (A-style):
            - The measured k-space y and ATy are fixed.
            - The previous reconstruction is used as the initialization of the next refinement round.
            - Intermediate reconstructions and metrics are saved.
            """
            cnt = 0
            psnr_avg = 0
            ssim_avg = 0
            nrmse_avg = 0

            # If args.num_refine is not defined in main.py, use 3 by default.
            num_refine = getattr(args, "num_refine", 3)
            all_metrics = {}

            for idx in range(x_orig.shape[0]):
                # [MODIFIED 4] Slice-wise LoRA reset.
                # For single-slice experiments this has no practical effect.
                # For multi-slice folders, each slice starts from the same
                # pretrained DDIP + freshly initialized LoRA state.
                if args.adaptation:
                    self._restore_trainable_state(model, self.initial_trainable_state)

                ATy_idx = ATy[idx:idx+1, ...].to(self.device)
                y_idx = y[idx:idx+1, ...].to(self.device)
                mps_idx = mps_orig[idx:idx+1, ...].to(self.device)
                label_np = np.abs(clear(x_orig[idx]))
                Acg_idx = functools.partial(Acg, mps=mps_idx, gamma=self.args.gamma)

                # [SUPPORT MASK]
                # The k-space undersampling mask cannot be multiplied directly in image domain.
                # Instead, we estimate an image-domain support mask from A^H y (ATy_idx).
                # This suppresses background/head-outside noise in sampling outputs and metrics.
                support_threshold = float(getattr(args, "support_threshold", 0.05))
                support_dilation_iters = int(getattr(args, "support_dilation_iters", 6))
                apply_support_mask = bool(getattr(args, "apply_support_mask", True))

                support_mask = make_support_mask_from_aty(
                    ATy_idx,
                    threshold_ratio=support_threshold,
                    dilation_iters=support_dilation_iters,
                )
                support_np = clear(support_mask).astype(np.float32)
                label_np_masked = label_np * support_np

                plt.imsave(
                    str(save_root / "support_mask" / f"{str(idx).zfill(3)}.png"),
                    support_np,
                    cmap="gray"
                )

                prev_recon = None
                refine_metrics = []

                for refine_idx in range(num_refine):
                    print(f"\n===== Slice {idx} / Refinement {refine_idx + 1}/{num_refine} =====")

                    # [MODIFIED 2] Use start_t, end_t, and T_sampling in the
                    # actual reverse diffusion schedule.
                    skip = max(1, config.diffusion.num_diffusion_timesteps // args.T_sampling)

                    num_total = config.diffusion.num_diffusion_timesteps
                    start_t = min(max(int(args.start_t), 0), num_total - 1)
                    end_t = min(max(int(args.end_t), 0), start_t)

                    # Generate an increasing list first, then reverse it.
                    # Example: start_t=999, end_t=0, skip=20
                    # reverse pairs: (999, 980), ..., (20, 0), (0, -1)
                    times = list(range(end_t, start_t + 1, skip))
                    if len(times) == 0 or times[-1] != start_t:
                        times.append(start_t)
                    times = sorted(set(times))

                    times_next = [-1] + times[:-1]
                    times_pair = list(zip(reversed(times), reversed(times_next)))
                    adapt_every_k = max(1, int(args.adapt_every_k))

                    # [MODIFIED 5] Refinement restart should be consistent with
                    # the diffusion timestep.
                    #
                    # prev_recon is an x0-like clean reconstruction from the
                    # previous refinement. It should NOT be fed directly as a
                    # high-timestep x_t. Instead, convert it to x_start_t by
                    # adding the amount of noise corresponding to start_t:
                    #
                    #   x_t = sqrt(alpha_t) * x0 + sqrt(1 - alpha_t) * eps
                    #
                    # This keeps the previous reconstruction as the base image
                    # while making the input distribution compatible with the
                    # reverse diffusion step.
                    if prev_recon is None:
                        # [ORDER FIX - INITIALIZATION]
                        # Use the measured-data initialization as the first x_t.
                        # This corresponds to x_0 = A^H y in the high-level
                        # explanation, instead of starting from pure random noise.
                        x = apply_support_mask_complex(ATy_idx.clone().to(self.device), support_mask)
                    else:
                        x0_prev = prev_recon.detach().to(self.device)

                        t_start = torch.full(
                            (x0_prev.shape[0],),
                            start_t,
                            device=self.device,
                            dtype=torch.long
                        )
                        a_start = compute_alpha(self.betas, t_start)

                        noise = torch.randn_like(x0_prev)
                        x = a_start.sqrt() * x0_prev + (1.0 - a_start).sqrt() * noise
                        if apply_support_mask:
                            x = apply_support_mask_complex(x, support_mask)

                    n = x.size(0)
                    x0_preds = []
                    xs = [x]
                    adapt_losses = []

                    # Reverse diffusion sampling
                    for step_idx, (i, j) in enumerate(tqdm.tqdm(
                        times_pair,
                        total=len(times_pair),
                        desc=f"slice {idx}, refine {refine_idx}"
                    )):
                        t = (torch.ones(n, device=self.device) * i)
                        next_t = (torch.ones(n, device=self.device) * j)

                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())

                        """
                        [ORDER FIX + IMAGE-DOMAIN LOSS]
                        Desired inner-step order:

                            1. DDIP denoising
                               DDIP forward: et = model(x_t, t)
                               z_t = x0_t

                            2. regularized CG-SENSE
                               z_dc = argmin_x ||A_SENSE x - y||^2 + lambda ||x - z_t||^2

                            3. Image-domain LoRA adaptation
                               loss = ||z_t - stop_gradient(z_dc)||^2
                               Update LoRA parameters using this image-domain loss.

                            4. DDIM update
                               Use z_dc from this timestep to produce x_{t_next}.

                        Important:
                        - Measured k-space y is still used in the CG-SENSE step.
                        - LoRA adaptation itself no longer uses the k-space loss
                          ||A_SENSE(z_dc) - y||^2.
                        - We do NOT run an additional DDIP inference after LoRA update
                          within the same timestep.
                        - Therefore, the updated LoRA parameters are reflected from the
                          next timestep's DDIP denoising step.
                        """
                        xt_comp = xs[-1].to(self.device)
                        xt_real = comp_to_nchw_real(xt_comp)

                        do_adapt = args.adaptation and (step_idx % adapt_every_k == 0)

                        if do_adapt:
                            optim = torch.optim.AdamW(
                                [p for p in model.parameters() if p.requires_grad],
                                lr=args.lr
                            )

                            # Step 1 -> Step 2 -> Step 3
                            # Recompute DDIP prior z_t for each LoRA step.
                            # CG-SENSE still uses measured k-space y through ATy_idx,
                            # but the LoRA loss is computed in the image domain.
                            # The z_dc used for DDIM below is the one obtained before
                            # the final optimizer step, so the just-updated LoRA affects
                            # the next timestep, not the already-computed current z_dc.
                            for _ in range(args.num_steps):
                                optim.zero_grad()

                                # 1. DDIP denoising / forward
                                et = model(xt_real, t)[:, :2]
                                z_t_real = (xt_real - et * (1 - at).sqrt()) / at.sqrt()
                                z_t = real_to_nchw_comp(z_t_real)

                                # 2. regularized CG-SENSE
                                # z_dc is used as an image-domain pseudo-target.
                                # Detach z_t for the CG target path so the LoRA gradient
                                # comes only from z_t in the image-domain loss below.
                                with torch.no_grad():
                                    z_t_target = z_t.detach()
                                    if apply_support_mask:
                                        z_t_target = apply_support_mask_complex(z_t_target, support_mask)
                                    bcg = z_t_target + self.args.gamma * ATy_idx
                                    z_dc = CG(Acg_idx, bcg, z_t_target, n_inner=5)
                                    if apply_support_mask:
                                        z_dc = apply_support_mask_complex(z_dc, support_mask)

                                # 3. Image-domain LoRA adaptation
                                # Previous k-space loss:
                                #     loss = adapt_loss_fn(A(z_dc, mps_idx), y_idx)
                                # New image-domain loss:
                                #     loss = ||z_t - stop_gradient(z_dc)||^2
                                z_t_for_loss = z_t
                                if apply_support_mask:
                                    z_t_for_loss = apply_support_mask_complex(z_t_for_loss, support_mask)

                                loss = F.mse_loss(
                                    comp_to_nchw_real(z_t_for_loss),
                                    comp_to_nchw_real(z_dc.detach())
                                )
                                loss.backward()
                                optim.step()

                                adapt_losses.append(float(loss.detach().cpu()))

                            # Use the current timestep's CG-SENSE result for DDIM update.
                            # The just-updated LoRA affects the next timestep.
                            z_dc_for_ddim = z_dc.detach()
                            et_for_ddim = et.detach()

                        else:
                            # No LoRA update at this timestep.
                            # Still follow Step 1 -> Step 2, then proceed to Step 4.
                            with torch.no_grad():
                                # 1. DDIP denoising / forward
                                et = model(xt_real, t)[:, :2]
                                z_t_real = (xt_real - et * (1 - at).sqrt()) / at.sqrt()
                                z_t = real_to_nchw_comp(z_t_real)

                                # 2. regularized CG-SENSE
                                if apply_support_mask:
                                    z_t = apply_support_mask_complex(z_t, support_mask)
                                bcg = z_t + self.args.gamma * ATy_idx
                                z_dc = CG(Acg_idx, bcg, z_t, n_inner=5)
                                if apply_support_mask:
                                    z_dc = apply_support_mask_complex(z_dc, support_mask)

                                z_dc_for_ddim = z_dc
                                et_for_ddim = et

                        # 4. DDIM update: move to the next diffusion timestep.
                        with torch.no_grad():
                            eta = self.args.eta
                            c1 = (1 - at_next).sqrt() * eta
                            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                            if j != 0:
                                et_comp = real_to_nchw_comp(et_for_ddim)
                                xt_next = (
                                    at_next.sqrt() * z_dc_for_ddim
                                    + c1 * torch.randn_like(z_dc_for_ddim)
                                    + c2 * et_comp
                                )
                            else:
                                xt_next = z_dc_for_ddim

                            if apply_support_mask:
                                xt_next = apply_support_mask_complex(xt_next, support_mask)
                                z_dc_for_ddim = apply_support_mask_complex(z_dc_for_ddim, support_mask)

                            x0_preds.append(z_dc_for_ddim.to("cpu"))
                            xs.append(xt_next.to("cpu"))

                    # Current refinement result
                    prev_recon = xs[-1].to(self.device)
                    if apply_support_mask:
                        prev_recon = apply_support_mask_complex(prev_recon, support_mask)

                    recon_np = np.abs(clear(prev_recon))
                    recon_np_masked = recon_np * support_np

                    metric_data_range = float(label_np_masked.max() - label_np_masked.min())
                    if metric_data_range <= 1e-12:
                        metric_data_range = float(label_np.max() - label_np.min() + 1e-12)

                    psnr = PSNR(recon_np_masked, label_np_masked)
                    ssim = SSIM(recon_np_masked, label_np_masked, data_range=metric_data_range)
                    nrmse = NRMSE(recon_np_masked, label_np_masked)

                    avg_adapt_loss = float(np.mean(adapt_losses)) if len(adapt_losses) > 0 else None
                    last_adapt_loss = float(adapt_losses[-1]) if len(adapt_losses) > 0 else None

                    refine_metrics.append({
                        "refine_idx": int(refine_idx),
                        "PSNR": float(psnr),
                        "SSIM": float(ssim),
                        "NRMSE": float(nrmse),
                        "adapt_loss_mean": avg_adapt_loss,
                        "adapt_loss_last": last_adapt_loss,
                    })

                    print(
                        f"Refine {refine_idx}: "
                        f"PSNR={psnr:.4f}, SSIM={ssim:.4f}, NRMSE={nrmse:.6f}, "
                        f"adapt_loss_mean={avg_adapt_loss}"
                    )

                    # Save intermediate reconstruction image
                    plt.imsave(
                        str(save_root / "progress" / f"{str(idx).zfill(3)}_refine_{str(refine_idx).zfill(2)}.png"),
                        recon_np_masked,
                        cmap="gray"
                    )

                    # Save intermediate reconstruction array
                    np.save(
                        str(save_root / "progress" / f"{str(idx).zfill(3)}_refine_{str(refine_idx).zfill(2)}.npy"),
                        prev_recon.detach().cpu().numpy()
                    )

                # Final reconstruction for this slice
                final_recon = np.abs(clear(prev_recon))
                final_recon_masked = final_recon * support_np

                metric_data_range = float(label_np_masked.max() - label_np_masked.min())
                if metric_data_range <= 1e-12:
                    metric_data_range = float(label_np.max() - label_np.min() + 1e-12)

                final_psnr = PSNR(final_recon_masked, label_np_masked)
                final_ssim = SSIM(final_recon_masked, label_np_masked, data_range=metric_data_range)
                final_nrmse = NRMSE(final_recon_masked, label_np_masked)

                psnr_avg += final_psnr
                ssim_avg += final_ssim
                nrmse_avg += final_nrmse
                cnt += 1

                plt.imsave(
                    str(save_root / "recon" / f"{str(idx).zfill(3)}.png"),
                    final_recon_masked,
                    cmap="gray"
                )

                all_metrics[f"slice_{str(idx).zfill(3)}"] = refine_metrics

            psnr_avg /= cnt
            ssim_avg /= cnt
            nrmse_avg /= cnt

            summary = {}
            summary["results"] = {
                "PSNR": float(psnr_avg),
                "SSIM": float(ssim_avg),
                "NRMSE": float(nrmse_avg),
            }
            summary["support_mask"] = {
                "source": "A^H y / ATy_idx",
                "threshold_ratio": float(getattr(args, "support_threshold", 0.05)),
                "dilation_iters": int(getattr(args, "support_dilation_iters", 6)),
                "applied_during_sampling": bool(getattr(args, "apply_support_mask", True)),
                "note": "This is an image-domain support mask, not the k-space undersampling mask."
            }
            summary["progress"] = all_metrics

            with open(str(save_root / "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)


def NRMSE(recon, label):
    return np.linalg.norm(recon - label) / (np.linalg.norm(label) + 1e-12)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

