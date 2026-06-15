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


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

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
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

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
        if self.args.adaptation:
            adapt_kwargs = {'r': int(self.args.lora_rank)}
            adapt_model(model, adapt_kwargs=adapt_kwargs)

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

            irl_types = ['input', 'recon', 'label', 'progress']
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
            ).to("cuda")
            A_funcs = MulticoilMRI(mask=mask)
            
            # Alias
            A = lambda z, mps: A_funcs._A(z, mps)
            AT = lambda z, mps: A_funcs._AT(z, mps)
            Ap = lambda z, mps: A_funcs._Adagger(z, mps)
            
            def Acg(x, mps, gamma):
                return x + gamma * A_funcs._AT(A_funcs._A(x, mps), mps)
            
            y = torch.zeros_like(mps_orig)
            ATy = torch.zeros_like(x_orig)
            for idx in range(x_orig.shape[0]):
                x_idx = x_orig[idx:idx+1, ...].to(self.device)
                mps_idx = mps_orig[idx:idx+1, ...].to(self.device)
                y_idx = A(x_idx, mps_idx)
                y += torch.randn_like(y) * self.args.sigma_y
                ATy_idx = AT(y_idx, mps_idx)
                y[idx, ...] = y_idx
                ATy[idx, ...] = ATy_idx
                input = np.abs(clear(ATy_idx))
                label = np.abs(clear(x_idx))
                plt.imsave(str(save_root / "input" / f"{str(idx).zfill(3)}.png"), input, cmap='gray')
                plt.imsave(str(save_root / "label" / f"{str(idx).zfill(3)}.png"), label, cmap='gray')
                
            """
            Actual inference running...
            MoDL-style DDIP self-refinement:
            - 10b: DDIP/LoRA produces a diffusion prior z_n.
            - 10a: Explicit CG data-consistency step combines z_n with fixed measured k-space y.
            - The measured k-space y and ATy are fixed across all refinement iterations.
            - The next refinement starts from the DC-corrected reconstruction x_{n+1}.
            """
            cnt = 0
            psnr_avg = 0
            ssim_avg = 0
            nrmse_avg = 0

            num_refine = getattr(args, "num_refine", 3)
            dc_cg_iter = getattr(args, "dc_cg_iter", 10)
            all_metrics = {}

            # additional folders for MoDL-style intermediate outputs
            for t in ["prior", "dc"]:
                (save_root / t).mkdir(parents=True, exist_ok=True)

            for idx in range(x_orig.shape[0]):
                ATy_idx = ATy[idx:idx+1, ...].to(self.device)
                y_idx = y[idx:idx+1, ...].to(self.device)
                mps_idx = mps_orig[idx:idx+1, ...].to(self.device)
                label_np = np.abs(clear(x_orig[idx]))
                Acg_idx = functools.partial(Acg, mps=mps_idx, gamma=self.args.gamma)

                # x_n in MoDL. First iteration starts from random noise, then uses DC-corrected x_{n+1}.
                cur_recon = None
                refine_metrics = []

                for refine_idx in range(num_refine):
                    print(f"\n===== Slice {idx} / MoDL-style Refinement {refine_idx + 1}/{num_refine} =====")

                    if cur_recon is None:
                        x = torch.randn_like(x_orig[idx:idx+1, ...]).to(self.device)
                    else:
                        x = cur_recon.detach().to(self.device)

                    skip = config.diffusion.num_diffusion_timesteps // args.T_sampling
                    n = x.size(0)
                    xs = [x]
                    x0_preds = []

                    times = range(0, 1000, skip)
                    times_next = [-1] + list(times[:-1])
                    times_pair = zip(reversed(times), reversed(times_next))

                    adapt_losses = []

                    # ------------------------------------------------------------
                    # 10b-like step: DDIP / LoRA denoising prior generation
                    # Produces z_n, but does not stop there.
                    # ------------------------------------------------------------
                    for i, j in tqdm.tqdm(
                        times_pair,
                        total=len(times),
                        desc=f"slice {idx}, refine {refine_idx} - DDIP prior"
                    ):
                        t = (torch.ones(n) * i).to("cuda")
                        next_t = (torch.ones(n) * j).to("cuda")

                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())

                        # LoRA test-time adaptation.
                        # The denoiser input is image-domain xt, but the current adaptation loss
                        # still uses k-space consistency: adapt_loss_fn(A(x0_t), y).
                        if args.adaptation:
                            xt = xs[-1].to("cuda")
                            xt = comp_to_nchw_real(xt)

                            optim = torch.optim.AdamW(
                                [p for p in model.parameters() if p.requires_grad],
                                lr=args.lr
                            )

                            for _ in range(args.num_steps):
                                optim.zero_grad()

                                et = model(xt, t)[:, :2]
                                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                                x0_t = real_to_nchw_comp(x0_t)

                                # Light inner DC used only to stabilize adaptation.
                                bcg_adapt = x0_t + self.args.gamma * ATy_idx
                                x0_t = CG(Acg_idx, bcg_adapt, x0_t, n_inner=1)

                                loss = adapt_loss_fn(A(x0_t, mps_idx), y_idx)
                                loss.backward()
                                optim.step()

                                adapt_losses.append(float(loss.detach().cpu()))

                        # Inference after adaptation.
                        with torch.no_grad():
                            xt = xs[-1].to("cuda")
                            xt = comp_to_nchw_real(xt)

                            et = model(xt, t)[:, :2]

                            # Tweedie estimate: image-domain estimate from diffusion model.
                            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                            x0_t = real_to_nchw_comp(x0_t)

                            # Keep the original DDIP inner DC step. This stabilizes each diffusion step.
                            bcg_inner = x0_t + self.args.gamma * ATy_idx
                            x0_t = CG(Acg_idx, bcg_inner, x0_t, n_inner=5)

                            eta = self.args.eta
                            c1 = (1 - at_next).sqrt() * eta
                            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                            if j != 0:
                                et = real_to_nchw_comp(et)
                                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
                            else:
                                xt_next = x0_t

                            x0_preds.append(x0_t.to("cpu"))
                            xs.append(xt_next.to("cpu"))

                    # z_n: DDIP denoiser/prior output of this refinement.
                    prior_img = xs[-1].to(self.device)
                    prior_np = np.abs(clear(prior_img))

                    # ------------------------------------------------------------
                    # 10a-like step: explicit MoDL/SENSE-style data consistency.
                    # x_{n+1} = argmin_x gamma*||A(x)-y||^2 + ||x-z_n||^2
                    # Equivalent normal equation in code:
                    #     (I + gamma A^H A)x = z_n + gamma A^H y
                    # ------------------------------------------------------------
                    with torch.no_grad():
                        bcg_dc = prior_img + self.args.gamma * ATy_idx
                        dc_recon = CG(Acg_idx, bcg_dc, prior_img, n_inner=dc_cg_iter)

                    # Use DC-corrected reconstruction as x_{n+1} for the next refinement.
                    cur_recon = dc_recon.detach()
                    recon_np = np.abs(clear(cur_recon))

                    psnr = PSNR(recon_np, label_np)
                    ssim = SSIM(recon_np, label_np, data_range=label_np.max())
                    nrmse = NRMSE(recon_np, label_np)

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

                    # Save DDIP prior z_n and DC-corrected x_{n+1} separately.
                    fname = f"{str(idx).zfill(3)}_refine_{str(refine_idx).zfill(2)}"
                    plt.imsave(str(save_root / "prior" / f"{fname}.png"), prior_np, cmap="gray")
                    np.save(str(save_root / "prior" / f"{fname}.npy"), prior_img.detach().cpu().numpy())

                    plt.imsave(str(save_root / "dc" / f"{fname}.png"), recon_np, cmap="gray")
                    np.save(str(save_root / "dc" / f"{fname}.npy"), cur_recon.detach().cpu().numpy())

                    # Keep progress folder as the main visual history of final DC-corrected outputs.
                    plt.imsave(str(save_root / "progress" / f"{fname}.png"), recon_np, cmap="gray")
                    np.save(str(save_root / "progress" / f"{fname}.npy"), cur_recon.detach().cpu().numpy())

                # Final reconstruction for this slice
                final_recon = np.abs(clear(cur_recon))
                final_psnr = PSNR(final_recon, label_np)
                final_ssim = SSIM(final_recon, label_np, data_range=label_np.max())
                final_nrmse = NRMSE(final_recon, label_np)

                psnr_avg += final_psnr
                ssim_avg += final_ssim
                nrmse_avg += final_nrmse
                cnt += 1

                plt.imsave(str(save_root / "recon" / f"{str(idx).zfill(3)}.png"), final_recon, cmap="gray")
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
            summary["progress"] = all_metrics
            summary["method"] = {
                "name": "MoDL-style DDIP self-refinement",
                "description": "Each refinement alternates DDIP/LoRA prior generation (10b-like) and explicit CG data consistency (10a-like).",
                "dc_cg_iter": int(dc_cg_iter),
                "gamma": float(self.args.gamma),
            }

            with open(str(save_root / "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)


def NRMSE(recon, label):
    return np.linalg.norm(recon - label) / (np.linalg.norm(label) + 1e-12)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
