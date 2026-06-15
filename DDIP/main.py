import argparse
import logging
import os
import sys
import traceback

import numpy as np
import torch
import yaml

from solver_mri_2d import Diffusion as Diffusion_MRI_2d


torch.set_printoptions(sci_mode=False)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="DDIP 2D MRI self-refinement runner")

    # config / runtime
    parser.add_argument("--config", type=str, required=True, help="Config file name under configs/vp or absolute path")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", type=str, default="info", choices=["debug", "info", "warning", "critical"])
    parser.add_argument("--save_root", type=str, default="./results")

    # MRI task
    parser.add_argument("--deg", type=str, default="MRI")
    parser.add_argument("--mask_type", type=str, default="mat")
    parser.add_argument("--acc_factor", type=int, default=4)
    parser.add_argument("--center_fraction", type=float, default=0.08)
    parser.add_argument("--sigma_y", type=float, default=0.0)

    # diffusion sampling
    parser.add_argument("--T_sampling", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.85)
    parser.add_argument("--gamma", type=float, default=5.0)

    # LoRA test-time adaptation
    parser.add_argument("--adaptation", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=3, help="LoRA optimization steps per diffusion step")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--start_t", type=int, default=1000)
    parser.add_argument("--end_t", type=int, default=0)
    parser.add_argument("--adapt_every_k", type=int, default=1)

    # proposed iterative self-refinement
    parser.add_argument("--num_refine", type=int, default=3, help="Number of outer self-refinement iterations")

    args = parser.parse_args()

    if os.path.isabs(args.config) or os.path.exists(args.config):
        config_path = args.config
    else:
        config_path = os.path.join("configs", "vp", args.config)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper())
    logging.basicConfig(
        level=level,
        format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: %s", device)
    config.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, config


def main():
    args, config = parse_args_and_config()

    try:
        runner = Diffusion_MRI_2d(args, config)
        runner.sample()
    except Exception:
        logging.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())