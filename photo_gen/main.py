"""
Training script for comparing different generative models (wGAN, VAE, etc.) with diffusion models.
This script follows the same hydra configuration structure as train4.py for consistency.
"""
import os
import datetime
from pathlib import Path

import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from photo_gen.evaluation.evaluation import evaluate_model
from photo_gen.utils.utils import make_wandb_run, normalize, debug_cleanup

from icecream import ic, install

ic.configureOutput(includeContext=True)
install()
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg):
    ic(cfg.train_set_size)
    OmegaConf.set_struct(cfg, False)

    # Apply debug configurations if in debug mode
    if cfg.debug and hasattr(cfg, 'debug_config'):
        for key, value in cfg.debug_config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
                print(f"Debug mode: Setting cfg.{key} = {value}")
            elif hasattr(cfg.model, key):
                setattr(cfg.model, key, value)
                print(f"Debug mode: Setting cfg.model.{key} = {value}")

    # Use config for savedir construction
    base_savedir = cfg.output.base_path
    savedir = Path(os.environ.get("SCRATCH", "")) / base_savedir
    if cfg.debug:
        savedir = savedir / cfg.output.debug_subdir
    jobid = os.environ.get("SLURM_JOB_ID", "local_run")
    # timestamp = datetime.datetime.now().strftime(cfg.output.timestamp_format)
    hydra_overrides = HydraConfig.get().overrides.task
    ic(hydra_overrides)
    savedir = savedir / (f"{jobid}" + "_".join(hydra_overrides))
    # savedir = savedir / f"{jobid}_{timestamp}"

    model_name = cfg.model.name
    savedir = savedir / model_name

    if cfg.inference_only:
        checkpoint_path = os.path.expanduser(cfg.checkpoint_load_path)
    else:
        checkpoint_path = savedir / "checkpoint.pt"

    train_fn = hydra.utils.instantiate(cfg.train)
    inference_fn = hydra.utils.instantiate(cfg.model.inference)

    os.makedirs(savedir, exist_ok=True)
    datapath = os.path.expanduser(cfg.data_path)
    images_savepath = savedir / "images"
    os.makedirs(images_savepath, exist_ok=True)

    data = np.load(datapath)
    print(f"Loaded data shape: {data.shape}")
    print(f"Data dtype: {data.dtype}, min: {data.min():.3f}, max: {data.max():.3f}")

    n_samples = cfg.train_set_size 
    ic(n_samples)
    data = data[:n_samples]
    if len(data.shape) == 3:
        data = data[:, None, :, :]

    assert data.shape[-2:] == cfg.image_shape, f"Data image size {data.shape[-2:]} does not match config image size {cfg.image_shape}"

    cfg.model.debug = cfg.debug
    
    n_epochs = int(cfg.n_compute_steps / n_samples)
    assert n_epochs > 0, ic(n_epochs, cfg.model.n_compute_steps, n_samples)
    cfg.model.n_epochs = n_epochs

    # Log model configuration if training (parameter estimation removed due to missing dependencies)
    if cfg.inference_only is False:
        target_params = cfg.get('target_params', None)
        if target_params:
            print(f"Target parameter count: {target_params:,}")
        print(f"Model configuration:")
        print(f"  - Model type: {cfg.model.get('_target_', 'unknown')}")
        print(f"  - Latent dim: {cfg.model.get('latent_dim', 'N/A')}")
        print(f"  - Hidden dim: {cfg.model.get('hidden_dim', 'N/A')}")
        print(f"  - Image channels: {cfg.model.get('img_channels', 'N/A')}")
        print(f"  - Image size: {cfg.model.get('img_size', 'N/A')}")

    if cfg.inference_only is False and cfg.evaluation_only is False:
        # Copy training parameters from model config to top level for UNet compatibility
        training_params = cfg.training.params_to_copy
        for param in training_params:
            if hasattr(cfg.model, param):
                setattr(cfg, param, getattr(cfg.model, param))
                print(f"Copied {param}: {getattr(cfg.model, param)} to top-level config")

        run = None
        if cfg.logger.enabled:
            run = make_wandb_run(cfg, savedir)

        train_fn(data=data, checkpoint_path=checkpoint_path,
                 savedir=savedir, run=run, cfg=cfg)

    if not cfg.evaluation_only:
        # Generate images using inference
        images = inference_fn(checkpoint_path=checkpoint_path, 
                              savepath=images_savepath, cfg=cfg)
        
        # Save the images path for future evaluation_only runs
        config_save_path = savedir / "evaluation_config.yaml" 
        evaluation_config = {
            "evaluation_images_path": str(images_savepath),
            "original_savedir": str(savedir),
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(config_save_path, 'w') as f:
            import yaml
            yaml.dump(evaluation_config, f, default_flow_style=False)
        print(f"Saved evaluation config to: {config_save_path}")
        print(f"To run evaluation only on these images, use: evaluation_only=True evaluation_images_path={images_savepath}")

    if cfg.evaluation_only:
        # Load images from previous run
        if cfg.evaluation_images_path is None:
            raise ValueError("evaluation_images_path must be set when evaluation_only=True")
       
        import glob
        images_path = os.path.expanduser(cfg.evaluation_images_path)
        print(f"Loading images for evaluation from: {images_path}")
        
        # Load all .npy files from the specified directory
        image_files = glob.glob(os.path.join(images_path, "*.npy"))
        if not image_files:
            raise FileNotFoundError(f"No .npy files found in {images_path}")
        
        print(f"Found {len(image_files)} .npy files: {image_files}")
        # Load the first images file (assuming generated images are saved as .npy)
        images = np.load(image_files[0])
        print(f"Loaded {len(images)} images for evaluation from {image_files[0]}")

    results = evaluate_model(images, savedir, cfg)

    if cfg.logger.enabled:
        run.log(results)

    # if debuggin, ask the user if they want to delete the savedir
    # Debug cleanup
    if cfg.debug:
        debug_cleanup(savedir, interactive=True)

    return

if __name__ == '__main__':
    main()
