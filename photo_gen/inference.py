import sys
import pdb
import os
import time
from icecream import ic

import torch
import numpy as np

import matplotlib.pyplot as plt

import hydra
from omegaconf import OmegaConf


from train3 import inference_parallele
from evaluation.evalgen import eval_metrics


def info(type, value, tb):
    """ Cette fonction est appelée lorsqu'une exception non gérée se produit.
    Elle nous met dans Pdb post-mortem pour nous permettre de déboguer
    l'erreur. 
        """
    import traceback
    # if the exception is a KeyboardInterrupt, we don't want to enter pdb
    if isinstance(value, KeyboardInterrupt):
        print("KeyboardInterrupt caught, exiting without pdb.")
        return
    # if in pdb already, we don't want to enter pdb again
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        print("Not entering pdb, already in interactive mode or stderr is not a tty.")
        return
    traceback.print_exception(type, value, tb)
    pdb.post_mortem(tb)


@hydra.main(config_path="config", config_name="config")
def test__inference(cfg):
    # sys.excepthook = info
    if cfg.debug:
        cfg.model.num_time_steps = 2  # Override num_time_steps when debug is True
    times = dict()
    # for n in range(2, 7):
    for n in [6]:
        n_images = 2 ** n
        cfg["n_images"] = n_images
        OmegaConf.set_struct(cfg, False)
        checkpoint_path = os.path.expanduser(cfg.checkpoint_load_path)
        images_savepath = "test"
        os.makedirs(images_savepath, exist_ok=True)
        t0 = time.time()
        images = inference_parallele(cfg=cfg,
                                checkpoint_path=checkpoint_path,
                                savepath=images_savepath,
                                meep_eval=True)
        t1 = time.time()
        t = t1-t0
        ic(t, n_images)
        ic(t/n_images)
        times[n_images] = t
        N = 2
        _, axes = plt.subplots(N, N)
        for i, ax in enumerate(axes.flatten()):
            ax.axis('off')
            ax.imshow(images[i].squeeze())
        plt.tight_layout()
        os.makedirs("test", exist_ok=True)
        plt.savefig(os.path.join("test", f"test_{n_images}.png"))
    print(times)
    # save times dict to a log file
    times_path = os.path.join("test", "times.txt")
    with open(times_path, 'w') as f:
        for n_images, time_taken in times.items():
            f.write(f"{n_images}: {time_taken:.2f} seconds\n")


@hydra.main(config_path="config", config_name="inference")
def main(cfg):
    if cfg.debug:
        cfg.model.num_time_steps = 2  # Override num_time_steps when debug is True
    OmegaConf.set_struct(cfg, False)
    savedir = 'nanophoto/diffusion/train3/'
    savedir = os.path.join(os.environ["SCRATCH"], savedir)
    if cfg.debug:
        savedir = os.path.join(savedir, 'debug')
    else:
        jobid = os.environ["SLURM_JOB_ID"]
        savedir = os.path.join(savedir, jobid)
    if cfg.inference_only:
        checkpoint_path = os.path.expanduser(cfg.checkpoint_load_path)
    else:
        checkpoint_path = os.path.join(savedir, "checkpoint.pt")

    os.makedirs(savedir, exist_ok=True)

    for model_cfg in cfg.models:
        modconfig = cfg.model
        modconfig.n_epochs = int(modconfig.n_compute_steps / n_samples)

        images_savepath = os.path.join(savedir, "images")
        os.makedirs(images_savepath, exist_ok=True)
        t0 = time.time()
        images, fom = inference(checkpoint_path=checkpoint_path, savepath=images_savepath,
                                cfg=cfg)
        t1 = time.time()
        ic(t1-t0, "inference time")
        plt.hist(fom, bins=100)
        plt.title("fom histogram")
        plt.savefig(os.path.join(savedir, "hist.png"))
        plt.close()
        dataset_cfg = OmegaConf.create([{"name": os.environ["SLURM_JOB_ID"],
                                         "path": images_savepath}])
        eval_metrics(dataset_cfg, os.path.dirname(datapath))


if __name__ == "__main__":
    test__inference()
    # main()
