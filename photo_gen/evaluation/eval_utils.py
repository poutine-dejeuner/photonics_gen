import os
from pathlib import Path
import yaml
import warnings

import torch

from icecream import ic


def normalise(images):
    """rescales the array to have only values in [0,1]"""
    if images.max() - images.min() == 0:
        return images
    images = (images - images.min())/(images.max() - images.min())

    return images


def tonumpy(array):
    if type(array) is torch.Tensor:
        return array.detach().cpu().numpy()
    else:
        return array



def update_stats_yaml(stats_path: Path, new_stats: dict) -> None:
    """
    Update or add a key-value pair in a YAML file.
    If the key exists, overwrite it. If not, add it.
    """

    # Load existing data or create empty dict
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            try:
                stats = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                stats = {}
    else:
        stats = {}

    stats = new_stats | stats

    # Write back to file
    with open(stats_path, 'w', encoding='utf-8') as f:
        yaml.dump(stats, f, default_flow_style=False)

def make_config(path):
    """
    search subdirectories of path. If dir/images contains an images.npy and
    some subdirectory of wandb contains a files/config.yaml, then search
    config.yaml for a "n_samples" key and record the following entries in a
    list
    "name": dir
    "path": dir/images
    "n_samples": config["n_samples"]
    Finally, save the discovered entries in a config.yaml file
    """
    datasets = []
    for subdir in Path(path).iterdir():
        if subdir.is_dir() and (subdir / 'images.npy').exists():
            images_path = subdir / 'images.npy'
            config_path = subdir / 'wandb' / 'files' / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                n_samples = config.get('n_samples', None)
                datasets.append({
                    "name": subdir.name,
                    "path": str(images_path),
                    "n_samples": n_samples
                })
    with open(os.path.join(path, 'datasets.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(datasets, f)


def find_files(rootdir: Path, filenames: list[str])-> list[ Path]:
    """
    Search for files with specified names in the current directory and its subdirectories.

    Args:
        rootdir: Root directory to search in
        filenames (list[str]): List of filenames to search for.

    Returns:
        dict[str, Path] | None: A dictionary where keys are filenames and values are Paths where the files were found,
                               or None if not all files were found.
    """
    found_files = dict()

    for dirpath, dirnames, files in os.walk(Path(rootdir)):
        for filename in filenames:
            if filename in files and filename not in found_files:
                filepath = (Path(dirpath) / filename)
                found_files[filename] = filepath

        if set(file.name for file in found_files.values()) == set(filenames):
            if len(filenames) == 1:
                found_files = [filepath]
            else:
                found_files = [found_files[key] for key in filenames]
            return found_files
    warnings.warn(f"Some files were not found {found_files}, {filenames}")
    return None

def make_config_chercheuse(rootdir: str|None = None) -> None:
    """
    For all subdir in rootdir, in subdir search for a file named images.npy and
    one named config.yaml. return those files
    """
    if rootdir is None:
        rootdir = Path(os.getcwd())
    else:
        rootdir = Path(rootdir)

    #enumerates all subdirectories of rootdir
    datasets = []
    for subdir in Path(rootdir).iterdir():
        if subdir.is_dir():
            #walk subdirs of subdir searching for images.npy and config.yaml
            images_path = None
            config_path = None
            for dirpath, dirnames, filenames in os.walk(subdir):
                for filename in filenames:
                    if filename == 'images.npy':
                        images_path = os.path.join(dirpath, filename)
                    elif filename == 'config.yaml':
                        config_path = os.path.join(dirpath, filename)
                if images_path and config_path:
                    break
            if images_path and config_path:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                n_samples = config.get('train_set_size', {}).get('value', None)
                datasets.append({
                    "name": subdir.name,
                    "path": images_path,
                    "n_samples": n_samples
                })
    with open(os.path.join(rootdir, 'datasets.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(datasets, f) 
