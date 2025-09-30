import os
import random
from typing import List, Callable, Any, Union, Tuple
import json
import functools
import psutil

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from einops import rearrange
from omegaconf import OmegaConf


def load_wandb_config(raw_cfg: dict):
    import ast
    """
    prend une config sauvegardée par wandb et fait un objet omegaconf avec
    """
    parsed = {}
    for k, v in raw_cfg.items():
        val = v["value"] if isinstance(v, dict) and "value" in v else v
        # si c'est une string qui ressemble à un dict ou une liste → parser
        if isinstance(val, str):
            try:
                val = ast.literal_eval(val)
            except Exception:
                pass  # laisser en string si ça ne marche pas
        parsed[k] = val
    return OmegaConf.create(parsed)


def save_checkpoint(model, optimizer, ema, epoch, scaler, checkpoint_path):
    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")


def set_seed(seed: int = 42):
    if seed is -1:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def find_file(root_dir, name):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if name in filenames:
            filepath = os.path.join(dirpath, name)
            return filepath
    return None


def select_diverse_images(images: Union[np.ndarray, torch.Tensor], 
                         M: int, 
                         distance_metric: str = 'euclidean') -> Tuple[np.ndarray, List[int]]:
    """
    Select M highly different images from an array of N images using a greedy approach.
    
    This function uses a greedy algorithm that starts with a random image and iteratively
    selects the image that maximizes the minimum distance to all previously selected images.
    
    Args:
        images (np.ndarray or torch.Tensor): Array of images with shape (N, H, W) or (N, C, H, W)
        M (int): Number of images to select (must be <= N)
        distance_metric (str): Distance metric to use ('euclidean', 'cosine', 'manhattan')
    
    Returns:
        tuple: (selected_images, selected_indices)
            - selected_images: Array of M selected images with same shape as input
            - selected_indices: List of indices of selected images in the original array
    
    Example:
        >>> images = np.random.rand(100, 64, 64)  # 100 images of size 64x64
        >>> selected, indices = select_diverse_images(images, 10, 'euclidean')
        >>> print(f"Selected {len(selected)} diverse images with indices: {indices}")
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    N = images.shape[0]
    if M >= N:
        return images, list(range(N))
    
    if M <= 0:
        raise ValueError("M must be positive")
    
    # Flatten images for distance computation
    flattened = images.reshape(N, -1)
    
    # Normalize for cosine distance if needed
    if distance_metric == 'cosine':
        norms = np.linalg.norm(flattened, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        flattened = flattened / norms
    
    # Start with a random image
    np.random.seed(42)  # For reproducibility
    selected_indices = [np.random.randint(0, N)]
    
    # Greedily select the most different images
    for _ in range(M - 1):
        max_min_distance = -1
        best_candidate = -1
        
        for candidate in range(N):
            if candidate in selected_indices:
                continue
            
            # Calculate minimum distance to already selected images
            min_distance = float('inf')
            for selected_idx in selected_indices:
                if distance_metric == 'euclidean':
                    dist = np.linalg.norm(flattened[candidate] - flattened[selected_idx])
                elif distance_metric == 'cosine':
                    # For normalized vectors, cosine distance = 2 - 2*dot_product
                    dot_product = np.dot(flattened[candidate], flattened[selected_idx])
                    dist = 2 - 2 * dot_product
                elif distance_metric == 'manhattan':
                    dist = np.sum(np.abs(flattened[candidate] - flattened[selected_idx]))
                else:
                    raise ValueError(f"Unknown distance metric: {distance_metric}")
                
                min_distance = min(min_distance, dist)
            
            # Select the candidate with maximum minimum distance
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_candidate = candidate
        
        selected_indices.append(best_candidate)
    
    selected_images = images[selected_indices]
    return selected_images, selected_indices


def select_diverse_images_kmeans(images: Union[np.ndarray, torch.Tensor], 
                                M: int) -> Tuple[np.ndarray, List[int]]:
    """
    Alternative method using K-means clustering to select diverse images.
    
    This function clusters the images into M clusters and selects one representative
    image from each cluster (the one closest to the cluster centroid).
    
    Args:
        images (np.ndarray or torch.Tensor): Array of images with shape (N, H, W) or (N, C, H, W)
        M (int): Number of images to select (must be <= N)
    
    Returns:
        tuple: (selected_images, selected_indices)
            - selected_images: Array of M selected images
            - selected_indices: List of indices of selected images
    
    Raises:
        ImportError: If scikit-learn is not installed
    
    Example:
        >>> images = np.random.rand(100, 64, 64)  # 100 images of size 64x64
        >>> selected, indices = select_diverse_images_kmeans(images, 10)
        >>> print(f"Selected {len(selected)} diverse images using K-means")
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("sklearn is required for K-means clustering. Install with: pip install scikit-learn")
    
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    N = images.shape[0]
    if M >= N:
        return images, list(range(N))
    
    if M <= 0:
        raise ValueError("M must be positive")
    
    # Flatten images for clustering
    flattened = images.reshape(N, -1)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=M, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(flattened)
    
    # Select one image from each cluster (closest to centroid)
    selected_indices = []
    for cluster_id in range(M):
        cluster_mask = cluster_labels == cluster_id
        if not np.any(cluster_mask):
            continue
            
        cluster_images = flattened[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        centroid = kmeans.cluster_centers_[cluster_id]
        
        # Find the image closest to the centroid
        distances = np.linalg.norm(cluster_images - centroid, axis=1)
        closest_idx = np.argmin(distances)
        selected_indices.append(cluster_indices[closest_idx])
    
    # If we have fewer clusters than requested, fill with random selection
    remaining_indices = [i for i in range(N) if i not in selected_indices]
    np.random.seed(42)  # For reproducibility
    while len(selected_indices) < M and remaining_indices:
        idx = np.random.choice(remaining_indices)
        selected_indices.append(idx)
        remaining_indices.remove(idx)
    
    selected_images = images[selected_indices]
    return selected_images, selected_indices


def calculate_diversity_score(images: Union[np.ndarray, torch.Tensor], 
                             distance_metric: str = 'euclidean') -> float:
    """
    Calculate the average pairwise distance between images as a diversity score.
    Higher scores indicate more diverse sets of images.
    
    Args:
        images (np.ndarray or torch.Tensor): Array of images with shape (N, H, W) or (N, C, H, W)
        distance_metric (str): Distance metric to use ('euclidean', 'cosine', 'manhattan')
    
    Returns:
        float: Average pairwise distance (diversity score)
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    N = images.shape[0]
    if N < 2:
        return 0.0
    
    flattened = images.reshape(N, -1)
    
    if distance_metric == 'cosine':
        norms = np.linalg.norm(flattened, axis=1, keepdims=True)
        norms[norms == 0] = 1
        flattened = flattened / norms
    
    total_distance = 0.0
    count = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            if distance_metric == 'euclidean':
                dist = np.linalg.norm(flattened[i] - flattened[j])
            elif distance_metric == 'cosine':
                dot_product = np.dot(flattened[i], flattened[j])
                dist = 2 - 2 * dot_product
            elif distance_metric == 'manhattan':
                dist = np.sum(np.abs(flattened[i] - flattened[j]))
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")
            
            total_distance += dist
            count += 1
    
    return total_distance / count


def memory_monitor(enabled: bool = True, save_stats: bool = True, save_path:
        str|None = None):
    """
    Decorator to monitor both RAM and GPU memory (VRAM) usage for training and inference functions.
    
    Args:
        enabled: Whether to enable memory monitoring
        save_stats: Whether to save memory statistics to a JSON file
        save_path: Path to save the memory stats (default: current working directory)
    """
    def decorator(func: Callable) -> Callable:
        if not enabled:
            return func
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get initial RAM state
            process = psutil.Process()
            initial_ram = process.memory_info().rss / 1e9  # RSS in GB
            total_ram = psutil.virtual_memory().total / 1e9
            
            if not torch.cuda.is_available():
                print("CUDA not available, monitoring RAM only")
                
                print(f"\n=== MEMORY MONITOR START: {func.__name__} ===")
                print(f"Total RAM: {total_ram:.2f}GB")
                print(f"Initial RAM: {initial_ram:.3f}GB")
                
                # Track peak RAM during execution
                peak_ram = initial_ram
                
                # Monkey patch to track RAM during execution
                original_forward = None
                if hasattr(torch.nn.Module, 'forward'):
                    original_forward = torch.nn.Module.__call__
                    
                    def monitored_forward(self, *forward_args, **forward_kwargs):
                        nonlocal peak_ram
                        result = original_forward(self, *forward_args, **forward_kwargs)
                        current_ram = process.memory_info().rss / 1e9
                        peak_ram = max(peak_ram, current_ram)
                        return result
                    
                    torch.nn.Module.__call__ = monitored_forward
                
                try:
                    result = func(*args, **kwargs)
                    final_ram = process.memory_info().rss / 1e9
                    peak_ram = max(peak_ram, final_ram)
                    
                    ram_overhead = peak_ram - initial_ram
                    ram_efficiency = (peak_ram / total_ram) * 100
                    
                    print(f"\n=== MEMORY MONITOR RESULTS: {func.__name__} ===")
                    print(f"Initial RAM: {initial_ram:.3f}GB")
                    print(f"Peak RAM: {peak_ram:.3f}GB")
                    print(f"Final RAM: {final_ram:.3f}GB")
                    print(f"RAM overhead: {ram_overhead:.3f}GB")
                    print(f"RAM efficiency: {ram_efficiency:.1f}%")
                    
                    if save_stats:
                        stats = {
                            'function_name': func.__name__,
                            'total_ram_gb': total_ram,
                            'initial_ram_gb': initial_ram,
                            'peak_ram_gb': peak_ram,
                            'final_ram_gb': final_ram,
                            'ram_overhead_gb': ram_overhead,
                            'ram_efficiency_percent': ram_efficiency,
                            'cuda_available': False
                        }
                        
                        filename = f"memory_stats_{func.__name__}.json"
                        filepath = os.path.join(save_path or os.getcwd(), filename)
                        
                        with open(filepath, 'w') as f:
                            json.dump(stats, f, indent=2)
                        print(f"Memory stats saved to: {filepath}")
                    
                    return result
                    
                finally:
                    if original_forward:
                        torch.nn.Module.__call__ = original_forward
                    print(f"=== MEMORY MONITOR END: {func.__name__} ===\n")
            
            # Clear cache before starting
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Initial memory state (GPU)
            initial_vram = torch.cuda.memory_allocated() / 1e9
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"\n=== MEMORY MONITOR START: {func.__name__} ===")
            print(f"Total RAM: {total_ram:.2f}GB")
            print(f"Total VRAM: {total_vram:.2f}GB")
            print(f"Initial RAM: {initial_ram:.3f}GB")
            print(f"Initial VRAM: {initial_vram:.3f}GB")
            
            # Start timing
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            
            # Track peak memory during execution
            peak_vram = initial_vram
            peak_ram = initial_ram
            memory_samples = [initial_vram]
            ram_samples = [initial_ram]
            
            # Monkey patch to track memory during execution
            original_forward = None
            if hasattr(torch.nn.Module, 'forward'):
                original_forward = torch.nn.Module.__call__
                
                def monitored_forward(self, *forward_args, **forward_kwargs):
                    nonlocal peak_vram, peak_ram
                    result = original_forward(self, *forward_args, **forward_kwargs)
                    current_vram = torch.cuda.memory_allocated() / 1e9
                    current_ram = process.memory_info().rss / 1e9
                    peak_vram = max(peak_vram, current_vram)
                    peak_ram = max(peak_ram, current_ram)
                    return result
                
                torch.nn.Module.__call__ = monitored_forward
            
            try:
                # Execute the original function
                result = func(*args, **kwargs)
                
                # Final memory measurement
                torch.cuda.synchronize()
                end_time.record()
                torch.cuda.synchronize()
                
                final_vram = torch.cuda.memory_allocated() / 1e9
                reserved_vram = torch.cuda.memory_reserved() / 1e9
                final_ram = process.memory_info().rss / 1e9
                peak_vram = max(peak_vram, final_vram)
                peak_ram = max(peak_ram, final_ram)
                execution_time = start_time.elapsed_time(end_time)
                
                # Memory statistics
                vram_overhead = peak_vram - initial_vram
                ram_overhead = peak_ram - initial_ram
                vram_efficiency = (peak_vram / total_vram) * 100
                ram_efficiency = (peak_ram / total_ram) * 100
                
                print(f"\n=== MEMORY MONITOR RESULTS: {func.__name__} ===")
                print(f"Execution time: {execution_time:.2f}ms")
                print(f"Initial RAM: {initial_ram:.3f}GB")
                print(f"Peak RAM: {peak_ram:.3f}GB")
                print(f"Final RAM: {final_ram:.3f}GB")
                print(f"RAM overhead: {ram_overhead:.3f}GB")
                print(f"RAM efficiency: {ram_efficiency:.1f}%")
                print(f"Initial VRAM: {initial_vram:.3f}GB")
                print(f"Peak VRAM: {peak_vram:.3f}GB")
                print(f"Final VRAM: {final_vram:.3f}GB")
                print(f"Reserved VRAM: {reserved_vram:.3f}GB")
                print(f"VRAM overhead: {vram_overhead:.3f}GB")
                print(f"VRAM efficiency: {vram_efficiency:.1f}%")
                
                # Save detailed statistics
                if save_stats:
                    stats = {
                        'function_name': func.__name__,
                        'total_ram_gb': total_ram,
                        'initial_ram_gb': initial_ram,
                        'peak_ram_gb': peak_ram,
                        'final_ram_gb': final_ram,
                        'ram_overhead_gb': ram_overhead,
                        'ram_efficiency_percent': ram_efficiency,
                        'total_vram_gb': total_vram,
                        'initial_vram_gb': initial_vram,
                        'peak_vram_gb': peak_vram,
                        'final_vram_gb': final_vram,
                        'reserved_vram_gb': reserved_vram,
                        'vram_overhead_gb': vram_overhead,
                        'vram_efficiency_percent': vram_efficiency,
                        'execution_time_ms': execution_time,
                        'memory_samples_vram': memory_samples,
                        'memory_samples_ram': ram_samples,
                        'cuda_available': True
                    }
                    
                    # Add function-specific parameters if available
                    if 'cfg' in kwargs or (args and hasattr(args[0], '__dict__')):
                        try:
                            cfg = kwargs.get('cfg') or (args[1] if len(args) > 1 else None)
                            if cfg and hasattr(cfg, 'n_images'):
                                stats['n_images'] = cfg.n_images
                            if cfg and hasattr(cfg, 'model') and hasattr(cfg.model, 'n_time_steps'):
                                stats['n_time_steps'] = cfg.model.n_time_steps
                            if cfg and hasattr(cfg, 'image_shape'):
                                stats['image_shape'] = list(cfg.image_shape)
                        except:
                            pass
                    
                    filename = f"memory_stats_{func.__name__}.json"
                    filepath = os.path.join(save_path or os.getcwd(), filename)
                    
                    with open(filepath, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(f"Memory stats saved to: {filepath}")
                
                return result
                
            finally:
                # Restore original forward method
                if original_forward:
                    torch.nn.Module.__call__ = original_forward
                    
                # Clean up
                torch.cuda.empty_cache()
                print(f"=== MEMORY MONITOR END: {func.__name__} ===\n")
        
        return wrapper
    return decorator


def get_model(config):
    if config.model == "UNet":
        from photo_gen.models.unet import UNET  # Import here to avoid circular import
        return UNET(config.model)


def normalize(x: torch.Tensor | np.ndarray):
    if x.min()<0 or x.max()>1 and x.min() != x.max():
        return (x - x.min()) / (x.max() - x.min())
    else:
        return x


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_wandb_run(cfg, savedir):
    """
    Create a wandb run using configuration parameters.
    
    Args:
        cfg: Configuration object containing logger settings and model info
        savedir: Path where wandb logs will be stored
    """
    model_name = cfg.model.name
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
    # Format group and run names using config templates
    group_name = cfg.logger.group_name_template.format(model_name=model_name)
    run_name = cfg.logger.run_name_template.format(model_name=model_name, job_id=job_id)
    
    wandb_dir = os.path.join(savedir, "wandb")
    wandb_dir = os.path.expanduser(wandb_dir)
    if not os.path.isdir(wandb_dir):
        os.makedirs(wandb_dir, exist_ok=True)
    
    print(f"Starting wandb run: {run_name}")
    run = wandb.init(
        project=cfg.logger.project, 
        config=dict(cfg), 
        entity=cfg.logger.entity,
        group=group_name, 
        name=run_name, 
        dir=wandb_dir
    )
    return run


def debug_cleanup(savedir, interactive=True):
    """
    Handle cleanup of debug directories.
    
    Args:
        savedir: Path to the directory to potentially delete
        interactive: If True, ask user for confirmation. If False, don't delete.
    
    Returns:
        bool: True if directory was deleted, False otherwise
    """
    if not interactive:
        print(f"Debug mode: Kept {savedir}")
        return False
        
    delete = input(f"Debug mode: Delete savedir {savedir}? (y/n): ")
    if delete.lower() == 'y':
        import shutil
        shutil.rmtree(savedir)
        print(f"Deleted {savedir}")
        return True
    else:
        print(f"Kept {savedir}")
        return False
