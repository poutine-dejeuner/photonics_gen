import os
from tqdm import tqdm
import numpy as np
from typing import Tuple, List, Union
import torch


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
    for _ in tqdm(range(M - 1)):
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


def main():
    """
    Demonstrate the usage of the diverse image selection functions.
    """
    
    path = "~/scratch/nanophoto/lowfom/distselect/"
    path = os.path.expanduser(path)
    print("loading")
    sample_images = np.load(os.path.join(path, "images.npy"))
    fom = np.load(os.path.join(path, "fom.npy"))
    # Sort sample_images by fom and pick top 1024
    print("done")
    top_k = 1024
    sorted_indices = np.argsort(fom)[-top_k:]
    sample_images = sample_images[sorted_indices]

    M = 256
    
    # Method 1: Greedy selection with different distance metrics
    print("Method 1: Greedy Selection")
    print("-" * 25)
    
    metrics = ['euclidean', 'cosine', 'manhattan']
    for metric in metrics:
        print(metric)
        selected, indices = select_diverse_images(sample_images, M, metric)
        diversity_score = calculate_diversity_score(selected, metric)
        print(f"{metric.capitalize()} selection:")
        print(f"  Selected indices: {indices}")
        print(f"  Diversity score: {diversity_score:.4f}")
        np.save(f"{metric}.npy", selected)
    

    
    # Method 2: K-means clustering (if sklearn is available)
    print("Method 2: K-means Clustering")
    print("-" * 26)
    try:
        selected_kmeans, indices_kmeans = select_diverse_images_kmeans(sample_images, M)
        diversity_score_kmeans = calculate_diversity_score(selected_kmeans, 'euclidean')
        print(f"K-means selection:")
        print(f"  Selected indices: {indices_kmeans}")
        print(f"  Diversity score: {diversity_score_kmeans:.4f}")
        np.save("kmeans.npy", selected_kmeans)
    except ImportError: 
        print("sklearn not available for K-means method")

if __name__ == '__main__':
    main()
