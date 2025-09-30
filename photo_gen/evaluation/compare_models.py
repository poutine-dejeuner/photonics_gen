"""
Script to compare evaluation metrics across different datasets.
Reads datasets from evaluation config, loads stats.yaml from each path,
and plots metrics vs n_samples.
"""

import os
import yaml
from typing import Dict, List, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf
import hydra
from photo_gen.evaluation.eval_utils import find_files
from photo_gen.evaluation.evaluation import (
    Entropy, PCAProjPerDimEntropy, ImageAverageEntropy, 
    PairwiseDistanceEntropy, BinarizationLoss
)

from icecream import ic


def load_datasets_config(config_path: str) -> List[Dict[str, Any]]:
    """Load datasets configuration from YAML file."""
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config['datasets'], resolve=True)


def expand_env_vars(path: str) -> str:
    """Expand environment variables in path."""
    return os.path.expandvars(path)


def load_stats_yaml(stats_path: Path) -> Dict[str, Any]:
    """Load stats.yaml file if it exists."""
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def compute_training_set_metrics(cfg, output_dir: str = "training_metrics") -> Dict[str, float]:
    """Compute evaluation metrics for the training set."""
    # Load training data
    train_data_path = os.path.expanduser(cfg.data_path)
    train_images = np.load(train_data_path)
    
    print(f"Computing metrics for training set: {train_data_path}")
    print(f"Training set shape: {train_images.shape}")
    
    # Load FOM data from the same directory
    fom_path = os.path.join(os.path.dirname(train_data_path), "fom.npy")
    fom_data = None
    if os.path.exists(fom_path):
        try:
            fom_data = np.load(fom_path)
            print(f"Loaded FOM data from: {fom_path}")
            print(f"FOM data shape: {fom_data.shape}")
        except Exception as e:
            print(f"Failed to load FOM data: {e}")
    else:
        print(f"FOM file not found: {fom_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {}
    
    # Compute FOM statistics if available
    if fom_data is not None:
        try:
            fom_mean = np.mean(fom_data)
            fom_std = np.std(fom_data)
            metrics['FOM_mean'] = fom_mean
            metrics['FOM_std'] = fom_std
            print(f"  ✓ FOM mean: {fom_mean:.4f}")
            print(f"  ✓ FOM std: {fom_std:.4f}")
        except Exception as e:
            print(f"  ✗ FOM computation failed: {e}")
    
    # Compute Entropy
    try:
        entropy_func = Entropy()
        entropy_val = entropy_func(train_images, output_dir, "training_set", cfg=cfg)
        metrics['Entropy'] = entropy_val
        print(f"  ✓ Entropy: {entropy_val:.4f}")
    except Exception as e:
        print(f"  ✗ Entropy failed: {e}")
    
    # Compute PCAProjPerDimEntropy
    try:
        pca_entropy_func = PCAProjPerDimEntropy(dim=50, n_neighbors=4)
        pca_entropy_val = pca_entropy_func(train_images)
        metrics['PCAProjPerDimEntropy'] = pca_entropy_val
        print(f"  ✓ PCAProjPerDimEntropy: {pca_entropy_val:.4f}")
    except Exception as e:
        print(f"  ✗ PCAProjPerDimEntropy failed: {e}")
    
    # Compute ImageAverageEntropy
    try:
        img_entropy_func = ImageAverageEntropy(n_neighbors=4)
        img_entropy_val = img_entropy_func(train_images, output_dir, "training_set", cfg=cfg)
        metrics['ImageAverageEntropy'] = img_entropy_val
        print(f"  ✓ ImageAverageEntropy: {img_entropy_val:.4f}")
    except Exception as e:
        print(f"  ✗ ImageAverageEntropy failed: {e}")
    
    # Compute PairwiseDistanceEntropy
    try:
        pairwise_entropy_func = PairwiseDistanceEntropy(n_neighbors=4)
        pairwise_entropy_val = pairwise_entropy_func(train_images, output_dir, "training_set", cfg=cfg)
        metrics['PairwiseDistanceEntropy'] = pairwise_entropy_val
        print(f"  ✓ PairwiseDistanceEntropy: {pairwise_entropy_val:.4f}")
    except Exception as e:
        print(f"  ✗ PairwiseDistanceEntropy failed: {e}")
    
    # Compute BinarizationLoss
    try:
        binarization_func = BinarizationLoss()
        binarization_val = binarization_func(train_images, output_dir, "training_set", cfg=cfg)
        metrics['BinarizationLoss'] = binarization_val
        print(f"  ✓ BinarizationLoss: {binarization_val:.4f}")
    except Exception as e:
        print(f"  ✗ BinarizationLoss failed: {e}")
    
    return metrics


def collect_metrics_data(datasets: List[Dict[str, Any]]) -> pd.DataFrame:
    """Collect metrics data from all datasets."""
    data_rows = []

    for dataset in datasets:
        name = dataset['name']
        path = Path(dataset['path']).parent.parent
        n_samples = dataset['n_samples']
        stats_path = find_files(path, filenames=["stats.yaml"])[0]
        assert stats_path is not None, ic(stats_path)
        stats = load_stats_yaml(stats_path)

        if stats:
            row = {
                'dataset_name': name,
                'path': path,
                'n_samples': n_samples
            }

            # Add all metrics from stats
            for metric_name, metric_value in stats.items():
                if isinstance(metric_value, (int, float)):
                    row[metric_name] = metric_value
                elif isinstance(metric_value, dict):
                    # Handle nested metrics (like mean/std from NNDistanceTrainSet)
                    for sub_key, sub_value in metric_value.items():
                        if isinstance(sub_value, (int, float)):
                            row[f"{metric_name}_{sub_key}"] = sub_value

            data_rows.append(row)
            print(
                f"  ✓ Found {len([k for k in row.keys() if k not in ['dataset_name', 'path', 'n_samples']])} metrics")
        else:
            print("  ✗ No stats.yaml found or empty")

    if not data_rows:
        print("No data found!")
        return pd.DataFrame()

    df = pd.DataFrame(data_rows)
    return df


def plot_metrics_comparison(df: pd.DataFrame, training_metrics: Dict[str, float], output_dir: str = "metric_plots"):
    """Plot each metric vs n_samples."""
    os.makedirs(output_dir, exist_ok=True)

    # Get all metric columns (exclude metadata columns)
    metadata_cols = ['dataset_name', 'path', 'n_samples']
    all_metric_cols = [col for col in df.columns if col not in metadata_cols]

    if not all_metric_cols:
        print("No metrics found to plot!")
        return

    print(f"Found metric columns: {all_metric_cols}")

    # Group metrics by their base name (before _mean or _std suffix)
    metric_groups = {}
    simple_metrics = []

    for col in all_metric_cols:
        if col.endswith('_mean'):
            base_name = col[:-5]  # Remove '_mean'
            if base_name not in metric_groups:
                metric_groups[base_name] = {}
            metric_groups[base_name]['mean'] = col
        elif col.endswith('_std'):
            base_name = col[:-4]  # Remove '_std'
            if base_name not in metric_groups:
                metric_groups[base_name] = {}
            metric_groups[base_name]['std'] = col
        else:
            simple_metrics.append(col)

    # Only keep metric groups that have both mean and std
    complete_groups = {name: group for name, group in metric_groups.items()
                       if 'mean' in group and 'std' in group}

    # Add simple metrics that don't have corresponding mean/std pairs
    for base_name in metric_groups:
        group = metric_groups[base_name]
        if 'mean' in group and 'std' not in group:
            simple_metrics.append(group['mean'])
        elif 'std' in group and 'mean' not in group:
            simple_metrics.append(group['std'])

    print(f"Metrics with mean/std pairs: {list(complete_groups.keys())}")
    print(f"Simple metrics: {simple_metrics}")

    # Sort by n_samples for better plotting
    df_sorted = df.sort_values('n_samples')

    # Create individual plots for metrics with error bars
    for metric_name, group in complete_groups.items():
        plt.figure(figsize=(10, 6))

        mean_col = group['mean']
        std_col = group['std']

        # Group data by n_samples to handle multiple values
        grouped = df_sorted.groupby('n_samples')

        # Calculate statistics for each n_samples group
        n_samples_list = []
        means_list = []
        stds_list = []

        for n_samples, group_data in grouped:
            # Get all values for this n_samples
            values = group_data[mean_col].dropna()
            if len(values) > 0:
                n_samples_list.append(n_samples)
                means_list.append(values.mean())
                stds_list.append(values.std() if len(values) > 1 else 0)

        # Plot individual data points
        plt.scatter(df_sorted['n_samples'], df_sorted[mean_col],
                    alpha=0.5, s=30, color='lightblue', label='Individual values')

        # Plot mean with error bars
        plt.errorbar(n_samples_list, means_list, yerr=stds_list,
                     fmt='o', capsize=5, capthick=2, color='darkblue',
                     alpha=0.8, markersize=8, label='Mean ± std across runs')
        plt.plot(n_samples_list, means_list, alpha=0.7,
                 linestyle='--', color='darkblue')

        # Add training set baseline if available
        if metric_name in training_metrics:
            train_value = training_metrics[metric_name]
            plt.axhline(y=train_value, color='red', linestyle='-', alpha=0.8, 
                       linewidth=2, label='Training set')

        plt.xlabel('Number of Samples')
        plt.ylabel(f'{metric_name} (mean ± std)')
        plt.title(f'{metric_name} vs Number of Samples')
        plt.grid(True, alpha=0.3)

        plt.legend()
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, f'{metric_name}_vs_n_samples.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved plot with error bars: {plot_path}")

    # Create individual plots for simple metrics (no error bars)
    for metric in simple_metrics:
        plt.figure(figsize=(10, 6))

        # Group data by n_samples to handle multiple values
        grouped = df_sorted.groupby('n_samples')

        # Calculate statistics for each n_samples group
        n_samples_list = []
        means_list = []
        stds_list = []

        for n_samples, group_data in grouped:
            # Get all values for this n_samples
            values = group_data[metric].dropna()
            if len(values) > 0:
                n_samples_list.append(n_samples)
                means_list.append(values.mean())
                stds_list.append(values.std() if len(values) > 1 else 0)

        # Plot individual data points
        plt.scatter(df_sorted['n_samples'], df_sorted[metric],
                    alpha=0.5, s=30, color='lightcoral', label='Individual values')

        # Plot mean with error bars if there are multiple runs
        if any(std > 0 for std in stds_list):
            plt.errorbar(n_samples_list, means_list, yerr=stds_list,
                         fmt='o', capsize=5, capthick=2, color='darkred',
                         alpha=0.8, markersize=8, label='Mean ± std across runs')
            plt.plot(n_samples_list, means_list, alpha=0.7,
                     linestyle='--', color='darkred')
        else:
            # If no std (single values), just plot the means
            plt.plot(n_samples_list, means_list, 'o-', color='darkred',
                     alpha=0.8, markersize=8, label='Mean values')

        # Add training set baseline if available
        if metric in training_metrics:
            train_value = training_metrics[metric]
            plt.axhline(y=train_value, color='red', linestyle='-', alpha=0.8, 
                       linewidth=2, label='Training set')

        plt.xlabel('Number of Samples')
        plt.ylabel(metric)
        plt.title(f'{metric} vs Number of Samples')
        plt.grid(True, alpha=0.3)

        plt.legend()

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, f'{metric}_vs_n_samples.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved plot: {plot_path}")

    # Create a summary plot with multiple metrics
    all_plot_metrics = list(complete_groups.keys()) + simple_metrics
    if len(all_plot_metrics) > 1:
        n_metrics = len(all_plot_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        plot_idx = 0

        # Plot metrics with error bars
        for metric_name, group in complete_groups.items():
            ax = axes[plot_idx]
            mean_col = group['mean']
            std_col = group['std']

            ax.errorbar(df_sorted['n_samples'], df_sorted[mean_col],
                        yerr=df_sorted[std_col], fmt='o', capsize=3, capthick=1,
                        alpha=0.7, markersize=4)
            ax.plot(df_sorted['n_samples'],
                    df_sorted[mean_col], alpha=0.5, linestyle='--')

            # Add training set baseline if available
            if metric_name in training_metrics:
                train_value = training_metrics[metric_name]
                ax.axhline(y=train_value, color='red', linestyle='-', alpha=0.8, 
                          linewidth=1.5, label='Training set')

            ax.set_xlabel('Number of Samples')
            ax.set_ylabel(f'{metric_name} (±std)')
            ax.set_title(metric_name)
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Plot simple metrics
        for metric in simple_metrics:
            if plot_idx >= len(axes):
                break
            ax = axes[plot_idx]
            breakpoint()
            ax.scatter(df_sorted['n_samples'],
                       df_sorted[metric], alpha=0.7, s=40)

            ax.plot(df_sorted['n_samples'], df_sorted[metric],
                    alpha=0.5, linestyle='--')

            # Add training set baseline if available
            if metric in training_metrics:
                train_value = training_metrics[metric]
                ax.axhline(y=train_value, color='red', linestyle='-', alpha=0.8, 
                          linewidth=1.5, label='Training set')

            ax.set_xlabel('Number of Samples')
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Hide empty subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save summary plot
        summary_path = os.path.join(output_dir, 'all_metrics_summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics of the metrics."""
    metadata_cols = ['dataset_name', 'path', 'n_samples']
    metric_cols = [col for col in df.columns if col not in metadata_cols]

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"Total datasets: {len(df)}")
    print(f"Sample range: {df['n_samples'].min()} - {df['n_samples'].max()}")
    print(f"Metrics found: {len(metric_cols)}")

    if metric_cols:
        print("\nMetric Summary:")
        for metric in metric_cols:
            values = df[metric].dropna()
            if len(values) > 0:
                print(f"  {metric}:")
                print(f"    Mean: {values.mean():.4f}")
                print(f"    Std:  {values.std():.4f}")
                print(f"    Range: {values.min():.4f} - {values.max():.4f}")

        # Correlation with n_samples
        print("\nCorrelation with n_samples:")
        for metric in metric_cols:
            values = df[metric].dropna()
            if len(values) > 1:
                corr = np.corrcoef(
                    df.loc[values.index, 'n_samples'], values)[0, 1]
                print(f"  {metric}: {corr:.4f}")


@hydra.main(version_base="1.1", config_path="../config",
            config_name="config")
def main(config):
    from photo_gen.evaluation.evaluation import main as evalmain
    """Main function to run the metric comparison analysis."""

    datasets = config.evaluation.datasets
    # evalmain(config)
    print(f"Found {len(datasets)} datasets")

    # Collect metrics data
    print("\nCollecting metrics data...")
    df = collect_metrics_data(datasets)

    if df.empty:
        print("No metrics data found!")
        return

    # Compute training set metrics
    print("\nComputing training set metrics...")
    training_metrics = compute_training_set_metrics(config)

    # Save collected data
    df.to_csv("metrics_data.csv", index=False)
    print("\nSaved collected data to: metrics_data.csv")

    # Print summary statistics
    print_summary_statistics(df)

    # Create plots
    print("\nCreating plots...")
    plot_metrics_comparison(df, training_metrics)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("Check the 'metric_plots' directory for generated plots.")
    print("Raw data saved in 'metrics_data.csv'")


if __name__ == "__main__":
    main()
