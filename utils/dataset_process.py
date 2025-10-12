import argparse
import json
import os

import numpy as np
from datasets import Array3D, Dataset, Features, Value

from dataset_prop import STATS, SUPPORTED_DATASETS, get_dataset_info


def get_dataset_config(dataset_name: str):
    """Get dataset-specific configuration (channels, shape, etc.)."""
    if dataset_name == "ns-nonbounded-temporal":
        return {"channels": 10, "shape": (10, 128, 128)}
    elif "jfm" in dataset_name:
        return {"channels": 24, "shape": (24, 89, 133)}
    else:
        # Default: NS, Darcy, Burgers, Helmholtz, Poisson, etc.
        return {"channels": 2, "shape": (2, 128, 128)}


def data_generator(dataset_name: str, training: bool):
    """Generator function to yield samples one at a time."""
    config = get_dataset_config(dataset_name)
    channels = config["channels"]
    expected_shape = config["shape"]
    
    stats = STATS[dataset_name]
    means = stats["mean"]
    stds = stats["std"]

    # Support for loading stats from .npy files (for large datasets)
    if isinstance(means, str) and isinstance(stds, str):
        means = np.load(means)
        stds = np.load(stds)
        print(f"Loaded stats from .npy files: shape {means.shape}")

    # Get dataset info
    info = get_dataset_info(dataset_name, training)
    loader = info["loader"]
    path_pattern = info["path_pattern"]
    file_range = info["range"]
    sample_idx = 0  # Global counter for all samples

    # Process each file
    for _i, j in enumerate(file_range):
        file_path = path_pattern.format(j)

        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        # Load raw data
        try:
            a, u = loader(file_path)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            continue

        # Special handling for JFM dataset
        if dataset_name == "jfm":
            if _i == 0 and a.dtype != np.float64:
                print(f"Converting dtype from {a.dtype} to float64")
                a = a.astype(np.float64)
            assert u == 0, f"Unexpected value for u: {u}"
            # Normalize with repeated means/stds for JFM
            means_repeated = np.repeat(means, 12, axis=0)[np.newaxis, :, :, :]
            stds_repeated = np.repeat(stds, 12, axis=0)[np.newaxis, :, :, :]
            combined = (a - means_repeated) * (0.5 / stds_repeated)
        else:
            # Standard normalization for NS and other datasets
            if _i == 0 and (a.dtype != np.float64 or u.dtype != np.float64):
                print(f"Converting dtypes ({a.dtype}, {u.dtype}) to float64")
                a = a.astype(np.float64)
                u = u.astype(np.float64)

            # Normalization
            a_normalized = (a - means[0]) * (0.5 / stds[0])
            u_normalized = (u - means[1]) * (0.5 / stds[1])

            # Dataset-specific reshaping
            if dataset_name == "ns-nonbounded-temporal":
                a_normalized = a_normalized[:, None, :, :]  # Add channel dim
            if dataset_name == "jfm_old":
                a_normalized = np.transpose(a_normalized, (0, 3, 1, 2))
                u_normalized = np.transpose(u_normalized, (0, 3, 1, 2))

            # Stack channels - use np.stack to create channel dimension
            combined = np.stack((a_normalized, u_normalized), axis=1)

        # Validate shape
        assert combined.shape[-3:] == expected_shape, \
            f"Unexpected shape: {combined.shape}, expected {expected_shape}"

        # Yield each sample
        batch_size = a.shape[0]
        for i in range(batch_size):
            yield {"id": sample_idx, "data": combined[i]}
            sample_idx += 1

        # Clean up memory
        del a, u


def process_dataset(dataset_name: str, training=True, custom_output_suffix=None) -> Dataset:
    """Process dataset and save in HuggingFace datasets format."""
    config = get_dataset_config(dataset_name)
    
    # Create output directory
    if custom_output_suffix:
        output_dir = f"data/DiffPDE/{dataset_name}_{custom_output_suffix if training else 'test_' + custom_output_suffix}"
    else:
        output_dir = f"data/DiffPDE/{dataset_name if training else dataset_name + '_test'}_hf"
    os.makedirs(output_dir, exist_ok=True)

    # Define features
    features = Features(
        {
            "id": Value("int32"),
            "data": Array3D(shape=config["shape"], dtype="float64"),
        }
    )

    # Create dataset using generator
    print(f"Processing {'training' if training else 'test'} dataset: {dataset_name}")
    dataset = Dataset.from_generator(
        generator=lambda: data_generator(dataset_name, training), 
        features=features
    )
    dataset.save_to_disk(output_dir)

    # Save metadata
    num_samples = len(dataset)
    metadata = {
        "name": dataset_name,
        "stats": STATS[dataset_name],
        "shape": config["shape"],
        "num_samples": num_samples,
        "__version__": STATS["__version__"],
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {num_samples} samples to {output_dir}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and store PDE datasets in HuggingFace format.")
    parser.add_argument("dataset", type=str, help=f"Name of the dataset ({', '.join(SUPPORTED_DATASETS)}) or 'all' to process all datasets")
    parser.add_argument("--test", "-t", action="store_true", help="Process the test dataset")
    parser.add_argument("--suffix", type=str, default=None, help="Custom output directory suffix (e.g., 'new_hf' for ns-nonbounded_new)")
    args = parser.parse_args()

    # Special handling for ns-nonbounded_new
    if args.dataset == "ns-nonbounded_new" and args.suffix is None:
        args.suffix = "hf"

    if args.dataset.lower() == "all":
        for dataset in SUPPORTED_DATASETS:
            print(f"\n{'='*60}")
            print(f"Processing {dataset} dataset...")
            print('='*60)
            process_dataset(dataset, not args.test, args.suffix)
            print(f"✓ Finished processing {dataset} dataset.\n")
    else:
        if args.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {args.dataset} not supported. Supported datasets are: {', '.join(SUPPORTED_DATASETS)}")
        process_dataset(args.dataset, not args.test, args.suffix)

    print("\n" + "="*60)
    print("✓ Finished processing all files.")
    print("="*60)
