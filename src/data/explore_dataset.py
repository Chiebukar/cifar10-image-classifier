
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision.datasets import CIFAR10


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_datasets():
    """Load the downloaded CIFAR-10 datasets."""

    train_dataset = CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
    )

    test_dataset = CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
    )

    return train_dataset, test_dataset


def create_output_directories():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def inspect_dataset(train_dataset, test_dataset):
    """Inspect image dimensions, labels and data types."""

    train_images = train_dataset.data
    test_images = test_dataset.data

    print("\nDATASET STRUCTURE")
    print("-" * 45)

    print(f"Training images: {len(train_images):,}")
    print(f"Test images: {len(test_images):,}")

    print(f"Training image shape: {train_images.shape}")
    print(f"Test image shape: {test_images.shape}")

    print(f"Image data type: {train_images.dtype}")
    print(f"Minimum pixel value: {train_images.min()}")
    print(f"Maximum pixel value: {train_images.max()}")

    print(
        "Training labels:",
        len(set(train_dataset.targets)),
    )

    print(
        "Test labels:",
        len(set(test_dataset.targets)),
    )

    assert train_images.shape == (50000, 32, 32, 3)
    assert test_images.shape == (10000, 32, 32, 3)

    assert train_images.dtype == np.uint8
    assert test_images.dtype == np.uint8

    assert train_images.min() >= 0
    assert train_images.max() <= 255

    print("\nBasic dataset integrity checks passed.")


def analyse_class_distribution(train_dataset, test_dataset):
    """Calculate and visualise class frequencies."""

    train_counts = np.bincount(
        train_dataset.targets,
        minlength=10,
    )

    test_counts = np.bincount(
        test_dataset.targets,
        minlength=10,
    )

    distribution = pd.DataFrame({
        "class": CLASS_NAMES,
        "training_samples": train_counts,
        "test_samples": test_counts,
        "total_samples": train_counts + test_counts,
    })

    distribution.to_csv(
        TABLES_DIR / "class_distribution.csv",
        index=False,
    )

    print("\nCLASS DISTRIBUTION")
    print("-" * 45)
    print(distribution.to_string(index=False))

    x = np.arange(len(CLASS_NAMES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x - width / 2,
        train_counts,
        width,
        label="Training",
    )

    ax.bar(
        x + width / 2,
        test_counts,
        width,
        label="Test",
    )

    ax.set_title("CIFAR-10 Class Distribution")
    ax.set_xlabel("Image Class")
    ax.set_ylabel("Number of Images")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=35)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "class_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return distribution


def visualise_samples(train_dataset):
    """Display representative images from each class."""

    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(14, 7),
    )

    images = train_dataset.data
    labels = np.array(train_dataset.targets)

    for class_index, ax in enumerate(axes.flat):
        image_index = np.where(labels == class_index)[0][0]

        ax.imshow(images[image_index])
        ax.set_title(CLASS_NAMES[class_index])
        ax.axis("off")

    fig.suptitle(
        "Representative CIFAR-10 Images",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "representative_samples.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def analyse_pixel_statistics(train_dataset):
    """Calculate RGB channel statistics and pixel distributions."""

    images = train_dataset.data.astype(np.float32) / 255.0

    channel_mean = images.mean(axis=(0, 1, 2))
    channel_std = images.std(axis=(0, 1, 2))

    statistics = {
        "channel_mean": channel_mean.tolist(),
        "channel_std": channel_std.tolist(),
        "normalised_pixel_min": float(images.min()),
        "normalised_pixel_max": float(images.max()),
    }

    with open(
        TABLES_DIR / "pixel_statistics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(statistics, file, indent=4)

    print("\nRGB CHANNEL STATISTICS")
    print("-" * 45)

    for index, channel in enumerate(["Red", "Green", "Blue"]):
        print(
            f"{channel}: "
            f"mean={channel_mean[index]:.4f}, "
            f"std={channel_std[index]:.4f}"
        )

    print(
        f"Normalised pixel range: "
        f"{images.min():.4f} to {images.max():.4f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    channel_names = ["Red", "Green", "Blue"]

    for index, ax in enumerate(axes):
        channel_pixels = images[..., index].ravel()

        ax.hist(
            channel_pixels,
            bins=50,
            density=True,
        )

        ax.set_title(f"{channel_names[index]} Channel")
        ax.set_xlabel("Normalised pixel intensity")
        ax.set_ylabel("Density")

    fig.suptitle("CIFAR-10 RGB Pixel Intensity Distributions")
    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "pixel_intensity_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return statistics


def check_exact_duplicates(train_dataset, test_dataset):
    """Check for exact image matches across the official splits."""

    train_hashes = {
        image.tobytes()
        for image in train_dataset.data
    }

    test_hashes = {
        image.tobytes()
        for image in test_dataset.data
    }

    duplicate_count = len(train_hashes.intersection(test_hashes))

    print("\nCROSS-SPLIT EXACT DUPLICATE CHECK")
    print("-" * 45)
    print(f"Exact image matches: {duplicate_count}")

    return duplicate_count


def main():
    create_output_directories()

    train_dataset, test_dataset = load_datasets()

    inspect_dataset(train_dataset, test_dataset)

    analyse_class_distribution(
        train_dataset,
        test_dataset,
    )

    visualise_samples(train_dataset)

    analyse_pixel_statistics(train_dataset)

    duplicate_count = check_exact_duplicates(
        train_dataset,
        test_dataset,
    )

    summary = {
        "training_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "number_of_classes": len(CLASS_NAMES),
        "image_dimensions": [32, 32, 3],
        "exact_cross_split_duplicates": duplicate_count,
    }

    with open(
        TABLES_DIR / "dataset_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nEDA COMPLETED SUCCESSFULLY")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
