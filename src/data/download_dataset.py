
import os
from pathlib import Path

import certifi

# Use certifi's maintained CA bundle when the local Python installation has no
# configured trust store. This keeps HTTPS certificate verification enabled.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from torchvision.datasets import CIFAR10


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_cifar10() -> None:
    """Download the official CIFAR-10 training and test datasets."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading CIFAR-10 training dataset...")

    train_dataset = CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
    )

    print("Downloading CIFAR-10 test dataset...")

    test_dataset = CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
    )

    print("\nDataset download completed.")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Dataset location: {DATA_DIR}")


if __name__ == "__main__":
    download_cifar10()
