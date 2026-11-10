
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_DIR = PROJECT_ROOT / "configs"

RANDOM_SEED = 42
VALIDATION_SIZE = 0.20


def create_splits():
    """Create reproducible, stratified training and validation splits."""

    dataset = CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
    )

    indices = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    split_data = {
        "random_seed": RANDOM_SEED,
        "validation_size": VALIDATION_SIZE,
        "training_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "official_test_set": "CIFAR-10 official test split",
    }

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    output_path = CONFIG_DIR / "data_splits.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(split_data, file)

    print("DATA SPLITTING COMPLETED")
    print("-" * 40)
    print(f"Training samples: {len(train_indices):,}")
    print(f"Validation samples: {len(validation_indices):,}")
    print(f"Official test samples: 10,000")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Saved to: {output_path}")

    # Verify that the splits do not overlap.
    assert len(set(train_indices).intersection(validation_indices)) == 0
    assert len(train_indices) + len(validation_indices) == 50_000

    print("Split integrity checks passed.")


if __name__ == "__main__":
    create_splits()
