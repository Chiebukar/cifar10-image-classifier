
import copy
import json
import logging
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models.custom_cnn import CustomResidualCNN


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_DIR = PROJECT_ROOT / "configs"
MODEL_DIR = PROJECT_ROOT / "models"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "custom_cnn"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

SPLIT_FILE = CONFIG_DIR / "data_splits.json"

RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 0

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# --------------------------------------------------
# 2. Reproducibility and device
# --------------------------------------------------

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    # Default to CPU because some macOS/PyTorch combinations abort inside MPS.
    # Set CIFAR10_DEVICE=auto, cuda, or mps to opt into acceleration.
    requested_device = os.getenv("CIFAR10_DEVICE", "cpu").lower()

    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CIFAR10_DEVICE=cuda, but CUDA is unavailable.")
        return torch.device("cuda")
    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("CIFAR10_DEVICE=mps, but MPS is unavailable.")
        return torch.device("mps")
    if requested_device != "auto":
        raise ValueError(
            "CIFAR10_DEVICE must be one of: auto, cuda, mps."
        )

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------
# 3. Load fixed experimental splits
# --------------------------------------------------

def load_splits() -> tuple[list[int], list[int]]:
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(SPLIT_FILE)

    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    train_indices = split_data["training_indices"]
    validation_indices = split_data["validation_indices"]

    if split_data["random_seed"] != RANDOM_SEED:
        raise ValueError("Unexpected data split seed.")

    if len(train_indices) != 40000:
        raise ValueError("Expected 40,000 training samples.")

    if len(validation_indices) != 10000:
        raise ValueError("Expected 10,000 validation samples.")

    if set(train_indices).intersection(validation_indices):
        raise ValueError("Training and validation sets overlap.")

    return train_indices, validation_indices


# --------------------------------------------------
# 4. Prepare datasets
# --------------------------------------------------

def prepare_dataloaders(
    device: torch.device,
) -> tuple[DataLoader, DataLoader, list[float], list[float]]:

    train_indices, validation_indices = load_splits()

    raw_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
    )

    # Calculate statistics from the training subset only.
    training_images = (
        raw_dataset.data[train_indices].astype(np.float32) / 255.0
    )

    mean = training_images.mean(axis=(0, 1, 2)).tolist()
    std = training_images.std(axis=(0, 1, 2)).tolist()

    # No augmentation in this experiment.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=transform,
    )

    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    return train_loader, validation_loader, mean, std


# --------------------------------------------------
# 5. Training
# --------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# --------------------------------------------------
# 6. Validation
# --------------------------------------------------

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# --------------------------------------------------
# 7. Save graphs and history
# --------------------------------------------------

def save_outputs(history: list[dict]) -> None:

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(history)

    dataframe.to_csv(
        EXPERIMENT_DIR / "training_history.csv",
        index=False,
    )

    # Loss plot
    plt.figure(figsize=(9, 5))
    plt.plot(dataframe["epoch"], dataframe["train_loss"],
             label="Training loss")
    plt.plot(dataframe["epoch"], dataframe["val_loss"],
             label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Custom CNN: Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "custom_cnn_loss.png", dpi=300)
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(9, 5))
    plt.plot(dataframe["epoch"], dataframe["train_accuracy"],
             label="Training accuracy")
    plt.plot(dataframe["epoch"], dataframe["val_accuracy"],
             label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Custom CNN: Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "custom_cnn_accuracy.png", dpi=300)
    plt.close()


# --------------------------------------------------
# 8. Main experiment
# --------------------------------------------------

def main() -> None:

    configure_logging()
    set_seed(RANDOM_SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    logging.info("Device: %s", device)
    logging.info("Preparing datasets...")

    train_loader, validation_loader, mean, std = (
        prepare_dataloaders(device)
    )

    model = CustomResidualCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    history = []

    best_accuracy = 0.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    start_time = time.time()

    logging.info("Starting CustomResidualCNN training...")

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_accuracy = validate(
            model,
            validation_loader,
            criterion,
            device,
        )

        scheduler.step(val_accuracy)

        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr,
            "duration_seconds": epoch_duration,
        })

        logging.info(
            "Epoch %02d/%02d | "
            "Train Loss: %.4f | Train Acc: %.2f%% | "
            "Val Loss: %.4f | Val Acc: %.2f%% | "
            "LR: %.6f | Time: %.1fs",
            epoch,
            EPOCHS,
            train_loss,
            train_accuracy * 100,
            val_loss,
            val_accuracy * 100,
            current_lr,
            epoch_duration,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            logging.info("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)

    checkpoint_path = MODEL_DIR / "custom_cnn_best.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "CustomResidualCNN",
            "num_classes": 10,
            "class_names": CLASS_NAMES,
            "normalisation_mean": mean,
            "normalisation_std": std,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "random_seed": RANDOM_SEED,
        },
        checkpoint_path,
    )

    save_outputs(history)

    total_time = time.time() - start_time

    summary = {
        "model": "CustomResidualCNN",
        "device": str(device),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "total_training_time_seconds": total_time,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "loss_function": "CrossEntropyLoss",
        "random_seed": RANDOM_SEED,
        "data_augmentation": False,
        "normalisation_mean": mean,
        "normalisation_std": std,
    }

    with open(
        EXPERIMENT_DIR / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    logging.info("Custom CNN training completed.")
    logging.info("Best epoch: %d", best_epoch)
    logging.info("Best validation accuracy: %.2f%%",
                 best_accuracy * 100)
    logging.info("Checkpoint: %s", checkpoint_path)
    logging.info("Total training time: %.2f seconds", total_time)


if __name__ == "__main__":
    main()
