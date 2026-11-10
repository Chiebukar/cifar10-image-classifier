
"""
MobileNetV2 Transfer Learning Experiment
Dataset: CIFAR-10
Task: 10-class image classification
"""

import copy
import json
import logging
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models.mobilenet_v2 import MobileNetV2Classifier


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_DIR = PROJECT_ROOT / "configs"

MODEL_DIR = PROJECT_ROOT / "models"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "mobilenet_v2"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

SPLIT_FILE = CONFIG_DIR / "data_splits.json"

CHECKPOINT_PATH = MODEL_DIR / "mobilenet_v2_best.pth"

HISTORY_PATH = EXPERIMENT_DIR / "training_history.csv"
SUMMARY_PATH = EXPERIMENT_DIR / "training_summary.json"

ACCURACY_FIGURE = FIGURE_DIR / "mobilenet_v2_accuracy.png"
LOSS_FIGURE = FIGURE_DIR / "mobilenet_v2_loss.png"


# ============================================================
# 2. HYPERPARAMETERS
# ============================================================

RANDOM_SEED = 42

NUM_CLASSES = 10
IMAGE_SIZE = 224

BATCH_SIZE = 64
EPOCHS = 15

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

NUM_WORKERS = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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


# ============================================================
# 3. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 5. DEVICE CONFIGURATION
# ============================================================

def get_device() -> torch.device:

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. "
            "Enable a GPU runtime in Google Colab."
        )

    device = torch.device("cuda")

    logger.info("Device: %s", device)
    logger.info(
        "GPU: %s",
        torch.cuda.get_device_name(0),
    )

    return device


# ============================================================
# 6. LOAD EXISTING DATA SPLITS
# ============================================================

def load_splits() -> tuple[list[int], list[int]]:

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Data split file not found: {SPLIT_FILE}"
        )

    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    train_indices = split_data["training_indices"]
    validation_indices = split_data["validation_indices"]

    if len(train_indices) != 40000:
        raise ValueError(
            f"Expected 40,000 training indices, "
            f"found {len(train_indices)}"
        )

    if len(validation_indices) != 10000:
        raise ValueError(
            f"Expected 10,000 validation indices, "
            f"found {len(validation_indices)}"
        )

    if set(train_indices).intersection(validation_indices):
        raise ValueError(
            "Training and validation indices overlap."
        )

    logger.info(
        "Training samples: %s",
        f"{len(train_indices):,}",
    )

    logger.info(
        "Validation samples: %s",
        f"{len(validation_indices):,}",
    )

    return train_indices, validation_indices


# ============================================================
# 7. DATA PREPARATION
# ============================================================

def prepare_dataloaders(
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:

    logger.info("Preparing datasets...")

    train_indices, validation_indices = load_splits()

    # Training transformations:
    # Random crop and horizontal flip are applied only
    # to the training images.

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    # Validation transformations:
    # No random augmentation is applied.

    validation_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    # Load CIFAR-10 once and share its underlying image
    # arrays between the two dataset views.

    base_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=None,
    )

    train_base = copy.copy(base_dataset)
    validation_base = copy.copy(base_dataset)

    train_base.transform = train_transform
    validation_base.transform = validation_transform

    train_dataset = Subset(
        train_base,
        train_indices,
    )

    validation_dataset = Subset(
        validation_base,
        validation_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    logger.info(
        "Training batches: %s",
        len(train_loader),
    )

    logger.info(
        "Validation batches: %s",
        len(validation_loader),
    )

    return train_loader, validation_loader


# ============================================================
# 8. MODEL INITIALISATION
# ============================================================

def initialise_model(
    device: torch.device,
) -> nn.Module:

    logger.info(
        "Loading ImageNet-pretrained MobileNetV2..."
    )

    model = MobileNetV2Classifier(
        num_classes=NUM_CLASSES,
        pretrained=True,
    )

    model = model.to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    logger.info(
        "Total parameters: %s",
        f"{total_parameters:,}",
    )

    logger.info(
        "Trainable parameters: %s",
        f"{trainable_parameters:,}",
    )

    return model


# ============================================================
# 9. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[float, float]:

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 10. VALIDATION FUNCTION
# ============================================================

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                outputs = model(images)

                loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

    validation_loss = running_loss / total

    validation_accuracy = correct / total

    return validation_loss, validation_accuracy


# ============================================================
# 11. SAVE TRAINING HISTORY AND GRAPHS
# ============================================================

def save_outputs(
    history: list[dict],
) -> None:

    dataframe = pd.DataFrame(history)

    dataframe.to_csv(
        HISTORY_PATH,
        index=False,
    )

    # Accuracy graph

    plt.figure(figsize=(10, 6))

    plt.plot(
        dataframe["epoch"],
        dataframe["train_accuracy"],
        label="Training accuracy",
    )

    plt.plot(
        dataframe["epoch"],
        dataframe["validation_accuracy"],
        label="Validation accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "MobileNetV2: Training and Validation Accuracy"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        ACCURACY_FIGURE,
        dpi=300,
    )

    plt.close()

    # Loss graph

    plt.figure(figsize=(10, 6))

    plt.plot(
        dataframe["epoch"],
        dataframe["train_loss"],
        label="Training loss",
    )

    plt.plot(
        dataframe["epoch"],
        dataframe["validation_loss"],
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")

    plt.title(
        "MobileNetV2: Training and Validation Loss"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        LOSS_FIGURE,
        dpi=300,
    )

    plt.close()

    logger.info("Training history and graphs saved.")


# ============================================================
# 12. MAIN TRAINING PIPELINE
# ============================================================

def main() -> None:

    set_seed()

    device = get_device()

    # Create output directories.

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Prepare data.

    train_loader, validation_loader = (
        prepare_dataloaders(device)
    )

    # Initialise model.

    model = initialise_model(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler("cuda")

    history = []

    best_accuracy = 0.0
    best_epoch = 0
    best_validation_loss = float("inf")

    total_training_time = 0.0

    logger.info(
        "Starting MobileNetV2 fine-tuning..."
    )

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        validation_loss, validation_accuracy = validate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(validation_loss)

        epoch_time = time.time() - epoch_start

        total_training_time += epoch_time

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
        })

        logger.info(
            "Epoch %02d/%02d | "
            "Train Loss: %.4f | "
            "Train Acc: %.2f%% | "
            "Val Loss: %.4f | "
            "Val Acc: %.2f%% | "
            "LR: %.6f | "
            "Time: %.1fs",

            epoch,
            EPOCHS,
            train_loss,
            train_accuracy * 100,
            validation_loss,
            validation_accuracy * 100,
            current_lr,
            epoch_time,
        )

        # Save best model using validation accuracy.

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_validation_loss = validation_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_validation_accuracy": best_accuracy,
                    "validation_loss": validation_loss,
                    "class_names": CLASS_NAMES,
                    "architecture": "MobileNetV2",
                    "num_classes": NUM_CLASSES,
                    "input_size": IMAGE_SIZE,
                    "imagenet_mean": IMAGENET_MEAN,
                    "imagenet_std": IMAGENET_STD,
                },
                CHECKPOINT_PATH,
            )

            logger.info(
                "New best checkpoint saved at epoch %d",
                epoch,
            )

    # --------------------------------------------------------
    # Save experiment outputs.
    # --------------------------------------------------------

    save_outputs(history)

    summary = {
        "model": "MobileNetV2",
        "experiment": "ImageNet Transfer Learning",
        "pretrained_weights": "ImageNet",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "best_validation_loss": best_validation_loss,
        "total_training_time_seconds": total_training_time,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "loss_function": "CrossEntropyLoss",
        "mixed_precision": True,
        "random_seed": RANDOM_SEED,
        "num_classes": NUM_CLASSES,
        "input_size": IMAGE_SIZE,
        "data_augmentation": True,
        "augmentation_techniques": [
            "RandomCrop(size=32, padding=4)",
            "RandomHorizontalFlip(p=0.5)",
        ],
        "normalisation_mean": IMAGENET_MEAN,
        "normalisation_std": IMAGENET_STD,
        "total_parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "trainable_parameters": sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    logger.info(
        "MobileNetV2 training completed."
    )

    logger.info(
        "Best epoch: %d",
        best_epoch,
    )

    logger.info(
        "Best validation accuracy: %.2f%%",
        best_accuracy * 100,
    )

    logger.info(
        "Checkpoint: %s",
        CHECKPOINT_PATH,
    )

    logger.info(
        "Total training time: %.2f seconds",
        total_training_time,
    )


if __name__ == "__main__":
    main()
