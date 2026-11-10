cd /content/cifar10-image-classifier

cat > src/evaluation/evaluate_test_set.py <<'PY'
"""
Final evaluation of four CIFAR-10 classifiers on the official test set.

The official CIFAR-10 test set is used only for final evaluation.
No training, validation, or hyperparameter tuning occurs here.
"""

import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.baseline_cnn import BaselineCNN
from src.models.custom_cnn import CustomResidualCNN
from src.models.mobilenet_v2 import MobileNetV2Classifier


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"

DATA_DIR = PROJECT_ROOT / "data" / "raw"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64
NUM_WORKERS = 2
RANDOM_SEED = 42

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

CIFAR_MEAN = (
    0.4910959005355835,
    0.48214635252952576,
    0.44657397270202637,
)

CIFAR_STD = (
    0.24700748920440674,
    0.24354411661624908,
    0.2616426646709442,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------
# 2. LOGGING AND REPRODUCIBILITY
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 3. MODEL CONFIGURATION
# ---------------------------------------------------------

MODEL_CONFIGS = [
    {
        "name": "Baseline CNN",
        "checkpoint": "baseline_cnn_best.pth",
        "architecture": BaselineCNN,
        "input_size": 32,
        "mean": CIFAR_MEAN,
        "std": CIFAR_STD,
    },
    {
        "name": "Custom Residual CNN",
        "checkpoint": "custom_cnn_best.pth",
        "architecture": CustomResidualCNN,
        "input_size": 32,
        "mean": CIFAR_MEAN,
        "std": CIFAR_STD,
    },
    {
        "name": "Custom Residual CNN (Augmented)",
        "checkpoint": "custom_cnn_augmented_best.pth",
        "architecture": CustomResidualCNN,
        "input_size": 32,
        "mean": CIFAR_MEAN,
        "std": CIFAR_STD,
    },
    {
        "name": "MobileNetV2",
        "checkpoint": "mobilenet_v2_best.pth",
        "architecture": MobileNetV2Classifier,
        "input_size": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
]


# ---------------------------------------------------------
# 4. DATASET AND TRANSFORMATIONS
# ---------------------------------------------------------

def build_test_transform(config):
    """Create deterministic test preprocessing."""

    operations = []

    if config["input_size"] == 224:
        operations.append(
            transforms.Resize((224, 224))
        )

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["mean"],
            std=config["std"],
        ),
    ])

    return transforms.Compose(operations)


def load_test_dataset(config):
    """Load the official CIFAR-10 test dataset."""

    dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=build_test_transform(config),
    )

    if len(dataset) != 10000:
        raise ValueError(
            f"Expected 10,000 test images, found {len(dataset)}."
        )

    return dataset


# ---------------------------------------------------------
# 5. MODEL LOADING
# ---------------------------------------------------------

def load_model(config):
    """Load a model and strictly validate its checkpoint."""

    checkpoint_path = MODEL_DIR / config["checkpoint"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    logger.info(
        "Loading model: %s",
        config["name"],
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Missing model_state_dict in {checkpoint_path}"
        )

    if checkpoint.get("num_classes") != 10:
        raise ValueError(
            f"Unexpected class count in {checkpoint_path}"
        )

    checkpoint_classes = checkpoint.get("class_names")

    if checkpoint_classes != CLASS_NAMES:
        raise ValueError(
            f"Class names do not match for {config['name']}"
        )

    # Avoid downloading ImageNet weights again.
    if config["name"] == "MobileNetV2":
        model = config["architecture"](
            num_classes=10,
            pretrained=False,
        )
    else:
        model = config["architecture"](num_classes=10)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model, checkpoint


# ---------------------------------------------------------
# 6. INFERENCE AND METRICS
# ---------------------------------------------------------

def evaluate_model(model, loader, model_name):
    """Evaluate a model and calculate classification metrics."""

    all_predictions = []
    all_targets = []

    inference_time = 0.0
    total_images = 0

    logger.info(
        "Starting test evaluation: %s",
        model_name,
    )

    with torch.inference_mode():

        for images, targets in loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            outputs = model(images)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            inference_time += time.perf_counter() - start_time

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                targets.cpu().numpy()
            )

            total_images += targets.size(0)

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_predictions)

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(10)),
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(10)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "model": model_name,
        "test_images": total_images,
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "inference_time_seconds": float(inference_time),
        "average_inference_ms_per_image": float(
            (inference_time / total_images) * 1000
        ),
        "throughput_images_per_second": float(
            total_images / inference_time
        ),
    }

    return results, report, cm


# ---------------------------------------------------------
# 7. CONFUSION MATRIX VISUALISATION
# ---------------------------------------------------------

def save_confusion_matrix(cm, model_name):
    """Save confusion matrix as a PNG image."""

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    figure, axis = plt.subplots(figsize=(10, 8))

    image = axis.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted Class",
        ylabel="Actual Class",
        title=f"Confusion Matrix - {model_name}",
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
    )

    threshold = cm.max() / 2

    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            axis.text(
                column,
                row,
                format(cm[row, column], "d"),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[row, column] > threshold
                    else "black"
                ),
                fontsize=8,
            )

    figure.tight_layout()

    output_path = (
        FIGURE_DIR / f"{safe_name}_confusion_matrix.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.info(
        "Saved confusion matrix: %s",
        output_path,
    )


# ---------------------------------------------------------
# 8. MAIN EVALUATION PIPELINE
# ---------------------------------------------------------

def main():

    logger.info("=" * 65)
    logger.info("CIFAR-10 FINAL TEST SET EVALUATION")
    logger.info("=" * 65)

    logger.info("Device: %s", DEVICE)

    if torch.cuda.is_available():
        logger.info(
            "GPU: %s",
            torch.cuda.get_device_name(0),
        )

    summary_results = []
    class_results = []
    confusion_matrices = {}

    for config in MODEL_CONFIGS:

        model_name = config["name"]

        logger.info("-" * 65)
        logger.info("Evaluating: %s", model_name)

        model, checkpoint = load_model(config)

        dataset = load_test_dataset(config)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
        )

        results, report, cm = evaluate_model(
            model,
            loader,
            model_name,
        )

        results["best_validation_accuracy"] = float(
            checkpoint["best_validation_accuracy"]
        )

        results["input_size"] = config["input_size"]

        results["total_parameters"] = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        results["trainable_parameters"] = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        summary_results.append(results)

        confusion_matrices[model_name] = cm.tolist()

        for class_name in CLASS_NAMES:
            class_metrics = report[class_name]

            class_results.append({
                "model": model_name,
                "class": class_name,
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "f1_score": class_metrics["f1-score"],
                "support": class_metrics["support"],
            })

        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        pd.DataFrame(report).transpose().to_csv(
            TABLE_DIR / f"{safe_name}_classification_report.csv"
        )

        save_confusion_matrix(cm, model_name)

        logger.info(
            "%s | Test accuracy: %.4f | Macro F1: %.4f",
            model_name,
            results["accuracy"],
            results["f1_macro"],
        )

        logger.info(
            "Inference: %.4f ms/image | Throughput: %.2f images/sec",
            results["average_inference_ms_per_image"],
            results["throughput_images_per_second"],
        )

        del model
        del dataset
        del loader

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save overall model comparison.
    summary_df = pd.DataFrame(summary_results)

    summary_df.to_csv(
        TABLE_DIR / "final_test_results.csv",
        index=False,
    )

    # Save per-class comparison.
    class_df = pd.DataFrame(class_results)

    class_df.to_csv(
        TABLE_DIR / "per_class_test_results.csv",
        index=False,
    )

    # Save all confusion matrices.
    with open(
        TABLE_DIR / "confusion_matrices.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            confusion_matrices,
            file,
            indent=4,
        )

    # Save evaluation metadata.
    metadata = {
        "evaluation_dataset": "CIFAR-10 official test set",
        "test_samples": 10000,
        "number_of_classes": 10,
        "device": str(DEVICE),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "test_augmentation": False,
        "models_evaluated": len(MODEL_CONFIGS),
        "evaluation_purpose": "Final independent test evaluation",
    }

    with open(
        TABLE_DIR / "evaluation_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    logger.info("=" * 65)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 65)

    display_columns = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "average_inference_ms_per_image",
    ]

    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    best_model = summary_df.loc[
        summary_df["accuracy"].idxmax()
    ]

    logger.info(
        "Highest test accuracy: %s (%.4f)",
        best_model["model"],
        best_model["accuracy"],
    )

    logger.info(
        "Evaluation completed successfully."
    )

    logger.info(
        "Results saved in: %s",
        TABLE_DIR,
    )

    logger.info(
        "Figures saved in: %s",
        FIGURE_DIR,
    )


if __name__ == "__main__":
    main()
PY