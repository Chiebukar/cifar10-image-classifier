# CIFAR-10 Image Classifier

A comparative image-classification project built with PyTorch and Streamlit. The app lets you upload an image, choose one of four trained models, and inspect its predicted CIFAR-10 class, confidence scores, and inference time.

**Try the deployed app:** [cifar-classifier.streamlit.app](https://cifar-classifier.streamlit.app/)

## What it does

- Classifies an uploaded PNG or JPEG into one of the ten CIFAR-10 categories.
- Shows the selected model's prediction, per-class confidence distribution, and inference time.
- Compares four trained models using accuracy on the official 10,000-image CIFAR-10 test set.
- Loads the included checkpoints and runs on CPU or CUDA when available.

The classifier is designed for CIFAR-10-like, single-object images. Images outside that domain may still receive a label; confidence is the model's softmax score and should not be treated as a calibrated probability.

## Model results

| Model | Official test accuracy | Input size |
|---|---:|---:|
| Baseline CNN | 81.43% | 32 × 32 |
| Custom Residual CNN | 84.92% | 32 × 32 |
| Augmented Residual CNN | 85.11% | 32 × 32 |
| MobileNetV2 (ImageNet-pretrained) | 95.40% | 224 × 224 |

Results are recorded in [`reports/tables/final_test_results.csv`](reports/tables/final_test_results.csv). MobileNetV2 uses ImageNet normalization; the custom CNNs use CIFAR-10 normalization.

## Run the app locally

From the project root, create a Python environment and install the listed dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit (typically http://localhost:8501). The app's model weights are in `models/`; the `data/` directory is only needed to reproduce training and evaluation, not for image prediction.

## CIFAR-10 classes

Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Reproduce the experiments

The dataset contains 50,000 training images and 10,000 official test images, each 32 × 32 RGB. The training scripts create and use a stratified 80/20 train/validation split (seed 42); the official test split is kept separate.

```bash
# Download the dataset into data/raw/
python -m src.data.download_dataset

# Create the reproducible train/validation split
python -m src.data.create_splits

# Train each model
python -m src.training.train_baseline
python -m src.training.train_custom_cnn
python -m src.training.train_custom_cnn_augmented
python -m src.training.train_mobilenet_v2
```

Training requires the Python dependencies above and can take substantially longer than running the app; a CUDA-capable GPU is recommended. Training scripts save checkpoints under `models/` and run histories and summaries under `experiments/`. Evaluation reports and plots are under `reports/`.

## Project layout

```text
app/                 Streamlit application
configs/             Reproducible dataset split configuration
experiments/         Training histories and summaries
models/              Trained model checkpoints
reports/             Evaluation tables and figures
src/data/            Dataset download, split, and exploration scripts
src/evaluation/      Model evaluation scripts
src/models/          PyTorch model definitions
src/training/        Training scripts
```

## Technologies

Python, PyTorch, torchvision, and Streamlit. See [`requirements.txt`](requirements.txt) for the full dependency list.
