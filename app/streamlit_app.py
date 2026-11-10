"""
CIFAR-10 Image Classification Application.

Streamlit interface for evaluating trained CIFAR-10 classifiers.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline_cnn import BaselineCNN
from src.models.custom_cnn import CustomResidualCNN
from src.models.mobilenet_v2 import MobileNetV2Classifier


MODEL_DIR = PROJECT_ROOT / "models"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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


MODEL_CONFIGS = {
    "MobileNetV2": {
        "checkpoint": "mobilenet_v2_best.pth",
        "architecture": MobileNetV2Classifier,
        "input_size": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    "Augmented Residual CNN": {
        "checkpoint": "custom_cnn_augmented_best.pth",
        "architecture": CustomResidualCNN,
        "input_size": 32,
        "mean": CIFAR_MEAN,
        "std": CIFAR_STD,
    },
    "Custom Residual CNN": {
        "checkpoint": "custom_cnn_best.pth",
        "architecture": CustomResidualCNN,
        "input_size": 32,
        "mean": CIFAR_MEAN,
        "std": CIFAR_STD,
    },
    "Baseline CNN": {
        "checkpoint": "baseline_cnn_best.pth",
        "architecture": BaselineCNN,
        "input_size": 32,
        "mean": CIFAR_MEAN,
        "std": CIFAR_STD,
    },
}


MODEL_TEST_ACCURACY = {
    "MobileNetV2": 95.40,
    "Augmented Residual CNN": 85.11,
    "Custom Residual CNN": 84.92,
    "Baseline CNN": 81.43,
}


# ---------------------------------------------------------
# 2. PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# 3. MODEL LOADING
# ---------------------------------------------------------

@st.cache_resource
def load_model(model_name):

    config = MODEL_CONFIGS[model_name]

    checkpoint_path = MODEL_DIR / config["checkpoint"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_class = config["architecture"]

    if model_name == "MobileNetV2":
        model = model_class(
            num_classes=10,
            pretrained=False,
        )
    else:
        model = model_class(num_classes=10)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model, checkpoint["class_names"]


# ---------------------------------------------------------
# 4. IMAGE PREPROCESSING
# ---------------------------------------------------------

def preprocess_image(image, model_name):

    config = MODEL_CONFIGS[model_name]

    image = image.convert("RGB")

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

    transform = transforms.Compose(operations)

    tensor = transform(image)

    return tensor.unsqueeze(0)


# ---------------------------------------------------------
# 5. PREDICTION
# ---------------------------------------------------------

def predict(image, model_name):

    model, class_names = load_model(model_name)

    input_tensor = preprocess_image(
        image,
        model_name,
    ).to(DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model(input_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1,
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_time = (
        time.perf_counter() - start_time
    ) * 1000

    probabilities = probabilities.squeeze(0).cpu()

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    prediction = class_names[predicted_index]

    confidence = float(
        probabilities[predicted_index].item()
    )

    scores = pd.DataFrame({
        "Class": class_names,
        "Confidence": probabilities.numpy(),
    })

    scores = scores.sort_values(
        by="Confidence",
        ascending=False,
    )

    return prediction, confidence, scores, inference_time


# ---------------------------------------------------------
# 6. APPLICATION INTERFACE
# ---------------------------------------------------------

st.title("CIFAR-10 Image Classification")
st.write(
    "Upload an image to classify it into one of ten object categories "
    "using a trained deep-learning model."
)

st.caption(
    "Research project: Comparative Development and Deployment "
    "of Deep Learning Models for Multiclass Image Classification"
)

st.divider()


# Sidebar
with st.sidebar:

    st.header("Model Configuration")

    selected_model = st.selectbox(
        "Select classification model",
        options=list(MODEL_CONFIGS.keys()),
        index=0,
    )

    st.write("**Selected model test accuracy**")

    st.metric(
        label="Official CIFAR-10 test accuracy",
        value=f"{MODEL_TEST_ACCURACY[selected_model]:.2f}%",
    )

    st.caption(
        "Test accuracy is measured on the official 10,000-image "
        "CIFAR-10 test set."
    )

    st.divider()

    st.subheader("Supported Classes")

    classes = [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]

    for class_name in classes:
        st.write(f"• {class_name}")


# Main upload interface
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
    help="Upload an image containing an object from one of the ten CIFAR-10 categories.",
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    left_column, right_column = st.columns(
        [1, 1],
        gap="large",
    )

    with left_column:

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Input image",
            use_container_width=True,
        )

        st.caption(
            f"Original image dimensions: {image.width} × {image.height}"
        )

    with right_column:

        st.subheader("Classification Results")

        try:

            with st.spinner(
                f"Classifying image using {selected_model}..."
            ):

                (
                    prediction,
                    confidence,
                    scores,
                    inference_time,
                ) = predict(
                    image,
                    selected_model,
                )

            st.success("Classification completed")

            st.markdown("**Predicted class**")

            st.header(prediction.title())

            st.metric(
                label="Prediction confidence",
                value=f"{confidence * 100:.2f}%",
            )

            st.metric(
                label="Model inference time",
                value=f"{inference_time:.3f} ms",
            )

        except Exception as error:

            st.error(
                f"Unable to classify image: {error}"
            )

            st.exception(error)

    st.divider()

    st.subheader("Class Confidence Distribution")

    chart_data = scores.copy()

    chart_data["Confidence (%)"] = (
        chart_data["Confidence"] * 100
    )

    chart_data["Class"] = (
        chart_data["Class"].str.title()
    )

    st.bar_chart(
        chart_data.set_index("Class")["Confidence (%)"],
        horizontal=True,
    )

    with st.expander("View detailed confidence scores"):

        display_scores = chart_data[
            ["Class", "Confidence (%)"]
        ].copy()

        display_scores["Confidence (%)"] = (
            display_scores["Confidence (%)"].map(
                lambda value: f"{value:.2f}%"
            )
        )

        st.dataframe(
            display_scores,
            use_container_width=True,
            hide_index=True,
        )

else:

    st.info(
        "Upload an image to begin classification."
    )


# ---------------------------------------------------------
# 7. MODEL PERFORMANCE COMPARISON
# ---------------------------------------------------------

st.divider()

st.subheader("Model Performance Comparison")

st.write(
    "The following results show the performance of all four "
    "models on the same official CIFAR-10 test set."
)

comparison_df = pd.DataFrame({
    "Model": [
        "Baseline CNN",
        "Custom Residual CNN",
        "Augmented Residual CNN",
        "MobileNetV2",
    ],
    "Test Accuracy (%)": [
        81.43,
        84.92,
        85.11,
        95.40,
    ],
})

st.bar_chart(
    comparison_df.set_index("Model")
)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "MobileNetV2 uses ImageNet-pretrained weights and 224 × 224 "
    "input images. The custom CNN models are trained from scratch "
    "using 32 × 32 CIFAR-10 images."
)


# ---------------------------------------------------------
# 8. FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Deep Learning Image Classification Research Project | "
    "CIFAR-10 Dataset | PyTorch and Streamlit"
)
