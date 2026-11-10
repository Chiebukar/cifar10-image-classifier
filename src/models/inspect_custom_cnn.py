
import torch

from src.models.custom_cnn import CustomResidualCNN


def main() -> None:
    model = CustomResidualCNN(num_classes=10)

    dummy_input = torch.randn(4, 3, 32, 32)

    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Custom CNN architecture verification")
    print("-" * 45)
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    assert output.shape == (4, 10), (
        f"Unexpected output shape: {output.shape}"
    )

    assert torch.isfinite(output).all(), (
        "Model produced non-finite values."
    )

    print("\nArchitecture verification successful.")


if __name__ == "__main__":
    main()
