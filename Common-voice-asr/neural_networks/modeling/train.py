from pathlib import Path

from loguru import logger
import torch
import typer

from neural_networks.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    logger.info("Starting toy model training...")

    input = torch.ones(5)
    expected_output = torch.zeros(3)

    weight = torch.randn(5,3,requires_grad=True)
    bias = torch.randn(3, requires_grad=True)

    z = torch.matmul(input, weight) + bias
    loss_function = torch.nn.functional.binary_cross_entropy_with_logits(z, expected_output)

    logger.info(f"Loss computed: {loss_function.item(): .4f}")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

