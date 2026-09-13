import numpy as np
import matplotlib.pyplot as plt


def plot_history(history):
    """
    Plot training and validation loss and accuracy.

    Parameters
    ----------
    history : dict
        Training history containing "loss" and "accuracy".
        Optionally contains "val_loss" and "val_accuracy".

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing loss and accuracy plots.
    """
    required_keys = {"loss", "accuracy"}

    missing_keys = required_keys - history.keys()

    if missing_keys:
        raise ValueError(f"history is missing required keys: {sorted(missing_keys)}")

    epochs = np.arange(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(
        epochs,
        history["loss"],
        label="Training Loss",
    )

    if "val_loss" in history:
        axes[0].plot(
            epochs,
            history["val_loss"],
            label="Validation Loss",
        )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(
        epochs,
        history["accuracy"],
        label="Training Accuracy",
    )

    if "val_accuracy" in history:
        axes[1].plot(
            epochs,
            history["val_accuracy"],
            label="Validation Accuracy",
        )

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    return fig
