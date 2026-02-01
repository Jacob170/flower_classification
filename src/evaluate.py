# Evaluation fucntion

import matplotlib.pyplot as plt
import os


def plot_cross_entropy_loss(history, model_name, split_name, test_loss):
    """
    Calculate accuracy
    Generate confusion matrix
    Plot accuracy and loss curves
    """

    # the losses and epochs lists
    train_losses = history["train_loss"]
    val_losses = history["val_loss"]

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, "g-", label="Validation Loss", linewidth=2)
    plt.axhline(
        y=test_loss,
        color="r",
        linestyle="--",
        label=f"Test Loss: {test_loss:.4f}",
        linewidth=2,
    )
    plt.title(f"Cross-Entropy Loss - {model_name.upper()} ({split_name})", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("results/graphs", exist_ok=True)
    save_path = f"results/graphs/{model_name}_{split_name}_loss.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()  # Close to free memory
    print(f"Saved: {save_path}")


def plot_accuracy_graph(history, model_name, split_name, test_acc):
    epochs = range(1, len(history["train_loss"]) + 1)

    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, "b-", label="Train Accuracy", linewidth=2)
    plt.plot(epochs, val_acc, "g-", label="Validation Accuracy", linewidth=2)
    plt.axhline(
        y=test_acc,
        color="r",
        linestyle="--",
        label=f"Test Accuracy: {test_acc:.2f}%",
        linewidth=2,
    )
    plt.title(f"Accuracy - {model_name.upper()} ({split_name})", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs("results/graphs", exist_ok=True)
    save_path = f"results/graphs/{model_name}_{split_name}_accuracy.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()  # Close to free memory
    print(f"Saved: {save_path}")


print("defined plotting methods")
