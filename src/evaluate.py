import matplotlib.pyplot as plt


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

    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Validation")
    # Test is single value - show as horizontal line
    plt.axhline(
        y=test_loss, color="red", linestyle="--", label=f"Test: {test_loss:.2f}"
    )

    # Labels and title
    plt.title(f"Loss over Epochs - {model_name.upper()} ({split_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid()
    plt.show()


def plot_accuracy_graph(history, model_name, split_name, test_acc):
    epochs = range(1, len(history["train_loss"]) + 1)

    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")

    plt.axhline(y=test_acc, color="red", linestyle="--", label=f"Test: {test_acc:.2f}")
    # Labels and title
    plt.title(f"Accuracy over Epochs - {model_name.upper()} ({split_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()
    plt.show()
