import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_losses(history) -> None:
    train_losses = [x["train_loss"] for x in history]
    val_losses = [x["val_loss"] for x in history]
    train_acc = [x["train_acc"] for x in history]
    val_acc = [x["val_acc"] for x in history]
    plt.plot(train_losses, "-bx")
    plt.plot(val_losses, "-rx")
    plt.plot(val_acc, "-gx")
    plt.plot(train_acc, "-mx")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid()
    plt.legend(["train_loss", "val_loss", "val_acc", "train_acc"])
    plt.title("Loss vs. NO. of epochs")


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "Receiver Operating Characteristic (ROC) Curve",
    figsize: tuple[int, int] = (8, 6),
    pos_label: int = 1,
) -> plt.Figure:
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr: Sequence of false positive rates.
        tpr: Sequence of true positive rates.
        auc_score: Optional Area Under the Curve score to display in the legend.
        title: The title for the plot.
        figsize: The figure size for the plot.
    """
    # Check if y_scores is for multi-class or binary
    if y_scores.ndim > 1 and y_scores.shape[1] > 1:
        # Assuming y_scores contains probabilities for multiple classes,
        # select the scores for the positive class
        if pos_label >= y_scores.shape[1]:
            raise ValueError(
                f"pos_label {pos_label} is out of bounds for y_scores with shape {y_scores.shape}"
            )
        y_scores = y_scores[:, pos_label]  # Select column corresponding to pos_label

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=figsize)  # Capture the figure object
    ax = fig.add_subplot(111)  # Get axes object

    # Plot the ROC curve
    label = f"ROC curve (AUC = {roc_auc:.4f})"
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=label)

    # Plot the diagonal line (random guessing)
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")

    # Set plot limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])  # Add a little space at the top
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)  # Add a grid for easier reading

    plt.show()  # Remove this line - let the caller decide to show or save
    plt.close(fig)  # Close the plot window to prevent display if not needed

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: str,
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
    normalize: str | None = None,  # Options: None, 'true', 'pred', 'all'
    fmt: str = "d",  # Format for annotations (e.g., 'd' for integer, '.2f' for float)
) -> plt.Figure:
    """
    Computes and plots a confusion matrix using seaborn heatmap.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        class_names: Optional list of class names for axis labels.
                     If None, integer labels (0, 1, 2...) are used.
        figsize: Figure size for the plot.
        cmap: Colormap for the heatmap.
        title: Title for the plot.
        normalize: Normalizes confusion matrix over the true (rows), predicted (columns)
                   conditions or all the population. If None, confusion matrix will not be
                   normalized. Options: 'true', 'pred', 'all'.
        fmt: String formatting code to use when adding annotations.
             Use '.2f' if normalizing, 'd' otherwise.

    Returns:
        matplotlib.figure.Figure: The figure object containing the confusion matrix plot.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    # Adjust format if normalizing
    if normalize:
        fmt = ".2f"

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 12},
    )  # Adjust annotation font size if needed

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )  # Improve x-label readability
    plt.setp(ax.get_yticklabels(), rotation=0)  # Keep y-labels horizontal
    fig.tight_layout()  # Adjust layout to prevent labels overlapping

    plt.show()  # Optional: remove if saving programmatically
    plt.close(fig)  # Close the plot window to prevent display if not needed
    return fig


def plot_image_mask_predictions(
    images: list | np.ndarray | torch.Tensor,
    true_masks: list | np.ndarray | torch.Tensor,
    pred_masks: list | np.ndarray | torch.Tensor,
    true_labels: list | np.ndarray | torch.Tensor,
    pred_labels: list | np.ndarray | torch.Tensor,
    class_names: list[str] | None = None,
    num_samples: int = 9,
    figsize: tuple[int, int] = (10, 20),
    mask_alpha: float = 0.5,
    mask_cmap: str = "Reds",  # Colormap for masks
    seed: int | None = None,  # Optional seed for reproducibility if selecting random samples
) -> plt.Figure:
    """
    Plots a grid comparing original images, ground truth masks, and predicted masks.

    Args:
        images: List, NumPy array, or Torch tensor of original images.
                Expected shape: (N, H, W) or (N, C, H, W) or (N, H, W, C).
        true_masks: List, NumPy array, or Torch tensor of ground truth masks.
                    Expected shape: (N, H, W).
        pred_masks: List, NumPy array, or Torch tensor of predicted masks.
                    Expected shape: (N, H, W). Should be binary (0 or 1) or probabilities.
                    If probabilities, they will be thresholded at 0.5.
        true_labels: List, NumPy array, or Torch tensor of ground truth labels (indices or strings).
                     Expected shape: (N,).
        pred_labels: List, NumPy array, or Torch tensor of predicted labels (indices or strings).
                     Expected shape: (N,).
        class_names: Optional list of class names to map label indices to readable names.
        num_samples: Number of samples to plot (default is 9).
        figsize: Figure size.
        mask_alpha: Transparency level for mask overlays.
        mask_cmap: Colormap used for displaying masks.
        seed: Optional random seed for selecting samples if len(images) > num_samples.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    total_available = len(images)
    if total_available < num_samples:
        print(
            f"Warning: Requested {num_samples} samples, but only {total_available} are available. Plotting all available samples."
        )
        num_samples = total_available
        if num_samples == 0:
            print("No samples to plot.")
            return plt.figure(figsize=figsize)  # Return an empty figure

    # --- Data Selection ---
    if total_available > num_samples:
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.choice(total_available, num_samples, replace=False)
    else:
        indices = np.arange(total_available)

    # --- Data Preparation ---
    def _prepare_data(data, index):
        item = data[index]
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()

        # Handle image dimensions (assuming grayscale H,W or C,H,W or H,W,C)
        if item.ndim == 3:
            if item.shape[0] in [1, 3]:  # Check if channel-first (C, H, W)
                # Convert to H, W, C for matplotlib
                item = np.transpose(item, (1, 2, 0))
            # If shape is (H, W, C), it's already fine
        # If shape is (H, W), it's grayscale, also fine

        # Squeeze single-channel dimensions for grayscale images if needed
        if item.ndim == 3 and item.shape[-1] == 1:
            item = item.squeeze(-1)

        # Normalize images to [0, 1] if they aren't already (basic check)
        if item.max() > 1.0 and np.issubdtype(item.dtype, np.floating):
            item = (item - item.min()) / (item.max() - item.min())
        elif np.issubdtype(item.dtype, np.integer):
            item = item / 255.0  # Assuming uint8

        return item

    def _prepare_mask(mask_data, index):
        mask = mask_data[index]
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        # Ensure mask is 2D (H, W)
        if mask.ndim != 2:
            raise ValueError(f"Masks should be 2D (H, W), but got shape {mask.shape}")
        # Threshold if probabilities are provided
        if mask.max() > 1.0 or (mask.min() < 0.0 and mask.max() <= 1.0):  # Check if not binary 0/1
            mask = (mask > 0.5).astype(float)  # Threshold probabilities
        return mask

    def _get_label_name(label_data, index):
        label = label_data[index]
        if isinstance(label, torch.Tensor):
            label = label.item()  # Get scalar value
        if class_names:
            try:
                return class_names[int(label)]
            except (IndexError, ValueError):
                return f"Unknown ({label})"  # Handle invalid index
        return str(label)  # Return label as string if no class_names

    # --- Plotting ---
    fig, axes = plt.subplots(num_samples, 3, figsize=figsize)

    # Handle case where num_samples = 1 separately for indexing
    if num_samples == 1:
        axes = np.array([axes])  # Make it 2D for consistent indexing

    for i, idx in enumerate(indices):
        img = _prepare_data(images, idx)
        true_mask = _prepare_mask(true_masks, idx)
        pred_mask = _prepare_mask(pred_masks, idx)
        true_label_name = _get_label_name(true_labels, idx)
        pred_label_name = _get_label_name(pred_labels, idx)

        # Determine cmap for image based on dimensions
        img_cmap = "gray" if img.ndim == 2 else None

        # Column 1: Original Image + True Label
        ax = axes[i, 0]
        ax.imshow(img, cmap=img_cmap)
        ax.set_title(f"True: {true_label_name}")
        ax.axis("off")

        # Column 2: Image + True Mask + True Label
        ax = axes[i, 1]
        ax.imshow(img, cmap=img_cmap)
        ax.imshow(true_mask, cmap=mask_cmap, alpha=mask_alpha, vmin=0, vmax=1)  # Overlay true mask
        ax.set_title(f"True Mask: {true_label_name}")
        ax.axis("off")

        # Column 3: Image + Predicted Mask + Predicted Label
        ax = axes[i, 2]
        ax.imshow(img, cmap=img_cmap)
        ax.imshow(
            pred_mask, cmap=mask_cmap, alpha=mask_alpha, vmin=0, vmax=1
        )  # Overlay predicted mask
        ax.set_title(f"Pred Mask: {pred_label_name}")
        ax.axis("off")

    plt.tight_layout(pad=0.5)  # Adjust spacing
    # plt.show() # Optional: remove if saving programmatically or showing later
    return fig
