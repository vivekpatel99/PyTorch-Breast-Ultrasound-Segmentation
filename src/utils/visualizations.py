import logging
import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve

log = logging.getLogger(__name__)


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

    # plt.show()
    plt.close(fig)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: str,
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
    normalize: str | None = None,
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

    # plt.show()
    plt.close(fig)  # Close the plot window to prevent display if not needed
    return fig


def create_prediction_gif(
    images: list | np.ndarray | torch.Tensor,
    true_masks: list | np.ndarray | torch.Tensor,
    pred_masks: list | np.ndarray | torch.Tensor,
    true_labels: list | np.ndarray | torch.Tensor,
    pred_labels: list | np.ndarray | torch.Tensor,
    gif_path: str,
    class_names: list[str] | None = None,
    duration: float = 1.0,  # Duration per frame in SECONDS
    mask_alpha: float = 0.5,
    mask_cmap: str = "Reds",
    samples_per_frame: int = 3,
    figsize_scale: float = 4.0,
    fps: int | None = None,  # Optional: Frames per second (overrides duration if set)
) -> None:
    """
    Creates an animated GIF using Matplotlib Animation comparing images,
    true masks, and predicted masks. Each frame displays a grid.

    Args:
        images: List/array/tensor of original images.
        true_masks: List/array/tensor of ground truth masks.
        pred_masks: List/array/tensor of predicted masks (probabilities or binary).
        true_labels: List/array/tensor of ground truth labels.
        pred_labels: List/array/tensor of predicted labels.
        gif_path: Path to save the output GIF.
        class_names: Optional list of class names.
        duration: Duration (seconds) per frame in the GIF. Used if fps is None.
        mask_alpha: Transparency for mask overlays.
        mask_cmap: Colormap for masks.
        samples_per_frame: How many samples to show in each frame (determines rows).
        figsize_scale: Controls the size of each subplot within the frame.
        fps: Optional frames per second for the animation. If set, overrides duration.
    """
    total_available = len(images)
    if total_available == 0:
        log.warning("No samples provided to create GIF.")
        return
    if samples_per_frame <= 0:
        log.warning("samples_per_frame must be positive. Defaulting to 3.")
        samples_per_frame = 3

    num_cols = 3  # Fixed: Image, True Mask, Pred Mask
    num_frames = math.ceil(total_available / samples_per_frame)

    log.info(f"Preparing animation with {num_frames} frames using Matplotlib Animation...")

    # --- Data Preparation Helpers (Keep as they were) ---
    # ... (_prepare_data, _prepare_mask, _get_label_name remain unchanged) ...
    def _prepare_data(data, index):
        # (Same code as before)
        item = data[index]
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if item.ndim == 3:
            if item.shape[0] in [1, 3]:
                item = np.transpose(item, (1, 2, 0))
        if item.ndim == 3 and item.shape[-1] == 1:
            item = item.squeeze(axis=-1)
        if np.issubdtype(item.dtype, np.floating) and (item.min() < 0.0 or item.max() > 1.0):
            min_val, max_val = item.min(), item.max()
            if max_val > min_val:
                item = (item - min_val) / (max_val - min_val)
            else:
                item = np.zeros_like(item)
        elif np.issubdtype(item.dtype, np.integer):
            max_val = np.iinfo(item.dtype).max if np.issubdtype(item.dtype, np.integer) else 255.0
            item = item.astype(np.float32) / max_val if max_val > 0 else item.astype(np.float32)
        item = np.clip(item, 0.0, 1.0)
        return item

    def _prepare_mask(mask_data, index):
        # (Same code as before)
        mask = mask_data[index]
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        original_shape = mask.shape
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(
                f"Masks should be reducible to 2D (H, W), got {original_shape} -> {mask.shape} at index {index}"
            )
        is_proba = (
            mask.min() >= 0.0
            and mask.max() <= 1.0
            and not np.all(np.logical_or(mask == 0, mask == 1))
        )
        if mask.max() > 1.0 or is_proba:
            mask = (mask > 0.5).astype(float)
        return mask

    def _get_label_name(label_data, index):
        # (Same code as before)
        label = label_data[index]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if class_names:
            try:
                int_label = int(round(label))
                if 0 <= int_label < len(class_names):
                    return class_names[int_label]
                else:
                    return f"Idx {int_label} (Out of {len(class_names)})"
            except (ValueError, TypeError):
                return f"Invalid Label ({label})"
        return str(label)

    # --- Set up the figure and axes (only once) ---
    fig, axes = plt.subplots(
        nrows=samples_per_frame,
        ncols=num_cols,
        figsize=(num_cols * figsize_scale, samples_per_frame * figsize_scale),
    )
    # Ensure axes is always a 2D array
    if samples_per_frame == 1:
        axes = np.array([axes])

    # --- Define the update function for animation ---
    def update(frame_idx):
        """Updates the plot content for a given frame index."""
        log.debug(
            f"Updating frame {frame_idx+1}/{num_frames}"
        )  # Use debug level for less verbose output

        # Calculate the range of sample indices for this frame
        start_idx = frame_idx * samples_per_frame
        end_idx = min(start_idx + samples_per_frame, total_available)
        batch_indices = range(start_idx, end_idx)

        # Clear previous content and plot new data
        for row_idx in range(samples_per_frame):
            for col_idx in range(num_cols):
                ax = axes[row_idx, col_idx]
                ax.clear()  # Clear previous drawings
                ax.axis("off")  # Turn off axis by default
                ax.set_visible(False)  # Hide by default

        # Plot samples for the current frame
        for row_idx, sample_idx in enumerate(batch_indices):
            try:
                img = _prepare_data(images, sample_idx)
                true_mask = _prepare_mask(true_masks, sample_idx)
                pred_mask = _prepare_mask(pred_masks, sample_idx)
                true_label_name = _get_label_name(true_labels, sample_idx)
                pred_label_name = _get_label_name(pred_labels, sample_idx)
                img_cmap = "gray" if img.ndim == 2 else None

                # Make axes visible before plotting
                axes[row_idx, 0].set_visible(True)
                axes[row_idx, 1].set_visible(True)
                axes[row_idx, 2].set_visible(True)

                # Plot Image
                ax = axes[row_idx, 0]
                ax.imshow(img, cmap=img_cmap)
                ax.set_title(
                    f"Sample {sample_idx+1}\nImage (True: {true_label_name})", fontsize=10
                )  # Adjust font size if needed
                ax.axis("off")

                # Plot True Mask Overlay
                ax = axes[row_idx, 1]
                ax.imshow(img, cmap=img_cmap)
                ax.imshow(true_mask, cmap=mask_cmap, alpha=mask_alpha, vmin=0, vmax=1)
                ax.set_title(f"Sample {sample_idx+1}\nTrue Mask Overlay", fontsize=10)
                ax.axis("off")

                # Plot Predicted Mask Overlay
                ax = axes[row_idx, 2]
                ax.imshow(img, cmap=img_cmap)
                ax.imshow(pred_mask, cmap=mask_cmap, alpha=mask_alpha, vmin=0, vmax=1)
                ax.set_title(
                    f"Sample {sample_idx+1}\nPred Mask (Pred: {pred_label_name})", fontsize=10
                )
                ax.axis("off")

            except Exception as e:
                log.error(
                    f"Error plotting sample index {sample_idx} in frame {frame_idx}: {e}",
                    exc_info=True,
                )
                # Optionally mark the subplot as errored
                axes[row_idx, 0].set_title(
                    f"Sample {sample_idx+1}\nError Loading", color="red", fontsize=10
                )
                axes[row_idx, 1].set_visible(False)
                axes[row_idx, 2].set_visible(False)

        fig.suptitle(
            f"Frame {frame_idx + 1}/{num_frames}", fontsize=12
        )  # Optional: Add frame number to title
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust rect to make space for suptitle

        # Return the axes that were modified (important for blitting optimization, though not strictly needed here)
        return [ax for row in axes for ax in row]

    # --- Create the animation ---
    # Calculate interval in milliseconds
    interval_ms = int(duration * 1000) if fps is None else int(1000 / fps)

    # Note: blit=True can improve performance but might cause issues with changing titles/layouts.
    # Set blit=False for more robustness if experiencing problems.
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval_ms,
        blit=False,  # Set to False for simplicity/robustness with title changes
        repeat=True,  # Corresponds to loop=0 in imageio/pillow
    )

    # --- Save the animation ---
    log.info(f"Saving animation to {gif_path}...")
    try:
        # You might need to install ffmpeg or imagemagick for saving animations
        # Pillow writer is often available by default for GIFs
        # ani.save(gif_path, writer='pillow', fps=(fps if fps else int(1.0/duration)))
        ani.save(gif_path, writer="pillow", fps=2, dpi=75)
        log.info("GIF saved successfully using Matplotlib Animation.")
    except Exception as e:
        log.error(f"Failed to save animation: {e}", exc_info=True)
        log.warning(
            "Ensure you have a suitable writer installed (e.g., 'pillow', 'imagemagick'). Try installing Pillow: pip install Pillow"
        )
    finally:
        plt.close(fig)  # Close the figure after saving or if an error occurs
