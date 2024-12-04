import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image
from torchvision import transforms


def load_image(img_path: str, max_size: int = 400, shape: tuple = None) -> torch.Tensor:
    """
    Loads an image, resizes it, and converts it to a normalized PyTorch tensor.

    Parameters:
        img_path (str): Path to the image file to load.
        max_size (int, optional): The maximum size for resizing the image (default is 400).
        shape (tuple, optional): The exact target size (height, width) for resizing (default is None).

    Returns:
        torch.Tensor: A 4D tensor of shape (1, channels, height, width) representing the image.
    """
    # Load the image from the given path and convert it to RGB format
    image = Image.open(img_path).convert("RGB")

    # Set the target size based on the provided shape or resize using max_size
    if shape is not None:
        target_size = shape
    else:
        target_size = min(max(image.size), max_size)

    # Define the transformation pipeline for the input image
    in_transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    # Apply the transformations and unsqueeze to add the batch dimension (1, C, H, W)
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image  # Return the image tensor with shape (1, 3, H, W)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized PyTorch tensor into a denormalized image suitable for visualization.

    Parameters:
        tensor (torch.Tensor): A PyTorch tensor of shape (batch_size, channels, height, width) or (channels, height, width).
        It is assumed to be normalized with ImageNet statistics (mean and std).

    Returns:
        np.ndarray: A NumPy array representing the image in the range [0, 1] with shape (height, width, channels).
    """
    # Define a transformation to invert the normalization applied during preprocessing
    denormalize = transforms.Normalize(
        mean=[
            -0.485 / 0.229,
            -0.456 / 0.224,
            -0.406 / 0.225,
        ],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    # Remove the batch dimension if it exists
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Denormalize the tensor, convert it to a NumPy array, rearrange dimensions, and clip values
    image = denormalize(tensor).cpu().numpy()
    image = image.squeeze().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)

    return image


def plot_style_transfer(images: list, total_losses: list, steps: int, checkpoints: int):
    """
    Plots a series of images along with a graph of total loss versus iteration.

    Parameters:
        images (list): List of image arrays to be plotted. Typically represents intermediate and final outputs of a model.
        total_losses (list): List of total loss values, where each entry corresponds to a specific iteration.
        steps (int): Total number of iterations used in the process.
        checkpoints (int): Number of checkpoints (intermediate stages) at which images were saved.
    """
    # Create a plot with 2 rows and 6 columns
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))

    # Flatten axes for easy iteration
    axes = axes.flatten()
    for i, image in enumerate(images):
        title = (
            f"Iteration {i * (steps // checkpoints):,}"
            if i > 0 and i < len(images) - 1
            else "Initial Image (Noise)"
            if i == 0
            else "Final Image"
        )
        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title(title)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Plot the total loss values
    plt.figure(figsize=(10, 5))
    plt.plot(total_losses, label="Total Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("Total Loss vs Iteration")
    plt.legend()
    plt.show()


def plot_images(
    fig_size: tuple,
    rows: int,
    cols: int,
    images: list,
    titles: list = None,
    axis: str | bool = "off",
):
    """
    Plots a grid of images with optional titles and axis visibility.

    Parameters:
        fig_size (tuple): Tuple specifying the figure size (width, height).
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        images (list): List of image arrays to be plotted.
        titles (list, optional): List of strings for the titles of the images. Defaults to None.
        axis (str or bool, optional): Axis visibility. Use 'on', 'off', True, or False. Defaults to 'off'.

    Raises:
        ValueError: If the number of images exceeds rows * cols.
    """
    total_plots = rows * cols

    if len(images) > total_plots:
        raise ValueError(
            f"Number of images ({len(images)}) exceeds grid size ({total_plots})."
        )

    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    axes = axes.ravel() if total_plots > 1 else [axes]

    for i in range(total_plots):
        if i < len(images):
            axes[i].imshow(
                images[i], cmap="gray" if len(images[i].shape) == 2 else None
            )
            if titles and i < len(titles):
                axes[i].set_title(titles[i], fontsize=10)
            axes[i].axis(axis)
        else:
            axes[i].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=1)
    plt.show()
