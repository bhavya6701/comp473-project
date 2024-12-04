import torch
from tqdm import tqdm
from torch import optim
from torchvision import models

from image_utils import tensor_to_image


def load_models(device: torch.device) -> dict:
    """
    Loads pre-trained VGG models (VGG16 and VGG19) from torchvision, freezes their parameters,
    and moves them to the specified device (CPU or GPU).

    Parameters:
        device (torch.device): The device to move the model to (e.g., 'cuda' or 'cpu').

    Returns:
        dict: A dictionary containing the feature extractor parts of VGG16 and VGG19 models.
    """
    # Load pre-trained VGG16 and VGG19 models from torchvision
    model_dict = {
        "vgg-16": models.vgg16(weights=models.VGG16_Weights.DEFAULT).features,
        "vgg-19": models.vgg19(weights=models.VGG19_Weights.DEFAULT).features,
    }

    # Freeze all model parameters (no backpropagation updates) and move to the device
    for model in model_dict.values():
        for param in model.parameters():
            param.requires_grad = False
        model.to(device)

    return model_dict


def extract_features(image: torch.Tensor, model: torch.nn.Module, layers: dict) -> dict:
    """
    Extracts feature maps from specified layers of a neural network model for a given image.

    Parameters:
        image (torch.Tensor): The input image tensor, typically of shape (batch_size, channels, height, width).
        model (torch.nn.Module): The neural network model (e.g., a pretrained VGG network).
        layers (dict): A dictionary mapping layer names in the model to descriptive feature names.

    Returns:
        dict: A dictionary containing the extracted feature maps, where keys are the descriptive feature names
        specified in `layers` and values are the corresponding feature maps as tensors.
    """
    # Initialize an empty dictionary to store the extracted features
    features = {}
    x = image

    # Iterate through the model layers in sequential order
    for name, layer in model._modules.items():
        # Apply the current layer to the input
        x = layer(x)

        # Store the feature map if the layer is in the given layers dictionary
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gram matrix for a given feature map tensor.

    Parameters:
        tensor (torch.Tensor): A 4D tensor with shape (batch_size, depth, height, width) representing the feature map of an image.

    Returns:
        torch.Tensor: The Gram matrix of shape (depth, depth), representing the pairwise correlations between feature map channels.
    """
    # Get the dimensions of the tensor: batch size, depth (channels), height, and width
    b, d, h, w = tensor.size()

    # Reshape the tensor to combine the spatial dimensions (height and width)
    # Resulting shape: (batch_size, depth, height * width)
    tensor = tensor.view(b, d, h * w)

    # Compute the Gram matrix as the dot product of the tensor with its transpose
    # Shape of gram: (batch_size, depth, depth)
    gram = torch.bmm(tensor, tensor.transpose(1, 2)) / (d * h * w)

    return gram


def calculate_content_loss(
    target_features: dict, content_features: dict, layer: str
) -> float:
    """
    Calculates the content loss between the target image and the content image.

    Parameters:
        target_features (dict): A dictionary of feature maps from the target image, keyed by layer names (e.g., 'conv4_2').
        content_features (dict): A dictionary of feature maps from the content image, keyed by the same layer names as `target_features`.
        layer (str): The layer name at which the content loss is to be calculated.

    Returns:
        float: The computed content loss, a scalar value.
    """
    # Detach the content features to prevent gradient computation for the content image
    content_features_layer = content_features[layer].detach()

    # Compute the mean squared error (MSE) between the target and content features
    content_loss = 0.5 * torch.mean(
        (target_features[layer] - content_features_layer) ** 2
    )

    return content_loss  # Return the computed content loss


def calculate_style_loss(
    target_features: dict, style_features: dict, model_style_weights: dict
) -> float:
    """
    Calculates the style loss between the target image and the style image based on their feature maps.

    Parameters:
        target_features (dict): A dictionary of feature maps from the target image, keyed by layer names (e.g., 'conv1_1').
        style_features (dict): A dictionary of feature maps from the style image, keyed by the same layer names as `target_features`.
        model_style_weights (dict): A dictionary mapping layer names to their respective weights for computing the style loss.

    Returns:
        float: The computed style loss, a scalar value.
    """
    # Initialize the total style loss
    style_loss = 0

    # Iterate through the layers in the model_style_weights dictionary
    for layer, weight in model_style_weights.items():
        # Extract feature maps for the current layer
        target_feature = target_features[layer]
        style_feature = style_features[layer]

        # Get the dimensions of the feature map
        _, d, h, w = target_feature.shape

        # Compute the Gram matrices for the target and style feature maps
        target_gram = gram_matrix(target_feature)
        style_gram = gram_matrix(style_feature)

        # Calculate the style loss for the current layer
        # Mean squared error between Gram matrices, weighted by layer weight
        layer_style_loss = weight * torch.mean((target_gram - style_gram) ** 2)

        # Normalize the loss by the total number of elements in the feature map
        style_loss += layer_style_loss

    return style_loss


def style_transfer_from_content(
    model_name: str,
    model_dict: dict,
    content: torch.Tensor,
    style: torch.Tensor,
    data: dict,
    device: torch.device,
):
    """
    Performs style transfer starting from the content image.

    Parameters:
        model_name (str): Name of the model to use (e.g., 'vgg-16').
        model_dict (dict): Dictionary containing pre-trained models.
        content (torch.Tensor): Content image tensor of shape (1, 3, H, W).
        style (torch.Tensor): Style image tensor of shape (1, 3, H, W).
        data (dict): Dictionary containing layers, weights, and hyperparameters.
        device (torch.device): Device to perform computation on (CPU or GPU).

    Returns:
        tuple: A list of images at checkpoints and a list of total losses.
    """
    # Extract configuration from the `data` dictionary
    layers = data["layers"]
    style_weights = data["style_weights"]
    content_loss_layer = data["content_layer"]
    params = data["hyperparameters"]

    # Define hyperparameters
    alpha = params["alpha"]
    beta = params["beta"]
    lr = params["lr"]
    iterations = params["iters"]
    checkpoints = params["ckpts"]

    # Extract features for the content and style images
    content_features = extract_features(
        content, model_dict[model_name], layers[model_name]
    )
    style_features = extract_features(style, model_dict[model_name], layers[model_name])
    model = model_dict[model_name]

    # Initialize the target image as a copy of the content image
    target = content.clone().requires_grad_(True).to(device)

    # Define the optimizer (Adam is used for efficient optimization)
    optimizer = optim.Adam([target], lr=lr)

    # Lists to store images at checkpoints and total loss values
    images = [tensor_to_image(target.detach())]
    total_losses = []

    # Optimization loop
    for step in tqdm(range(1, iterations + 1)):
        # Extract features from the current target image
        target_features = extract_features(target, model, layers[model_name])

        # Calculate content loss
        content_loss = calculate_content_loss(
            target_features, content_features, content_loss_layer
        )

        # Calculate style loss
        style_loss = calculate_style_loss(
            target_features, style_features, style_weights
        )

        # Compute total loss as weighted sum of content and style losses
        total_loss = alpha * content_loss + beta * style_loss

        # Update the target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Store total loss for tracking
        total_losses.append(total_loss.item())

        # Save the target image at specified checkpoints
        if step % (iterations // checkpoints) == 0:
            images.append(tensor_to_image(target.detach()))

    # Log final step information
    print(
        f"Step {step}/{iterations} - Total loss: {total_losses[-1]:.16f}, "
        f"Content loss: {alpha * content_loss.item():.16f}, Style loss: {beta * style_loss.item():.16f}"
    )

    # Save the final target image
    images.append(tensor_to_image(target.detach()))

    return images, total_losses


def style_transfer_from_noise(
    model_name: str,
    model_dict: dict,
    content: torch.Tensor,
    style: torch.Tensor,
    data: dict,
    device: torch.device,
) -> tuple:
    """
    Performs style transfer starting from a noise image.

    Parameters:
        model_name (str): Name of the model to use (e.g., 'vgg-16').
        model_dict (dict): Dictionary containing pre-trained models.
        content (torch.Tensor): Content image tensor of shape (1, 3, H, W).
        style (torch.Tensor): Style image tensor of shape (1, 3, H, W).
        data (dict): Dictionary containing layers, weights, and hyperparameters.
        device (torch.device): Device to perform computation on (CPU or GPU).

    Returns:
        tuple: A list of images at checkpoints and a list of total losses.
    """
    # Extract configuration from the `data` dictionary
    layers = data["layers"]
    style_weights = data["style_weights"]
    content_loss_layer = data["content_layer"]
    params = data["hyperparameters"]

    # Define hyperparameters
    alpha = params["alpha"]
    beta = params["beta"]
    lr = params["lr"]
    iterations = params["iters"]
    checkpoints = params["ckpts"]

    # Extract features for the content and style images
    content_features = extract_features(
        content, model_dict[model_name], layers[model_name]
    )
    style_features = extract_features(style, model_dict[model_name], layers[model_name])
    model = model_dict[model_name]

    # Initialize the target image as random noise, matching the content image size
    target = torch.randn_like(content).requires_grad_(True).to(device)

    # Define the optimizer
    optimizer = optim.LBFGS([target], lr=lr)

    # Lists to store images at checkpoints and total loss values
    images = [tensor_to_image(target.detach())]  # Initial random noise image
    total_losses = []

    # Define the optimization closure
    def closure():
        optimizer.zero_grad()

        # Extract features from the current target image
        target_features = extract_features(target, model, layers[model_name])

        # Calculate content loss
        content_loss = calculate_content_loss(
            target_features, content_features, content_loss_layer
        )

        # Calculate style loss
        style_loss = calculate_style_loss(
            target_features, style_features, style_weights
        )

        # Compute total loss as weighted sum of content and style losses
        total_loss = alpha * content_loss + beta * style_loss

        # Backpropagate total loss
        total_loss.backward()

        # Store total loss for tracking
        total_losses.append(total_loss.item())

        return total_loss

    # Optimization loop
    for step in tqdm(range(1, iterations + 1)):
        optimizer.step(closure)

        # Save the target image at specified checkpoints
        if step % (iterations // checkpoints) == 0:
            images.append(tensor_to_image(target.detach()))

    # Log final step information
    print(f"Step {step}/{iterations} - Total loss: {total_losses[-1]:.16f}")

    # Save the final target image
    images.append(tensor_to_image(target.detach()))

    return images, total_losses
