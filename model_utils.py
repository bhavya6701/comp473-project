import torch
from torchvision import models


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
    content_loss = torch.mean((target_features[layer] - content_features_layer) ** 2)

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
