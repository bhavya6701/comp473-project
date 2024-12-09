# COMP473 - Artistic Style Transfer Using Convolutional Neural Networks

## Project Overview

This project implements a sophisticated **Neural Style Transfer** system that leverages Convolutional Neural Networks (CNNs) to blend the content of one image with the artistic style of another. Inspired by the groundbreaking paper "*A Neural Algorithm of Artistic Style*" by Gatys et al., our implementation demonstrates the power of deep learning in image transformation.

## Team Members

- Ruturajsinh Vihol (40154693)
- Shibin Koshy (40295019)
- Bhavya Ruparelia (40164863)

## Key Features

- **Advanced Feature Extraction**: Utilizes pre-trained **VGG-16**, **VGG-19** and **ResNet-18** networks
- **Flexible Style Transfer**: Supports multiple optimization techniques
- **Comprehensive Analysis**: Includes comparisons of:
  - Different CNN architectures
  - Content extraction layers
  - Style-to-content weight ratios
  - Optimization algorithms (Adam vs L-BFGS)

## Technical Methodology

### Feature Representation

- **Content Features**: Extracted from intermediate CNN layers to preserve image structure
- **Style Features**: Captured using Gram matrices to represent texture and artistic characteristics
- **Loss Calculation**: Combines content and style losses to guide image synthesis

### Optimization Techniques

1. **Adam Optimizer**
   - Efficient gradient-based optimization
   - Provides stable convergence

2. **L-BFGS Optimizer**
   - Memory-efficient large-scale optimization
   - Approximates inverse Hessian matrix
   - Generates unique stylized outputs

## Configurable Parameters

- **Content Layers**: Customize feature extraction (`Conv1_1` to `Conv5_1`)
- **Style Weights**: Control balance between content and style
- **Iterations**: Define optimization process duration
- **Initial Image**: Start from content image or random noise

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/bhavya6701/comp473-project.git
   ```

2. Install the required dependencies (preferably in a virtual environment like `conda` or `venv`):

   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the content and style images in the `images` directory

4. Configure the model parameters in `model_config.json`

5. Open the Jupyter notebook with IDE or terminal:

   ```bash
   jupyter notebook
   ```

## Visualization

The notebook provides comprehensive visualizations:

- Original content and style images
- Stylized outputs from different models
- Comparative analysis of layer and weight variations

## Performance Insights

- **VGG-19** captures more intricate details compared to **VGG-16** and **ResNet-18** provides poor results for style transfer
- Deeper content layers preserve more structural information
- Lower α/β ratios emphasize artistic style
- Higher α/β ratios maintain content structure

## References

Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image Style Transfer Using Convolutional Neural Networks. CVPR 2016.

## Acknowledgments

Special thanks to the course instructors and the deep learning research community.
