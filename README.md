# COMP473 - Pattern Recognition
## Artistic Style Transfer Using Convolutional Neural Networks

## Project Overview
This project implements an **image style transfer system** that combines the **content of one image** with the **style of another**. Based on the work of Gatys et al. (2016), the system leverages **Convolutional Neural Networks (CNNs)**, specifically the **VGG16** architecture, to extract and recombine content and style features from input images.

## Key Objectives
- Extract content representations from higher CNN layers.
- Capture style features using Gram matrices across multiple layers.
- Synthesize a new image that blends content and style through optimization of a combined loss function.

## Methodology
1. **Data Preparation**: Input images are loaded and preprocessed, with a content image representing the structure and a style image representing the texture.
2. **Model and Feature Extraction**: We use a pretrained **VGG16** model to:
   - Extract **content features** from higher layers.
   - Capture **style features** by computing Gram matrices at multiple layers.
3. **Training (Style Transfer Optimization)**: Using backpropagation, we minimize a combined **content and style loss** to generate a new image. PyTorch and the L-BFGS optimizer facilitate efficient optimization.
4. **Testing and Evaluation**: The style transfer is evaluated through visual inspection of synthesized images and a qualitative balance of style vs. content retention.

## Inference
Run the style transfer on any pair of content and style images using the provided inference script or notebook:
```python
python style_transfer_inference.py --content path/to/content.jpg --style path/to/style.jpg --output path/to/output.jpg
```

## Results
The output is a synthesized image that visually combines the content of the content image with the style of the style image. Sample results show the flexibility of the system across various styles and content structures, with visual comparisons included in the results/ directory.

## References
Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image Style Transfer Using Convolutional Neural Networks. CVPR 2016.
