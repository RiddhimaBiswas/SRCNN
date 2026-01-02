# Super-Resolution CNN with Custom Loss

## Overview
A dynamic CNN for image super-resolution using custom filters, adaptive activations,
and a perceptual loss combining SSIM, edge strength, entropy, and noise.

## Features
- Custom GELU, Swish, Fixed-PReLU
- Edge-aware + entropy-aware loss
- LoG, Gabor, Sobel, Scharr kernels
- Bicubic skip connection

## Training
```bash
python train.py
