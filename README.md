# Deep Learning Assignment (2024-25)

This repository contains all code, experiments, and documentation for the MSc Data Science Deep Learning course assignment.

## Part 1: Classic Image Classification

We worked on two popular datasets:

- **Fashion-MNIST**: Grayscale images of clothing items.
- **CIFAR-10**: RGB images of various object classes.

### Models:

- `mnist_mlp.ipynb`: MLP built using the Functional API for Fashion-MNIST.
- `cifar10_cnn.ipynb`: Convolutional Neural Network for CIFAR-10 using Conv2D + MaxPooling + Dropout.

## Part 2: X-ray Classification (MURA Dataset)

We tackled the MURA dataset to detect abnormal radiographs and classify the body part as an auxiliary task.

### Models:

- `mura_transfer_cnn_finetune.ipynb`: Custom CNN built from scratch (multitask: binary + body part)
- `mura_transfer_resnet.ipynb`: Transfer learning using ResNet50 pretrained on ImageNet
- `mura_transfer_efficientnet.ipynb`: ❌ Failed transfer learning with EfficientNetB0 (val accuracy ~4.3%)
- `mura_transfer_inceptionv3.ipynb`: ✅ Final multitask model using InceptionV3 + MaxPooling

### Dataset Handling:

- Grayscale → RGB conversion
- tf.data pipeline with caching
- Sample weighting using `sklearn.utils.compute_sample_weight`
- Data augmentation: horizontal flip, contrast, brightness
- Training on Google Colab Pro (A100 GPU)

### Challenges:

- Severe overfitting and domain mismatch with EfficientNetB0
- Multitask output shape mismatches in transfer learning pipelines
- Out-of-memory crashes on Colab (A100) during multitask training
- Transition from grayscale to RGB inputs for pretrained backbones
- Large dataset loading on low-end hardware

## Report

Final report is in `Deep Learning Assignment Report.pdf` and includes all metrics, plots, decisions, and experimental details.

## Author

Iason-Christoforos Asproudis  
Student ID: p3352318  
[GitHub Repository](https://github.com/jasproudis/deep-learning-assignment)
