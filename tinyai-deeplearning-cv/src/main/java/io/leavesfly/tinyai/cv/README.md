# TinyAI Computer Vision Module

This module provides computer vision functionalities for the TinyAI deep learning framework.

## Features

- Image preprocessing and augmentation
- Classic CNN architectures (LeNet, AlexNet, VGG, ResNet)
- Feature visualization tools
- Transfer learning support
- CIFAR-10 image classification example

## Modules

### Image Processing
- Normalization and standardization
- Data augmentation (rotation, flipping, cropping, etc.)
- Batch processing pipeline

### CNN Architectures
- LeNet-5 implementation
- Simplified AlexNet
- VGG network blocks
- ResNet residual blocks

### Visualization
- Convolutional kernel visualization
- Feature map visualization
- Class Activation Mapping (CAM)

### Transfer Learning
- Feature extraction
- Model fine-tuning
- Pre-trained model management

## Usage

Add this dependency to your project:

```xml
<dependency>
    <groupId>io.leavesfly</groupId>
    <artifactId>tinyai-deeplearning-cv</artifactId>
    <version>1.0.0</version>
</dependency>
```

## Example

```java
// Create a LeNet-5 model
LeNet5 lenet = new LeNet5();

// Load CIFAR-10 dataset
Cifar10Dataset dataset = new Cifar10Dataset("path/to/cifar10");
dataset.loadTrainData();

// Preprocess images
ImagePreprocessor preprocessor = new ImagePreprocessorImpl();
BatchProcessor batchProcessor = new BatchProcessor();

// Train the model
// ... (see Cifar10TrainingExample for complete training code)
```