# cnn-digit-classification

This project extends a basic convolutional neural network (Net-5) for classifying 16×16 grayscale images of handwritten digits. Through iterative modifications—such as deeper architectures, ReLU activations, batch normalization, dropout, and max pooling—this project explores architectural choices that improve model generalization and accuracy.

## Overview

Starting from a baseline CNN, the project conducts six trials by varying network components to study their impact on performance. The final enhanced model achieves 100% test accuracy, demonstrating the power of combined design principles in convolutional neural networks.

## Dataset

- 480 grayscale images of handwritten ZIP digits (0–9), each 16×16 pixels.
- Stratified 320/160 train/test split to preserve class balance.
- Labels were one-hot encoded; images reshaped with single-channel dimension for CNN compatibility.

## Models & Trials

| Model                        | Description                                        | Accuracy   |
|-----------------------------|----------------------------------------------------|------------|
| Net-1                       | Fully connected (logistic regression)              | 86.88%     |
| Net-5                       | 2 conv layers + FC + tanh activations              | 95.00%     |
| Trial 1: Net5 Original      | Retrained baseline                                 | 88.75%     |
| Trial 2: Net5 ReLU          | Replaced tanh with ReLU                            | 93.75%     |
| Trial 3: ReLU + More Channels | Increased filter count                            | 96.25%     |
| Trial 4: Net5 Deep          | Added 3rd convolutional layer                      | 97.50%     |
| Trial 5: Net5 Deep + BN     | Added batch normalization to deep model            | 95.62%     |
| Trial 6: Net5 Strong        | Added MaxPooling + Dropout + deeper layers         | **100.00%**|

## Key Techniques

- **ReLU Activation**: Accelerates convergence and improves gradient flow vs. tanh.
- **Increased Filter Count**: Enables richer feature extraction.
- **Network Depth**: Improves abstraction of hierarchical visual features.
- **Batch Normalization**: Stabilizes training (but less effective with small datasets).
- **Max Pooling**: Adds translational invariance.
- **Dropout**: Prevents overfitting by randomly deactivating neurons during training.

## Training Setup

- **Optimizer**: Adam  
- **Loss Function**: Cross-entropy  
- **Net-1 / Net-5**: 30 epochs, learning rate 0.004  
- **Trials**: 50 epochs, learning rate 0.002  
- **Batch Size**: 32  
- Training was conducted on Google Colab using GPU acceleration.

## Experiment Workflow

1. **Data Preparation**  
   - Load and preprocess `zip_numerals.bmp` into 480 digit samples.  
   - Normalize pixel values and apply one-hot encoding to labels.  
   - Split into training (320) and test (160) sets with stratified sampling.

2. **Model Definition**  
   - Implement baseline models (`Net1`, `Net5`) and define trial architectures.  
   - Use modular PyTorch classes to isolate design changes (activations, depth, pooling).

3. **Training & Evaluation**  
   - Train each model using cross-entropy loss and track test accuracy across epochs.  
   - Log and visualize training performance to compare architecture effectiveness.

4. **Filter Visualization & Interpretation**  
   - Extract and display learned filters from early layers to understand pattern sensitivity.  
   - Quantify vertical vs. horizontal gradients for interpretability.

## Key Takeaways

- Compound improvements outperform isolated tweaks—depth, regularization, and activation must be balanced.
- Dropout and max pooling had the largest impact on generalization in small datasets.
- Batch normalization may not always help on low-data regimes.

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Matplotlib, scikit-learn, PIL

---

This project highlights how careful design of CNN components can drastically improve performance in image classification tasks. It serves as a reproducible experiment in model refinement and a hands-on study of architectural trade-offs.
