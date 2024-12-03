# Learning Rate Range Test with Linear Regression in PyTorch
 
## Overview
This code is a general-purpose implementation to perform a **Learning Rate Range Test** for a simple linear regression model using PyTorch. 
The test helps identify the optimal learning rate by observing the loss across a range of learning rates. This approach can be applied to any regression or similar task to improve model training efficiency.

---

## How the Code Works

### 1. Dataset Loading
- The data is loaded from an external file (e.g., Excel, CSV) into a pandas DataFrame.
- The dataset should include:
  - **Features**: Independent variables used for predictions.
  - **Target**: Dependent variable to be predicted.

### 2. Tensor Conversion and DataLoader Creation
- The features and target are converted to PyTorch tensors for compatibility with PyTorch models.
- A `TensorDataset` is created to pair the features and target, followed by a `DataLoader` to enable mini-batch processing and shuffling of data during training.

### 3. Model Definition
- A simple linear regression model is defined using PyTorch's `nn.Module`.
  - The model consists of a single linear layer with an input dimension matching the number of features and an output dimension of 1 (for regression).

### 4. Loss Function and Optimizer
- The **Mean Squared Error (MSE)** loss function is used to measure the difference between predictions and actual values.
- The **Stochastic Gradient Descent (SGD)** optimizer is used for model weight updates.

### 5. Learning Rate Range Test
- A custom function, `find_lr`, performs the following steps:
  - Iterates through a logarithmic range of learning rates (from `1e-7` to `10`).
  - For each learning rate:
    - Updates the optimizer's learning rate dynamically.
    - Performs a forward pass to calculate the loss for the current learning rate.
    - Tracks the loss and identifies the learning rate with the lowest loss.
  - Plots the loss values against the learning rates to visualize the relationship.

### 6. Visualization
- The results of the learning rate test are visualized on a logarithmic scale.
- The graph helps identify the optimal learning rate, which is also printed to the console.

---

## Purpose of the Code
- To determine the optimal learning rate for training a regression model efficiently.
- To provide a visualization of the impact of different learning rates on model performance.
- To create a reusable framework for learning rate range testing in PyTorch.

---

## General Use Cases
- Optimizing learning rates for regression or similar tasks.
- Testing dynamic learning rates for various PyTorch models.
- Understanding the relationship between learning rate and loss for better model training.

---

## Requirements
- Input dataset with clearly defined features and target variables.
- Libraries:
  - `torch`, `torch.nn`, `torch.utils.data` (for PyTorch model and data handling)
  - `pandas` (for data manipulation)
  - `matplotlib` (for visualization)
  - `numpy` (for numerical computations)

---

## Notes
- The code can be adapted for more complex models by modifying the architecture of the `LinearRegression` class.
- Adjust the `start_lr`, `end_lr`, and `num_iter` parameters in the `find_lr` function to suit different datasets or tasks.

---

This modular code structure allows users to easily replace the dataset, modify the features and target, or experiment with different learning rate ranges.