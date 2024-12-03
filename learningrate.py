import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Load Dataset
file_path = 'DATA1FUCKME.xlsx'
dataset = pd.read_excel(file_path)

# **Select Features and Target**
# Assuming 'CONDUCTANCE (FIELD)' is the target and others are features
features = dataset[['SPECIFIC CONDUCTANCE', 'PH', 'TDS', 'TURBIDITY']]
target = dataset['IRON TOTAL']

# **Convert to PyTorch Tensors and Create Data Loader**
# Convert to Tensors
X = torch.tensor(features.values, dtype=torch.float32)
y = torch.tensor(target.values, dtype=torch.float32).view(-1, 1)  # Ensure target is 2D for LinearRegression

# Create TensorDataset and DataLoader
tensor_dataset = TensorDataset(X, y)
data_loader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)

# **Simple Linear Regression Model**
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return out

# **Update Input Dimension to Match Number of Features**
model = LinearRegression(input_dim=features.shape[1], output_dim=1)

# **Criterion and Optimizer (with dynamic learning rate)**
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)  # Initial LR (will be updated)

# **Learning Rate Range Test**
def find_lr(model, data_loader, criterion, optimizer, start_lr=1e-7, end_lr=10, num_iter=100):
    lr_values = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
    losses = []
    best_loss = float('inf')
    optimal_lr = None
    
    for i, lr in enumerate(lr_values):
        # Update LR for this iteration
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        inputs, labels = next(iter(data_loader))
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Store the loss
        losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            optimal_lr = lr
        
        # Print progress
        if i % 10 == 0:
            print(f"Iter: {i+1}, LR: {lr:.2e}, Loss: {loss.item():.4f}")
    
    print(f"Optimal LR found: {optimal_lr:.2e}")
    
    # Plotting
    plt.plot(lr_values, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Range Test')
    plt.show()

# **Execute Learning Rate Range Test**
find_lr(model, data_loader, criterion, optimizer)
