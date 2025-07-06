import os
import torch
import torch.nn as nn
import numpy as np
from mcnn import MCNN
from mcnnCBAM import MCNN_CBAM
from dataset import Dataset
import matplotlib.pyplot as plt

# Image and model paths
train_img_dir = r"D:\Final Project (University)\ShanghaiTech\part_a\train_data\images"
train_density = r"D:\Final Project (University)\ShanghaiTech\part_a\train_data\density_maps"
test_img_dir = r"D:\Final Project (University)\ShanghaiTech\part_a\test_data\images"
test_density = r"D:\Final Project (University)\ShanghaiTech\part_a\test_data\density_maps"
os.makedirs("Model Checkpoints", exist_ok=True)
model_path = "Model Checkpoints/best_model_CBAM.pth" # The weights for the model to be generated

# Hyperparemters and model
# Batch size of 1 to preserve image quality, if we used a higher batch size the images would have to all be the same size rather 
# than downsampled by a factor of 4. It can be infered the MCNN model used batch size of 1 as they didn;t resize and so must have used a batch of 1 otherwise there would be an image size mismatch
batchSize = 1
epochs = 400
learning_rate = 1e-6

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mcnn_model = MCNN_CBAM().to(device) # Model change
criterion = nn.MSELoss(reduction='sum').to(device)
optimizer = torch.optim.SGD(mcnn_model.parameters(), lr=learning_rate, momentum=0.90)

# Dataset
train_dataset = Dataset(train_img_dir, train_density)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
test_dataset = Dataset(test_img_dir, test_density)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

# Evaluation metrics
def head_count(density_map):
    return density_map.sum().item()

# Function to calculate MAE between predicted and ground truth density maps
def MAE(predictions, ground_truths):
    total_error = 0
    for i in range(len(predictions)):
        pred_heads = head_count(predictions[i])
        gt_heads = head_count(ground_truths[i])
        total_error += np.abs(pred_heads - gt_heads)
    return total_error / len(predictions)

# Train evaluation
def evaluate_model():
    
    # Model evaluation mode
    mcnn_model.eval()
    total_loss = 0.0
    total_mae = 0.0

    # go through test dataset for evaluation during training 
    with torch.no_grad():
        for images, gts in test_loader:
            images, gts = images.to(device), gts.to(device)
            preds = mcnn_model(images)
            total_loss += criterion(preds, gts).item()
            total_mae += MAE(preds, gts)
        
    # Work out metrics
    num_batches = len(test_loader)
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches

    print(f"Avg MSE Loss: {avg_loss:.4f}")
    print(f"Avg MAE: {avg_mae:.2f}")

    return avg_loss, avg_mae

# Train loop
def train_model():
    
    best_mae = float('inf')
    
    # Metrics for figures
    all_train_mae = []
    all_train_loss = []
    all_test_loss = [] 
    
    for epoch in range(epochs):
        
        print(f"Training for epoch [{epoch + 1}/{epochs}]\n")
        # Model train mode
        mcnn_model.train() 
        total_loss = 0.0
        total_mae = 0.0

        for images, gts in train_loader:
            
            images, gts = images.to(device), gts.to(device)
            optimizer.zero_grad()
            preds = mcnn_model(images)
            loss = criterion(preds, gts)
            loss.backward() 
            optimizer.step()
            total_loss += loss.item()
            total_mae += MAE(preds, gts)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_mae = total_mae / len(train_loader)

        # print out evaluation metrics
        test_loss, test_mae = evaluate_model()
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.2f}")
        print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.2f}\n")
        
        all_train_loss.append(avg_train_loss)
        all_train_mae.append(avg_train_mae)
        all_test_loss.append(test_loss)
   
        # Save the best model weights 
        if test_mae < best_mae:
            best_mae = test_mae
            torch.save(mcnn_model.state_dict(), model_path)
            print(f"Best model saved (Test MAE: {best_mae:.2f})\n")
        
        # Do at the end of training  
        if epoch == (epochs - 1):
            # Plot train MAE over epcohs
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, epochs + 1), all_train_mae, linestyle='-', color='green', label="Train MAE")
            plt.xlabel("Epoch")
            plt.ylabel("Train MAE")
            plt.title("Train MAE Over Epochs")
            plt.legend()
            plt.grid(True)
            plt.show()
            
            # Plot training loss and testing loss
            plt.figure(figsize=(8, 6))
            plt.plot(all_train_loss, linestyle='-', color='r', label='Train Loss')
            plt.plot(all_test_loss, linestyle='-', color='b', label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Train vs Test Loss Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.show()

train_model()

