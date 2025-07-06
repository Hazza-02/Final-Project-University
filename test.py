import torch
import torch.nn as nn
import numpy as np
from mcnn import MCNN  
from dataset import Dataset  
import matplotlib.pyplot as plt
from mcnnCBAM import MCNN_CBAM

"""

--- Testing logic ---
            
MSE - MSE Loss is ideal for predicting density maps, the loss function will 
be the predicted density map from the model against the ground truth density map

MAE - Mean Absolute Error is ideal for comparing the accuracy of our model by comparing
the predicted number of people against the ground truth count

"""

# Batch size of 1 to preserve image quality, if we used a higher batch size the images would have to all be the same size rather 
# than downsampled by a factor of 4. It can be infered the MCNN model used batch size of 1 as they didn;t resize and so must have used a batch of 1 otherwise there would be an image size mismatch 
batchSize = 1
test_img_dir = r"D:\Final Project (University)\ShanghaiTech\part_A\test_data\images"
test_density = r"D:\Final Project (University)\ShanghaiTech\part_A\test_data\density_maps"
model_path = "Model Checkpoints/117_CBAM.pth"  # Model change
 
# Dataset
test_dataset=Dataset(test_img_dir,test_density)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mcnn_model = MCNN_CBAM().to(device) # Model change
mcnn_model.load_state_dict(torch.load(model_path, weights_only=True))

# MSE Loss function
criterion = nn.MSELoss(reduction='sum').to(device)

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

# Test evaluation 
def evaluate_model():
    
    # Model evaluation mode
    mcnn_model.eval()
    total_loss = 0.0
    total_mae = 0.0
    
    batch_loss = []
    batch_mae = []

    # go through test dataset for evaluation during training 
    with torch.no_grad():
        for images, gts in test_loader:
            images, gts = images.to(device), gts.to(device)
            preds = mcnn_model(images)
            
            loss = criterion(preds, gts).item() # for figure
            mae = MAE(preds, gts) # for average
        
            total_loss += criterion(preds, gts).item() # for average
            total_mae += MAE(preds, gts) # for average
            
            batch_loss.append(loss)
            batch_mae.append(mae)
        
    # Work out metrics
    num_batches = len(test_loader)
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    print(f"Avg MSE Loss: {avg_loss:.4f}")
    print(f"Avg MAE: {avg_mae:.2f}")

    return batch_loss, batch_mae, avg_loss, avg_mae

if __name__ == "__main__":
    batch_loss, batch_mae, loss, mae = evaluate_model()
    
    # Plot test MAE over the batch
    plt.figure(figsize=(8, 6))
    plt.plot(batch_mae, linestyle='-', color='green', label="Test MAE")
    plt.xlabel("Sample")
    plt.ylabel("Test MAE")
    plt.title("Test MAE for each sample")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot testing loss over the batch
    plt.figure(figsize=(8, 6))
    plt.plot(batch_loss, linestyle='-', color='b', label='Test Loss')
    plt.xlabel('Sample')
    plt.ylabel('Test Loss')
    plt.title('Test Loss for each sample')
    plt.legend()
    plt.grid(True)
    plt.show()