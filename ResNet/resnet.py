# %% 
# Import libraries and load the datasets
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

path = os.path.abspath(os.getcwd())
training_data = pd.read_csv(path + 'training.csv')
test_data = pd.read_csv(path + 'test.csv')
sample_submission = pd.read_csv(path + 'SampleSubmission.csv')
id_lookup_table = pd.read_csv(path + 'IdLookupTable.csv')

# %% 
# Data Preprocessing

# Handle missing values: Drop rows with missing keypoints
# training_data_clean = training_data.dropna()
training_data_clean = training_data

# Extract Image and Keypoints data
images = []
for idx, sample in training_data_clean.iterrows():
    image = np.fromstring(sample['Image'], sep=' ').astype(np.float32).reshape(96, 96)
    images.append(image)
images = np.array(images) / 255.0

# Extract Image data from test dataset
test_images = []
for idx, sample in test_data.iterrows():
    image = np.fromstring(sample['Image'], sep=' ').astype(np.float32).reshape(96, 96)
    test_images.append(image)
test_images = np.array(test_images) / 255.0 

keypoints = training_data_clean.drop(columns=['Image']).to_numpy().astype(np.float32)

# Shuffle and Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, keypoints, test_size=0.2, random_state=42)

# %% 
# Define the FacialKeypoints Dataset class

class FacialKeypointsDataset(torch.utils.data.Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        
        img = self.X[i]
        keypts = self.y[i]
        
        img = np.reshape(img, (1, 96, 96))
        img = torch.tensor(img, dtype=torch.float32)
        keypts = torch.tensor(keypts, dtype=torch.float32)

        return img, keypts
    
class FacialKeypointsTestDataset(torch.utils.data.Dataset):
    
    def __init__(self, X):
        self.X = [np.fromstring(x, sep=' ') for x in X]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        
        img = self.X[i]
        img = np.reshape(img, (1, 96, 96)).astype(np.float32) / 255.0 
        img = torch.tensor(img, dtype=torch.float32)

        return img

# Create datasets and dataloaders
dataset_train = FacialKeypointsDataset(X_train, y_train)
dataset_val = FacialKeypointsDataset(X_val, y_val)
dataset_test = FacialKeypointsTestDataset(test_data['Image'].to_numpy())

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=False) 
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False)

x_batch, y_batch = next(iter(dataloader_train))

# %% Define the neural network model

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        self.b1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(32), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(32, 32, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(32, 64, 2))
        self.b4 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b5 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b6 = nn.Sequential(*resnet_block(256, 512, 2))

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 30)

    def forward(self, x):
        x = self.b6(self.b5(self.b4(self.b3(self.b2(self.b1(x))))))
        x = self.fc(self.flatten(self.pool(x)))

        return x

# %%
# Set device and initialize the model
model = ResNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

class MSELoss_Ignore_Nan(nn.Module):

    def forward(self, pred, label):

        count = torch.isfinite(label).sum().item()
        label = torch.where(torch.isnan(label), pred, label)
        loss = torch.sum((pred-label)**2)

        return loss / count

# Loss Function & Optimizer
# criterion = nn.SmoothL1Loss()
# criterion = nn.MSELoss()
criterion = MSELoss_Ignore_Nan()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# %%
# Training Function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, keypoints in dataloader:
        images, keypoints = images.to(device), keypoints.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataset_train)

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataset_val)

    return epoch_loss

# Prediction function
def predict(model, dataloader, device):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            outputs = model(images)
            all_predictions.extend(outputs.cpu().numpy())

    return np.vstack(all_predictions)

# %%
# Early Stopping
best_val_loss = float('inf')
patience_counter = 0
patience = 5
# %% 
# Training loop and plot the training and validation 
num_epochs = 100

L_history_train = []
L_history_val = []

for epoch in range(num_epochs):
    train_loss = train(model, dataloader_train, criterion, optimizer, device)
    val_loss = validate(model, dataloader_val, criterion, device)
    
    # Early stopping
    if val_loss < best_val_loss:  # Check if the validation loss improved
        best_val_loss = val_loss
        patience_counter = 0  # reset the patience counter
        torch.save(model.state_dict(), 'best_model.pth')  # Save the model
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break

    L_history_train.append(train_loss)
    L_history_val.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    scheduler.step()

plt.plot(L_history_train, label='Training loss')
plt.plot(L_history_val, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
# Make Predictions
predictions = []
# predictions = predict(model, dataloader_test, device)
predictions = predict(model, dataloader_test, device)
predictions = np.clip(predictions, 0, 96)

# %%
def show_keypoints(image, keypoints):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.scatter(keypoints[0::2], keypoints[1::2], s=24, marker='.', c='r')
    plt.axis('off')
    plt.show()
    
#%%
with torch.no_grad():
    x_batch, y_batch = next(iter(dataloader_train))
    y_pred = model(x_batch)
i = np.random.randint(16)
x = x_batch[i].numpy().reshape(96, 96)
y = y_pred[i]
show_keypoints(x, y)
#%%
i = np.random.randint(1782) # Select a random index for the image with keypoints
img = test_images[i]
key_pts = predictions[i]

# The image should be scaled back to 0-255 if it was normalized
img = img * 255.0
img = img.astype(np.uint8)

# Display the image and keypoints
show_keypoints(img, key_pts)


# %% 
# # Create a submission file
submission_data = []
feature_names = [
    'left_eye_center_x', 'left_eye_center_y',
    'right_eye_center_x', 'right_eye_center_y',
    'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
    'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
    'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
    'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
    'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
    'nose_tip_x', 'nose_tip_y',
    'mouth_left_corner_x', 'mouth_left_corner_y',
    'mouth_right_corner_x', 'mouth_right_corner_y',
    'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
    'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'
]

# Iterate over the id_lookup_table and fill in the predicted keypoints
for i, row in id_lookup_table.iterrows():
    image_id = int(row['ImageId']) - 1  # Adjust for zero indexing
    feature_name = row['FeatureName']
    
    # Find the index of the feature in the list of keypoints
    feature_index = feature_names.index(feature_name)
    
    # Find the prediction corresponding to the current image_id and feature_name
    predicted_value = predictions[image_id, feature_index]
    
    # Append the row to the submission data
    submission_data.append({
        'RowId': int(row['RowId']),
        'ImageId': image_id + 1,  # Convert back to one indexing
        'FeatureName': feature_name,
        'Location': predicted_value
    })

# Convert the submission data to a DataFrame
submission_df = pd.DataFrame(submission_data)

# Save the DataFrame to a CSV file
output_path = path + 'submission.csv'
submission_df.to_csv(output_path, index=False, columns=['RowId', 'Location'])
# %%

## TODO:
# 1. Code Refactoring
# 2. Try to improve the model (e.g. data augmentation (using albumentations), 
# add more layers, change the optimizer, etc.)
# 3. Try to improve the loss function 
# (自定义loss function，比如当一组数据有缺失的keypoints时，
# 不要不计算loss或者排除这组数据而是应该忽略这个keypoint的loss，
# 然后取其他keypoints的loss的平均值作为这组数据的loss)   