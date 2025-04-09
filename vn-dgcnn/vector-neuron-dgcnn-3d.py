import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import h5py
import os

# --- Load ModelNet40 Dataset ---
class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, num_points=1024):
        with h5py.File(h5_file, "r") as f:
            self.data = f["data"][:num_points]
            self.labels = f["label"][:num_points].flatten()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.long)

# --- kNN for 3D Graph Construction ---
def knn(x, k):
    """ Compute k-nearest neighbors (kNN) for 3D point clouds """
    batch_size, num_points, _ = x.size()
    inner = -2 * torch.matmul(x, x.transpose(2, 1))  # Pairwise distance
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # Get kNN indices
    return idx

# --- Extract Graph Features for 3D ---
def get_graph_feature(x, k=20):
    """ Compute edge features for 3D point clouds """
    batch_size, num_points, num_dims = x.size()
    idx = knn(x, k)  # Get kNN indices

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    idx = idx + idx_base.to(x.device)
    idx = idx.view(-1)

    x = x.view(batch_size * num_points, -1)[idx, :]
    x = x.view(batch_size, num_points, k, num_dims)
    return x.permute(0, 3, 1, 2)  # Reshape to (batch, channels, points, k)

# --- Vector Neuron Layer ---
class VectorNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(VectorNeuronLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bn = nn.BatchNorm1d(out_features)  # Normalize features

    def forward(self, x):
        x = torch.matmul(self.weights, x)
        x = self.bn(x)
        return F.relu(x)

# --- VN-DGCNN for 3D Point Cloud Classification ---
class VNDGCNN3D(nn.Module):
    def __init__(self, num_classes=40, k=20):
        super(VNDGCNN3D, self).__init__()
        self.k = k

        self.layer1 = VectorNeuronLayer(3, 64)  # Input: (x, y, z)
        self.layer2 = VectorNeuronLayer(64, 128)
        self.layer3 = VectorNeuronLayer(128, 256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)  # Graph Features
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))

        x = x.max(dim=-1, keepdim=False)[0]  # Global Max Pooling

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Load ModelNet40 Data ---
train_dataset = ModelNet40Dataset("modelnet40_train.h5", num_points=1024)
test_dataset = ModelNet40Dataset("modelnet40_test.h5", num_points=1024)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Training Function ---
def train(model, train_loader, num_epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        for points, labels in train_loader:
            points = points.to(torch.float32)
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# --- Testing Function ---
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for points, labels in test_loader:
            points = points.to(torch.float32)
            outputs = model(points)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# --- Run Training & Testing ---
model = VNDGCNN3D(num_classes=40)
train(model, train_loader, num_epochs=10)
test(model, test_loader)
