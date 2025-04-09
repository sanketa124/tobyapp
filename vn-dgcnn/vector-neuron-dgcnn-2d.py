import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

# --- Define kNN Graph Helper Function ---
def knn(x, k):
    """ Compute k-nearest neighbors (kNN) graph """
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # Get k nearest neighbors
    return idx

# --- Construct Edge Features ---
def get_graph_feature(x, k=20):
    """ Generate edge features from kNN graph """
    batch_size, num_dims, num_points = x.size()
    idx = knn(x, k)  # Get kNN indices
    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    idx = idx + idx_base.to(x.device)  # Adjust indices for batch processing
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature

# --- Vector Neuron Layer ---
class VectorNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(VectorNeuronLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return torch.matmul(self.weights, x)

# --- VN-DGCNN Backbone ---
class VNDGCNN(nn.Module):
    def __init__(self, num_classes=10, k=20):
        super(VNDGCNN, self).__init__()
        self.k = k
        self.layer1 = VectorNeuronLayer(6, 64)  # First VN layer
        self.layer2 = VectorNeuronLayer(64, 128)
        self.layer3 = VectorNeuronLayer(128, 256)
        self.fc = nn.Linear(256, num_classes)  # Fully connected layer for classification

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)  # Extract graph features
        x = self.layer1(x)
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.max(dim=-1, keepdim=False)[0]  # Global max pooling
        x = self.fc(x.view(x.size(0), -1))
        return x

# --- Load Rotated MNIST Dataset ---
transform = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(30)])  # Augmentation
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Convert images into 2D vectors
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Training Function ---
def train(model, train_loader, num_epochs=5, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.view(images.size(0), 1, -1).to(torch.float32)  # Convert to 2D vector neurons
            optimizer.zero_grad()
            outputs = model(images)
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
        for images, labels in test_loader:
            images = images.view(images.size(0), 1, -1).to(torch.float32)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# --- Run Training & Testing ---
model = VNDGCNN(num_classes=10)
train(model, train_loader, num_epochs=5)
test(model, test_loader)
