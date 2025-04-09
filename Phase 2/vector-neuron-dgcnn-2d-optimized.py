import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

# --- Optimized kNN Graph Function ---
def knn(x, k):
    """ Compute k-nearest neighbors (kNN) graph dynamically """
    batch_size, num_dims, num_points = x.size()
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # Compute pairwise distance
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # kNN indices
    return idx

# --- Adaptive Graph Feature Extraction ---
def get_graph_feature(x, k=20, dynamic=True):
    """ Generate adaptive edge features using EdgeConv """
    batch_size, num_dims, num_points = x.size()
    idx = knn(x, k)  # Get kNN indices

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    idx = idx + idx_base.to(x.device)
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3)  # Edge feature: (neighbor - center, center)
    return feature.permute(0, 3, 1, 2)  # Reshape to (batch, channels, points, k)

# --- Vector Neuron Layer with Feature Normalization ---
class VectorNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(VectorNeuronLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bn = nn.BatchNorm1d(out_features)  # BatchNorm for stability

    def forward(self, x):
        x = torch.matmul(self.weights, x)
        x = self.bn(x)  # Normalize features
        return F.relu(x)  # Non-linearity

# --- VN-DGCNN with Multi-Scale Aggregation ---
class VNDGCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(VNDGCNN, self).__init__()
        self.k1, self.k2, self.k3 = 10, 20, 30  # Multi-scale kNN

        self.layer1 = VectorNeuronLayer(6, 64)
        self.layer2 = VectorNeuronLayer(64, 128)
        self.layer3 = VectorNeuronLayer(128, 256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Multi-scale graph feature extraction
        x1 = get_graph_feature(x, k=self.k1)
        x2 = get_graph_feature(x, k=self.k2)
        x3 = get_graph_feature(x, k=self.k3)

        # Feature aggregation
        x = torch.cat((self.layer1(x1), self.layer2(x2), self.layer3(x3)), dim=1)
        x = x.max(dim=-1, keepdim=False)[0]  # Global Max Pooling

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Load Rotated MNIST Dataset ---
transform = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(30)])  # Augmentation
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Training Function ---
def train(model, train_loader, num_epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.view(images.size(0), 1, -1).to(torch.float32)
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
train(model, train_loader, num_epochs=10)
test(model, test_loader)
