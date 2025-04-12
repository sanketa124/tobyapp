import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import h5py
import os

# --- CUDA Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load ModelNet40 Dataset ---
class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, num_points=1024):
        with h5py.File(h5_file, "r") as f:
            self.data = f["data"][:]
            self.labels = f["label"][:].flatten()
        self.data = self.data[:, :num_points, :]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.long)

# --- kNN for 3D Graph Construction ---
def knn(x, k):
    batch_size, num_points, _ = x.size()
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

# --- Extract Graph Feature ---
def get_graph_feature(x, k=20):
    batch_size, num_points, num_dims = x.size()
    idx = knn(x, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.view(batch_size * num_points, -1)
    neighbors = x[idx, :].view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    edge_feature = torch.cat((neighbors - x, x), dim=3).permute(0, 3, 1, 2)
    return edge_feature

# --- Vector Neuron Layer ---
class VectorNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(VectorNeuronLayer, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

# --- VN-DGCNN Model ---
class VNDGCNN3D(nn.Module):
    def __init__(self, num_classes=40, k=20):
        super(VNDGCNN3D, self).__init__()
        self.k = k

        self.layer1 = VectorNeuronLayer(6, 64)
        self.layer2 = VectorNeuronLayer(64, 128)
        self.layer3 = VectorNeuronLayer(128, 256)

        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)  # (B, 6, N, k)
        x = self.layer1(x)                  # (B, 64, N, k)
        x = x.max(dim=-1)[0]                # (B, 64, N)

        x = self.layer2(x.unsqueeze(-1))    # (B, 128, N, 1)
        x = x.max(dim=-1)[0]                # (B, 128, N)

        x = self.layer3(x.unsqueeze(-1))    # (B, 256, N, 1)
        x = x.max(dim=-1)[0]                # (B, 256, N)

        x = x.max(dim=-1)[0]                # Global Max Pooling → (B, 256)

        x = F.relu(self.fc1(x))             # (B, 128)
        x = self.dropout(x)
        x = self.fc2(x)                     # (B, num_classes)
        return x

# --- Training Function ---
def train(model, loader, epochs=10, lr=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss / len(loader):.4f}")

# --- Testing Function ---
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# --- Main Entry Point (IMPORTANT on Windows) ---
if __name__ == "__main__":
    # Load Datasets
    train_dataset = ModelNet40Dataset("modelnet40_train.h5", num_points=1024)
    test_dataset = ModelNet40Dataset("modelnet40_test.h5", num_points=1024)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Train & Test Model
    model = VNDGCNN3D(num_classes=40)
    train(model, train_loader, epochs=10)
    test(model, test_loader)

    # Save Model
    torch.save(model.state_dict(), "model.pth")
    print("✅ Model saved to model.pth")