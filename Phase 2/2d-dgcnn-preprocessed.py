import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

# --- Define kNN Graph Helper Function ---
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # [B, N, N]
    xx = torch.sum(x ** 2, dim=1, keepdim=True)      # [B, 1, N]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

# --- Construct Edge Features ---
def get_graph_feature(x, k=20):
    batch_size, num_dims, num_points = x.size()
    idx = knn(x, k)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature  # shape: [B, 2*D, N, k]

# --- Vector Neuron Layer ---
class VectorNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(VectorNeuronLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        B, C, N, K = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        x = torch.matmul(x, self.weights.t())  # [B*N*K, out_features]
        x = x.view(B, N, K, -1).permute(0, 3, 1, 2)
        return x

# --- VN-DGCNN Backbone ---
class VNDGCNN(nn.Module):
    def __init__(self, num_classes=10, k=20):
        super(VNDGCNN, self).__init__()
        self.k = k
        self.layer1 = VectorNeuronLayer(6, 64)
        self.layer2 = VectorNeuronLayer(64, 128)
        self.layer3 = VectorNeuronLayer(128, 256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.max(dim=-1)[0].max(dim=-1)[0]  # Global max pooling
        x = self.fc(x)
        return x

# --- Load MNIST Dataset with Transforms ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(30),
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Preprocess images to include pixel coords ---
def preprocess_images(images):
    B, _, H, W = images.shape
    images = images.view(B, 1, -1)  # [B, 1, 784]

    coords = torch.stack(torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij'), dim=0)
    coords = coords.reshape(2, -1).unsqueeze(0).repeat(B, 1, 1).to(images.device)  # [B, 2, 784]

    out = torch.cat([images, coords], dim=1)  # [B, 3, 784]
    return out

# --- Training ---
def train(model, train_loader, num_epochs=5, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images = preprocess_images(images).to(torch.float32)
            labels = labels.to(images.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# --- Testing ---
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = preprocess_images(images).to(torch.float32)
            labels = labels.to(images.device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# --- Run Training and Testing ---
if __name__ == "__main__":
    model = VNDGCNN(num_classes=10)
    train(model, train_loader, num_epochs=5)
    test(model, test_loader)
    torch.save(model.state_dict(), "model.pth")  # Save model for custom testing
