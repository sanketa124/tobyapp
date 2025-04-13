# visualize_predictions.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import argparse
import json

# ----- Device Setup -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- ShapeNet40 Class Names -----
shapenet40_classes = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]

# ----- Dataset Loader -----
class ShapeNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, num_points=1024):
        with h5py.File(h5_file, "r") as f:
            self.data = f["data"][:]
            self.labels = f["label"][:].flatten()
        self.data = self.data[:, :num_points, :]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.labels[idx]

# ----- kNN Graph Construction -----
def knn(x, k):
    batch_size, num_points, _ = x.size()
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

# ----- Graph Feature Extractor -----
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

# ----- Vector Neuron Layer -----
class VectorNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(VectorNeuronLayer, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

# ----- VN-DGCNN Model -----
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
        x = get_graph_feature(x, k=self.k)
        x = self.layer1(x).max(dim=-1)[0]

        x = self.layer2(x.unsqueeze(-1)).max(dim=-1)[0]
        x = self.layer3(x.unsqueeze(-1)).max(dim=-1)[0]

        x = x.max(dim=-1)[0]  # Global Max Pooling
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ----- Visualization -----
def visualize(model, dataset, num_classes=10):
    model.eval()

    class_to_indices = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_to_indices.setdefault(label, []).append(idx)

    chosen_classes = random.sample(list(class_to_indices.keys()), num_classes)
    fig = plt.figure(figsize=(15, num_classes * 3))

    for i, class_id in enumerate(chosen_classes):
        idx = random.choice(class_to_indices[class_id])
        points, true_label = dataset[idx]

        input_tensor = points.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred_label = torch.argmax(output, dim=1).item()

        ax = fig.add_subplot(num_classes, 1, i + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=2)
        ax.set_title(f"True: {shapenet40_classes[true_label]} | Predicted: {shapenet40_classes[pred_label]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

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

# ----- Helper to get prediction JSON -----
def get_prediction_json(model, dataset, index):
    model.eval()
    points, true_label = dataset[index]
    input_tensor = points.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_label = torch.argmax(output, dim=1).item()

    result = {
        "index": index,
        "true_label": shapenet40_classes[true_label],
        "predicted_label": shapenet40_classes[pred_label]
    }
    return json.dumps(result, indent=4)

# ----- Helper to save point cloud as JPG -----
def save_pointcloud_visualization(dataset, index, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    points, label = dataset[index]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=2)
    ax.set_title(f"Point Cloud for Index {index} ({shapenet40_classes[label]})")
    ax.axis("off")
    plt.tight_layout()
    filename = f"pointcloud_{shapenet40_classes[label]}_{index}.jpg"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# ----- Load Model and Dataset -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VN-DGCNN ShapeNet40 Inference")
    parser.add_argument("--index", type=int, help="Sample index for prediction/visualization")
    parser.add_argument("--visualize", action="store_true", help="Save point cloud visualization as JPG")
    args = parser.parse_args()

    model_path = "model.pth"
    test_file = "shapenet40_test.h5"  

    model = VNDGCNN3D(num_classes=40)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    test_dataset = ShapeNet40Dataset(test_file, num_points=1024)

    if args.index is not None:
        print(get_prediction_json(model, test_dataset, args.index))
        if args.visualize:
            save_pointcloud_visualization(test_dataset, args.index)
            print("Saved point cloud visualization in 'output/' folder")
    else:
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        test(model, test_loader)
        visualize(model, test_dataset, num_classes=10)
