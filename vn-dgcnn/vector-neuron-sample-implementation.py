import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define a simple Vector Neuron Layer
class VectorNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(VectorNeuronLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return torch.matmul(self.weights, x)

# Define a Vector Neuron Network for classification
class VectorNeuronClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(VectorNeuronClassifier, self).__init__()
        self.layer1 = VectorNeuronLayer(input_size, 128)  # 128-dimensional vector neurons
        self.layer2 = VectorNeuronLayer(128, 64)
        self.fc = nn.Linear(64, num_classes)  # Fully connected layer for classification
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.fc(x)
        return x

# Rotation function
def rotate_image(image, angle):
    """ Rotates an image by a given angle """
    transform = transforms.Compose([
        transforms.RandomRotation([angle, angle]),  # Rotate by a fixed angle
        transforms.ToTensor()
    ])
    return transform(image)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Apply rotation augmentation (label remains unchanged)
def augment_dataset(dataset, angles=[30, 60, 90]):
    augmented_images = []
    augmented_labels = []
    for img, label in dataset:
        img = transforms.ToPILImage()(img)  # Convert tensor to PIL Image
        augmented_images.append(transforms.ToTensor()(img))  # Original
        augmented_labels.append(label)
        for angle in angles:
            rotated_img = rotate_image(img, angle)
            augmented_images.append(rotated_img)
            augmented_labels.append(label)  # Label remains the same
    return torch.stack(augmented_images), torch.tensor(augmented_labels)

# Augment training dataset with rotations
train_images, train_labels = augment_dataset(train_dataset)
test_images, test_labels = augment_dataset(test_dataset)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(list(zip(train_images, train_labels)), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(list(zip(test_images, test_labels)), batch_size=batch_size, shuffle=False)

# Initialize network
input_size = 28 * 28  # MNIST image size (flattened)
num_classes = 10
model = VectorNeuronClassifier(input_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.view(images.size(0), -1)  # Flatten images
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate on test dataset
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.size(0), -1)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")