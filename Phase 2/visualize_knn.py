import torch
from torchvision import datasets, transforms
from model_vndgcnn import VNDGCNN, preprocess_images, knn  # import your model & helpers
import matplotlib.pyplot as plt

# Load MNIST for test digit
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(30)
])
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VNDGCNN(num_classes=10)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Visualization Function
def visualize_knn_graph(image_tensor, k=20):
    with torch.no_grad():
        image_tensor = preprocess_images(image_tensor.unsqueeze(0).to(device))  # [1, 3, 784]
        x = image_tensor
        B, C, N = x.shape
        x_coords = x[0, 1].cpu().numpy()
        y_coords = x[0, 2].cpu().numpy()
        idx = knn(x, k=k)[0]

        plt.figure(figsize=(6, 6))
        for i in range(0, N, 30):
            xi, yi = x_coords[i], y_coords[i]
            plt.scatter(xi, yi, color='red', s=10)
            for j in idx[i]:
                xj, yj = x_coords[j], y_coords[j]
                plt.plot([xi, xj], [yi, yj], 'gray', linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.title(f"k-NN Graph (k={k})")
        plt.show()

# Run on a sample digit
sample_img, label = test_dataset[0]
print(f"True Label: {label}")
visualize_knn_graph(sample_img)
