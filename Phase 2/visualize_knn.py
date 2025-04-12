import torch
from torchvision import datasets, transforms
from model_vndgcnn import VNDGCNN, preprocess_images, knn
import matplotlib.pyplot as plt

# Load MNIST dataset
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

# --- Visualization ---
def visualize_knn_graph(image_tensor, label, k=20):
    with torch.no_grad():
        preprocessed = preprocess_images(image_tensor.unsqueeze(0).to(device))  # [1, 3, 784]
        x = preprocessed
        B, C, N = x.shape
        x_coords = x[0, 1].cpu().numpy()
        y_coords = x[0, 2].cpu().numpy()
        idx = knn(x, k=k)[0]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot original image
        axes[0].imshow(image_tensor.squeeze().cpu(), cmap='gray')
        axes[0].set_title(f"Original Image (Label: {label})")
        axes[0].axis("off")

        # Plot k-NN graph
        for i in range(0, N, 30):  # plot fewer points to reduce clutter
            xi, yi = x_coords[i], y_coords[i]
            axes[1].scatter(xi, yi, color='red', s=10)
            for j in idx[i]:
                xj, yj = x_coords[j], y_coords[j]
                axes[1].plot([xi, xj], [yi, yj], 'gray', linewidth=0.5)
        axes[1].invert_yaxis()
        axes[1].set_title(f"k-NN Graph (k={k})")

        plt.tight_layout()
        plt.show()

# Pick a sample digit from test set
sample_img, label = test_dataset[0]
print(f"True Label: {label}")
visualize_knn_graph(sample_img, label)
