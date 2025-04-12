import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Load class names for ModelNet40 (you can adjust based on your dataset)
modelnet40_classes = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]

def visualize_predictions(model, dataset, num_samples=5):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)

    fig = plt.figure(figsize=(15, 3 * num_samples))

    for i, idx in enumerate(indices):
        points, label = dataset[idx]
        with torch.no_grad():
            input_tensor = points.unsqueeze(0).to(device)
            pred = model(input_tensor)
            pred_label = torch.argmax(pred, dim=1).item()

        # Plot
        ax = fig.add_subplot(num_samples, 1, i + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=2)
        ax.set_title(f"True: {modelnet40_classes[label]} | Predicted: {modelnet40_classes[pred_label]}", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
