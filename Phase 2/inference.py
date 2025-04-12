import os
import cv2
import torch
import numpy as np
from model import VNDGCNN, preprocess_images  # Reuse your model + helper

# Load model and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VNDGCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
print(f"Model loaded on {device}")

# --- Preprocessing for a digit region ---
def preprocess_region(region):
    # Resize with padding to retain aspect ratio
    h, w = region.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=np.uint8)
    padded[(size - h) // 2:(size - h) // 2 + h, (size - w) // 2:(size - w) // 2 + w] = region

    # Resize to 28x28
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
    normed = resized.astype(np.float32) / 255.0

    tensor = torch.tensor(normed).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
    tensor = preprocess_images(tensor)
    return tensor.to(device)

# --- Detect digits in a single image ---
def detect_digits(image_path, visualize=True):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Couldn't read {image_path}")
        return []

    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = thresh[y:y+h, x:x+w]

        if w > 10 and h > 10 and w*h > 100:
            tensor = preprocess_region(roi)
            with torch.no_grad():
                output = model(tensor)
                pred = torch.argmax(output, dim=1).item()
                results.append((pred, (x, y, w, h)))
                if visualize:
                    cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    cv2.putText(image_rgb, str(pred), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    if visualize:
        cv2.imshow(f"Detected - {os.path.basename(image_path)}", image_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results

# --- Loop over all images in a folder ---
def test_on_folder(folder_path):
    supported_exts = (".jpg", ".jpeg", ".png", ".bmp")
    all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_exts)]

    for img_file in all_images:
        img_path = os.path.join(folder_path, img_file)
        print(f"\n🔍 Processing {img_file}")
        results = detect_digits(img_path)
        if results:
            print("Digits detected:", [r[0] for r in results])
        else:
            print("No digits detected.")

# --- Run ---
if __name__ == "__main__":
    test_on_folder("image_data")  # Your folder path
