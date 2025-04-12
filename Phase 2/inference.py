import cv2
import torch
import numpy as np
from torchvision import transforms
from model import VNDGCNN, preprocess_images  # reuse your existing functions

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VNDGCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
print(f"Model loaded on {device}")

# --- Preprocessing single digit region ---
def preprocess_region(region):
    region = cv2.resize(region, (28, 28))
    region = region.astype(np.float32) / 255.0
    tensor = torch.tensor(region).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
    tensor = preprocess_images(tensor)
    return tensor.to(device)

# --- Inference on full image with multiple digits ---
def detect_digits(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (each digit)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = thresh[y:y+h, x:x+w]
        if w > 5 and h > 5:  # filter small noise
            tensor = preprocess_region(roi)
            with torch.no_grad():
                output = model(tensor)
                pred = torch.argmax(output, dim=1).item()
                results.append((pred, (x, y, w, h)))
                cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.putText(image_rgb, str(pred), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    cv2.imshow("Detected Digits", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return results

# --- Run ---
if __name__ == "__main__":
    image_path = "your_image_with_digits.png"  # Update this
    results = detect_digits(image_path)
    print("Detected digits:", results)
