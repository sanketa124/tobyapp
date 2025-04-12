# 🧠 VN-DGCNN for MNIST Classification

This project implements a **Vector Neuron-based Dynamic Graph CNN (VN-DGCNN)** to classify MNIST digit images by treating them as **2D point clouds**. Each image is converted into a set of 2D coordinates with pixel intensity, enabling point-based learning on a traditionally grid-based dataset.

---

## 📚 Overview

Conventional CNNs operate on structured grids, but this project explores applying graph neural networks and vector neuron layers on pixel-based point cloud representations. The MNIST digits (28x28 grayscale images) are treated as 784 points with associated (x, y, intensity) features.

### 🔍 Why VN-DGCNN?

- Learns from unordered sets of pixels (point clouds)
- Captures local geometric structures using dynamic k-NN graphs
- Uses **vector neuron layers** for orientation-aware transformations

---

## 🏗️ Architecture

### 🔁 Key Components

- **k-NN Graph Construction:** Dynamically computed during each forward pass.
- **Edge Feature Construction:** Computes pairwise differences with neighbors.
- **Vector Neuron Layers:** Applied as deep feature extractors.
- **Global Max Pooling:** Aggregates across spatial dimensions.
- **Linear Classifier:** Predicts digit class (0-9).

### 📈 Pipeline

1. **Input:**
   - Grayscale image → Flattened to 784 points
   - Add 2D pixel coordinates → `[intensity, x, y]`
2. **Feature Graph Construction:**
   - k-NN neighbors per point
   - Compute edge features
3. **Network:**
   - 3 Vector Neuron Layers (64 → 128 → 256)
   - ReLU activation
   - Max pooling over neighborhood and points
4. **Classifier:**
   - Fully connected layer → 10 classes

---

## 🧠 Dataset: MNIST

- **Train set:** 60,000 handwritten digits
- **Test set:** 10,000 handwritten digits
- Format: `(1, 28, 28)` grayscale images
- Transformed into 784-point cloud with 3D features

---

## 📁 File Structure

