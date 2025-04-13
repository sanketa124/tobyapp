# 🧠 VN-DGCNN for 3D Point Cloud Classification on ShapeNet40

This project implements a **Vector Neuron-based Dynamic Graph Convolutional Neural Network (VN-DGCNN)** for classifying 3D point cloud objects using the **ShapeNet40** dataset.

---

## 📚 Overview

3D point clouds are unordered sets of points in 3D space. Unlike images, point clouds lack a regular structure, making it challenging for standard CNNs to process them directly. This project addresses this challenge using dynamic graph convolution and vector neuron layers to learn spatially meaningful features from raw point clouds.

---

## 🏗️ Model Architecture

### 🔁 Key Concepts

- **k-Nearest Neighbor (kNN) Graph:** Built dynamically during each forward pass to capture local neighborhood features.
- **Edge Features:** For each point, the model computes the difference between its neighbors and itself and concatenates it with the original point.
- **Vector Neuron Layers:** Orientation-aware convolution layers to extract rich geometric features.

### 🧮 Layer-by-Layer Breakdown

1. **Input:** Point cloud of shape `(B, N, 3)`
2. **Graph Feature Extraction:**
   - Builds edges via kNN.
   - Constructs features of shape `(B, 6, N, k)` using relative position.
3. **Vector Neuron Layers:**
   - Conv2D + BatchNorm over edge features
   - Output dims: `64 → 128 → 256`
4. **Aggregation:**
   - Max over neighbors `k`
   - Max over points `N` for global pooling
5. **Classification Head:**
   - Fully connected layers with dropout
   - Output logits for 40 classes

---

## 📊 Dataset: ShapeNet40

ShapeNet40 is a 3D object dataset containing **12,311 CAD models** from **40 categories**.

### Format

- Provided as `.h5` files
- Shape of each sample:  
  - `data`: `(B, 1024, 3)`  
  - `label`: `(B,)`

### Files Needed

- `shapenet40_train.h5`
- `shapenet40_test.h5`

---

## 🚀 Training Pipeline

### ✅ Dataset Loader

Loads 3D point cloud data from HDF5 format.

```python
train_dataset = ShapeNet40Dataset("shapenet40_train.h5", num_points=1024)
test_dataset = ShapeNet40Dataset("shapenet40_test.h5", num_points=1024)
