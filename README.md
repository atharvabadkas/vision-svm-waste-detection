# 🧠 Food Waste Classification via OpenCLIP Feature Embeddings and SVM Tuning

---

## 📖 Overview

This project implements a resource-efficient food waste classification pipeline using **OpenCLIP ViT-L/14** for embedding generation and a **linear Support Vector Machine (SVM)** for classification. It is optimized for real-world kitchen environments where identifying waste types accurately and quickly is crucial for **sustainability analysis** and **smart waste management**.

---

## 🚀 Features

- 🧠 OpenCLIP ViT-L/14 for zero-shot, domain-agnostic feature embeddings  
- 🎯 Linear SVM classifier trained on top of embeddings for speed and accuracy  
- 🔁 Test-Time Augmentations (TTA) to improve inference robustness  
- 💾 Pre-trained `waste_label_classifier.pkl` and `label_encoder.pkl` included  
- 📊 Validated with real-world waste datasets and Django test environments  
- 📈 Results tracked in CSV format across multiple experiments  

---

## 🛠 Tech Stack

| Component        | Technology                   |
|------------------|------------------------------|
| Embedding Model  | OpenCLIP (ViT-L/14)  
| Classifier       | Scikit-learn Linear SVM  
| Training         | PyTorch, OpenCLIP, Sklearn  
| Inference        | OpenCLIP, NumPy, Pickle  
| Augmentation     | torchvision.transforms  
| Visualization    | pandas, matplotlib (optional)  

---

## 📁 File Breakdown

| File | Description |
|------|-------------|
| `varandah_waste_training.py` | Generates OpenCLIP embeddings from labeled data and trains a linear SVM |
| `varandah_waste_model.py` | Loads pre-trained model and predicts the waste class for new input images |
| `waste_label_classifier.pkl` | Serialized sklearn SVM classifier |
| `waste_label_encoder.pkl` | Label encoder to map string classes to numeric targets |
| `Vision Model Experiments - Clip Models.csv` | Results from various OpenCLIP variants |
| `Vision Model Experiments - Waste Data Feature Tuning (T1).csv` | Feature-tuned accuracy comparisons |
| `Vision Model Experiments - Waste Dataset - Django Testing (T2).csv` | Real-world Django pipeline performance  

---

## 🧠 Model Workflow

### 1. Training

python varandah_waste_training.py
  --data_dir ./data/waste_images
  --save_model waste_label_classifier.pkl
  --save_encoder waste_label_encoder.pkl

### 2. Inference

python varandah_waste_model.py
  --input_image ./test_samples/paneer.jpg
  --classifier waste_label_classifier.pkl
  --encoder waste_label_encoder.pkl

## Project Structure
.
├── varandah_waste_model.py
├── varandah_waste_training.py
├── waste_label_classifier.pkl
├── waste_label_encoder.pkl
├── data/
│   └── waste_images/
├── outputs/
│   └── predictions.csv
├── test_samples/
├── *.csv (result tracking)

## 🔁 Test-Time Augmentation (TTA)

To improve robustness and generalization, the inference pipeline applies multiple augmentations to each input image before generating predictions. The final output is averaged across all augmented views.

**Augmentations Used:**
- Horizontal Flip
- Resize and Center Crop
- Brightness & Contrast Adjustment
- Color Jitter

This ensures that predictions remain stable under different lighting, orientation, and noise conditions.

## 📈 Results Summary

Model performance was evaluated across multiple OpenCLIP variants, embedding configurations, and testing scenarios. Below are the summarized outcomes.

### 📊 `Vision Model Experiments - Clip Models.csv`

| Model Variant | Accuracy (%) |
|---------------|--------------|
| ViT-B/32      | 84.1         |
| ViT-L/14      | **92.5**     |
| RN50x64       | 86.8         |

> ✅ OpenCLIP ViT-L/14 provided the best performance across all variants.

### 📊 `Vision Model Experiments - Waste Data Feature Tuning (T1).csv`

| Experiment Type          | Accuracy (%) | Notes                      |
|--------------------------|--------------|----------------------------|
| Raw Embedding + SVM      | 87.2         | Baseline                   |
| Augmented + Normalized   | **92.5**     | Best performance achieved  |
| No Normalization Applied | 78.6         | Degraded performance       |

### 🧪 `Vision Model Experiments - Waste Dataset - Django Testing (T2).csv`

| Real-World Scenario      | Accuracy (%) |
|--------------------------|--------------|
| Pan with Oil Residue     | 94.2         |
| Plate with Rice Grains   | 91.3         |
| Mixed Waste on Tray      | 89.6         |

> These results demonstrate robustness in real kitchen conditions integrated via a Django-based interface.
