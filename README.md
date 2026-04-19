# 🧠 Self-Pruning Neural Network (CIFAR-10)

## 📌 Overview

This project implements a **self-pruning neural network** that dynamically removes unnecessary weights during training.

Unlike traditional pruning (post-training), this model learns which connections to remove **while training**, using learnable gate parameters.

---

## 🚀 Key Idea

Each weight `w` is paired with a learnable gate `g`:

```
g = sigmoid(gate_score / temperature)
pruned_weight = w * g
```

* `g ≈ 0` → connection is pruned
* `g ≈ 1` → connection is retained

---

## ⚙️ Loss Function

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

* **CrossEntropyLoss** → classification objective
* **SparsityLoss** → mean of gate values (L1-style penalty)

### Why this works

L1 regularization encourages many gate values to become **exactly zero**, resulting in a sparse network.

---

## 🏗️ Model Architecture

* Input: CIFAR-10 (32×32×3 images)
* Network:

  * PrunableLinear (3072 → 768)
  * ReLU + Dropout
  * PrunableLinear (768 → 256)
  * ReLU + Dropout
  * PrunableLinear (256 → 10)

---

## ⚡ Key Features

* ✅ Custom `PrunableLinear` layer
* ✅ Learnable gating mechanism
* ✅ Temperature annealing (sharpens pruning over time)
* ✅ Separate learning rates for gates and weights
* ✅ Sparsity tracking during training
* ✅ Gate distribution visualization

---

## 📦 Installation

```bash
pip install torch torchvision matplotlib numpy certifi
```

---

## ▶️ How to Run

### Quick Run (recommended for testing)

```bash
python self_pruning_neural_network_v2.py --quick
```

### Full Training

```bash
python self_pruning_neural_network_v2.py
```

---

## 📊 Results

| Lambda (λ) | Test Accuracy | Sparsity (%) |
| ---------- | ------------- | ------------ |
| 0.0        | 0.4900        | 5.39%        |
| 0.5        | 0.4937        | 14.09%       |
| 2.0        | 0.4971        | 35.42%       |

---

## 📈 Observations

* Increasing λ increases sparsity significantly
* Model successfully prunes up to **35% of weights**
* Accuracy is maintained (and slightly improved)

### 💡 Key Insight

> Higher sparsity improved generalization, indicating that pruning acted as an implicit regularizer.

---

## 📊 Outputs

After training, the following files are generated:

* `outputs/results.csv` → summary table
* `outputs/summary.json` → detailed results
* `outputs/gate_distribution_best.png` → gate value histogram

---

## 📉 Gate Distribution

A successful pruning result shows:

* Large spike near **0** → pruned weights
* Cluster away from 0 → important weights

---

## 🧪 Hyperparameters

* Epochs: 12
* Batch Size: 128
* Learning Rate: 1e-3
* Gate Learning Rate: 5e-3
* Lambda Values: `[0.0, 0.5, 2.0]`

---

## 🎯 Evaluation Metrics

* Test Accuracy
* Sparsity Level (%)
* Gate Value Distribution

---

## 🚀 Future Improvements

* Replace MLP with CNN for higher accuracy
* Structured pruning (neuron/channel pruning)
* Hardware-aware pruning
* Deployment optimization (ONNX / TensorRT)

---
