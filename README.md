# 🧠 Digit Recognizer (from Scratch using NumPy)

A neural network implementation from scratch (without using deep learning libraries like TensorFlow or PyTorch) to classify handwritten digits from the MNIST dataset.

This project demonstrates core machine learning principles such as forward propagation, backpropagation, ReLU activation, Softmax, one-hot encoding, and gradient descent — all written manually using NumPy.

---

## 📌 Features

- Neural Network with 1 hidden layer
- Implemented using **NumPy only**
- Supports **784 input features**, **10 hidden neurons**, and **10 output classes**
- Custom implementation of:
  - Forward Propagation
  - ReLU and Softmax Activation
  - One-Hot Encoding
  - Backpropagation with Gradient Descent
- Trained and tested on the MNIST digit dataset

---

## 🧾 Dataset

- 📁 [Kaggle Digit Recognizer Dataset](https://www.kaggle.com/c/digit-recognizer)
- You can also use any version of the MNIST dataset (CSV format recommended).

---

 ## 🔢 Neural Network Architecture

- **Input Layer**: 784 (28x28 pixels)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax (for multi-class classification)

---
## 🧮 Forward and Backward Propagation Equations

  ## 🔁 Forward Propagation Equations:
   
 Let:
- `X` be the input of shape (784, m)
- `W1`, `B1` → weights and bias for hidden layer
- `W2`, `B2` → weights and bias for output layer

1. **Hidden Layer (ReLU Activation)**
   Z1 = W1 ⋅ X + B1
   A1 = ReLU(Z1)

2. **Output Layer (Softmax Activation)**
  Z2 = W2 ⋅ A1 + B2
  A2 = softmax(Z2)

  where:
     softmax(z_i) = exp(z_i) / Σ exp(z_j)


### 🔁 Backward Propagation Equations:

Let `Y` be the one-hot encoded true labels of shape (10, m)

1. **Output Layer Gradient**
   dZ2 = A2 - Y
   dW2 = (1/m) ⋅ dZ2 ⋅ A1ᵀ
   dB2 = (1/m) ⋅ Σ dZ2

2. **Hidden Layer Gradient (using ReLU derivative)**
   dZ1 = W2ᵀ ⋅ dZ2 ⊙ ReLU'(Z1)
   dW1 = (1/m) ⋅ dZ1 ⋅ Xᵀ
   dB1 = (1/m) ⋅ Σ dZ1

    Where:
    - `⊙` is element-wise multiplication
    - `ReLU'(Z1)` is 1 where Z1 > 0, else 0


## ✅ Results

- Final Accuracy: ~90%+ after training (varies based on hyperparameters)
- Visual inspection of predictions using matplotlib



## 🚀 Getting Started

### 🔧 Requirements
- Python 3.x
- NumPy
- Matplotlib (for visualizing digits)


🌟 Credits
Kaggle - Digit Recognizer Challenge
Inspired by deep learning foundations and Stanford CS229: Machine Learning Full Course


### 📦 Installation

Clone the repository:
```bash
git clone https://github.com/Anish-Pal/digit-recognizer.git
cd digit-recognizer


