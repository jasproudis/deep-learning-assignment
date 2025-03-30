# Deep Learning Assignment 2024–25

**Author**: Iason-Christoforos Asproudis  
**Student ID**: p3352318

This repository contains code and documentation for the Deep Learning assignment of the MSc Data Science program at Athens University of Economics and Business.

---

## 📁 Project Structure

```
.
├── fashion_mnist_mlp.ipynb              # MLP for Fashion-MNIST
├── fashion_mnist_cnn.ipynb              # CNN for Fashion-MNIST
├── cifar10_mlp.ipynb                    # MLP for CIFAR-10
├── cifar10_cnn2.ipynb                   # CNN for CIFAR-10
├── mura_loader.ipynb                    # Loader and preprocessing for MURA
├── mura_transfer_resnet.ipynb           # Transfer learning with ResNet50
├── mura_transfer_cnn_finetune.ipynb     # Custom CNN for MURA
├── models/                              # Saved Keras models (.h5)
├── screenshots/                         # Training logs, model summaries, confusion matrices
├── Deep Learning Assignment Report.docx # Final report (structured, with screenshots)
└── README.md                            # This file
```

---

## 🧠 Part 1 – Image Classification

### 🔹 Fashion-MNIST
- **Dataset**: Grayscale 28×28 images of clothing items
- **Models**:
  - `fashion_mnist_mlp.ipynb`: MLP with two Dense layers and dropout
  - `fashion_mnist_cnn.ipynb`: CNN with convolutional blocks + batch norm + dropout
- **Summary**: Although CNNs work better, we explored both MLP and CNN to compare architectures.

### 🔹 CIFAR-10
- **Dataset**: RGB 32×32 images of real-world objects
- **Models**:
  - `cifar10_mlp.ipynb`: Simple MLP for baseline comparison
  - `cifar10_cnn2.ipynb`: CNN with Glorot init, BatchNorm, Dropout, EarlyStopping
- **Summary**: CNN is the model of choice due to the complex spatial nature of the data.

---

## 🩻 Part 2 – X-ray Study Classification (MURA Dataset)

- **Dataset**: Radiographs labeled as normal or abnormal
- **Preprocessing**: Handled image path parsing, resizing, and batching via `tf.data.Dataset`

### Models:
- `mura_transfer_cnn_finetune.ipynb`: Custom CNN built from scratch
- `mura_transfer_resnet.ipynb`: Transfer learning using ResNet50 pretrained on ImageNet

### Challenges:
- Large dataset loading on low-end hardware
- Training without GPU locally was time-consuming; Colab was explored

---

## 📊 Screenshots
All plots and model summaries can be found under the `screenshots/` directory. These include:
- Accuracy/Loss curves
- Confusion matrices
- Model diagrams

---

## 📝 Report
The main findings, challenges, and experimental results are summarized in:

📄 `Deep Learning Assignment Report.docx`

---

## 🔗 Submission Info
- 📂 GitHub Repo: [https://github.com/jasproudis/deep-learning-assignment](https://github.com/jasproudis/deep-learning-assignment)
- 📎 Accompanied by a full report and screenshots

---

*Thank you for reviewing this work!*
