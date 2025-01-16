

# 🖊️ Handwritten Digit Recognition

This repository demonstrates a powerful application of **Convolutional Neural Networks (CNNs)** to recognize handwritten digits from 0 to 9 using the famous MNIST dataset. Dive into this project to learn, test, and visualize how neural networks can revolutionize digit recognition.

---

## 🌟 Features

### 🧠 Neural Network Model
- **Architecture**: Built using TensorFlow and Keras, the model leverages a **Convolutional Neural Network (CNN)**, designed to excel in image recognition tasks.
- **Layers**:
  - **Convolutional Layers**: Extract spatial features using filters to detect patterns like edges and curves witth 64 filters.
  - **Batch Normalization**: Stabilizes and speeds up training by normalizing inputs to each layer.
  - **Max Pooling**: Reduces spatial dimensions to retain essential features while reducing computation.
  - **Dropout Layer**: Prevents overfitting by randomly dropping connections during training.
  - **Dense Layers**: Fully connected layers to classify features into digit classes.
- **Output**: A 10-class softmax layer producing probabilities for each digit softmax is going to turn output to probabilty distribution.

### 📊 Dataset
- Utilizes the **MNIST dataset**, containing:
  - **60,000 training images**
  - **10,000 test images**
- Each image is a 28x28 grayscale representation of a handwritten digit.
- **Preprocessing**:
  - Normalization: Pixel values scaled to the range [0, 1].
  - One-Hot Encoding: Converts labels into binary vectors for classification.

### 🔧 Training and Data Augmentation
- **Training Features**:
  - Optimizer: Adam for efficient gradient-based optimization.
  - Loss Function: Categorical cross-entropy for multi-class classification.
  - Metrics: Accuracy to evaluate model performance.
- **Data Augmentation**:
  - Adds variability by applying random transformations like rotation, zoom, and shifts.
  - Improves model robustness to new data.

### 🎨 Interactive Drawing Interface
- **Grid-based Canvas**:
  - A 28x28 pixel grid where users can draw digits.
  - Dynamic updating of pixel intensities as the user draws.
- **Buttons**:
  - "Reset": Clears the canvas.
  - "Classify": Uses the trained model to predict the drawn digit.
- **Real-Time Classification**:
  - Displays the predicted digit alongside the canvas.

### 🚀 Performance
- Achieves **high accuracy (~99%)** on the MNIST test set.
- Robust to slight variations due to data augmentation.

### 💾 Model Saving and Loading
- **Save Model**:
  - Trained models can be saved with a ".keras" extension.
- **Load Model**:
  - Load pre-trained models for classification without retraining.

---

## 📂 File Structure

- `handwriting.py`: Contains the neural network training code.
- `recognition.py`: Implements the interactive interface for handwritten digit recognition.
- `assets/fonts/`: Includes the OpenSans font for rendering text in the interface.

---

## 🤖 About Convolutional Neural Networks (CNNs)
- **What are CNNs?**
  - Specialized neural networks designed for processing grid-like data such as images.
  - Utilize spatial hierarchies, learning local patterns in early layers and global patterns in deeper layers.

- **Key Features of CNNs in This Project**:
  - **Convolutional Layers**:
    - Learn spatial patterns by sliding filters across the image.
    - Detect features like edges, corners, and textures.
  - **Pooling Layers**:
    - Downsample feature maps, reducing dimensions and focusing on significant information.
  - **Fully Connected Layers**:
    - Combine extracted features for final classification.
  - **Softmax Output**:
    - Converts raw scores into probabilities for multi-class predictions.

- **Advantages of CNNs**:
  - Automatically learn relevant features from raw images.
  - Effective for image-related tasks, reducing the need for manual feature engineering.

---

## 📜 How to Run

### Prerequisites
- Python 3.10 or above
- TensorFlow 2.0+
- Pygame library for the interactive interface

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/ArianDilfanian/handwritten-digit-recognition.git
   ```
2. Navigate to the project directory:
   ```bash
   cd handwritten-digit-recognition
   ```
3. Train the model:
   ```bash
   python handwriting.py model_filename.keras
   ```
4. Launch the interactive interface:
   ```bash
   python recognition.py model_filename.keras
   ```

---

# 🎥 Video



https://github.com/user-attachments/assets/4b7b33fe-809c-4b86-92bc-df282fa336ef




---

## 🛠️ Future Enhancements
- Add support for recognizing multiple digits in a single canvas.
---

## 📧 Contact
For any questions or feedback, feel free to reach out to **Arian Dilfanian** via GitHub.

---



