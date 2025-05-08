# 🏏 Sports Player Classification using Haar Cascades, Wavelet Transform, and ResNet-50

This project performs **image-based classification of 30 legendary sports players** (cricketers) using deep learning and computer vision techniques. It combines traditional image processing methods (like **Haar Cascades** for face and eye detection) with modern deep learning (using **ResNet-50**) and **wavelet-based feature extraction** to achieve robust performance, even on a small dataset (~750–800 samples).

---

## 📁 Dataset
- ~750–800 images of **30 cricket players**
- Each player has around 25 samples
- Images are preprocessed using face detection to improve feature learning

---

## 🔍 Project Workflow

1. **Face & Eye Detection using Haar Cascades**
   - Detect **frontal faces** using `haarcascade_frontalface_default.xml`
   - Confirm detection using **eyes** from `haarcascade_eye.xml`
   - This ensures the model trains on properly cropped faces

2. **Preprocessing and Feature Extraction**
   - Faces are:
     - Resized to 224x224
     - Converted to grayscale
     - Normalized
   - **Wavelet Transform** is applied to extract high-frequency texture features
     - Helps in enhancing edges and patterns critical to identity
     - Combined with raw image for hybrid feature representation

3. **Advanced Data Augmentation**
   To combat the small dataset, we use:
   - Random rotations, flips, and zoom
   - Brightness and contrast jittering
   - Color normalization
   - **MixUp and CutMix (optional)**: Merges features and labels of two images to regularize the model

4. **Model - Transfer Learning with ResNet-50**
   - Pretrained on ImageNet
   - Final layer modified to output 30 classes
   - All earlier layers are frozen during initial training (can be fine-tuned later)
   - Uses `CrossEntropyLoss` and Adam optimizer

5. **Training Strategy**
   - Stratified data split (Train/Validation)
   - Use of **early stopping** and **learning rate scheduling**
   - Evaluation using Accuracy and F1-score

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV** (for Haar Cascade detection)
- **PyWavelets** (for wavelet transform)
- **PyTorch** (for deep learning)
- **TorchVision** (data augmentations and pretrained models)
- **NumPy, Pandas, Matplotlib** (utilities and visualization)

---

## 📊 Results

- Achieved up to **80–85% validation accuracy**
- Effective even with small data thanks to wavelets and augmentation
- Face and eye validation significantly improved model confidence and reduced noise

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sports-player-classification.git
   cd sports-player-classification
