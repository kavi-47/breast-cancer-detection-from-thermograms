Breast Cancer Detection from Thermograms

Overview:
  This project implements a deep learning model for fully automated breast cancer detection using thermal images. The model employs U-Net for image segmentation and a Convolutional Neural Network (CNN) for classification, achieving impressive performance metrics.

Model Performance:

  Accuracy: 99.33%
  Specificity: 98.67%
  Sensitivity: 100%
These results indicate that the model is highly effective in detecting breast cancer, with a very low false positive rate and perfect detection of positive cases.

Methodology:

1. Data Preprocessing
  Loading Images: The thermal images and masks are loaded from a specified folder.
  Resizing: Images are resized to 228 × 228 pixels for faster computation.
  Normalization: Both resized images and masks are normalized.
  Data Splitting: The dataset is split into training, validation, and testing sets in a 70:15:15 ratio, randomly.

2. Image Segmentation (U-Net)
  Architecture: U-Net consists of a contracting path (left side) and an expansive path (right side).
  Contracting Path:
    Two 3x3 convolutions, each followed by a ReLU activation and a 2x2 max-pooling operation with stride 2 for downsampling.
  Expansive Path:
    Upsampling of the feature map, followed by a 2x2 convolution that halves the number of feature channels.
    Concatenation with the cropped feature map from the contracting path.
    Two 3x3 convolutions, each followed by a ReLU activation.
  Training:
    Optimizer: Adaptive Moment Estimation (ADAM)
    Epochs: 30
    Initial Learning Rate: 1.0e-3, with a piecewise schedule dropping by a factor of 0.3 every 10 epochs.
    Batch Size: 8
4. Classification (CNN)
  Architecture:
    Input Image Size: 228x228 pixels
    Convolutional Layers:
      64 filters (7x7), depth 3, stride 6
      128 filters (3x3x64)
      256 filters (3x3x128)
      256 filters (3x3x256)
      256 filters (3x3x256)
      256 filters (3x3x256) with max-pooling
    Max-pooling applied after the first and sixth convolutional layers.
    Fully Connected Layers: Two layers with 1024 neurons each.
    Output Layer: Number of neurons equal to the number of classes (2 for normal and abnormal breast tissue).
    ReLU activation applied after each convolutional layer.
    Dropout layers may be applied after fully connected layers to prevent overfitting.
  Training:
    Optimizer: Adaptive Moment Estimation (ADAM)
    Epochs: 30
    Initial Learning Rate: 2.0e-3
    Batch Size: 60
   
Comparison with Pretrained Models:

  We compared the performance of our custom model with the following pretrained models:
    AlexNet
    ResNet-18
    GoogLeNet
    VGG-16
  These models were fine-tuned and evaluated using the same dataset and preprocessing steps for a fair comparison.
