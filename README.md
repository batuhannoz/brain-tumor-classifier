# Brain Tumor Detector and Segmenter

This MATLAB application provides tools for detecting and segmenting brain tumors in medical images.

## Features
- Load medical images from various formats (JPG, PNG, TIFF, etc.)
- Tumor detection using a pre-trained YOLOv2 model
- Tumor segmentation using Medical Segment Anything Model (MedSAM)
- Visualization of results with bounding boxes and segmentation overlays
- Metrics for segmented tumor area percentage

## Requirements
- MATLAB (R2024b or newer)
- Computer Vision Toolbox
- Deep Learning Toolbox
- Medical Imaging Toolbox
- Medical Imaging Toolbox Model for MedSAM

## Usage
1. Launch `TumorDetectorSegmenterApp.m`
2. Click "Load Image" to select a medical image
3. Click "Detect Tumors" to identify tumors in the image
4. If a tumor is detected, click "Segment Tumor" for detailed segmentation

## Implementation Details
- The application uses YOLOv2 for initial tumor detection
- MedSAM (Medical Segment Anything Model) is used for precise segmentation
- Detected tumors are highlighted with a red bounding box
- Segmented tumors are displayed with a semi-transparent red overlay

MATLAB-based brain tumor classification project using deep learning approaches.

## Project Structure

```
brain-tumor-classifier/
├── archive/                    # Dataset directory
│   ├── yes/                   # Brain MRI images with tumors
│   └── no/                    # Brain MRI images without tumors
│
├── brain_tumor_detection_transfer_learning.m    # Transfer learning implementation using MobileNetV2
├── brain_tumor_detection_custom_cnn.m           # Custom CNN implementation
├── BrainTumorClassifierApp.m                   # MATLAB GUI application
└── README.md                                   # Project documentation
```

## Files Description

### Main Scripts

- **brain_tumor_detection_transfer_learning.m**
  - Implementation using transfer learning with MobileNetV2
  - Includes data preprocessing, model training, and evaluation
  - Features comprehensive performance metrics and visualizations

- **brain_tumor_detection_custom_cnn.m**
  - Custom CNN architecture implementation
  - Includes data augmentation and class imbalance handling
  - Implements early stopping and model evaluation

- **BrainTumorClassifierApp.m**
  - MATLAB GUI application for real-time tumor classification
  - Features model selection and image upload functionality
  - Provides confidence scores and visual results

### Dataset Structure

The project expects brain MRI images to be organized in the following structure:
```
archive/
├── yes/    # Contains MRI images with tumors
└── no/     # Contains MRI images without tumors
```

## Model Outputs

The training scripts generate several output files:
- Trained model files (*.mat)
- Performance metrics and visualizations
- Confusion matrices and ROC curves

## Requirements

- MATLAB R2020b or later
- Deep Learning Toolbox
- Image Processing Toolbox
- Deep Learning Toolbox Model for MobileNetV2 Network (for transfer learning version)