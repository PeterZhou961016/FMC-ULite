FMC-ULite: Fast Multi-scale Convolutional U-Net Lite for Semantic Segmentation
===============================================================================

Overview
----------
FMC-ULite is an efficient semantic segmentation model designed for disaster scene analysis and building damage assessment. The architecture combines MobileNetV3-Large as a backbone with advanced attention mechanisms and frequency domain processing for improved feature extraction and computational efficiency.

Key Features
----------
1.Simplified MobileNetV3 Backbone: Leverages lightweight MobileNetV3-Large for efficient feature extraction

2.FFT Frequency Fusion: Incorporates Fast Fourier Transform (FFT) processing with learnable Gaussian low-pass filters for enhanced frequency domain features

3.Multi-scale Cross-level Fusion: Advanced feature fusion across different encoder levels with attention mechanisms

4.Disaster Attention Gate: Specialized attention mechanism for disaster scene segmentation

5.Adaptive Loss Function: Combines CrossEntropy and Dice loss with adaptive weighting during training

Model Architecture
-------------------
The FMC-ULite architecture consists of:

1.FFT Preprocessing Module: Processes input images in frequency domain using Gaussian low-pass filters

2.MobileNetV3-Large Encoder: Extracts multi-scale features with efficient depthwise convolutions

3.Cross-level Feature Fusion: Fuses features from different encoder levels using MultiScaleCrossLevelFusion

4.Decoder with Attention Gates: Upsamples features with DisasterAttentionGate for precise localization

5.Output Segmentation Head: Produces final semantic segmentation masks

Installation
---------------
Prerequisites
Python 3.8 or higher

CUDA-compatible GPU (recommended for training)

Dependencies
----------------
The main dependencies are:

torch>=2.0.0

torchvision>=0.15.0

numpy>=1.24.0

opencv-python>=4.8.0

scikit-learn>=1.3.0

pandas>=2.0.0

tqdm>=4.65.0

See requirements.txt for complete dependency list.

Project Structure
-----------------
FMC-ULite/
├── config.py                    
├── main.py                      
├── requirements.txt             
├── README.md                   
├── model/                       
│   ├── __init__.py
│   ├── attention.py             
│   ├── fft_fusion.py           
│   ├── fusion_modules.py       
│   └── unet_mobilenet.py       
├── data/                        
│   ├── __init__.py
│   ├── dataset.py              
│   └── transforms.py          
├── loss/                       
│   ├── __init__.py
│   └── adaptive_loss.py       
├── train/                       
│   ├── __init__.py
│   ├── trainer.py              
│   ├── validator.py            
│   └── tester.py               
└── utils/                      
    ├── __init__.py     
    ├── metrics.py              
    ├── visualization.py        
    └── checkpoint.py       
    
Data Preparation
The model expects data in the following structure:

RescueNet/
├── trainset/
│   ├── images/         
│   └── masks/          
├── validationset/
│   ├── images/         
│   └── masks/          
└── testset/
    ├── images/         
    └── masks/          

Maintenance
--------------------------------
This project is actively maintained with the following commitment:

Regular Updates

Code Maintenance: Regular bug fixes and performance improvements

Dependency Updates: Periodic updates to ensure compatibility with latest libraries

Documentation: Continuous improvement of documentation and examples

Issue Management

Bug Reports: Prompt investigation and resolution of reported issues

Feature Requests: Consideration of community suggestions for enhancements

Pull Requests: Review and integration of community contributions

Support
---------------------
Compatibility: Support for recent PyTorch versions

Platform Support: Testing on major platforms (Linux, Windows with GPU support)

Community Support: Assistance through GitHub issues for installation and usage problems

Development Roadmap

Planned improvements include:

Support for additional datasets

Export to ONNX/TensorRT for deployment

Integration with popular ML frameworks

Performance optimization for edge devices

Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

