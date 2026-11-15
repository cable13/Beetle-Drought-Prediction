# 🪲 Imageomics HDR-SMood Challenge: Beetle Drought Prediction

This repository hosts the official 1st Place submission for the multi-institute Imageomics HDR-SMood Challenge.

Our architecture leverages a Dual-Branch Convolutional Neural Network (CNN) to predict environmental drought severity (Standardized Precipitation Evapotranspiration Index - SPEI) across 30-day, 1-year, and 2-year timeframes. By fusing 20,000+ high-resolution biological images with tabular metadata, we achieved a top R² score of 0.840.

## 🚀 Engineering Highlights
- **Multi-Modal**: Architecture: Engineered a custom TensorFlow model combining an EfficientNetV2B2 image feature extractor with dense layers for categorical metadata (Scientific Name, Domain ID).
- **Data Pipeline Optimization**: Mitigated GPU data-starvation and memory bottlenecks by implementing local SSD caching and asynchronous data batching via tf.data.AUTOTUNE.
- **Dynamic Batch Inference**: Developed a robust inference pipeline capable of dynamically resizing variable-resolution imagery and handling missing tensors in real-time.

## 🧠 Model Architecture

## 📊 Model Performance

- **30-Day SPEI**: MAE: 0.590
- **1-Year SPEI**: MAE: 0.328 | R²: 0.743
- **2-Year SPEI**: MAE: 0.267 | R²: 0.840
## Team Information

**Team Name:** nebiyu

**Team Member:** Nebiyeleul Yifru (University of Maryland, Baltimore County - UMBC)

## 📁 Repository Structure

- **model.py**: Contains the Model class with the load() and predict() functions for batch inference. Handles dynamic image resizing, error fallback, and metadata integration.

- **requirements.txt**: Python dependencies required to run the inference code.

- **best_beetle_model.keras**: (Hosted externally due to GitHub file size limits).


## 📥 Model Weights Download

To comply with FAIR data storage standards, the trained `.keras` model weights are hosted securely on Hugging Face.

**[Download Model Weights from Hugging Face]
(https://huggingface.co/Nebiyu7/beetle-drought-prediction)**

To run the code locally, download the model from the link above and place it in the same directory as model.py.

## ⚙️ Quick Start

1. Clone the repository and install dependencies:

   pip install -r requirements.txt

2. Initialize the model and run predictions:

    from model import Model
    predictor = Model()
    predictor.load()
    #### Pass a list of dictionaries containing image paths and metadata
    predictions = predictor.predict(inputs)
