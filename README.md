###🪲 Imageomics HDR-SMood Challenge: Beetle Drought Prediction

This repository hosts the official submission for the Imageomics HDR-SMood Challenge.

Our model leverages deep learning to predict environmental drought indices (Standardized Precipitation Evapotranspiration Index - SPEI) across multiple timeframes (30-day, 1-year, and 2-year). By combining robust image processing of anatomical beetle specimens with ecological metadata feature extraction, this architecture is designed for high-accuracy batch inference and achieved top-tier predictive performance on the Codabench leaderboard.

Team Information

Team Name: nebiyu

Team Member: Nebiyeleul Yifru (University of Maryland, Baltimore County - UMBC)

📁 Repository Structure

model.py: Contains the Model class with the load() and predict() functions for batch inference. Handles dynamic image resizing, error fallback, and metadata integration.

requirements.txt: Python dependencies required to run the inference code.

best_beetle_model.keras: (Hosted externally due to GitHub file size limits).

📥 Model Weights Download

Because the trained .keras model exceeds GitHub's 100MB file limit, the weights are hosted securely on Google Drive.

https://drive.google.com/drive/folders/1Nbews_ud8c-qjs0RTMDK_qfpv_lSMEE0?usp=sharing

To run the code locally, download the model from the link above and place it in the same directory as model.py.

⚙️ Dependencies

To install the required packages, run:

pip install -r requirements.txt
