# Time-Series Forecasting of Engagement Levels

## Project Title
**Multimodal Time-Series Forecasting of Student Engagement Using EEG, Eye-Tracking, and GSR Signals**

---

## Project Overview
This project focuses on predicting future engagement levels using multimodal time-series data including EEG, eye-tracking, and GSR signals. The goal is to build sequence-to-sequence models (LSTM, GRU, Transformer) that can forecast engagement scores based on past physiological and behavioral signals.

---

## Team Information
### Group Details
- **Group Name**: Dualbots
- **Group ID**: T1_G04
- **Department**: Artificial Intelligence & Machine Learning (AIML)

### Team Members
- **Hrishikesh Shahane** (Group Leader)
- **Sayali Pundapal**
- **Krutika Kambale**

### Mentorship
- **Faculty Mentor**: Dr. Tanvi R. Patil
- **Internship Program**: IITB EdTech Internship 2025, DYPCET Track 1 - Educational Data Analysis (EDA)

---

## Problem Statement
**Problem ID**: 10  
**Objective**: Predict future engagement from past gaze, EEG, and GSR signals.  
**Methods**: RNNs, GRUs, LSTMs, Transformer models.

---

## Dataset Description
The dataset consists of the following files:

- `ENG.csv`: Engagement labels (continuous or discrete)
- `EEG.csv`: Brainwave bands (Delta, Theta, Alpha, Beta, Gamma)
- `EYE.csv`: Eye-tracking signals (fixation, saccade, pupil size)
- `IVT.csv`: Eye event classification (fixations, saccades, blinks)
- `GSR.csv`: Skin conductance/resistance values

---
## Workflow

### Step 1: Data Understanding & Preparation
- Identify target variable: Engagement score from `ENG.csv`
- Use sliding window approach:  
  - Input: Past N seconds of multimodal signals  
  - Output: Engagement score for the next M seconds

### Step 2: Preprocessing Pipeline
- Synchronize timestamps across all modalities
- Resample to a consistent frequency (e.g., 1 Hz)
- Extract features:
  - EEG: Band power (mean, variance)
  - GSR: Slope, peaks, recovery time
  - Eye: Fixation density, blink rate, pupil velocity
- Normalize using z-score per participant

### Step 3: Modeling Approaches
- **Baseline Models**: ARIMA, Random Forest, XGBoost
- **Deep Learning**: LSTM, GRU, BiLSTM with Attention
- **Advanced**: Multimodal Transformer encoder

### Step 4: Evaluation
- **Regression Metrics**: MAE, RMSE, R²
- **Classification Metrics**: Accuracy, F1-score, ROC-AUC
- **Interpretability**: Attention heatmaps, SHAP, LIME

### Step 5: Experimentation
- Compare per-participant vs. cross-participant models
- Test different window sizes and forecast horizons
- Apply data augmentation and multimodal dropout analysis

---

## Usage

### Prerequisites
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, torch, transformers, matplotlib, seaborn

### Running the Code
1. Clone the repository and navigate to the project directory.
2. Install required packages:
   ```bash
   pip install -r requirements.txt
