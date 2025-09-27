
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