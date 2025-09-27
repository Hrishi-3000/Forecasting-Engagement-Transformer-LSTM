# process/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, window_size=30, step_size=5):
        self.window_size = window_size
        self.step_size = step_size
    
    def safe_statistical_features(self, data, prefix):
        """Safe statistical feature extraction with type handling"""
        features = {}
        
        # Ensure data is numeric and convert if necessary
        try:
            data_numeric = pd.to_numeric(data, errors='coerce').dropna()
        except:
            data_numeric = np.array(data, dtype=float)
        
        if len(data_numeric) == 0:
            # Return zeros if no valid data
            features.update({
                f'{prefix}_mean': 0, f'{prefix}_std': 0, f'{prefix}_min': 0,
                f'{prefix}_max': 0, f'{prefix}_range': 0, f'{prefix}_median': 0
            })
            return features
        
        # Basic statistical features
        features[f'{prefix}_mean'] = float(np.mean(data_numeric))
        features[f'{prefix}_std'] = float(np.std(data_numeric))
        features[f'{prefix}_min'] = float(np.min(data_numeric))
        features[f'{prefix}_max'] = float(np.max(data_numeric))
        features[f'{prefix}_range'] = float(np.max(data_numeric) - np.min(data_numeric))
        features[f'{prefix}_median'] = float(np.median(data_numeric))
        
        # Only calculate these if we have enough data points
        if len(data_numeric) > 3:
            try:
                from scipy import stats
                features[f'{prefix}_skew'] = float(stats.skew(data_numeric))
                features[f'{prefix}_kurtosis'] = float(stats.kurtosis(data_numeric))
            except:
                features[f'{prefix}_skew'] = 0
                features[f'{prefix}_kurtosis'] = 0
        
        # Simple percentiles
        try:
            features[f'{prefix}_q25'] = float(np.percentile(data_numeric, 25))
            features[f'{prefix}_q75'] = float(np.percentile(data_numeric, 75))
        except:
            features[f'{prefix}_q25'] = 0
            features[f'{prefix}_q75'] = 0
        
        return features
    
    def create_sliding_windows_simple(self, df, engagement_col='engagement'):
        """Create sliding windows with robust error handling"""
        features_list = []
        targets = []
        
        total_samples = len(df)
        
        # Adjust parameters based on data size
        if total_samples < 50:
            window_size = min(10, total_samples - 1)
            step_size = max(1, window_size // 2)
        else:
            window_size = min(self.window_size, total_samples // 5)
            step_size = max(1, window_size // 3)
        
        print(f"   Using window_size={window_size}, step_size={step_size}")
        
        windows_created = 0
        for start_idx in range(0, total_samples - window_size, step_size):
            end_idx = start_idx + window_size
            target_idx = end_idx
            
            if target_idx >= total_samples:
                break
            
            try:
                window_data = df.iloc[start_idx:end_idx]
                target_data = df.iloc[target_idx]
                
                # Skip if target is not available
                if engagement_col not in target_data:
                    continue
                
                target_value = target_data[engagement_col]
                if not isinstance(target_value, (int, float, np.number)):
                    continue
                
                # Extract features from window
                window_features = {}
                
                # Process each numeric column
                numeric_cols = window_data.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col not in ['timestamp', engagement_col]]
                
                for col in numeric_cols[:20]:  # Limit to first 20 columns to avoid too many features
                    if col in window_data.columns:
                        col_data = window_data[col].values
                        col_features = self.safe_statistical_features(col_data, col)
                        window_features.update(col_features)
                
                # Only add if we have features
                if window_features:
                    features_list.append(window_features)
                    targets.append(float(target_value))
                    windows_created += 1
                    
            except Exception as e:
                continue  # Skip this window if any error occurs
        
        print(f"   Successfully created {windows_created} windows")
        
        if not features_list:
            return pd.DataFrame(), pd.Series(dtype=float)
        
        features_df = pd.DataFrame(features_list)
        targets_series = pd.Series(targets, name='engagement', dtype=float)
        
        return features_df, targets_series
    
    def engineer_features(self, processed_df):
        """Robust feature engineering pipeline"""
        print("Starting feature engineering...")
        
        if processed_df is None or len(processed_df) == 0:
            raise ValueError("No processed data available")
        
        print(f"   Input data shape: {processed_df.shape}")
        
        # Ensure engagement column exists and is numeric
        if 'engagement' not in processed_df.columns:
            # Create synthetic engagement if missing
            processed_df['engagement'] = np.linspace(0, 1, len(processed_df))
            print("   Created synthetic engagement column")
        
        # Convert engagement to numeric
        processed_df['engagement'] = pd.to_numeric(processed_df['engagement'], errors='coerce')
        processed_df = processed_df.dropna(subset=['engagement'])
        
        if len(processed_df) == 0:
            raise ValueError("No valid engagement data after cleaning")
        
        # Create sliding windows
        features_df, targets_series = self.create_sliding_windows_simple(processed_df)
        
        if features_df.empty or len(features_df) == 0:
            raise ValueError("No features could be created from the data")
        
        print(f"   Created {len(features_df)} samples with {len(features_df.columns)} features")
        
        # Clean the features dataframe
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        # Remove constant columns
        constant_cols = []
        for col in features_df.columns:
            if features_df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"   Removing {len(constant_cols)} constant columns")
            features_df = features_df.drop(columns=constant_cols)
        
        if features_df.empty:
            raise ValueError("All features were constant")
        
        # Combine features and targets
        final_df = pd.concat([features_df.reset_index(drop=True), 
                             targets_series.reset_index(drop=True)], axis=1)
        
        print(f"   Final features shape: {final_df.shape}")
        return final_df, None

    # -------------------------------
    # 🔥 Enhanced methods
    # -------------------------------
    def create_advanced_features(self, processed_df):
        """Create advanced features with better temporal patterns"""
        print("Creating advanced features...")
        
        # Use larger window for better context
        self.window_size = 60  # 60 seconds for better pattern recognition
        self.step_size = 10    # 10-second steps
        
        features_df, targets_series = self.create_sliding_windows_simple(processed_df)
        
        if features_df.empty:
            return features_df, targets_series
        
        # Add temporal features
        features_df = self.add_temporal_features(features_df, processed_df)
        
        # Add cross-modality features
        features_df = self.add_cross_modality_features(features_df)
        
        return features_df, targets_series

    def add_temporal_features(self, features_df, original_df):
        """Add temporal pattern features"""
        
        # Calculate engagement trends
        engagement_trend = original_df['engagement'].diff().rolling(window=10).mean()
        features_df['engagement_trend'] = engagement_trend.mean()
        
        # Variability features
        for col in original_df.columns:
            if any(modality in col.lower() for modality in ['eeg', 'gsr', 'eye']):
                rolling_std = original_df[col].rolling(window=15).std()
                features_df[f'{col}_variability'] = rolling_std.mean()
        
        return features_df

    def add_cross_modality_features(self, features_df):
        """Add features that capture interactions between modalities"""
        
        # EEG-GSR correlation (cognitive-physiological coupling)
        eeg_cols = [col for col in features_df.columns if 'eeg' in col.lower() and 'mean' in col]
        gsr_cols = [col for col in features_df.columns if 'gsr' in col.lower() and 'mean' in col]
        
        if eeg_cols and gsr_cols:
            # Simple interaction feature
            features_df['eeg_gsr_interaction'] = (
                features_df[eeg_cols[0]] * features_df[gsr_cols[0]]
            )
        
        return features_df


# Test function
def test_feature_engineering():
    """Test the feature engineering pipeline"""
    # Create sample data with proper types
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'timestamp': range(100),
        'EEG_1': np.random.normal(0, 1, 100).astype(float),
        'EEG_2': np.random.normal(0, 1, 100).astype(float),
        'GSR': np.random.exponential(1, 100).astype(float),
        'EYE': np.random.normal(5, 1, 100).astype(float),
        'engagement': np.sin(np.arange(100) * 0.1).astype(float) + 0.5
    })
    
    feature_engineer = FeatureEngineer(window_size=20, step_size=5)
    features_df, selector = feature_engineer.engineer_features(sample_data)
    adv_features_df, _ = feature_engineer.create_advanced_features(sample_data)
    
    print(f"Test passed! Basic features: {len(features_df)} samples, Advanced features: {len(adv_features_df)} samples")
    return features_df, adv_features_df

if __name__ == "__main__":
    test_feature_engineering()
