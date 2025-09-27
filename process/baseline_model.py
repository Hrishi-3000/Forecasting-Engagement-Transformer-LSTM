import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class BaselineModels:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'svr': SVR(kernel='rbf', C=1.0),
            'xgboost': XGBRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_score = float('inf')
    
    def prepare_data(self, features_df, test_size=0.2):
        """Prepare data for training and testing"""
        # Remove timestamp for modeling
        if 'timestamp' in features_df.columns:
            features_df = features_df.drop('timestamp', axis=1)
        
        # Separate features and target
        X = features_df.drop('engagement', axis=1)
        y = features_df['engagement']
        
        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """Evaluate a single model"""
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        return metrics, y_pred_test
    
    def cross_validate(self, model, X, y, cv_splits=5):
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        scores = cross_val_score(model, X, y, cv=tscv, 
                               scoring='neg_mean_squared_error')
        return np.sqrt(-scores)  # Return RMSE
    
    def train_all_models(self, features_df):
        """Train and evaluate all baseline models"""
        print("Training baseline models...")
        
        X_train, X_test, y_train, y_test = self.prepare_data(features_df)
        
        results = {}
        predictions = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Evaluate model
            metrics, y_pred = self.evaluate_model(model, X_train, X_test, y_train, y_test)
            results[model_name] = metrics
            predictions[model_name] = y_pred
            
            # Cross-validation
            cv_scores = self.cross_validate(model, pd.concat([X_train, X_test]), 
                                          pd.concat([y_train, y_test]))
            results[model_name]['cv_mean_rmse'] = np.mean(cv_scores)
            results[model_name]['cv_std_rmse'] = np.std(cv_scores)
            
            # Update best model
            if metrics['test_rmse'] < self.best_score:
                self.best_score = metrics['test_rmse']
                self.best_model = model
        
        return results, predictions
    
    def get_feature_importance(self, features_df):
        """Get feature importance from the best model"""
        if self.best_model is None:
            self.train_all_models(features_df)
        
        X = features_df.drop(['engagement', 'timestamp'], axis=1, errors='ignore')
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            # For linear models, use coefficients
            if hasattr(self.best_model, 'coef_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.abs(self.best_model.coef_)
                }).sort_values('importance', ascending=False)
            else:
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': [1] * len(X.columns)
                })
        
        return importance_df

if __name__ == "__main__":
    # Example usage
    baseline = BaselineModels()
    
    # Load features
    features_df = pd.read_csv('data/features/session1_features.csv')
    
    # Train models
    results, predictions = baseline.train_all_models(features_df)
    
    # Print results
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"Test MAE: {metrics['test_mae']:.4f}")
        print(f"Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
    
    # Get feature importance
    importance_df = baseline.get_feature_importance(features_df)
    print("\nTop 10 features:")
    print(importance_df.head(10))