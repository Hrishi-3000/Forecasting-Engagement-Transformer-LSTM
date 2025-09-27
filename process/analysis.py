import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ResultAnalyzer:
    def __init__(self, results_path='results/'):
        self.results_path = results_path
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_results(self, session_num):
        """Load results for a specific session"""
        try:
            baseline_results = pd.read_csv(f'{self.results_path}session{session_num}_baseline.csv')
            lstm_results = pd.read_csv(f'{self.results_path}session{session_num}_lstm.csv')
            transformer_results = pd.read_csv(f'{self.results_path}session{session_num}_transformer.csv')
            return baseline_results, lstm_results, transformer_results
        except FileNotFoundError:
            print(f"Results for session {session_num} not found")
            return None, None, None
    
    def compare_models(self, session_results):
        """Compare performance across different models"""
        baseline, lstm, transformer = session_results
        
        comparison_data = []
        
        if baseline is not None:
            comparison_data.append({
                'Model': 'Random Forest',
                'MAE': baseline['test_mae'].iloc[0],
                'RMSE': baseline['test_rmse'].iloc[0],
                'R2': baseline['test_r2'].iloc[0]
            })
        
        if lstm is not None:
            comparison_data.append({
                'Model': 'LSTM',
                'MAE': lstm['mae'].iloc[0],
                'RMSE': lstm['rmse'].iloc[0],
                'R2': lstm['r2'].iloc[0]
            })
        
        if transformer is not None:
            comparison_data.append({
                'Model': 'Transformer',
                'MAE': transformer['mae'].iloc[0],
                'RMSE': transformer['rmse'].iloc[0],
                'R2': transformer['r2'].iloc[0]
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name, session_num):
        """Plot predictions vs actual values"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.6)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Engagement')
        ax1.set_ylabel('Predicted Engagement')
        ax1.set_title(f'{model_name} - Predictions vs Actual\nSession {session_num}')
        
        # Time series plot
        ax2.plot(y_true.values, label='Actual', alpha=0.7)
        ax2.plot(y_pred, label='Predicted', alpha=0.7)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Engagement Level')
        ax2.set_title(f'{model_name} - Time Series Prediction\nSession {session_num}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.results_path}{model_name}_session{session_num}_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance_df, top_n=15):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(f'{self.results_path}feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_plot(self, y_true, predictions_dict, session_num):
        """Create interactive plot using Plotly"""
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Predictions vs Actual', 'Error Distribution'))
        
        # Time series plot
        fig.add_trace(
            go.Scatter(y=y_true, mode='lines', name='Actual', line=dict(color='blue')),
            row=1, col=1
        )
        
        colors = ['red', 'green', 'orange']
        for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
            fig.add_trace(
                go.Scatter(y=y_pred, mode='lines', name=f'{model_name} Predicted', 
                          line=dict(color=colors[i % len(colors)], dash='dot')),
                row=1, col=1
            )
        
        # Error distribution
        for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
            errors = y_true - y_pred
            fig.add_trace(
                go.Histogram(x=errors, name=f'{model_name} Errors', opacity=0.7,
                            marker_color=colors[i % len(colors)]),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title_text=f"Engagement Prediction Analysis - Session {session_num}")
        fig.write_html(f'{self.results_path}session{session_num}_interactive_analysis.html')
        
        return fig
    
    def calculate_confidence_intervals(self, predictions, n_bootstraps=1000):
        """Calculate confidence intervals using bootstrapping"""
        bootstrap_scores = []
        
        for _ in range(n_bootstraps):
            # Sample with replacement
            indices = np.random.randint(0, len(predictions), len(predictions))
            sample_pred = predictions[indices]
            bootstrap_scores.append(np.mean(sample_pred))
        
        lower = np.percentile(bootstrap_scores, 2.5)
        upper = np.percentile(bootstrap_scores, 97.5)
        
        return lower, upper
    
    def generate_report(self, session_num, comparison_df, feature_importance):
        """Generate a comprehensive analysis report"""
        report = f"""
        ENGAGEMENT PREDICTION ANALYSIS REPORT
        =====================================
        
        Session: {session_num}
        Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        MODEL PERFORMANCE COMPARISON:
        {comparison_df.to_string(index=False)}
        
        TOP FEATURES:
        {feature_importance.head(10).to_string(index=False)}
        
        KEY INSIGHTS:
        - Best performing model: {comparison_df.loc[comparison_df['R2'].idxmax(), 'Model']}
        - Best R² score: {comparison_df['R2'].max():.4f}
        - Best MAE: {comparison_df['MAE'].min():.4f}
        - Best RMSE: {comparison_df['RMSE'].min():.4f}
        
        RECOMMENDATIONS:
        - Consider using {comparison_df.loc[comparison_df['R2'].idxmax(), 'Model']} for future predictions
        - Focus on the top features identified for interpretability
        - Consider ensemble methods for improved performance
        """
        
        with open(f'{self.results_path}session{session_num}_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        return report

if __name__ == "__main__":
    # Example usage
    analyzer = ResultAnalyzer()
    
    # Load results for session 1
    session_results = analyzer.load_results(1)
    
    if all(r is not None for r in session_results):
        # Compare models
        comparison_df = analyzer.compare_models(session_results)
        print("Model Comparison:")
        print(comparison_df)
        
        # Generate report
        feature_importance = pd.read_csv('results/feature_importance.csv')  # Assuming this exists
        analyzer.generate_report(1, comparison_df, feature_importance)