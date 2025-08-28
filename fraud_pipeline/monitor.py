"""Monitoring module for fraud detection pipeline."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from .config import load_config
from .utils.logger import setup_logger


class DataDriftMonitor:
    """Monitor for detecting data drift in fraud detection pipeline."""
    
    def __init__(self, config_path: str = "fraud_pipeline/configs/default.yaml"):
        """Initialize the data drift monitor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logger("data_drift_monitor")
        
        # Load training data statistics
        self.training_stats = self._load_training_stats()
        
    def _load_training_stats(self) -> Dict[str, Any]:
        """Load training data statistics for comparison.
        
        Returns:
            Dictionary containing training data statistics
        """
        try:
            # Load training data
            train_path = Path(self.config.data.processed_path) / "X_train.parquet"
            if train_path.exists():
                train_df = pd.read_parquet(train_path)
                
                # Calculate statistics for each numeric column
                stats_dict = {}
                for col in train_df.select_dtypes(include=[np.number]).columns:
                    stats_dict[col] = {
                        'mean': float(train_df[col].mean()),
                        'std': float(train_df[col].std()),
                        'min': float(train_df[col].min()),
                        'max': float(train_df[col].max()),
                        'percentiles': {
                            '25': float(train_df[col].quantile(0.25)),
                            '50': float(train_df[col].quantile(0.50)),
                            '75': float(train_df[col].quantile(0.75))
                        }
                    }
                
                return stats_dict
            else:
                self.logger.warning(f"Training data not found at {train_path}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error loading training stats: {e}")
            return {}
    
    def detect_drift(self, new_data: pd.DataFrame, 
                    features: Optional[List[str]] = None,
                    alpha: float = 0.05) -> Dict[str, Any]:
        """Detect data drift using Kolmogorov-Smirnov test.
        
        Args:
            new_data: New data to compare against training data
            features: List of features to check for drift (if None, use all numeric)
            alpha: Significance level for drift detection
            
        Returns:
            Dictionary containing drift detection results
        """
        if not self.training_stats:
            return {"error": "No training statistics available"}
        
        if features is None:
            features = list(self.training_stats.keys())
        
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "alpha": alpha,
            "features_checked": features,
            "drift_detected": [],
            "no_drift": [],
            "ks_statistics": {},
            "p_values": {},
            "summary": {}
        }
        
        for feature in features:
            if feature not in self.training_stats:
                continue
                
            if feature not in new_data.columns:
                continue
            
            # Perform KS test
            try:
                # Get training data distribution (approximate using stored stats)
                # For simplicity, we'll use the new data to estimate the training distribution
                # In a real implementation, you'd store actual training data samples
                train_mean = self.training_stats[feature]['mean']
                train_std = self.training_stats[feature]['std']
                
                # Generate synthetic training data based on stored statistics
                n_samples = min(10000, len(new_data))
                synthetic_train = np.random.normal(train_mean, train_std, n_samples)
                
                # Perform KS test
                ks_stat, p_value = stats.ks_2samp(
                    new_data[feature].dropna(), 
                    synthetic_train
                )
                
                drift_results["ks_statistics"][feature] = float(ks_stat)
                drift_results["p_values"][feature] = float(p_value)
                
                # Determine if drift is detected
                if p_value < alpha:
                    drift_results["drift_detected"].append(feature)
                else:
                    drift_results["no_drift"].append(feature)
                    
            except Exception as e:
                self.logger.error(f"Error performing KS test for {feature}: {e}")
                continue
        
        # Add summary statistics
        drift_results["summary"] = {
            "total_features": len(features),
            "features_with_drift": len(drift_results["drift_detected"]),
            "drift_percentage": len(drift_results["drift_detected"]) / len(features) * 100 if features else 0,
            "avg_ks_statistic": np.mean(list(drift_results["ks_statistics"].values())) if drift_results["ks_statistics"] else 0,
            "avg_p_value": np.mean(list(drift_results["p_values"].values())) if drift_results["p_values"] else 1
        }
        
        return drift_results
    
    def generate_drift_plots(self, new_data: pd.DataFrame, 
                           drift_results: Dict[str, Any],
                           output_dir: str = "data/reports/plots") -> List[str]:
        """Generate drift detection plots.
        
        Args:
            new_data: New data used for drift detection
            drift_results: Results from drift detection
            output_dir: Directory to save plots
            
        Returns:
            List of generated plot file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = []
        
        # Create drift summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Drift detection summary
        features = drift_results.get("features_checked", [])
        drift_detected = drift_results.get("drift_detected", [])
        no_drift = drift_results.get("no_drift", [])
        
        ax1.bar(['Drift Detected', 'No Drift'], 
                [len(drift_detected), len(no_drift)],
                color=['red', 'green'])
        ax1.set_title('Data Drift Summary')
        ax1.set_ylabel('Number of Features')
        
        # Plot 2: KS statistics distribution
        ks_stats = list(drift_results.get("ks_statistics", {}).values())
        if ks_stats:
            ax2.hist(ks_stats, bins=20, alpha=0.7, color='blue')
            ax2.set_title('Distribution of KS Statistics')
            ax2.set_xlabel('KS Statistic')
            ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        drift_summary_path = output_path / "drift_summary.png"
        plt.savefig(drift_summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(drift_summary_path))
        
        # Create individual feature plots for drifted features
        for feature in drift_detected[:5]:  # Limit to first 5 drifted features
            if feature in new_data.columns and feature in self.training_stats:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot distributions
                ax.hist(new_data[feature].dropna(), bins=50, alpha=0.7, 
                       label='New Data', density=True)
                
                # Plot training distribution (approximate)
                train_mean = self.training_stats[feature]['mean']
                train_std = self.training_stats[feature]['std']
                x = np.linspace(train_mean - 3*train_std, train_mean + 3*train_std, 100)
                y = stats.norm.pdf(x, train_mean, train_std)
                ax.plot(x, y, 'r-', linewidth=2, label='Training Data (approx)')
                
                ax.set_title(f'Distribution Comparison: {feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Density')
                ax.legend()
                
                feature_plot_path = output_path / f"drift_{feature}.png"
                plt.savefig(feature_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(str(feature_plot_path))
        
        return plot_files
    
    def generate_monitoring_report(self, new_data: pd.DataFrame,
                                 output_path: str = "data/reports/monitoring.html") -> str:
        """Generate comprehensive monitoring report.
        
        Args:
            new_data: New data to monitor
            output_path: Path to save the monitoring report
            
        Returns:
            Path to the generated report
        """
        # Detect drift
        drift_results = self.detect_drift(new_data)
        
        # Generate plots
        plot_files = self.generate_drift_plots(new_data, drift_results)
        
        # Create HTML report
        html_content = self._create_html_report(new_data, drift_results, plot_files)
        
        # Save report
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Monitoring report saved to {report_path}")
        return str(report_path)
    
    def _create_html_report(self, new_data: pd.DataFrame,
                          drift_results: Dict[str, Any],
                          plot_files: List[str]) -> str:
        """Create HTML monitoring report.
        
        Args:
            new_data: New data analyzed
            drift_results: Results from drift detection
            plot_files: List of generated plot files
            
        Returns:
            HTML content as string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate basic statistics
        summary_stats = {
            "total_rows": len(new_data),
            "total_columns": len(new_data.columns),
            "missing_values": new_data.isnull().sum().sum(),
            "numeric_columns": len(new_data.select_dtypes(include=[np.number]).columns)
        }
        
        # Create HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fraud Detection Pipeline - Monitoring Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .drift-detected {{ color: red; font-weight: bold; }}
                .no-drift {{ color: green; }}
                .warning {{ color: orange; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fraud Detection Pipeline - Monitoring Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Data Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Rows</td><td>{summary_stats['total_rows']:,}</td></tr>
                    <tr><td>Total Columns</td><td>{summary_stats['total_columns']}</td></tr>
                    <tr><td>Missing Values</td><td>{summary_stats['missing_values']:,}</td></tr>
                    <tr><td>Numeric Columns</td><td>{summary_stats['numeric_columns']}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Drift Detection Results</h2>
                <p><strong>Significance Level (α):</strong> {drift_results.get('alpha', 0.05)}</p>
                <p><strong>Features Checked:</strong> {len(drift_results.get('features_checked', []))}</p>
                <p><strong>Drift Detected:</strong> <span class="drift-detected">{len(drift_results.get('drift_detected', []))}</span></p>
                <p><strong>No Drift:</strong> <span class="no-drift">{len(drift_results.get('no_drift', []))}</span></p>
                
                <h3>Drift Summary</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Features</td><td>{drift_results.get('summary', {}).get('total_features', 0)}</td></tr>
                    <tr><td>Features with Drift</td><td>{drift_results.get('summary', {}).get('features_with_drift', 0)}</td></tr>
                    <tr><td>Drift Percentage</td><td>{drift_results.get('summary', {}).get('drift_percentage', 0):.2f}%</td></tr>
                    <tr><td>Average KS Statistic</td><td>{drift_results.get('summary', {}).get('avg_ks_statistic', 0):.4f}</td></tr>
                    <tr><td>Average P-value</td><td>{drift_results.get('summary', {}).get('avg_p_value', 1):.4f}</td></tr>
                </table>
            </div>
        """
        
        # Add drifted features table
        if drift_results.get('drift_detected'):
            html += """
            <div class="section">
                <h3>Features with Detected Drift</h3>
                <table>
                    <tr><th>Feature</th><th>KS Statistic</th><th>P-value</th></tr>
            """
            
            for feature in drift_results['drift_detected']:
                ks_stat = drift_results['ks_statistics'].get(feature, 0)
                p_value = drift_results['p_values'].get(feature, 1)
                html += f"""
                    <tr>
                        <td class="drift-detected">{feature}</td>
                        <td>{ks_stat:.4f}</td>
                        <td>{p_value:.6f}</td>
                    </tr>
                """
            
            html += "</table></div>"
        
        # Add plots
        if plot_files:
            html += """
            <div class="section">
                <h2>Visualizations</h2>
            """
            
            for plot_file in plot_files:
                plot_name = Path(plot_file).name
                html += f"""
                <div class="plot">
                    <h3>{plot_name}</h3>
                    <img src="{plot_file}" alt="{plot_name}">
                </div>
                """
            
            html += "</div>"
        
        # Add recommendations
        drift_percentage = drift_results.get('summary', {}).get('drift_percentage', 0)
        if drift_percentage > 20:
            recommendation = "High drift detected. Consider retraining the model."
            recommendation_class = "drift-detected"
        elif drift_percentage > 10:
            recommendation = "Moderate drift detected. Monitor closely and consider retraining soon."
            recommendation_class = "warning"
        else:
            recommendation = "Low drift detected. Model appears stable."
            recommendation_class = "no-drift"
        
        html += f"""
            <div class="section">
                <h2>Recommendations</h2>
                <p class="{recommendation_class}">{recommendation}</p>
                <ul>
                    <li>Monitor drift percentage: {drift_percentage:.2f}%</li>
                    <li>Check feature distributions regularly</li>
                    <li>Consider model retraining if drift exceeds 20%</li>
                    <li>Review data preprocessing pipeline</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html
