"""Exploratory Data Analysis module for fraud detection pipeline."""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .utils.logger import setup_logger


class EDAAnalyzer:
    """Exploratory Data Analysis analyzer for fraud detection dataset."""
    
    def __init__(self, config):
        """Initialize EDA analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = setup_logger("eda_analyzer")
        self.reports_dir = Path(config.data.reports_path)
        self.plots_dir = self.reports_dir / "plots"
        
        # Create directories
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def load_data(self) -> pd.DataFrame:
        """Load the dataset.
        
        Returns:
            Loaded DataFrame
        """
        csv_path = Path(self.config.data.raw_path)
        if csv_path.exists():
            self.logger.info(f"Loading data from {csv_path}")
            return pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        self.logger.info("Generating summary statistics...")
        
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "null_counts": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Add descriptive statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number])
        if not numeric_cols.empty:
            summary["descriptive_stats"] = numeric_cols.describe().to_dict()
        
        return summary
    
    def analyze_class_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze class distribution.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with class distribution analysis
        """
        self.logger.info("Analyzing class distribution...")
        
        class_counts = df['Class'].value_counts().sort_index()
        total_rows = len(df)
        fraud_rate = (class_counts.get(1, 0) / total_rows) * 100
        
        return {
            "class_counts": class_counts.to_dict(),
            "total_rows": total_rows,
            "fraud_rate": fraud_rate,
            "imbalance_ratio": class_counts.get(0, 0) / class_counts.get(1, 0) if class_counts.get(1, 0) > 0 else float('inf')
        }
    
    def plot_class_distribution(self, df: pd.DataFrame) -> str:
        """Create class distribution bar plot.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Creating class distribution plot...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        class_counts = df['Class'].value_counts().sort_index()
        colors = ['#2E8B57', '#DC143C']  # Green for legitimate, Red for fraud
        
        bars = ax.bar(class_counts.index, class_counts.values, color=colors)
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class (0=Legitimate, 1=Fraud)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
        
        # Add percentage labels
        total = len(df)
        for i, (class_val, count) in enumerate(class_counts.items()):
            percentage = (count / total) * 100
            ax.text(i, count/2, f'{percentage:.2f}%', ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / "class_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, top_n: int = 15) -> str:
        """Create correlation heatmap for top correlated features with Class.
        
        Args:
            df: Input DataFrame
            top_n: Number of top correlated features to show
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Creating correlation heatmap...")
        
        # Calculate correlations with Class
        numeric_cols = df.select_dtypes(include=[np.number])
        correlations = numeric_cols.corr()['Class'].abs().sort_values(ascending=False)
        
        # Get top N features (excluding Class itself)
        top_features = correlations[correlations.index != 'Class'].head(top_n)
        top_features = ['Class'] + top_features.index.tolist()
        
        # Create correlation matrix for top features
        corr_matrix = df[top_features].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontsize=12)
        
        # Set ticks and labels
        ax.set_xticks(range(len(top_features)))
        ax.set_yticks(range(len(top_features)))
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.set_yticklabels(top_features)
        
        # Add correlation values as text
        for i in range(len(top_features)):
            for j in range(len(top_features)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(f'Correlation Heatmap - Top {top_n} Features with Class', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / "correlation_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def plot_amount_histogram(self, df: pd.DataFrame) -> str:
        """Create histogram for Amount feature.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Creating Amount histogram...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall Amount distribution
        ax1.hist(df['Amount'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Amount Distribution (All Transactions)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Amount', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Amount distribution by class
        legitimate = df[df['Class'] == 0]['Amount']
        fraudulent = df[df['Class'] == 1]['Amount']
        
        ax2.hist(legitimate, bins=30, alpha=0.7, label='Legitimate', color='green')
        ax2.hist(fraudulent, bins=30, alpha=0.7, label='Fraudulent', color='red')
        ax2.set_title('Amount Distribution by Class', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Amount', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / "amount_histogram.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def plot_time_histogram(self, df: pd.DataFrame) -> str:
        """Create histogram for Time feature.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Creating Time histogram...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall Time distribution
        ax1.hist(df['Time'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax1.set_title('Time Distribution (All Transactions)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Time distribution by class
        legitimate = df[df['Class'] == 0]['Time']
        fraudulent = df[df['Class'] == 1]['Time']
        
        ax2.hist(legitimate, bins=30, alpha=0.7, label='Legitimate', color='green')
        ax2.hist(fraudulent, bins=30, alpha=0.7, label='Fraudulent', color='red')
        ax2.set_title('Time Distribution by Class', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / "time_histogram.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def check_data_leakage(self, df: pd.DataFrame) -> Dict:
        """Check for data leakage issues.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with leakage check results
        """
        self.logger.info("Checking for data leakage...")
        
        results = {}
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        results["duplicate_rows"] = {
            "count": duplicate_rows,
            "percentage": (duplicate_rows / len(df)) * 100
        }
        
        # Check for identical features (perfect correlation)
        numeric_cols = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_cols.corr()
        
        # Find features with perfect correlation (excluding self-correlation)
        perfect_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) == 1.0:
                    perfect_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        results["perfect_correlations"] = perfect_corr_pairs
        
        # Check for features with zero variance
        zero_variance_features = []
        for col in numeric_cols.columns:
            if numeric_cols[col].var() == 0:
                zero_variance_features.append(col)
        
        results["zero_variance_features"] = zero_variance_features
        
        # Check for features with very low variance (potential leakage)
        low_variance_features = []
        for col in numeric_cols.columns:
            if col != 'Class' and numeric_cols[col].var() < 1e-6:
                low_variance_features.append(col)
        
        results["low_variance_features"] = low_variance_features
        
        return results
    
    def generate_eda_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive EDA report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with all EDA results
        """
        self.logger.info("Generating comprehensive EDA report...")
        
        report = {
            "summary_stats": self.get_summary_stats(df),
            "class_distribution": self.analyze_class_distribution(df),
            "leakage_checks": self.check_data_leakage(df)
        }
        
        # Generate plots
        report["plots"] = {
            "class_distribution": self.plot_class_distribution(df),
            "correlation_heatmap": self.plot_correlation_heatmap(df),
            "amount_histogram": self.plot_amount_histogram(df),
            "time_histogram": self.plot_time_histogram(df)
        }
        
        return report
