#!/usr/bin/env python3
"""
EDA Report Generation Script for Fraud Detection Pipeline
Generates comprehensive exploratory data analysis report.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fraud_pipeline.config import load_config
from fraud_pipeline.eda import EDAAnalyzer
from fraud_pipeline.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate EDA report for Credit Card Fraud dataset")
    parser.add_argument(
        "--config", 
        default="fraud_pipeline/configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force regenerate report even if it exists"
    )
    return parser.parse_args()


def create_html_report(report_data: dict, output_path: Path) -> None:
    """Create HTML report from EDA data.
    
    Args:
        report_data: Dictionary with EDA results
        output_path: Path to save HTML report
    """
    logger = setup_logger("html_generator")
    logger.info("Creating HTML report...")
    
    # Get plot paths relative to reports directory
    plots_dir = Path("plots")
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Card Fraud Detection - EDA Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: #2c3e50;
            margin-top: 25px;
        }}
        .summary-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px 15px;
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            font-weight: bold;
        }}
        .plot-container {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .success {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Credit Card Fraud Detection - EDA Report</h1>
        <p style="text-align: center; color: #7f8c8d;">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
        
        <h2>📊 Dataset Overview</h2>
        <div class="summary-box">
            <div class="metric">Total Rows: {report_data['summary_stats']['total_rows']:,}</div>
            <div class="metric">Total Columns: {report_data['summary_stats']['total_columns']}</div>
            <div class="metric">Memory Usage: {report_data['summary_stats']['memory_usage_mb']:.2f} MB</div>
            <div class="metric">Fraud Rate: {report_data['class_distribution']['fraud_rate']:.4f}%</div>
            <div class="metric">Imbalance Ratio: 1:{report_data['class_distribution']['imbalance_ratio']:.1f}</div>
        </div>
        
        <h2>📈 Class Distribution</h2>
        <div class="plot-container">
            <img src="{plots_dir}/class_distribution.png" alt="Class Distribution">
        </div>
        
        <h3>Class Distribution Details</h3>
        <table>
            <tr>
                <th>Class</th>
                <th>Count</th>
                <th>Percentage</th>
                <th>Description</th>
            </tr>
            <tr>
                <td>0</td>
                <td>{report_data['class_distribution']['class_counts'][0]:,}</td>
                <td>{(report_data['class_distribution']['class_counts'][0] / report_data['class_distribution']['total_rows'] * 100):.2f}%</td>
                <td>Legitimate Transactions</td>
            </tr>
            <tr>
                <td>1</td>
                <td>{report_data['class_distribution']['class_counts'][1]:,}</td>
                <td>{(report_data['class_distribution']['class_counts'][1] / report_data['class_distribution']['total_rows'] * 100):.2f}%</td>
                <td>Fraudulent Transactions</td>
            </tr>
        </table>
        
        <h2>🔥 Correlation Analysis</h2>
        <div class="plot-container">
            <img src="{plots_dir}/correlation_heatmap.png" alt="Correlation Heatmap">
        </div>
        
        <h2>💰 Amount Distribution</h2>
        <div class="plot-container">
            <img src="{plots_dir}/amount_histogram.png" alt="Amount Histogram">
        </div>
        
        <h2>⏰ Time Distribution</h2>
        <div class="plot-container">
            <img src="{plots_dir}/time_histogram.png" alt="Time Histogram">
        </div>
        
        <h2>🔍 Data Quality & Leakage Checks</h2>
        
        <h3>Duplicate Rows</h3>
        {f'<div class="warning">⚠️ Found {report_data["leakage_checks"]["duplicate_rows"]["count"]} duplicate rows ({report_data["leakage_checks"]["duplicate_rows"]["percentage"]:.2f}% of dataset)</div>' if report_data["leakage_checks"]["duplicate_rows"]["count"] > 0 else '<div class="success">✅ No duplicate rows found</div>'}
        
        <h3>Perfect Correlations</h3>
        {f'<div class="warning">⚠️ Found {len(report_data["leakage_checks"]["perfect_correlations"])} pairs of perfectly correlated features: {", ".join([f"{pair[0]} ↔ {pair[1]}" for pair in report_data["leakage_checks"]["perfect_correlations"]])}</div>' if report_data["leakage_checks"]["perfect_correlations"] else '<div class="success">✅ No perfect correlations found</div>'}
        
        <h3>Zero Variance Features</h3>
        {f'<div class="warning">⚠️ Found {len(report_data["leakage_checks"]["zero_variance_features"])} features with zero variance: {", ".join(report_data["leakage_checks"]["zero_variance_features"])}</div>' if report_data["leakage_checks"]["zero_variance_features"] else '<div class="success">✅ No zero variance features found</div>'}
        
        <h3>Low Variance Features</h3>
        {f'<div class="warning">⚠️ Found {len(report_data["leakage_checks"]["low_variance_features"])} features with very low variance: {", ".join(report_data["leakage_checks"]["low_variance_features"])}</div>' if report_data["leakage_checks"]["low_variance_features"] else '<div class="success">✅ No low variance features found</div>'}
        
        <h2>📋 Data Types Summary</h2>
        <table>
            <tr>
                <th>Column</th>
                <th>Data Type</th>
                <th>Null Count</th>
            </tr>
    """
    
    # Add data types table
    for col, dtype in report_data['summary_stats']['dtypes'].items():
        null_count = report_data['summary_stats']['null_counts'][col]
        html_content += f"""
            <tr>
                <td>{col}</td>
                <td>{dtype}</td>
                <td>{null_count}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <div class="footer">
            <p>Generated by Fraud Detection Pipeline EDA Tool</p>
            <p>This report provides insights into the dataset structure, class distribution, and potential data quality issues.</p>
        </div>
    </div>
</body>
</html>
    """
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to: {output_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger("eda_report")
    
    logger.info("🚀 Starting EDA report generation...")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    # Initialize EDA analyzer
    analyzer = EDAAnalyzer(config)
    
    # Check if report already exists
    html_path = Path(config.data.reports_path) / "eda.html"
    if not args.force and html_path.exists():
        logger.info("EDA report already exists. Use --force to regenerate.")
        return 0
    
    try:
        # Load data
        df = analyzer.load_data()
        logger.info(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # Generate EDA report
        report_data = analyzer.generate_eda_report(df)
        
        # Create HTML report
        create_html_report(report_data, html_path)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 EDA REPORT SUMMARY")
        print("="*60)
        print(f"📁 Report location: {html_path}")
        print(f"📈 Dataset size: {report_data['summary_stats']['total_rows']:,} rows × {report_data['summary_stats']['total_columns']} columns")
        print(f"🎯 Class distribution: {report_data['class_distribution']['class_counts']}")
        print(f"💰 Fraud rate: {report_data['class_distribution']['fraud_rate']:.4f}%")
        print(f"⚖️  Imbalance ratio: 1:{report_data['class_distribution']['imbalance_ratio']:.1f}")
        print(f"🔍 Duplicate rows: {report_data['leakage_checks']['duplicate_rows']['count']}")
        print(f"📊 Plots generated: {len(report_data['plots'])}")
        print("\n✅ EDA report generated successfully!")
        print(f"🌐 Open {html_path} in your browser to view the full report")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error generating EDA report: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
