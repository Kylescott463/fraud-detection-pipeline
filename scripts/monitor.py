#!/usr/bin/env python3
"""CLI script for data drift monitoring."""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fraud_pipeline.monitor import DataDriftMonitor
from fraud_pipeline.config import load_config
from fraud_pipeline.utils.logger import setup_logger


def main():
    """Main function for monitoring CLI."""
    parser = argparse.ArgumentParser(description="Monitor data drift in fraud detection pipeline")
    
    parser.add_argument(
        "--config",
        type=str,
        default="fraud_pipeline/configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input data file (CSV or Parquet)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/reports/monitoring.html",
        help="Path to output monitoring report"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for drift detection (default: 0.05)"
    )
    
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        help="Specific features to check for drift (if not specified, check all numeric)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger("monitor_cli", level=log_level)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Initialize monitor
        logger.info("Initializing data drift monitor")
        monitor = DataDriftMonitor(args.config)
        
        # Load input data
        logger.info(f"Loading input data from {args.input}")
        input_path = Path(args.input)
        
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        
        if input_path.suffix.lower() == '.csv':
            data = pd.read_csv(input_path)
        elif input_path.suffix.lower() == '.parquet':
            data = pd.read_parquet(input_path)
        else:
            logger.error(f"Unsupported file format: {input_path.suffix}")
            sys.exit(1)
        
        logger.info(f"Loaded {len(data)} rows and {len(data.columns)} columns")
        
        # Detect drift
        logger.info("Detecting data drift...")
        drift_results = monitor.detect_drift(data, features=args.features, alpha=args.alpha)
        
        if "error" in drift_results:
            logger.error(f"Drift detection failed: {drift_results['error']}")
            sys.exit(1)
        
        # Print summary
        summary = drift_results.get("summary", {})
        logger.info("Drift Detection Summary:")
        logger.info(f"  Total features checked: {summary.get('total_features', 0)}")
        logger.info(f"  Features with drift: {summary.get('features_with_drift', 0)}")
        logger.info(f"  Drift percentage: {summary.get('drift_percentage', 0):.2f}%")
        logger.info(f"  Average KS statistic: {summary.get('avg_ks_statistic', 0):.4f}")
        logger.info(f"  Average P-value: {summary.get('avg_p_value', 1):.4f}")
        
        # Generate monitoring report
        logger.info(f"Generating monitoring report to {args.output}")
        report_path = monitor.generate_monitoring_report(data, args.output)
        
        logger.info(f"✅ Monitoring report generated successfully: {report_path}")
        
        # Print drifted features if any
        drifted_features = drift_results.get("drift_detected", [])
        if drifted_features:
            logger.warning(f"Drift detected in {len(drifted_features)} features:")
            for feature in drifted_features[:10]:  # Show first 10
                ks_stat = drift_results["ks_statistics"].get(feature, 0)
                p_value = drift_results["p_values"].get(feature, 1)
                logger.warning(f"  - {feature}: KS={ks_stat:.4f}, p={p_value:.6f}")
            
            if len(drifted_features) > 10:
                logger.warning(f"  ... and {len(drifted_features) - 10} more features")
        else:
            logger.info("✅ No drift detected in any features")
        
        # Exit with appropriate code
        drift_percentage = summary.get('drift_percentage', 0)
        if drift_percentage > 20:
            logger.warning("High drift detected - consider retraining model")
            sys.exit(2)  # Warning exit code
        elif drift_percentage > 10:
            logger.warning("Moderate drift detected - monitor closely")
            sys.exit(1)  # Warning exit code
        else:
            logger.info("Low drift detected - model appears stable")
            sys.exit(0)  # Success
            
    except Exception as e:
        logger.error(f"Error during monitoring: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
