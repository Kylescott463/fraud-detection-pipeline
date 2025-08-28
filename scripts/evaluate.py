#!/usr/bin/env python3
"""
Model Evaluation Script for Fraud Detection Pipeline
Evaluates trained models and selects the best one with optimal threshold.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fraud_pipeline.config import load_config
from fraud_pipeline.evaluate import ModelEvaluator
from fraud_pipeline.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate fraud detection models")
    parser.add_argument(
        "--config", 
        default="fraud_pipeline/configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-evaluation even if reports exist"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger("evaluate")
    
    logger.info("🎯 Starting model evaluation...")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Check if evaluation reports already exist
    reports_dir = Path(config.data.reports_path)
    required_files = ["test_metrics.json", "final_report.md", "model_curves.png"]
    
    if not args.force and all((reports_dir / file).exists() for file in required_files):
        logger.info("Evaluation reports already exist. Use --force to re-evaluate.")
        
        # Load and display existing results
        try:
            import json
            metrics_path = reports_dir / "test_metrics.json"
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            print("\n" + "="*80)
            print("📊 EXISTING EVALUATION RESULTS")
            print("="*80)
            print(f"Best Model: {metrics['best_model']}")
            print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
            print(f"Minimum Recall Requirement: ≥{metrics['min_recall_requirement']:.2f}")
            
            print(f"\nModel Performance:")
            for model_name, model_metrics in metrics['test_metrics'].items():
                print(f"  {model_name}: PR-AUC={model_metrics['pr_auc']:.4f}, ROC-AUC={model_metrics['roc_auc']:.4f}")
            
            print(f"\n📁 Reports available at: {reports_dir}")
            return 0
            
        except Exception as e:
            logger.error(f"Error reading existing results: {e}")
            return 1
    
    try:
        # Evaluate models
        evaluation_results = evaluator.evaluate_models()
        
        # Print results
        evaluator.print_results(evaluation_results)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
