#!/usr/bin/env python3
"""
Kaggle CLI Verification Script
Checks that Kaggle CLI is properly installed and configured.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def print_status(message, status="INFO"):
    """Print a status message with color coding."""
    colors = {
        "PASS": "\033[92m",  # Green
        "FAIL": "\033[91m",  # Red
        "WARN": "\033[93m",  # Yellow
        "INFO": "\033[94m",  # Blue
    }
    reset = "\033[0m"
    
    if status in colors:
        print(f"{colors[status]}{status}:{reset} {message}")
    else:
        print(f"{colors['INFO']}INFO:{reset} {message}")


def check_kaggle_cli():
    """Check if Kaggle CLI is callable."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print_status(f"Kaggle CLI is available: {version}", "PASS")
            return True
        else:
            print_status(f"Kaggle CLI returned error: {result.stderr}", "FAIL")
            return False
    except FileNotFoundError:
        print_status("Kaggle CLI not found in PATH", "FAIL")
        print_status("Run the bootstrap script first: make bootstrap", "INFO")
        return False
    except subprocess.TimeoutExpired:
        print_status("Kaggle CLI command timed out", "FAIL")
        return False
    except Exception as e:
        print_status(f"Error checking Kaggle CLI: {e}", "FAIL")
        return False


def check_kaggle_credentials():
    """Check if Kaggle credentials file exists and is readable."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_dir.exists():
        print_status(".kaggle directory not found", "FAIL")
        print_status("Run the bootstrap script first: make bootstrap", "INFO")
        return False
    
    if not kaggle_json.exists():
        print_status("kaggle.json not found in ~/.kaggle/", "FAIL")
        print_status("Place kaggle.json in project root and run: make bootstrap", "INFO")
        print_status("See kaggle.json.example for the expected format", "INFO")
        return False
    
    # Check file permissions (Unix-like systems)
    if os.name != 'nt':  # Not Windows
        try:
            stat = kaggle_json.stat()
            mode = stat.st_mode & 0o777
            if mode != 0o600:
                print_status(f"kaggle.json has incorrect permissions: {oct(mode)} (should be 600)", "WARN")
                print_status("Run: chmod 600 ~/.kaggle/kaggle.json", "INFO")
            else:
                print_status("kaggle.json has correct permissions (600)", "PASS")
        except Exception as e:
            print_status(f"Could not check file permissions: {e}", "WARN")
    
    # Try to read and parse the JSON file
    try:
        with open(kaggle_json, 'r') as f:
            credentials = json.load(f)
        
        required_keys = ['username', 'key']
        missing_keys = [key for key in required_keys if key not in credentials]
        
        if missing_keys:
            print_status(f"kaggle.json missing required keys: {missing_keys}", "FAIL")
            return False
        
        print_status("kaggle.json is valid and contains required credentials", "PASS")
        return True
        
    except json.JSONDecodeError as e:
        print_status(f"kaggle.json is not valid JSON: {e}", "FAIL")
        return False
    except Exception as e:
        print_status(f"Error reading kaggle.json: {e}", "FAIL")
        return False


def check_kaggle_api():
    """Test Kaggle API by searching for the credit card fraud dataset."""
    try:
        print_status("Testing Kaggle API connection...", "INFO")
        result = subprocess.run(
            ["kaggle", "datasets", "list", "-s", "creditcardfraud"], 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if "creditcardfraud" in output.lower():
                print_status("Successfully found credit card fraud dataset", "PASS")
                return True
            else:
                print_status("API call succeeded but dataset not found in results", "WARN")
                print_status("This might be due to search terms or API changes", "INFO")
                return True  # API is working, just dataset not found
        else:
            error_msg = result.stderr.strip()
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                print_status("Kaggle API authentication failed", "FAIL")
                print_status("Check your kaggle.json credentials", "INFO")
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                print_status("Kaggle API rate limit exceeded", "WARN")
                print_status("Wait a few minutes and try again", "INFO")
            else:
                print_status(f"Kaggle API error: {error_msg}", "FAIL")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("Kaggle API call timed out", "FAIL")
        return False
    except Exception as e:
        print_status(f"Error testing Kaggle API: {e}", "FAIL")
        return False


def main():
    """Main verification function."""
    print("🔍 Kaggle CLI Verification")
    print("=" * 40)
    
    checks = [
        ("Kaggle CLI Installation", check_kaggle_cli),
        ("Kaggle Credentials", check_kaggle_credentials),
        ("Kaggle API Connection", check_kaggle_api),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        if check_func():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print_status("🎉 All checks passed! Kaggle is ready to use.", "PASS")
        print_status("You can now proceed with data download.", "INFO")
        return 0
    else:
        print_status("❌ Some checks failed. Please fix the issues above.", "FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
