#!/bin/bash

# Fraud Detection Pipeline - Bootstrap Script (macOS/Linux)
# Sets up Python environment and Kaggle CLI

set -e  # Exit on any error

echo "🚀 Fraud Detection Pipeline - Bootstrap Script"
echo "=============================================="

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

echo "📋 Detected OS: $OS"

# Check if python3 exists
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo ""
    if [[ "$OS" == "macOS" ]]; then
        echo "💡 To install Python 3 on macOS:"
        echo "   1. Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "   2. Install Python: brew install python"
    else
        echo "💡 To install Python 3 on Linux:"
        echo "   Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip"
        echo "   CentOS/RHEL: sudo yum install python3 python3-pip"
    fi
    echo ""
    echo "After installing Python 3, re-run this script."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Ensure pip is available
echo "📦 Ensuring pip is available..."
python3 -m ensurepip --upgrade --user

# Install Kaggle CLI
echo "🔧 Installing Kaggle CLI..."
python3 -m pip install --user --upgrade kaggle

# Get user base directory and add to PATH if needed
USERBASE=$(python3 -m site --user-base)
BIN="$USERBASE/bin"

echo "📁 User base directory: $USERBASE"
echo "📁 Bin directory: $BIN"

# Check if kaggle is in PATH
if ! command -v kaggle &> /dev/null; then
    echo "⚠️  Kaggle CLI not found in PATH"
    
    # Check if it's in the user bin directory
    if [[ -f "$BIN/kaggle" ]]; then
        echo "✅ Kaggle CLI found in $BIN"
        
        # Determine shell and update PATH
        SHELL_RC=""
        if [[ "$SHELL" == *"zsh"* ]]; then
            SHELL_RC="$HOME/.zshrc"
        else
            SHELL_RC="$HOME/.bashrc"
        fi
        
        # Add to PATH if not already there
        if ! grep -q "$BIN" "$SHELL_RC" 2>/dev/null; then
            echo "export PATH=\"\$PATH:$BIN\"" >> "$SHELL_RC"
            echo "✅ Added $BIN to PATH in $SHELL_RC"
            echo ""
            echo "🔄 Please restart your shell or run:"
            echo "   source $SHELL_RC"
        else
            echo "✅ $BIN already in PATH in $SHELL_RC"
        fi
    else
        echo "❌ Kaggle CLI not found in $BIN"
        exit 1
    fi
else
    echo "✅ Kaggle CLI found in PATH"
fi

# Set up Kaggle credentials
echo "🔐 Setting up Kaggle credentials..."

# Create .kaggle directory
mkdir -p "$HOME/.kaggle"

# Check if kaggle.json exists in project root
if [[ -f "kaggle.json" ]]; then
    echo "📄 Found kaggle.json in project root"
    mv kaggle.json "$HOME/.kaggle/kaggle.json"
    chmod 600 "$HOME/.kaggle/kaggle.json"
    echo "✅ Moved kaggle.json to $HOME/.kaggle/kaggle.json"
    echo "✅ Set permissions to 600"
else
    echo "❌ kaggle.json not found in project root"
    echo ""
    echo "💡 Please:"
    echo "   1. Download kaggle.json from your Kaggle account settings"
    echo "   2. Place it in the project root directory"
    echo "   3. Re-run this script"
    echo ""
    echo "   See kaggle.json.example for the expected format"
    echo "   Or manually create $HOME/.kaggle/kaggle.json with your credentials"
    exit 1
fi

echo ""
echo "🎉 Bootstrap completed successfully!"
echo "====================================="
echo "✅ Python 3 installed"
echo "✅ pip available"
echo "✅ Kaggle CLI installed"
echo "✅ Kaggle credentials configured"
echo ""
echo "Next steps:"
echo "1. If PATH was updated, restart your shell or run: source $SHELL_RC"
echo "2. Run: make kaggle-test"
