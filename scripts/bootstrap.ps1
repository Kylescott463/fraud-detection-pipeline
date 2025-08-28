# Fraud Detection Pipeline - Bootstrap Script (Windows PowerShell)
# Sets up Python environment and Kaggle CLI

param(
    [switch]$Force
)

Write-Host "🚀 Fraud Detection Pipeline - Bootstrap Script" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if ($isAdmin) {
    Write-Host "⚠️  Running as administrator - switching to user mode" -ForegroundColor Yellow
}

# Check if Python is available
$pythonCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
    Write-Host "✅ Found Python launcher (py)" -ForegroundColor Green
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
    Write-Host "✅ Found Python (python)" -ForegroundColor Green
} else {
    Write-Host "❌ Python is not installed" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 To install Python on Windows:" -ForegroundColor Yellow
    Write-Host "   1. Download from https://www.python.org/downloads/"
    Write-Host "   2. Run installer and check 'Add Python to PATH'"
    Write-Host "   3. Restart PowerShell and re-run this script"
    Write-Host ""
    Write-Host "   Or use winget: winget install Python.Python.3.11"
    exit 1
}

# Ensure pip is available
Write-Host "📦 Ensuring pip is available..." -ForegroundColor Cyan
try {
    if ($pythonCmd -eq "py") {
        py -m ensurepip --upgrade --user
    } else {
        python -m ensurepip --upgrade --user
    }
    Write-Host "✅ pip is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to ensure pip is available" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

# Install Kaggle CLI
Write-Host "🔧 Installing Kaggle CLI..." -ForegroundColor Cyan
try {
    if ($pythonCmd -eq "py") {
        py -m pip install --user --upgrade kaggle
    } else {
        python -m pip install --user --upgrade kaggle
    }
    Write-Host "✅ Kaggle CLI installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install Kaggle CLI" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

# Get user Scripts directory
Write-Host "📁 Locating user Scripts directory..." -ForegroundColor Cyan
try {
    if ($pythonCmd -eq "py") {
        $userBase = py -c "import site; print(site.USER_BASE)"
    } else {
        $userBase = python -c "import site; print(site.USER_BASE)"
    }
    $scriptsDir = Join-Path $userBase "Scripts"
    Write-Host "📁 User Scripts directory: $scriptsDir" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to locate user Scripts directory" -ForegroundColor Red
    exit 1
}

# Check if kaggle is in PATH
$kaggleInPath = Get-Command kaggle -ErrorAction SilentlyContinue
if ($kaggleInPath) {
    Write-Host "✅ Kaggle CLI found in PATH" -ForegroundColor Green
} else {
    Write-Host "⚠️  Kaggle CLI not found in PATH" -ForegroundColor Yellow
    
    # Check if it's in the user Scripts directory
    $kaggleExe = Join-Path $scriptsDir "kaggle.exe"
    if (Test-Path $kaggleExe) {
        Write-Host "✅ Kaggle CLI found in $scriptsDir" -ForegroundColor Green
        Write-Host ""
        Write-Host "💡 To add to PATH permanently:" -ForegroundColor Yellow
        Write-Host "   1. Open System Properties > Environment Variables"
        Write-Host "   2. Add '$scriptsDir' to your user PATH"
        Write-Host "   3. Restart PowerShell"
        Write-Host ""
        Write-Host "   Or run this command to add to current session:" -ForegroundColor Yellow
        Write-Host "   `$env:PATH += `";$scriptsDir`""
    else
        Write-Host "❌ Kaggle CLI not found in $scriptsDir" -ForegroundColor Red
        exit 1
    }
}

# Set up Kaggle credentials
Write-Host "🔐 Setting up Kaggle credentials..." -ForegroundColor Cyan

# Create .kaggle directory
$kaggleDir = Join-Path $env:USERPROFILE ".kaggle"
if (!(Test-Path $kaggleDir)) {
    New-Item -ItemType Directory -Path $kaggleDir -Force | Out-Null
    Write-Host "✅ Created .kaggle directory" -ForegroundColor Green
}

# Check if kaggle.json exists in project root
$kaggleJson = Join-Path (Get-Location) "kaggle.json"
$kaggleJsonDest = Join-Path $kaggleDir "kaggle.json"

if (Test-Path $kaggleJson) {
    Write-Host "📄 Found kaggle.json in project root" -ForegroundColor Green
    Move-Item -Path $kaggleJson -Destination $kaggleJsonDest -Force
    Write-Host "✅ Moved kaggle.json to $kaggleJsonDest" -ForegroundColor Green
    
    # Set restrictive permissions (Windows equivalent of chmod 600)
    try {
        $acl = Get-Acl $kaggleJsonDest
        $acl.SetAccessRuleProtection($true, $false)
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule($env:USERNAME, "FullControl", "Allow")
        $acl.AddAccessRule($rule)
        Set-Acl -Path $kaggleJsonDest -AclObject $acl
        Write-Host "✅ Set restrictive permissions on kaggle.json" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Could not set restrictive permissions (this is okay)" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ kaggle.json not found in project root" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Please:" -ForegroundColor Yellow
    Write-Host "   1. Download kaggle.json from your Kaggle account settings"
    Write-Host "   2. Place it in the project root directory"
    Write-Host "   3. Re-run this script"
    Write-Host ""
    Write-Host "   See kaggle.json.example for the expected format" -ForegroundColor Yellow
    Write-Host "   Or manually create $kaggleJsonDest with your credentials"
    exit 1
}

Write-Host ""
Write-Host "🎉 Bootstrap completed successfully!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "✅ Python installed" -ForegroundColor Green
Write-Host "✅ pip available" -ForegroundColor Green
Write-Host "✅ Kaggle CLI installed" -ForegroundColor Green
Write-Host "✅ Kaggle credentials configured" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. If PATH was updated, restart PowerShell"
Write-Host "2. Run: python scripts/verify_kaggle.py"
