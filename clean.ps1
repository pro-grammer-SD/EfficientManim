# ===============================
# Efficient Python Cleanup Script (BEAST MODE)
# ===============================

$Path = "."

function Invoke-SoftCommand($cmd) {
    Write-Host "→ $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠ Command reported issues but continuing..." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "===================================" -ForegroundColor Yellow
Write-Host " Python Codebase Cleanup - BEAST "
Write-Host "===================================" -ForegroundColor Yellow
Write-Host ""

# Ensure Python exists
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found. Install Python first." -ForegroundColor Red
    exit 1
}

# Upgrade pip
Invoke-SoftCommand "python -m pip install --upgrade pip"

# Ensure required tools installed
$packages = @("ruff", "mypy", "bandit")

foreach ($pkg in $packages) {
    python -c "import $pkg" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing $pkg..." -ForegroundColor Yellow
        Invoke-SoftCommand "python -m pip install --upgrade $pkg"
    }
}

Write-Host ""
Write-Host "=== PHASE 1: Ruff Deep Auto-Fix ===" -ForegroundColor Green

Invoke-SoftCommand "python -m ruff format $Path"
Invoke-SoftCommand "python -m ruff check $Path --fix --unsafe-fixes"
Invoke-SoftCommand "python -m ruff check $Path --fix --unsafe-fixes"

Write-Host ""
Write-Host "=== PHASE 2: MyPy Strict (Parallel) ===" -ForegroundColor Green
Invoke-SoftCommand "python -m mypy $Path --strict"

Write-Host ""
Write-Host "=== PHASE 3: Bandit Security Scan (Parallel) ===" -ForegroundColor Green
Invoke-SoftCommand "python -m bandit -r $Path -n 4"

Write-Host ""
Write-Host "===================================" -ForegroundColor Green
Write-Host " Cleanup Completed - Beast Mode   "
Write-Host "===================================" -ForegroundColor Green
Write-Host ""
Write-Host "Note: Remaining issues (if any) require manual fixes." -ForegroundColor DarkYellow
Write-Host ""
