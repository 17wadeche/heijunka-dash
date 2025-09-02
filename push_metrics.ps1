# C:\heijunka-data\push_metrics.ps1
$ErrorActionPreference = "Stop"
$repo = "C:\heijunka-data"
$csvSource = "C:\Users\wadec8\OneDrive - Medtronic PLC\metrics_aggregate.csv" # produced by your collector
$csvDest   = Join-Path $repo "metrics_aggregate.csv"r
git config --global --add safe.directory $repo 2>$null
if (-not (Test-Path $repo)) { throw "Repo folder not found: $repo" }
Set-Location $repo
git switch main 2>$null | Out-Null
git pull --rebase --autostash origin main | Out-Null
if (-not (Test-Path $csvSource)) { throw "Source CSV not found: $csvSource" }
Copy-Item $csvSource $csvDest -Force
git add metrics_aggregate.csv
if (git diff --cached --quiet) {
    Write-Host "No changes to commit."
} else {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm"
    git commit -m "Update metrics CSV ($ts)"
    git push origin main
    Write-Host "Pushed updated CSV to GitHub."
}
