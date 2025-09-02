# C:\heijunka-data\push_metrics.ps1
$ErrorActionPreference = "Stop"
$repoDir = "C:\heijunka"
$srcCsv  = "C:\Users\wadec8\OneDrive - Medtronic PLC\metrics_aggregate.csv"
$dstCsv  = Join-Path $repoDir "metrics_aggregate.csv"
New-Item -ItemType Directory -Path $repoDir -Force | Out-Null
Set-Location $repoDir
git fetch origin
git checkout -B main 2>$null
git pull --rebase --autostash --allow-unrelated-histories origin main
if (Test-Path $srcCsv) {
    Copy-Item $srcCsv $dstCsv -Force
} else {
    Write-Host "WARNING: $srcCsv not found; nothing to publish."
}
git add metrics_aggregate.csv
if ((git status --porcelain) -ne "") {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm"
    git commit -m "Update metrics CSV ($ts)"
} else {
    Write-Host "No local changes to commit."
}
git pull --rebase --autostash origin main
git push origin main
