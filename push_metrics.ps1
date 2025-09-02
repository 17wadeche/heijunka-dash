$srcXlsx = "C:\Users\wadec8\OneDrive - Medtronic PLC\metrics_aggregate.xlsx"
$srcCsv  = "C:\Users\wadec8\OneDrive - Medtronic PLC\metrics_aggregate.csv"
$repoDir = "C:\heijunka-data"
$dstCsv  = Join-Path $repoDir "metrics_aggregate.csv"
New-Item -ItemType Directory -Path $repoDir -Force | Out-Null
if (Test-Path $srcCsv) {
    Copy-Item $srcCsv $dstCsv -Force
} elseif (Test-Path $srcXlsx) {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $wb = $excel.Workbooks.Open($srcXlsx)
    try {
        $wb.Worksheets("All Metrics").SaveAs($dstCsv, 6) # 6 = xlCSV
    } finally {
        $wb.Close($false)
        $excel.Quit()
    }
} else {
    Write-Host "No source metrics found."
    exit 0
}
Set-Location $repoDir
git add metrics_aggregate.csv
if ((git status --porcelain) -ne "") {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm"
    git commit -m "Update metrics CSV ($ts)"
    git branch -M main 2>$null
    git push -u origin main
} else {
    Write-Host "No changes to commit."
}
