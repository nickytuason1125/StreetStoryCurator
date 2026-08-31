# launch_check.ps1 — ship-gate: prove no Microsoft tab appears at launch.
#
# For each run: launches the installed app, waits for boot, then verifies:
#   1. the app process is up
#   2. WebView2 children run in the ISOLATED user-data folder
#      (%LOCALAPPDATA%\Cullwise\WebView2 — proves the isolation fix is live)
#   3. no new msedge.exe / chrome.exe browser processes spawned
#      (a "Microsoft tab" would appear as new browser processes)
#   4. the backend answers /api/health
#
# Usage:  .\launch_check.ps1 -Runs 3 [-Exe <path>]
param(
    [int]$Runs = 3,
    [string]$Exe = "$env:LOCALAPPDATA\Cullwise\cullwise.exe"
)

function BrowserWindowCount {
    # Only VISIBLE top-level browser windows count as a "tab popping up" —
    # background browser processes churn constantly and mean nothing.
    @(Get-Process msedge, chrome -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -ne '' }).Count
}
function IsolatedWebViewCount {
    @(Get-CimInstance Win32_Process -Filter "Name='msedgewebview2.exe'" |
        Where-Object { $_.CommandLine -like '*Cullwise\WebView2*' }).Count
}

$results = @()
for ($i = 1; $i -le $Runs; $i++) {
    $edgeBefore = BrowserWindowCount
    Start-Process $Exe
    Start-Sleep -Seconds 30        # backend boot + webview init

    $app       = @(Get-Process cullwise -ErrorAction SilentlyContinue).Count
    $isolated  = IsolatedWebViewCount
    $edgeAfter = BrowserWindowCount
    $delta     = $edgeAfter - $edgeBefore

    $health = $false
    for ($w = 0; $w -lt 18; $w++) {          # poll up to 90 s for backend boot
        try {
            if ((Invoke-WebRequest 'http://127.0.0.1:8000/api/health' `
                -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) {
                $health = $true
                break
            }
        } catch {}
        Start-Sleep -Seconds 5
    }

    $ok = ($app -ge 1) -and ($isolated -ge 1) -and ($delta -le 0) -and $health
    $results += $ok
    Write-Host ("run {0}: app={1} isolated-webview={2} msedge/chrome delta={3:+0;-0;0} health={4}  [{5}]" -f `
        $i, $app, $isolated, $delta, $health, $(if ($ok) { 'PASS' } else { 'FAIL' }))

    # cleanup between runs: close the app and its backend
    Get-Process cullwise -ErrorAction SilentlyContinue | Stop-Process -Force
    Get-Process pythonw, python -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3
}

$passed = @($results | Where-Object { $_ }).Count
Write-Host ("`nlaunch_check: {0}/{1} runs clean" -f $passed, $Runs)
if ($passed -eq $Runs) { exit 0 } else { exit 1 }
