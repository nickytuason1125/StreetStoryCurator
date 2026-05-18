# kill_ports.ps1 — Force-kill all Python/Uvicorn/Node zombie processes on ports 8000, 3000, 5173.
# Usage: Right-click → "Run with PowerShell"  OR  pwsh -File kill_ports.ps1

$PORTS = @(8000, 3000, 5173)
$PROCS = @("python", "uvicorn", "node")

Write-Host ""
Write-Host "=== STEP 1: Kill by process name ==="
foreach ($name in $PROCS) {
    $hits = Get-Process -Name $name -ErrorAction SilentlyContinue
    if ($hits) {
        $hits | Stop-Process -Force
        Write-Host "  Killed $($hits.Count) x '$name' process(es)"
    } else {
        Write-Host "  No '$name' processes found"
    }
}

Write-Host ""
Write-Host "=== STEP 2: Kill by port (netstat fallback) ==="
foreach ($port in $PORTS) {
    $lines = netstat -ano | Select-String ":$port\s"
    if (-not $lines) {
        Write-Host "  Port $port — nothing listening"
        continue
    }
    $pids = $lines |
        ForEach-Object { ($_ -split '\s+')[-1] } |
        Select-Object -Unique |
        Where-Object { $_ -match '^\d+$' -and [int]$_ -ne 0 }

    foreach ($pid in $pids) {
        $proc = Get-Process -Id ([int]$pid) -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "  Port $port  → PID $pid ($($proc.ProcessName)) — killing"
            Stop-Process -Id ([int]$pid) -Force -ErrorAction SilentlyContinue
        } else {
            Write-Host "  Port $port  → PID $pid already gone"
        }
    }
}

Write-Host ""
Write-Host "=== STEP 3: Verify ports are free ==="
Start-Sleep -Milliseconds 500
foreach ($port in $PORTS) {
    $still = netstat -ano | Select-String ":$port\s"
    if ($still) {
        Write-Host "  Port $port  STILL IN USE — check manually"
    } else {
        Write-Host "  Port $port  clear"
    }
}

Write-Host ""
Write-Host "Done. Safe to run dev.ps1"
Write-Host ""
