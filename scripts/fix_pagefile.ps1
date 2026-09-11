# Fix the pagefile: fixed size, no auto-growth lag.
# Root cause this kills: auto-managed pagefile starts small after boot and
# expands lazily. The cull's model-load spikes commit by 10+ GB in seconds;
# when expansion lags, allocations fail ("paging file is too small" /
# bad_alloc / CUDA OOM) and workers die. A fixed 32-48 GB pagefile removes
# the lag entirely. Requires admin. Takes effect after reboot.

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "NOT ADMIN - relaunch me elevated"
    exit 1
}

$cs = Get-CimInstance Win32_ComputerSystem
if ($cs.AutomaticManagedPagefile) {
    Set-CimInstance -InputObject $cs -Property @{AutomaticManagedPagefile = $false}
    Write-Host "auto-managed pagefile disabled"
}

# drop any explicit setting, then create the fixed one
Get-CimInstance Win32_PageFileSetting -ErrorAction SilentlyContinue | Remove-CimInstance
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management" -Name "PagingFiles" -Value "C:\pagefile.sys 32768 49152" -Type MultiString
Write-Host "pagefile set: C:\ initial 32 GB, max 48 GB (fixed)"

Get-CimInstance Win32_PageFileSetting | ForEach-Object { Write-Host "setting: $($_.Name) init=$([math]::Round($_.InitialSize/1MB))GB max=$([math]::Round($_.MaximumSize/1MB))GB" }
Write-Host "REBOOT REQUIRED for it to take effect"