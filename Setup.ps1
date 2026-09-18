Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

# ── Palette ───────────────────────────────────────────────────────
$bg     = [System.Drawing.Color]::FromArgb(13,  13,  17 )
$surf   = [System.Drawing.Color]::FromArgb(17,  17,  21 )
$surf2  = [System.Drawing.Color]::FromArgb(26,  26,  33 )
$border = [System.Drawing.Color]::FromArgb(36,  36,  48 )
$accent = [System.Drawing.Color]::FromArgb(82,  130, 255)
$acLow  = [System.Drawing.Color]::FromArgb(82,  130, 255)
$txt    = [System.Drawing.Color]::FromArgb(232, 232, 237)
$txt2   = [System.Drawing.Color]::FromArgb(138, 138, 154)
$txt3   = [System.Drawing.Color]::FromArgb(68,  68,  90 )
$green  = [System.Drawing.Color]::FromArgb(75,  185, 105)
$amber  = [System.Drawing.Color]::FromArgb(210, 150, 55 )
$red    = [System.Drawing.Color]::FromArgb(215, 70,  70 )

function MakeFont($size, $bold=$false) {
    $style = if ($bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    New-Object System.Drawing.Font("Segoe UI", $size, $style)
}

# ── Main window ───────────────────────────────────────────────────
$form = New-Object System.Windows.Forms.Form
$form.Text            = "FirstCut — Setup"
$form.ClientSize      = New-Object System.Drawing.Size(500, 440)
$form.StartPosition   = "CenterScreen"
$form.BackColor       = $bg
$form.FormBorderStyle = "FixedSingle"
$form.MaximizeBox     = $false
$form.MinimizeBox     = $false
$form.Font            = MakeFont 10

# ── Sidebar ───────────────────────────────────────────────────────
$sidebar = New-Object System.Windows.Forms.Panel
$sidebar.Size     = New-Object System.Drawing.Size(148, 440)
$sidebar.Location = New-Object System.Drawing.Point(0, 0)
$sidebar.BackColor = $surf
$form.Controls.Add($sidebar)

# Aperture glyph (unicode ⦿ as stand-in)
$logo = New-Object System.Windows.Forms.Label
$logo.Text      = "◎"
$logo.Font      = MakeFont 38
$logo.ForeColor = $amber
$logo.AutoSize  = $false
$logo.Size      = New-Object System.Drawing.Size(148, 60)
$logo.Location  = New-Object System.Drawing.Point(0, 36)
$logo.TextAlign = "MiddleCenter"
$sidebar.Controls.Add($logo)

$appName = New-Object System.Windows.Forms.Label
$appName.Text      = "Street`nStory`nCurator"
$appName.Font      = MakeFont 11 $true
$appName.ForeColor = $txt
$appName.AutoSize  = $false
$appName.Size      = New-Object System.Drawing.Size(148, 72)
$appName.Location  = New-Object System.Drawing.Point(0, 102)
$appName.TextAlign = "MiddleCenter"
$sidebar.Controls.Add($appName)

# Step indicators
$stepLabels = @("Welcome", "Requirements", "Installing", "Complete")
$stepDots   = @()
$stepTexts  = @()
for ($i = 0; $i -lt 4; $i++) {
    $dot = New-Object System.Windows.Forms.Label
    $dot.Size      = New-Object System.Drawing.Size(8, 8)
    $dot.Location  = New-Object System.Drawing.Point(20, 230 + $i * 34)
    $dot.BackColor = $txt3
    $dot.Text      = ""
    $sidebar.Controls.Add($dot)
    $stepDots += $dot

    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Text      = $stepLabels[$i]
    $lbl.Font      = MakeFont 9
    $lbl.ForeColor = $txt3
    $lbl.AutoSize  = $false
    $lbl.Size      = New-Object System.Drawing.Size(108, 18)
    $lbl.Location  = New-Object System.Drawing.Point(36, 226 + $i * 34)
    $sidebar.Controls.Add($lbl)
    $stepTexts += $lbl
}

function Set-Step($idx) {
    for ($i = 0; $i -lt 4; $i++) {
        if ($i -eq $idx) {
            $stepDots[$i].BackColor  = $amber
            $stepTexts[$i].ForeColor = $txt
            $stepTexts[$i].Font      = MakeFont 9 $true
        } elseif ($i -lt $idx) {
            $stepDots[$i].BackColor  = $green
            $stepTexts[$i].ForeColor = $txt2
            $stepTexts[$i].Font      = MakeFont 9
        } else {
            $stepDots[$i].BackColor  = $txt3
            $stepTexts[$i].ForeColor = $txt3
            $stepTexts[$i].Font      = MakeFont 9
        }
    }
}

# ── Content area ─────────────────────────────────────────────────
$content = New-Object System.Windows.Forms.Panel
$content.Size      = New-Object System.Drawing.Size(352, 380)
$content.Location  = New-Object System.Drawing.Point(148, 0)
$content.BackColor = $bg
$form.Controls.Add($content)

# ── Bottom bar ────────────────────────────────────────────────────
$bar = New-Object System.Windows.Forms.Panel
$bar.Size      = New-Object System.Drawing.Size(352, 60)
$bar.Location  = New-Object System.Drawing.Point(148, 380)
$bar.BackColor = $surf
$form.Controls.Add($bar)

function Make-Button($text, $x, $primary=$false) {
    $b = New-Object System.Windows.Forms.Button
    $b.Text      = $text
    $b.Size      = New-Object System.Drawing.Size(100, 32)
    $b.Location  = New-Object System.Drawing.Point($x, 14)
    $b.FlatStyle = "Flat"
    $b.Font      = MakeFont 10 $primary
    $b.Cursor    = "Hand"
    if ($primary) {
        $b.BackColor = $accent
        $b.ForeColor = [System.Drawing.Color]::White
        $b.FlatAppearance.BorderSize = 0
    } else {
        $b.BackColor = $surf2
        $b.ForeColor = $txt2
        $b.FlatAppearance.BorderColor = $border
        $b.FlatAppearance.BorderSize  = 1
    }
    $bar.Controls.Add($b)
    return $b
}

$btnBack   = Make-Button "← Back"   16
$btnNext   = Make-Button "Next →"   238 $true
$btnCancel = Make-Button "Cancel"   126
$btnCancel.ForeColor = $txt3
$btnCancel.Add_Click({ $form.Close() })

# ══════════════════════════════════════════════════════════════════
# PAGE HELPERS
# ══════════════════════════════════════════════════════════════════

function Clear-Content { $content.Controls.Clear() }

function Add-Title($text) {
    $l = New-Object System.Windows.Forms.Label
    $l.Text      = $text
    $l.Font      = MakeFont 16 $true
    $l.ForeColor = $txt
    $l.AutoSize  = $false
    $l.Size      = New-Object System.Drawing.Size(316, 38)
    $l.Location  = New-Object System.Drawing.Point(18, 24)
    $content.Controls.Add($l)
}

function Add-Sub($text, $y) {
    $l = New-Object System.Windows.Forms.Label
    $l.Text      = $text
    $l.Font      = MakeFont 9.5
    $l.ForeColor = $txt2
    $l.AutoSize  = $false
    $l.Size      = New-Object System.Drawing.Size(316, 0)
    $l.MaximumSize = New-Object System.Drawing.Size(316, 200)
    $l.AutoSize  = $true
    $l.Location  = New-Object System.Drawing.Point(18, $y)
    $content.Controls.Add($l)
    return $l
}

function Add-Divider($y) {
    $p = New-Object System.Windows.Forms.Panel
    $p.Size      = New-Object System.Drawing.Size(316, 1)
    $p.Location  = New-Object System.Drawing.Point(18, $y)
    $p.BackColor = $border
    $content.Controls.Add($p)
}

# ══════════════════════════════════════════════════════════════════
# PAGE 1: WELCOME
# ══════════════════════════════════════════════════════════════════

function Show-Welcome {
    Set-Step 0
    Clear-Content
    $btnBack.Enabled = $false
    $btnBack.Visible = $true
    $btnNext.Text    = "Next →"
    $btnNext.Enabled = $true
    $btnCancel.Visible = $true

    Add-Title "Welcome"
    Add-Divider 68
    Add-Sub "This wizard will install FirstCut on your computer." 84
    Add-Sub "FirstCut analyses and grades your street photography, builds sequenced stories, and gives editorial feedback — all running locally on your machine." 120
    Add-Divider 210
    Add-Sub "Install location:" 226
    $pathBox = New-Object System.Windows.Forms.TextBox
    $pathBox.Text      = $ROOT
    $pathBox.Font      = New-Object System.Drawing.Font("Consolas", 8)
    $pathBox.ForeColor = $txt2
    $pathBox.BackColor = $surf2
    $pathBox.BorderStyle = "None"
    $pathBox.ReadOnly  = $true
    $pathBox.Size      = New-Object System.Drawing.Size(316, 18)
    $pathBox.Location  = New-Object System.Drawing.Point(18, 252)
    $content.Controls.Add($pathBox)
    Add-Divider 276

    Add-Sub "Disk space required: ~3 GB (libraries + models)" 292
    Add-Sub "Time for first launch: 5–10 minutes (downloads once)" 316

    # Repair mode (2026-09-17): an existing install offers repair, not a blind
    # reinstall — the wizard skips completed steps and verifies the rest.
    if (Test-Path (Join-Path $ROOT "venv\.setup_ok")) {
        $rep = New-Object System.Windows.Forms.CheckBox
        $rep.Text      = "Existing install detected — repair mode (skip what works, re-verify the rest)"
        $rep.Font      = MakeFont 8.5
        $rep.ForeColor = $txt2
        $rep.BackColor = $bg
        $rep.Checked   = $false
        $rep.AutoSize  = $true
        $rep.Location  = New-Object System.Drawing.Point(18, 344)
        $content.Controls.Add($rep)
        $script:lastRepairBox = $rep
    } else {
        $script:repairMode = $false
    }
}

# ══════════════════════════════════════════════════════════════════
# PAGE 2: REQUIREMENTS
# ══════════════════════════════════════════════════════════════════

$script:pyOk   = $false
$script:nodeOk = $false

function Check-Python {
    try {
        $v = & python --version 2>&1
        if ($v -match "Python (\d+)\.(\d+)") {
            $maj = [int]$Matches[1]; $min = [int]$Matches[2]
            return ($maj -gt 3 -or ($maj -eq 3 -and $min -ge 10)), $v
        }
    } catch {}
    return $false, ""
}

function Check-Node {
    try {
        $v = & node --version 2>&1
        if ($v -match "v(\d+)") { return ([int]$Matches[1] -ge 16), $v }
    } catch {}
    return $false, ""
}

# ── Platform preflight (2026-09-17) ──────────────────────────────────────────
# Mirrors src/machine_profile.detect()'s verdict so a doomed install STOPS at
# the wizard instead of failing 40 minutes into pip downloads. ARM64 Windows
# has no torch/llama wheels today (the app's own MachineProfile says "an ARM
# build is planned"), and 32-bit/pre-Win10 never worked — say so kindly, now.
function Check-Platform {
    try {
        $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
        $arch = $cpu.Architecture   # 9 = x64, 12 = ARM64, 0/5/6 = x86 family
        if ($arch -eq 12) {
            return $false, "ARM64 Windows — FirstCut needs torch and llama wheels that don't exist for Windows-on-ARM yet. An ARM build is planned. (x64 emulation is not supported for the GPU engines this app needs.)"
        }
        if ($arch -ne 9) {
            return $false, "32-bit Windows — FirstCut needs 64-bit Windows 10/11."
        }
        $osv = [System.Environment]::OSVersion.Version
        if ($osv.Major -lt 10) {
            return $false, "Windows 10 or newer is required."
        }
        return $true, "64-bit Windows $($osv.Major).$($osv.Minor) — supported"
    } catch {
        return $true, "platform check skipped ($_) — continuing"
    }
}

# ── Python provisioning (2026-09-17) ─────────────────────────────────────────
# PATH roulette was the #1 broken-install cause: the WindowsApps "python.exe"
# store shim opens the Store and exits (venv silently fails), and wrong-arch or
# too-old interpreters failed later with confusing pip errors. Resolution
# order: the wizard's OWN project-local interpreter (installed by this wizard)
# first, then PATH. Version >= 3.10, 64-bit only.
$script:pythonExe = $null

function Test-SuitablePython($exe) {
    try {
        $v = & $exe -c "import sys,platform;print('%d.%d %s'%(sys.version_info[0],sys.version_info[1],platform.machine().upper()))" 2>&1
        if ($LASTEXITCODE -ne 0) { return $false }
        if ($v -match "(\d+)\.(\d+) (AMD64|X86_64|X64)") {
            $maj = [int]$Matches[1]; $min = [int]$Matches[2]
            if ($maj -ne 3) { return $false, $v }
            # Pinned-wheel window: torch 2.5.1 / llama 0.3.23 publish wheels for
            # 3.10-3.12. A newer interpreter (3.13/3.14) "passes" the version
            # check and then fails 40 minutes into pip with "no matching wheel"
            # — reject it here so the auto-installer provisions 3.12 instead.
            if ($min -lt 10 -or $min -gt 12) { return $false, $v }
            return $true, $v
        }
        # matched version but wrong arch (ARM64 python, 32-bit)
        return $false, $v
    } catch { return $false, "" }
}

function Get-PythonInfo {
    # 1) The wizard's project-local interpreter (installed by this wizard).
    $local = Join-Path $ROOT "runtime\python\python.exe"
    if (Test-Path $local) {
        $ok, $v = Test-SuitablePython $local
        if ($ok) { return $true, $v, $local }
    }
    # 2) PATH python — but reject the WindowsApps store shim.
    try {
        $shim = Get-Command python -ErrorAction SilentlyContinue
        if ($shim -and $shim.Source -match '\\WindowsApps\\') {
            # Store shim: fake python that opens the Store. Treat as missing.
        } elseif ($shim) {
            $ok, $v = Test-SuitablePython "python"
            if ($ok) { return $true, $v, "python" }
        }
    } catch {}
    return $false, "", $null
}

function Install-PythonLocal {
    <# Download the official python.org x64 installer and install it into
    $ROOT\runtime\python — project-local, no PATH changes, no admin. #>
    Set-Progress 4 "Downloading Python (one-time, ~25 MB)..."
    $url = "https://www.python.org/ftp/python/3.12.8/amd64/python-3.12.8-amd64.exe"
    $dst = "$env:TEMP\python-3.12.8-amd64.exe"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        (New-Object Net.WebClient).DownloadFile($url, $dst)
    } catch {
        Log "  ✗ Python download failed: $_"
        return $false
    }
    Set-Progress 6 "Installing Python into this project (no admin needed)..."
    $target = Join-Path $ROOT "runtime\python"
    $p = Start-Process -FilePath $dst -Wait -PassThru -ArgumentList @(
        "/quiet", "InstallAllUsers=0", "PrependPath=0",
        "TargetDir=$target", "Include_launcher=0", "Include_test=0", "Shortcuts=0"
    )
    if ($p.ExitCode -ne 0 -or -not (Test-Path (Join-Path $target "python.exe"))) {
        Log "  ✗ Silent Python install failed (exit $($p.ExitCode))."
        return $false
    }
    Log "  ✓ Python installed to runtime\python"
    return $true
}

function Make-ReqRow($label, $y, $ok, $version, $url) {
    $dot = New-Object System.Windows.Forms.Panel
    $dot.Size      = New-Object System.Drawing.Size(10, 10)
    $dot.Location  = New-Object System.Drawing.Point(18, $y + 5)
    $dot.BackColor = if ($ok) { $green } else { $red }
    $content.Controls.Add($dot)

    $name = New-Object System.Windows.Forms.Label
    $name.Text      = $label
    $name.Font      = MakeFont 10 $true
    $name.ForeColor = $txt
    $name.AutoSize  = $true
    $name.Location  = New-Object System.Drawing.Point(36, $y)
    $content.Controls.Add($name)

    $status = New-Object System.Windows.Forms.Label
    $status.Text      = if ($ok) { $version } else { "Not found" }
    $status.Font      = MakeFont 9
    $status.ForeColor = if ($ok) { $green } else { $red }
    $status.AutoSize  = $true
    $status.Location  = New-Object System.Drawing.Point(36, $y + 20)
    $content.Controls.Add($status)

    if (-not $ok) {
        $btn = New-Object System.Windows.Forms.Button
        $btn.Text      = "Download →"
        $btn.Size      = New-Object System.Drawing.Size(100, 26)
        $btn.Location  = New-Object System.Drawing.Point(214, $y + 4)
        $btn.FlatStyle = "Flat"
        $btn.BackColor = $surf2
        $btn.ForeColor = $accent
        $btn.Font      = MakeFont 9
        $btn.Cursor    = "Hand"
        $btn.FlatAppearance.BorderColor = $border
        $u = $url
        $btn.Add_Click({ Start-Process $u })
        $content.Controls.Add($btn)
    }
}

function Show-Requirements {
    Set-Step 1
    Clear-Content
    $btnBack.Enabled = $true
    $btnNext.Text    = "Next →"

    Add-Title "Requirements"
    Add-Divider 68

    # Platform gate FIRST — a doomed install must stop here, not in pip.
    $platOk, $platMsg = Check-Platform
    if (-not $platOk) {
        $stop = New-Object System.Windows.Forms.Label
        $stop.Text      = "✗  This machine can't run FirstCut"
        $stop.Font      = MakeFont 11 $true
        $stop.ForeColor = $red
        $stop.AutoSize  = $true
        $stop.Location  = New-Object System.Drawing.Point(18, 82)
        $content.Controls.Add($stop)

        $why = Add-Sub $platMsg 108
        $why.ForeColor = $txt2

        Add-Divider 200
        Add-Sub "Nothing was installed. The wizard closes when you click Cancel." 216
        Add-Sub "Check github.com/…/FirstCut for the ARM/macOS roadmap." 244

        $btnNext.Enabled = $false
        return
    }
    $platLbl = Add-Sub "✔  $platMsg" 82
    $platLbl.ForeColor = $green

    $pyResult   = Get-PythonInfo
    $nodeResult = Check-Node
    $script:pyOk   = $pyResult[0]
    $script:nodeOk = $nodeResult[0]
    $script:pythonExe = $pyResult[2]

    Make-ReqRow "Python 3.10 or newer (64-bit)" 118 $script:pyOk $pyResult[1] "https://www.python.org/downloads/"
    Add-Divider 178
    Make-ReqRow "Node.js 16 or newer"  202 $script:nodeOk $nodeResult[1] "https://nodejs.org"
    Add-Divider 262

    if ((-not $script:pyOk) -and $platOk) {
        # Python is the #1 broken-install cause — offer to fix it HERE instead
        # of sending the user to python.org and hoping they pick the right
        # build. Installs project-local (runtime\python), no PATH changes.
        $auto = New-Object System.Windows.Forms.Button
        $auto.Text      = "Install Python for me (recommended)"
        $auto.Size      = New-Object System.Drawing.Size(240, 30)
        $auto.Location  = New-Object System.Drawing.Point(18, 276)
        $auto.FlatStyle = "Flat"
        $auto.BackColor = $accent
        $auto.ForeColor = [System.Drawing.Color]::White
        $auto.Font      = MakeFont 9.5 $true
        $auto.Cursor    = "Hand"
        $auto.FlatAppearance.BorderSize = 0
        $auto.Add_Click({
            $auto.Enabled = $false
            $auto.Text    = "Installing Python…"
            [System.Windows.Forms.Application]::DoEvents()
            if (Install-PythonLocal) {
                Show-Requirements          # re-run the whole page
            } else {
                $auto.Text    = "Install failed — install from python.org"
                $auto.Enabled = $true
            }
        })
        $content.Controls.Add($auto)
        $btnNext.Enabled = $false
        return
    }

    if ($script:pyOk -and $script:nodeOk) {
        $ok = New-Object System.Windows.Forms.Label
        $ok.Text      = "✔  All requirements met — ready to install."
        $ok.Font      = MakeFont 9.5
        $ok.ForeColor = $green
        $ok.AutoSize  = $true
        $ok.Location  = New-Object System.Drawing.Point(18, 280)
        $content.Controls.Add($ok)
        $btnNext.Enabled = $true
    } else {
        $warn = New-Object System.Windows.Forms.Label
        $warn.Text      = "Install the missing programs above, then click Refresh."
        $warn.Font      = MakeFont 9.5
        $warn.ForeColor = $amber
        $warn.AutoSize  = $true
        $warn.Location  = New-Object System.Drawing.Point(18, 280)
        $content.Controls.Add($warn)
        $btnNext.Enabled = $false

        $refresh = New-Object System.Windows.Forms.Button
        $refresh.Text      = "↻  Refresh"
        $refresh.Size      = New-Object System.Drawing.Size(100, 28)
        $refresh.Location  = New-Object System.Drawing.Point(18, 316)
        $refresh.FlatStyle = "Flat"
        $refresh.BackColor = $surf2
        $refresh.ForeColor = $txt2
        $refresh.Font      = MakeFont 9.5
        $refresh.Cursor    = "Hand"
        $refresh.FlatAppearance.BorderColor = $border
        $refresh.Add_Click({ Show-Requirements })
        $content.Controls.Add($refresh)
    }
}

# ══════════════════════════════════════════════════════════════════
# PAGE 3: INSTALLING
# ══════════════════════════════════════════════════════════════════

$script:progressBar = $null
$script:logBox      = $null

function Log($msg) {
    $script:logBox.AppendText("$msg`r`n")
    $script:logBox.ScrollToCaret()
    [System.Windows.Forms.Application]::DoEvents()
}

function Set-Progress($pct, $msg) {
    $script:progressBar.Value = [Math]::Min($pct, 100)
    Log $msg
}

function Run-Cmd($exe, $args, $desc) {
    Log "  → $desc"
    $p = Start-Process -FilePath $exe -ArgumentList $args `
        -WorkingDirectory $ROOT -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "$env:TEMP\ssc_out.txt" `
        -RedirectStandardError  "$env:TEMP\ssc_err.txt"
    while (-not $p.HasExited) {
        [System.Windows.Forms.Application]::DoEvents()
        Start-Sleep -Milliseconds 200
    }
    if ($p.ExitCode -ne 0) {
        $err = Get-Content "$env:TEMP\ssc_err.txt" -Raw -ErrorAction SilentlyContinue
        Log "  ✗ Failed (exit $($p.ExitCode))"
        if ($err) { Log $err.Trim() }
        return $false
    }
    return $true
}

function Show-Installing {
    Set-Step 2
    Clear-Content
    $btnBack.Enabled  = $false
    $btnNext.Enabled  = $false
    $btnCancel.Visible = $false

    Add-Title "Installing..."
    Add-Divider 68

    $script:progressBar = New-Object System.Windows.Forms.ProgressBar
    $script:progressBar.Size     = New-Object System.Drawing.Size(316, 14)
    $script:progressBar.Location = New-Object System.Drawing.Point(18, 84)
    $script:progressBar.Style    = "Continuous"
    $script:progressBar.Minimum  = 0
    $script:progressBar.Maximum  = 100
    $script:progressBar.Value    = 0
    $content.Controls.Add($script:progressBar)

    $script:logBox = New-Object System.Windows.Forms.RichTextBox
    $script:logBox.Size           = New-Object System.Drawing.Size(316, 250)
    $script:logBox.Location       = New-Object System.Drawing.Point(18, 108)
    $script:logBox.BackColor      = $surf2
    $script:logBox.ForeColor      = $txt2
    $script:logBox.Font           = New-Object System.Drawing.Font("Consolas", 8.5)
    $script:logBox.ReadOnly       = $true
    $script:logBox.BorderStyle    = "None"
    $script:logBox.ScrollBars     = "Vertical"
    $content.Controls.Add($script:logBox)

    [System.Windows.Forms.Application]::DoEvents()

    $python  = if ($script:pythonExe) { $script:pythonExe } else { "python" }
    $pip     = Join-Path $ROOT "venv\Scripts\pip.exe"
    $pythonV = Join-Path $ROOT "venv\Scripts\python.exe"
    $npm     = "npm"
    $ok = $true
    # Repair mode: re-running the wizard on an existing install skips what
    # already exists instead of re-downloading everything.
    $repair = $script:repairMode

    # Step 1 — venv
    Set-Progress 5 "Creating Python environment..."
    if (-not (Test-Path (Join-Path $ROOT "venv\Scripts\python.exe"))) {
        $ok = Run-Cmd $python "-m venv `"$(Join-Path $ROOT 'venv')`"" "python -m venv"
    } else { Log "  ✓ Environment already exists" }

    # Step 2 — pip upgrade
    if ($ok) {
        Set-Progress 8 "Upgrading pip..."
        Run-Cmd $pythonV "-m pip install --upgrade pip --quiet" "pip upgrade" | Out-Null
    }

    # Step 3 — PyTorch (detect GPU, install matching wheel)
    if ($ok) {
        $hasCuda = $false
        try {
            $nvOut = & nvidia-smi 2>&1
            if ($LASTEXITCODE -eq 0) { $hasCuda = $true }
        } catch {}

        if ($hasCuda) {
            Set-Progress 12 "GPU detected — installing PyTorch with CUDA 12.1..."
            Log "  GPU detected — installing torch cu121 wheel"
            $ok = Run-Cmd $pip `
                "install `"torch==2.5.1`" `"torchvision==0.20.1`" --index-url https://download.pytorch.org/whl/cu121 --quiet" `
                "torch + torchvision (CUDA 12.1)"
        } else {
            Set-Progress 12 "No GPU — installing PyTorch (CPU)..."
            Log "  No NVIDIA GPU — installing CPU torch wheel"
            $ok = Run-Cmd $pip `
                "install `"torch==2.5.1`" `"torchvision==0.20.1`" --index-url https://download.pytorch.org/whl/cpu --quiet" `
                "torch + torchvision (CPU)"
        }

        # onnxruntime must match the machine too. requirements.txt pins the CPU
        # wheel as a floor so a bare `pip install -r` works, but installing that
        # on a GPU machine is NOT harmless: run_profile.onnx_enabled() still
        # returns True, ORT silently falls back to CPUExecutionProvider, and the
        # default Pro-tier encoder drops from ~1.2 s/img to 11-14 s/img with no
        # error anywhere. Only the installer knows which wheel is right.
        if ($ok -and $hasCuda) {
            Set-Progress 16 "Installing ONNX runtime (GPU)..."
            $ok = Run-Cmd $pip "install `"onnxruntime-gpu==1.22.0`" --quiet" "onnxruntime-gpu"
        }

        # llama-cpp-python from a prebuilt wheel index. The PyPI sdist builds from
        # source and needs CMake + MSVC Build Tools, which a clean Windows machine
        # does not have — it is a hard install failure, not a warning. Installed
        # BEFORE requirements.txt so pip sees it already satisfied.
        if ($ok) {
            Set-Progress 18 "Installing the local model runtime..."
            $llamaIdx = if ($hasCuda) {
                "https://abetlen.github.io/llama-cpp-python/whl/cu121"
            } else {
                "https://abetlen.github.io/llama-cpp-python/whl/cpu"
            }
            # Non-fatal: this powers critique, annotations and Story Mode. Grading
            # is SigLIP + TOPIQ and does not touch it, so a failure here must not
            # cost the user the whole install.
            $llamaOk = Run-Cmd $pip `
                "install `"llama-cpp-python==0.3.23`" --extra-index-url $llamaIdx --quiet" `
                "llama-cpp-python"
            if (-not $llamaOk) {
                Log "  ! llama-cpp-python unavailable - critique and Story Mode will be disabled."
                Log "    Grading is unaffected. Continuing."
            }
        }
    }

    # Step 4 — HuggingFace + quantisation core
    if ($ok) {
        Set-Progress 48 "Installing AI model libraries..."
        $ok = Run-Cmd $pip "install transformers accelerate bitsandbytes --quiet" "transformers + accelerate + bitsandbytes"
    }

    # Step 5 — remaining dependencies (pyiqa, ultralytics, lancedb, etc.)
    if ($ok) {
        Set-Progress 65 "Installing remaining libraries..."
        $ok = Run-Cmd $pip "install -r `"$(Join-Path $ROOT 'requirements.txt')`" --quiet" "requirements.txt"
    }

    # Step 5b — model weights
    # The install used to end without a single model file on disk. Nothing called
    # a downloader, so the first grade either failed or triggered encode_worker's
    # open_clip fallback: a ~7 GB fp32 fetch peaking at 10.3 GB RAM, i.e. the
    # heaviest path, on whatever machine happened to run first.
    #
    # fetch_models.py asks tier_select which encoder THIS machine will actually
    # use and downloads only that — roughly 0.8 GB on a CPU laptop against the
    # 20+ GB an unconditional prefetch would have pulled. Optional models
    # (critique, Story Mode) are left for the user to request from the UI.
    if ($ok) {
        $sentinel = Join-Path $ROOT "models\.models_ready"
        if ($repair -and (Test-Path $sentinel)) {
            Set-Progress 74 "Model weights already present — skipped"
            Log "  ✓ Model weights already on disk"
        } else {
            Set-Progress 74 "Downloading AI models for this machine..."
            $fetchOk = Run-Cmd $pythonV "`"$(Join-Path $ROOT 'scripts\fetch_models.py')`"" "model download"
            if (-not $fetchOk) {
                Log "  ! Model download incomplete. The app will retry on first launch,"
                Log "    or you can run: venv\Scripts\python.exe scripts\fetch_models.py"
            }
        }
    }

    # Step 6 — frontend
    if ($ok) {
        Set-Progress 83 "Building the interface..."
        if (-not (Test-Path (Join-Path $ROOT "frontend\dist\index.html"))) {
            $ok = Run-Cmd $npm "--prefix `"$(Join-Path $ROOT 'frontend')`" install --silent" "npm install"
            if ($ok) {
                $ok = Run-Cmd $npm "--prefix `"$(Join-Path $ROOT 'frontend')`" run build" "npm run build"
            }
        } else { Log "  ✓ Interface already built" }
    }

    # ── VERIFY (2026-09-17): prove the install works BEFORE declaring victory ──
    # "Setup finished" used to mean "pip exited 0" — the first grade then
    # exploded on a broken torch/CUDA pair, a missing wheel, or an empty
    # models dir. These checks exercise the real stack and surface a one-line
    # fix per row; results also render on the Complete page.
    $script:verifyResults = @()
    Set-Progress 88 "Verifying the installation..."
    Log "`r`nVerifying:"

    function Test-Verify($name, $scriptblock) {
        $r = & $scriptblock
        $mark = switch ($r[0]) { "ok" { "  ✓"; break } "warn" { "  !"; break } default { "  ✗" } }
        $color = switch ($r[0]) { "ok" { $green; break } "warn" { $amber; break } default { $red } }
        $script:verifyResults += @(@($r[0]), $name, $r[1])
        Log "$mark $name — $($r[1])"
        return ($r[0] -ne "fail")
    }

    $vOk = $true
    # 1) torch + CUDA truth — nvidia-smi and torch MUST agree, or the encoder
    #    silently drops to CPU (11-14 s/img with no error anywhere).
    $vOk = (Test-Verify "PyTorch" {
        $out = & $pythonV -c "import torch;print('torch',torch.__version__);print('cuda',torch.cuda.is_available())" 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { return @("fail", "import failed: $($out.Trim())") }
        $cuda = ($out -match "cuda True")
        $hasNv = ($null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue))
        if ($hasNv -and -not $cuda) { return @("warn", "installed, but CUDA is not active — culls fall back to CPU (slow). Reinstall the cu121 torch wheel.") }
        if (-not $hasNv -and $cuda) { return @("warn", "CUDA active but nvidia-smi is missing — check the NVIDIA driver.") }
        return @("ok", $out.Trim())
    }) -and $vOk

    # 2) llama_cpp — critique/Story runtime; grading survives without it.
    $vOk = (Test-Verify "Local model runtime (llama-cpp)" {
        $out = & $pythonV -c "import llama_cpp;print('ok')" 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { return @("warn", "not installed — Grading works; critique and Story Mode are disabled. Re-run the wizard to fix.") }
        return @("ok", "importable")
    }) -and $vOk

    # 3) transformers stack.
    $vOk = (Test-Verify "Transformers / model libraries" {
        $out = & $pythonV -c "import transformers, accelerate;print('ok')" 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { return @("fail", "missing — grading cannot run. Re-run the wizard.") }
        return @("ok", "importable")
    }) -and $vOk

    # 4) Encoder weights actually on disk (the install used to end with none).
    $vOk = (Test-Verify "Model weights" {
        if (Test-Path (Join-Path $ROOT "models\.models_ready")) { return @("ok", "encoder weights present") }
        return @("warn", "no weights on disk yet — the app retries on first launch, or run: venv\Scripts\python.exe scripts\fetch_models.py")
    }) -and $vOk

    # 5) Interface build.
    $vOk = (Test-Verify "Interface (frontend build)" {
        if (Test-Path (Join-Path $ROOT "frontend\dist\index.html")) { return @("ok", "built") }
        return @("fail", "frontend\dist\index.html missing — re-run the wizard.")
    }) -and $vOk

    # 6) Server boot smoke — the moment of truth: the whole backend up + API
    #    answering. 90 s budget (first import on AV-scanned machines is slow).
    $vOk = (Test-Verify "Server boot (headless smoke test)" {
        $smoke = Start-Process -FilePath $pythonV `
            -ArgumentList "`"$(Join-Path $ROOT 'src\local_launcher.py')`" --server-only" `
            -WorkingDirectory $ROOT -PassThru -WindowStyle Hidden
        $up = $false
        for ($i = 0; $i -lt 45; $i++) {
            Start-Sleep -Seconds 2
            try {
                $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/config" `
                    -UseBasicParsing -TimeoutSec 3
                if ($resp.StatusCode -eq 200) { $up = $true; break }
            } catch {}
            if ($smoke.HasExited) { break }
        }
        try { if (-not $smoke.HasExited) { Stop-Process -Id $smoke.Id -Force -ErrorAction SilentlyContinue } } catch {}
        if ($up) { return @("ok", "backend booted and answered /api/config") }
        return @("fail", "server did not answer within 90 s — see crash.log in the project root")
    }) -and $vOk

    if (-not $vOk) { Log "`r`nSome checks failed — see above. You can still Finish and re-run the wizard to repair." }

    # Step 7 — shortcut
    if ($ok) {
        Set-Progress 96 "Creating desktop shortcut..."
        try {
            $desktop  = [Environment]::GetFolderPath('Desktop')
            $lnkPath  = Join-Path $desktop "FirstCut.lnk"
            $iconPath = Join-Path $ROOT "icon.ico"
            $vbsPath  = Join-Path $ROOT "launch_hidden.vbs"
            $sh       = New-Object -ComObject WScript.Shell
            $s        = $sh.CreateShortcut($lnkPath)
            $s.TargetPath       = "wscript.exe"
            $s.Arguments        = "`"$vbsPath`""
            $s.WorkingDirectory = $ROOT
            $s.IconLocation     = "$iconPath,0"
            $s.Description      = "FirstCut — AI Photo Curator"
            $s.Save()
            Log "  ✓ Shortcut created on Desktop"
        } catch { Log "  ⚠ Could not create shortcut: $_" }
    }

    # Write stamp — marks setup as complete so future launches skip install
    if ($ok) {
        try {
            Set-Content -Path (Join-Path $ROOT "venv\.setup_ok") -Value "ok"
        } catch {}
    }

    Set-Progress 100 ""

    if ($ok) {
        Log "`r`nInstallation complete."
        Show-Complete
    } else {
        Log "`r`nInstallation failed. See log above for details."
        $btnCancel.Visible = $true
        $btnCancel.Text    = "Close"
    }
}

# ══════════════════════════════════════════════════════════════════
# PAGE 4: COMPLETE
# ══════════════════════════════════════════════════════════════════

function Show-Complete {
    Set-Step 3
    Clear-Content
    $btnBack.Enabled   = $false
    $btnNext.Visible   = $false
    $btnCancel.Visible = $false

    $tick = New-Object System.Windows.Forms.Label
    $tick.Text      = "✔"
    $tick.Font      = MakeFont 42
    $tick.ForeColor = $green
    $tick.AutoSize  = $true
    $tick.Location  = New-Object System.Drawing.Point(18, 24)
    $content.Controls.Add($tick)

    Add-Title "All done!"
    $($content.Controls | Where-Object { $_ -is [System.Windows.Forms.Label] -and $_.Text -eq "All done!" }).Location = New-Object System.Drawing.Point(70, 38)

    Add-Divider 86
    if ($script:verifyResults) {
        Add-Sub "Verification results:" 102
        $y = 126
        foreach ($v in $script:verifyResults) {
            $mark = switch ($v[0]) { "ok" { "✓"; break } "warn" { "!"; break } default { "✗" } }
            $color = switch ($v[0]) { "ok" { $green; break } "warn" { $amber; break } default { $red } }
            $row = New-Object System.Windows.Forms.Label
            $row.Text      = "$mark  $($v[1])"
            $row.Font      = New-Object System.Drawing.Font("Consolas", 8.5)
            $row.ForeColor = $color
            $row.AutoSize  = $true
            $row.Location  = New-Object System.Drawing.Point(18, $y)
            $row.MaximumSize = New-Object System.Drawing.Size(316, 60)
            $content.Controls.Add($row)
            $y += 22
            if ($y -gt 300) { break }
        }
        Add-Divider ($y + 6)
        Add-Sub "A shortcut has been placed on your Desktop.`nDouble-click it any time to launch the app." ($y + 20)
        $chkY = $y + 74
    } else {
        Add-Sub "FirstCut is installed and ready." 102
        Add-Sub "A shortcut has been placed on your Desktop.`nDouble-click it any time to launch the app." 132
        Add-Divider 196
        $chkY = 216
    }

    $chk = New-Object System.Windows.Forms.CheckBox
    $chk.Text      = "Launch FirstCut now"
    $chk.Font      = MakeFont 10
    $chk.ForeColor = $txt
    $chk.BackColor = $bg
    $chk.Checked   = $true
    $chk.AutoSize  = $true
    $chk.Location  = New-Object System.Drawing.Point(18, $chkY)
    $content.Controls.Add($chk)

    $finish = New-Object System.Windows.Forms.Button
    $finish.Text      = "Finish"
    $finish.Size      = New-Object System.Drawing.Size(100, 32)
    $finish.Location  = New-Object System.Drawing.Point(18, 310)
    $finish.FlatStyle = "Flat"
    $finish.BackColor = $accent
    $finish.ForeColor = [System.Drawing.Color]::White
    $finish.Font      = MakeFont 10 $true
    $finish.Cursor    = "Hand"
    $finish.FlatAppearance.BorderSize = 0
    $finish.Add_Click({
        if ($chk.Checked) {
            $vbs = Join-Path $ROOT "launch_hidden.vbs"
            Start-Process "wscript.exe" "`"$vbs`""
        }
        $form.Close()
    })
    $content.Controls.Add($finish)
}

# ══════════════════════════════════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════════════════════════════════

$script:page = 0
$script:repairMode = $false
$script:verifyResults = @()

function Go-Next {
    $script:page++
    switch ($script:page) {
        1 { Show-Requirements }
        2 { Show-Installing   }
        3 { Show-Complete     }
    }
}

function Go-Back {
    $script:page--
    switch ($script:page) {
        0 { Show-Welcome      }
        1 { Show-Requirements }
    }
}

$btnNext.Add_Click({ Go-Next })
$btnBack.Add_Click({ Go-Back })
# Repair flag is captured ONCE here — Welcome shows the checkbox but handlers
# must not stack (Go-Back re-runs Show-Welcome, which would append duplicates).
$btnNext.Add_Click({
    try {
        $repCtrl = $script:lastRepairBox
        if ($repCtrl) { $script:repairMode = $repCtrl.Checked }
    } catch {}
})

# ── Start ─────────────────────────────────────────────────────────
Show-Welcome
[System.Windows.Forms.Application]::Run($form)
