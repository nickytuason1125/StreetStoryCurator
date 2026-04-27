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
$form.Text            = "Street Story Curator — Setup"
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
    Add-Sub "This wizard will install Street Story Curator on your computer." 84
    Add-Sub "Street Story Curator analyses and grades your street photography, builds sequenced stories, and gives editorial feedback — all running locally on your machine." 120
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
    Add-Sub "The following programs must be installed before continuing." 82

    $pyResult   = Check-Python
    $nodeResult = Check-Node
    $script:pyOk   = $pyResult[0]
    $script:nodeOk = $nodeResult[0]

    Make-ReqRow "Python 3.10 or newer" 118 $script:pyOk $pyResult[1] "https://www.python.org/downloads/"
    Add-Divider 178
    Make-ReqRow "Node.js 16 or newer"  202 $script:nodeOk $nodeResult[1] "https://nodejs.org"
    Add-Divider 262

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

    $python  = "python"
    $pip     = Join-Path $ROOT "venv\Scripts\pip.exe"
    $pythonV = Join-Path $ROOT "venv\Scripts\python.exe"
    $npm     = "npm"
    $ok = $true

    # Step 1 — venv
    Set-Progress 5 "Creating Python environment..."
    if (-not (Test-Path (Join-Path $ROOT "venv\Scripts\python.exe"))) {
        $ok = Run-Cmd $python "-m venv `"$(Join-Path $ROOT 'venv')`"" "python -m venv"
    } else { Log "  ✓ Environment already exists" }

    # Step 2 — pip upgrade
    if ($ok) {
        Set-Progress 10 "Upgrading pip..."
        Run-Cmd $pythonV "-m pip install --upgrade pip --quiet" "pip upgrade" | Out-Null
    }

    # Step 3 — PyTorch (largest step)
    if ($ok) {
        Set-Progress 14 "Installing PyTorch (CPU) — this takes a few minutes..."
        $ok = Run-Cmd $pip "install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet" "torch + torchvision"
    }

    # Step 4 — CLIP
    if ($ok) {
        Set-Progress 60 "Installing CLIP..."
        $clipOk = Run-Cmd $pip "install `"clip @ git+https://github.com/openai/CLIP.git`" --quiet" "openai/CLIP"
        if (-not $clipOk) { Log "  ⚠ CLIP install skipped (git may not be in PATH). Some features may be limited." }
    }

    # Step 5 — requirements
    if ($ok) {
        Set-Progress 70 "Installing remaining libraries..."
        $ok = Run-Cmd $pip "install -r `"$(Join-Path $ROOT 'requirements.txt')`" --quiet" "requirements.txt"
    }

    # Step 6 — frontend
    if ($ok) {
        Set-Progress 85 "Building the interface..."
        if (-not (Test-Path (Join-Path $ROOT "frontend\dist\index.html"))) {
            $ok = Run-Cmd $npm "--prefix `"$(Join-Path $ROOT 'frontend')`" install --silent" "npm install"
            if ($ok) {
                $ok = Run-Cmd $npm "--prefix `"$(Join-Path $ROOT 'frontend')`" run build" "npm run build"
            }
        } else { Log "  ✓ Interface already built" }
    }

    # Step 7 — shortcut
    if ($ok) {
        Set-Progress 96 "Creating desktop shortcut..."
        try {
            $desktop  = [Environment]::GetFolderPath('Desktop')
            $lnkPath  = Join-Path $desktop "Street Story Curator.lnk"
            $iconPath = Join-Path $ROOT "icon.ico"
            $vbsPath  = Join-Path $ROOT "launch_hidden.vbs"
            $sh       = New-Object -ComObject WScript.Shell
            $s        = $sh.CreateShortcut($lnkPath)
            $s.TargetPath       = "wscript.exe"
            $s.Arguments        = "`"$vbsPath`""
            $s.WorkingDirectory = $ROOT
            $s.IconLocation     = "$iconPath,0"
            $s.Description      = "Street Story Curator"
            $s.Save()
            Log "  ✓ Shortcut created on Desktop"
        } catch { Log "  ⚠ Could not create shortcut: $_" }
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
    Add-Sub "Street Story Curator is installed and ready." 102
    Add-Sub "A shortcut has been placed on your Desktop.`nDouble-click it any time to launch the app." 132

    Add-Divider 196

    $chk = New-Object System.Windows.Forms.CheckBox
    $chk.Text      = "Launch Street Story Curator now"
    $chk.Font      = MakeFont 10
    $chk.ForeColor = $txt
    $chk.BackColor = $bg
    $chk.Checked   = $true
    $chk.AutoSize  = $true
    $chk.Location  = New-Object System.Drawing.Point(18, 216)
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

# ── Start ─────────────────────────────────────────────────────────
Show-Welcome
[System.Windows.Forms.Application]::Run($form)
