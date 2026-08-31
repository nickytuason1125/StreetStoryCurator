# sign_release.ps1 — code-sign the Lumara build artifacts.
#
# Two supported certificate routes (pick one):
#
#   1. STORE  — OV/EV code-signing certificate as a PFX in the user store
#      (Sectigo/DigiCert/Certum, ~$100-500/yr). Find the SHA-1 thumbprint:
#          Get-ChildItem Cert:\CurrentUser\My
#      Usage:  .\sign_release.ps1 -Mode Store -Thumbprint <SHA1>
#
#   2. TRUSTEDSIGNING — Azure Trusted Signing (~$9.99/mo, Microsoft-hosted).
#      Fastest route to SmartScreen reputation. Requires the Azure.CodeSigning
#      dlib (dotnet add package Azure.CodeSigning.Dlib) and a metadata JSON:
#          { "CertificateProfileName": "<profile>",
#            "CodeSigningAccountName": "<account>",
#            "Endpoint": "https://eus.codesigning.azure.net/" }
#      Usage:  .\sign_release.ps1 -Mode TrustedSigning -Account <name> `
#                    -Profile <profile> -Endpoint <url>
#
# SmartScreen reality check:
#   - EV certs and Azure Trusted Signing get near-immediate reputation.
#   - OV certs build reputation over days/weeks of downloads.
#   - Unsigned or self-signed: always warned. There is no free cure — the
#     beta-user workaround is "More info -> Run anyway" (2 clicks).
#
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Store", "TrustedSigning")]
    [string]$Mode,

    [string]$Thumbprint = "",
    [string]$Account    = "",
    [string]$Profile    = "",
    [string]$Endpoint   = "https://eus.codesigning.azure.net/",
    [string]$DmdfPath   = ""
)

$root = Split-Path -Parent $PSScriptRoot
$targets = @(
    "$root\frontend\src-tauri\target\release\lumara.exe",
    "$root\frontend\src-tauri\target\release\bundle\nsis\Lumara_1.0.0_x64-setup.exe"
)

if ($Mode -eq "Store" -and -not $Thumbprint) {
    Write-Host "ERROR: Store mode requires -Thumbprint <SHA1>"; exit 1
}
if ($Mode -eq "TrustedSigning" -and (-not $Account -or -not $Profile -or -not $DmdfPath)) {
    Write-Host "ERROR: TrustedSigning mode requires -Account, -Profile, -DmdfPath"; exit 1
}

foreach ($file in $targets) {
    if (-not (Test-Path $file)) { Write-Host "SKIP (missing): $file"; continue }
    Write-Host "Signing: $file"

    if ($Mode -eq "Store") {
        & signtool sign /fd sha256 /td sha256 /tr http://timestamp.digicert.com `
            /sha1 $Thumbprint $file
    } else {
        & signtool sign /fd sha256 /td sha256 `
            /dlib Azure.CodeSigning.Dlib.dll /dmdf $DmdfPath $file
    }

    if ($LASTEXITCODE -ne 0) { Write-Host "SIGN FAILED: $file"; exit 1 }
}

Write-Host "`nAll artifacts signed. Verify with:"
Write-Host "  signtool verify /pa /all <file>"
