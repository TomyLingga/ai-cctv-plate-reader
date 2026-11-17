<#
.SYNOPSIS
    Install or update alpr-id on the inference PC from the development laptop, over SSH.

.DESCRIPTION
    Run on the laptop, from anywhere. Needs enable-ssh-for-alpr.ps1 to have been run once on
    the inference PC.

    Steps:
      1. Pack the project (code, config, models, .env) into a zip. Virtual environments,
         recordings, logs and the local event database are left out.
      2. Copy the zip to the inference PC and unpack it over the install folder. The data
         folder on the inference PC (events, media, logs) is never touched.
      3. Install uv there if it is missing, then build the environment from uv.lock.
      4. Print which ONNX Runtime providers the detector really uses.

    Autostart is a separate, explicit step: -InstallAutostart.

.EXAMPLE
    .\deploy.ps1 -User "IT SUPPORT"                      # install or update
    .\deploy.ps1 -User "IT SUPPORT" -InstallAutostart    # also register the scheduled task
    .\deploy.ps1 -PackageOnly                            # only build the zip, to inspect it
#>
param(
    [string]$User,
    [string]$HostName = '100.90.220.87',           # Tailscale address of the inference PC
    [string]$RemoteDir = 'C:\alpr-id',
    [ValidateSet('gpu', 'cpu')][string]$Extra = 'gpu',
    [string]$KeyFile = "$env:USERPROFILE\.ssh\alpr_rog_ed25519",
    [switch]$PackageOnly,
    [switch]$InstallAutostart
)

$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Stage = Join-Path $env:TEMP 'alpr-id-deploy'
$Zip = Join-Path $env:TEMP 'alpr-id-deploy.zip'

function Step([string]$Message) { Write-Host "[deploy] $Message" }

# -- 1. package -------------------------------------------------------------------------
Step "Mengemas $Project"
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Force $Stage | Out-Null
$excludeDirs = @('.git', '.venv', '.venv-gpu', '.pytest_cache', '.ruff_cache', '__pycache__', 'data', 'docs', 'tests', 'training')
# /XD takes names, so nested __pycache__ folders are skipped too. Exit codes below 8 mean success.
robocopy $Project $Stage /E /XD @excludeDirs /XF '*.pyc' 'uv.toml' /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy gagal ($LASTEXITCODE)" }
Remove-Item (Join-Path $Stage 'models\detector\src') -Recurse -Force -ErrorAction SilentlyContinue

# The inference PC gets its own setting for the ONNX Runtime build.
$envFile = Join-Path $Stage '.env'
if (-not (Test-Path $envFile)) { throw 'File .env tidak ada di proyek; tidak ada URL kamera untuk dikirim.' }
$lines = @(Get-Content $envFile | Where-Object { $_ -notmatch '^\s*ALPR_EXTRA\s*=' }) + "ALPR_EXTRA=$Extra"
[IO.File]::WriteAllLines($envFile, $lines, (New-Object Text.UTF8Encoding $false))
if (-not (Test-Path (Join-Path $Stage 'config\app.yaml'))) {
    Copy-Item (Join-Path $Stage 'config\app.example.yaml') (Join-Path $Stage 'config\app.yaml')
}

if (Test-Path $Zip) { Remove-Item $Zip -Force }
Compress-Archive -Path (Join-Path $Stage '*') -DestinationPath $Zip
$files = @(Get-ChildItem $Stage -Recurse -File)
Step ("Paket: {0} file, {1:N1} MB -> {2}" -f $files.Count, ((Get-Item $Zip).Length / 1MB), $Zip)
if ($PackageOnly) {
    $files | ForEach-Object { $_.FullName.Substring($Stage.Length + 1) } | Sort-Object
    return
}

# -- 2. copy and unpack -----------------------------------------------------------------
if (-not $User) { throw 'Sebutkan nama user Windows di PC inferensi: -User "nama"' }
$target = "$User@$HostName"
$sshArgs = @('-i', $KeyFile, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=10')
# -n: ssh must not wait on stdin. Without it a session started from a background job stayed
# open after the remote command had finished.
$sshOnly = @('-n')

function Invoke-Remote([string]$Command) {
    # The remote default shell is Windows PowerShell; -EncodedCommand avoids quoting trouble.
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes("`$ProgressPreference = 'SilentlyContinue'; $Command"))
    # ssh relays the remote stderr, where uv writes its progress. Under 'Stop', Windows
    # PowerShell turns those lines into terminating errors as soon as the caller of this
    # script redirects stderr, so the call is judged by its exit code only.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & ssh @sshOnly @sshArgs $target "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $encoded" 2>&1 |
            ForEach-Object { Write-Host "  $_" }
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0) { throw "Perintah di PC inferensi gagal ($LASTEXITCODE)" }
}

# The package holds .env with the camera credentials. Whatever happens below, no copy of it
# stays behind in the temp folder.
try {

Step "Menyambung ke $target"
Invoke-Remote "New-Item -ItemType Directory -Force '$RemoteDir' | Out-Null; Write-Host ('tersambung ke ' + `$env:COMPUTERNAME + ' sebagai ' + `$env:USERNAME)"

Step 'Menyalin paket'
& scp @sshArgs $Zip "${target}:$($RemoteDir -replace '\\', '/')/alpr-id-deploy.zip"
if ($LASTEXITCODE -ne 0) { throw 'scp gagal' }

Step 'Membuka paket (folder data di PC inferensi tidak disentuh)'
Invoke-Remote "Expand-Archive -Path '$RemoteDir\alpr-id-deploy.zip' -DestinationPath '$RemoteDir' -Force; Remove-Item '$RemoteDir\alpr-id-deploy.zip'"

# -- 3. environment ---------------------------------------------------------------------
Step "Menyiapkan lingkungan Python (ONNX Runtime $Extra); unduhan pertama bisa 2 GB lebih"
Invoke-Remote @"
`$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not `$uv) { `$uv = Join-Path `$env:USERPROFILE '.local\bin\uv.exe' }
if (-not (Test-Path `$uv)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}
Set-Location '$RemoteDir'
& `$uv sync --frozen --extra $Extra
if (`$LASTEXITCODE -ne 0) { throw 'uv sync gagal' }
"@

# -- 4. which providers are really used -------------------------------------------------
Step 'Memeriksa provider ONNX Runtime'
Invoke-Remote @"
Set-Location '$RemoteDir'
& .\.venv\Scripts\python.exe -c "from alpr_id.config import load_config; from alpr_id.logging_setup import configure_logging; from alpr_id.pipeline.engine import load_models; configure_logging(log_dir=None, level='WARNING'); d, o = load_models(load_config()); print('detector providers:', d.providers); print('ocr providers:', o.providers)"
"@

if ($InstallAutostart) {
    Step 'Mendaftarkan task terjadwal dan aturan firewall'
    Invoke-Remote "Set-Location '$RemoteDir'; & .\install-autostart.ps1 -Action Install"
}
Step 'Selesai.'

} finally {
    Remove-Item $Stage -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $Zip -Force -ErrorAction SilentlyContinue
}
