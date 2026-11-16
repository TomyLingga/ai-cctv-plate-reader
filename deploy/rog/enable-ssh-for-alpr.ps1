<#
.SYNOPSIS
    One-time setup on the inference PC so alpr-id can be installed and maintained remotely.

.DESCRIPTION
    Turns on the Windows OpenSSH server with these limits:
      - login only with the deploy key below, never with a password
      - port 22 reachable only from Tailscale addresses (100.64.0.0/10)
    Then sends a short info file back to the development laptop through Tailscale.

    To undo everything: run with -Remove.
#>
param([switch]$Remove)

$ErrorActionPreference = 'Stop'
$PublicKey = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGVHvfrQy1qNFC1AyYw1fmHo2UxzFnzZGvKBcmVEvXMx alpr-id-deploy@laptop'
$Laptop = 'laptop-i7qcpupa'
$RuleName = 'ALPR-ID SSH (Tailscale)'
$SshDir = Join-Path $env:ProgramData 'ssh'
$KeyFile = Join-Path $SshDir 'administrators_authorized_keys'

function Step([string]$Message) { Write-Host "[alpr-id] $Message" }

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Jalankan sebagai Administrator (klik dua kali Aktifkan-akses-SSH.bat).'
}

if ($Remove) {
    Step 'Menghapus akses SSH alpr-id...'
    Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    if (Test-Path $KeyFile) {
        $kept = @(Get-Content $KeyFile | Where-Object { $_ -ne $PublicKey })
        Set-Content -Path $KeyFile -Value $kept -Encoding ascii
    }
    Stop-Service sshd -ErrorAction SilentlyContinue
    Set-Service -Name sshd -StartupType Disabled -ErrorAction SilentlyContinue
    Step 'Selesai. Layanan sshd dimatikan, kunci dan aturan firewall dihapus.'
    return
}

Step '1/5 Memasang OpenSSH Server (butuh internet, bisa beberapa menit)...'
$capability = Get-WindowsCapability -Online -Name 'OpenSSH.Server*' | Select-Object -First 1
if ($capability.State -ne 'Installed') { Add-WindowsCapability -Online -Name $capability.Name | Out-Null }
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd   # the first start creates sshd_config

Step '2/5 Memasang kunci publik untuk laptop pengembangan...'
$existing = @(if (Test-Path $KeyFile) { Get-Content $KeyFile })
if ($existing -notcontains $PublicKey) { Add-Content -Path $KeyFile -Value $PublicKey -Encoding ascii }
# sshd ignores this file unless only Administrators and SYSTEM can write it.
icacls $KeyFile /inheritance:r /grant '*S-1-5-32-544:F' /grant '*S-1-5-18:F' | Out-Null

Step '3/5 Login hanya dengan kunci, tanpa kata sandi...'
$configFile = Join-Path $SshDir 'sshd_config'
$config = Get-Content $configFile -Raw
$config = [regex]::Replace($config, '(?m)^[ \t]*#?[ \t]*PasswordAuthentication[ \t]+\w+[ \t]*(\r?)$', 'PasswordAuthentication no$1')
if ($config -notmatch '(?m)^PasswordAuthentication no') { $config = "PasswordAuthentication no`r`n" + $config }
$config = [regex]::Replace($config, '(?m)^[ \t]*#?[ \t]*PubkeyAuthentication[ \t]+\w+[ \t]*(\r?)$', 'PubkeyAuthentication yes$1')
Set-Content -Path $configFile -Value $config -Encoding ascii
New-Item -Path 'HKLM:\SOFTWARE\OpenSSH' -Force | Out-Null
New-ItemProperty -Path 'HKLM:\SOFTWARE\OpenSSH' -Name DefaultShell -PropertyType String -Force `
    -Value "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" | Out-Null
Restart-Service sshd

Step '4/5 Firewall: port 22 hanya dari alamat Tailscale...'
Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue | Remove-NetFirewallRule
Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Protocol TCP -LocalPort 22 `
    -RemoteAddress '100.64.0.0/10' -Action Allow -Profile Any | Out-Null

Step '5/5 Mengirim info PC ini ke laptop pengembangan...'
$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1
$drives = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free -ne $null } |
    ForEach-Object { "{0}={1}GB" -f $_.Name, [math]::Round($_.Free / 1GB, 1) }
$info = @(
    "user=$env:USERNAME"
    "computer=$env:COMPUTERNAME"
    "time=$(Get-Date -Format s)"
    "os=$((Get-CimInstance Win32_OperatingSystem).Caption) build $([Environment]::OSVersion.Version.Build)"
    "free=$($drives -join ' ')"
    "gpu=$($gpu.Name) driver=$($gpu.DriverVersion)"
    "uv=$([bool](Get-Command uv -ErrorAction SilentlyContinue))"
    "port5000_in_use=$([bool](Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue))"
    "sshd=$((Get-Service sshd).Status)"
)
$infoFile = Join-Path $env:TEMP 'alpr-rog-info.txt'
Set-Content -Path $infoFile -Value $info -Encoding ascii
$tailscale = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'
$sent = $false
if (Test-Path $tailscale) {
    & $tailscale file cp $infoFile "${Laptop}:" 2>$null
    $sent = ($LASTEXITCODE -eq 0)
}

Write-Host ''
Step 'SELESAI. Akses SSH aktif untuk laptop pengembangan lewat Tailscale.'
Step "Nama user Windows di PC ini: $env:USERNAME"
if (-not $sent) { Step 'Info tidak terkirim otomatis. Sampaikan nama user di atas ke Claude.' }
Write-Host ''
$info | ForEach-Object { Write-Host "    $_" }
