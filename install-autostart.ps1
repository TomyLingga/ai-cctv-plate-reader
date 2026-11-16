<#
.SYNOPSIS
    Run alpr-id automatically when the PC starts, through Windows Task Scheduler.

.DESCRIPTION
    Install   registers the task, opens the API port in the firewall and starts the service.
              Needs an elevated (Administrator) PowerShell.
    Uninstall stops the service, removes the task and the firewall rule.
    Start / Stop control the service without removing the task.
    Status    shows the task state and whether the API answers.

    The task starts 30 seconds after boot, before anyone logs in, without a window, and is
    restarted by Task Scheduler if it stops. It runs the project's own Python directly, so
    stopping the task stops the service.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install-autostart.ps1
    powershell -ExecutionPolicy Bypass -File .\install-autostart.ps1 -Action Status
    powershell -ExecutionPolicy Bypass -File .\install-autostart.ps1 -Action Uninstall
#>
param(
    [ValidateSet('Install', 'Uninstall', 'Start', 'Stop', 'Status')]
    [string]$Action = 'Install',
    [string]$TaskName = 'ALPR-ID',
    # The account the service runs as. USERDOMAIN is not reliable here: in an SSH session it
    # is the workgroup name, which is not part of any account name.
    [string]$User = [Security.Principal.WindowsIdentity]::GetCurrent().Name
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$EnvFile = Join-Path $Root '.env'
$FirewallRule = "$TaskName API"

function Read-EnvValue([string]$Name, [string]$Default) {
    if (Test-Path $EnvFile) {
        $line = Get-Content $EnvFile | Where-Object { $_ -match "^\s*$Name\s*=" } | Select-Object -Last 1
        if ($line) { return ($line -split '=', 2)[1].Trim() }
    }
    return $Default
}

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    return ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-ServiceProcesses {
    # Python processes of this project that run the service, however they were started.
    $rootPattern = [regex]::Escape($Root)
    Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
        Where-Object { $_.CommandLine -match 'alpr[_-]id' -and $_.CommandLine -match '\bserve\b' -and ($_.CommandLine -match $rootPattern -or $_.ExecutablePath -match $rootPattern) }
}

function Stop-ServiceProcesses {
    foreach ($p in Get-ServiceProcesses) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Proses service dihentikan (PID $($p.ProcessId))."
    }
}

function Show-Status {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Host "Task '$TaskName' belum terdaftar."
    } else {
        $info = $task | Get-ScheduledTaskInfo
        Write-Host "Task '$TaskName': $($task.State). Terakhir dijalankan $($info.LastRunTime), hasil $($info.LastTaskResult)."
    }
    $port = Read-EnvValue 'ALPR_PORT' '5000'
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 3
        Write-Host "API menjawab di port ${port}: $($health.cameras_connected) dari $($health.cameras_total) kamera tersambung, provider $($health.providers -join ', ')."
    } catch {
        Write-Host "API tidak menjawab di port $port."
    }
    $rule = Get-NetFirewallRule -DisplayName $FirewallRule -ErrorAction SilentlyContinue
    if ($rule) {
        Write-Host "Aturan firewall '$FirewallRule' ada. Dari PC lain, coba http://<IP-PC-ini>:$port/health."
    } else {
        Write-Host "Aturan firewall '$FirewallRule' belum ada: PC lain mungkin tidak bisa mengakses port $port."
    }
}

switch ($Action) {
    'Status' { Show-Status }

    'Start' {
        Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
        Start-Sleep -Seconds 5
        Show-Status
    }

    'Stop' {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Stop-ServiceProcesses
        Write-Host "Service dihentikan. Task '$TaskName' tetap terdaftar dan akan jalan lagi saat PC menyala."
    }

    'Uninstall' {
        if (-not (Test-Admin)) { throw 'Jalankan PowerShell sebagai Administrator untuk menghapus task.' }
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Stop-ServiceProcesses
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Remove-NetFirewallRule -DisplayName $FirewallRule -ErrorAction SilentlyContinue
        Write-Host "Task '$TaskName' dan aturan firewall '$FirewallRule' dihapus."
    }

    'Install' {
        if (-not (Test-Admin)) { throw 'Jalankan PowerShell sebagai Administrator untuk mendaftarkan task.' }
        if (-not (Test-Path $EnvFile)) { throw 'File .env belum ada. Salin .env.example menjadi .env lalu isi URL kamera.' }
        if (-not (Test-Path (Join-Path $Root 'config\app.yaml'))) { throw 'File config\app.yaml belum ada. Salin dari config\app.example.yaml.' }

        # Stop a running instance first: its DLLs would block the environment update below,
        # and it would keep the port busy.
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Stop-ServiceProcesses
        Start-Sleep -Seconds 2

        # Make sure the environment exists and matches uv.lock before the task depends on it.
        $extra = Read-EnvValue 'ALPR_EXTRA' 'cpu'
        Write-Host "Menyiapkan lingkungan Python (ONNX Runtime $extra)..."
        # A fresh uv install is not on PATH yet in sessions that were already open.
        $uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
        if (-not $uv) { $uv = Join-Path $env:USERPROFILE '.local\bin\uv.exe' }
        if (-not (Test-Path $uv)) { throw 'uv tidak ditemukan. Pasang dari https://docs.astral.sh/uv/ lalu jalankan lagi.' }
        # uv reports progress on stderr. Under 'Stop' Windows PowerShell would treat those
        # lines as errors, so judge the call by its exit code only.
        $ErrorActionPreference = 'Continue'
        & $uv sync --frozen --extra $extra --project $Root 2>&1 | ForEach-Object { Write-Host "  $_" }
        $syncExit = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($syncExit -ne 0) { throw 'uv sync gagal.' }
        if (-not (Test-Path $Python)) { throw "Python proyek tidak ditemukan di $Python." }

        $taskAction = New-ScheduledTaskAction -Execute $Python -Argument '-m alpr_id serve' -WorkingDirectory $Root
        $trigger = New-ScheduledTaskTrigger -AtStartup
        $trigger.Delay = 'PT30S'   # wait for the network and the GPU driver
        $principal = New-ScheduledTaskPrincipal -UserId $User -LogonType S4U -RunLevel Limited
        $settings = New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit ([TimeSpan]::Zero) `
            -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
            -MultipleInstances IgnoreNew -StartWhenAvailable -Priority 4

        # -ErrorAction Stop is given explicitly on the task and firewall cmdlets: they live in
        # modules and do not see this script's $ErrorActionPreference, so a failure would
        # otherwise pass silently.
        Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger -Principal $principal `
            -Settings $settings -Description 'alpr-id: pembaca plat nomor dari CCTV timbangan (REST API)' `
            -Force -ErrorAction Stop | Out-Null
        Write-Host "Task '$TaskName' terdaftar untuk user $User."

        # A task has no window, so Windows never shows its "allow access" dialog. Without this
        # rule the API would answer on this PC only. The rule covers every network profile:
        # on the inference PC the office LAN is classified as Public, and a rule limited to
        # Domain and Private would leave the weighbridge PCs locked out.
        $port = Read-EnvValue 'ALPR_PORT' '5000'
        Remove-NetFirewallRule -DisplayName $FirewallRule -ErrorAction SilentlyContinue
        New-NetFirewallRule -DisplayName $FirewallRule -Direction Inbound -Protocol TCP -LocalPort $port `
            -Action Allow -Profile Any -ErrorAction Stop | Out-Null
        Write-Host "Port $port dibuka di firewall untuk semua profil jaringan."

        Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
        Write-Host 'Menunggu service menyala...'
        Start-Sleep -Seconds 20
        Show-Status
    }
}
