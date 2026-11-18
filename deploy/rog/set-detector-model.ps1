<#
Point the service at another detector model and restart it. Run on the inference PC:

    powershell -ExecutionPolicy Bypass -File C:\alpr-id\deploy\rog\set-detector-model.ps1 -Model models/detector/candidates/yolov8n-20261118b.onnx

Back to the detector the service started with:

    powershell -ExecutionPolicy Bypass -File C:\alpr-id\deploy\rog\set-detector-model.ps1 -Model models/detector/plat.onnx

Only the model_path line in the detector section of config\app.yaml changes. The model must give
the output the runtime reads: one class, rows of x1, y1, x2, y2, confidence, class (see
training\detector\export_onnx.py). The file as it was is kept next to it as
app.yaml.before-detector-switch. deploy.ps1 ships config\app.yaml from the laptop, so make the
same change there or the next deploy brings the old model back.
#>
param(
    [Parameter(Mandatory)][string]$Model,
    [switch]$NoRestart
)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot)
$config = Join-Path $root 'config\app.yaml'
if (-not (Test-Path (Join-Path $root $Model))) { throw "Model file not found under ${root}: $Model" }

$lines = [IO.File]::ReadAllLines($config)
$section = ''
$changed = $false
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^([A-Za-z_]+):') { $section = $Matches[1] }
    if ($section -eq 'detector' -and $lines[$i] -match '^(\s+model_path:\s*)(\S+)(.*)$') {
        Write-Output "detector.model_path: $($Matches[2]) -> $Model"
        $lines[$i] = $Matches[1] + $Model + $Matches[3]
        $changed = $true
    }
}
if (-not $changed) { throw 'No model_path line found in the detector section of config\app.yaml' }
Copy-Item $config "$config.before-detector-switch" -Force
# Without a byte order mark: the YAML loader would read it as part of the first key.
[IO.File]::WriteAllLines($config, $lines, (New-Object Text.UTF8Encoding($false)))

if (-not $NoRestart) {
    $autostart = Join-Path $root 'install-autostart.ps1'
    & $autostart -Action Stop
    & $autostart -Action Start
}
