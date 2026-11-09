<#
.SYNOPSIS
    Start whisper-server and llama-server, then health-check them (spec §6, M0).

.DESCRIPTION
    Both servers run as detached background processes with their output captured
    to logs\. PIDs are recorded to logs\*.pid so -Stop can shut them down.

    Servers are left running after the health check so later milestones can use
    them without a restart. Use -Stop to shut them down.

.PARAMETER Stop
    Stop any servers previously started by this script, then exit.

.PARAMETER Restart
    Stop existing servers first, then start fresh.

.PARAMETER CheckOnly
    Do not start anything; just run the health check against whatever is running.

.PARAMETER StaticOnly
    Check files and GPU only. Does not require the servers.
#>
[CmdletBinding()]
param(
    [switch]$Stop,
    [switch]$Restart,
    [switch]$CheckOnly,
    [switch]$StaticOnly
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$LogDir = Join-Path $ProjectRoot 'logs'

function Write-Step ($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Write-Ok   ($m) { Write-Host "  [ok]   $m" -ForegroundColor Green }
function Write-Info ($m) { Write-Host "  [info] $m" -ForegroundColor Gray }
function Write-Warn2($m) { Write-Host "  [warn] $m" -ForegroundColor Yellow }
function Fail       ($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $VenvPython)) { Fail ".venv not found - run .\setup.ps1 first" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Stop-Service2 {
    param([string]$Name)
    $pidFile = Join-Path $LogDir "$Name.pid"
    if (-not (Test-Path $pidFile)) { return }
    $recorded = (Get-Content $pidFile -Raw).Trim()
    if ($recorded) {
        $proc = Get-Process -Id $recorded -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Info "stopping $Name (pid $recorded)"
            Stop-Process -Id $recorded -Force -ErrorAction SilentlyContinue
            $proc.WaitForExit(5000) | Out-Null
        }
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

function Test-Running {
    param([string]$Name)
    $pidFile = Join-Path $LogDir "$Name.pid"
    if (-not (Test-Path $pidFile)) { return $false }
    $recorded = (Get-Content $pidFile -Raw).Trim()
    if (-not $recorded) { return $false }
    return [bool](Get-Process -Id $recorded -ErrorAction SilentlyContinue)
}

function Start-Service2 {
    param([string]$Name, [string]$Exe, [string[]]$Arguments)
    if (Test-Running $Name) { Write-Ok "$Name already running"; return }
    if (-not (Test-Path $Exe)) { Fail "$Name executable not found: $Exe" }
    # Windows will not let one process redirect stdout and stderr to the same
    # file, so they are split. src/health.py reads both.
    $out = Join-Path $LogDir "$Name.out.log"
    $err = Join-Path $LogDir "$Name.log"
    Remove-Item $out, $err -Force -ErrorAction SilentlyContinue
    Write-Info "$Name  $Exe $($Arguments -join ' ')"
    $proc = Start-Process -FilePath $Exe -ArgumentList $Arguments `
        -WorkingDirectory (Split-Path -Parent $Exe) `
        -RedirectStandardOutput $out -RedirectStandardError $err `
        -WindowStyle Hidden -PassThru
    Set-Content -Path (Join-Path $LogDir "$Name.pid") -Value $proc.Id -Encoding ascii

    # A bad argument kills these servers in well under a second. Catch that now
    # instead of burning the full health-check timeout waiting for a dead port.
    if ($proc.WaitForExit(2000)) {
        Write-Host "  [FAIL] $Name exited immediately (code $($proc.ExitCode))" -ForegroundColor Red
        Get-Content $err -Tail 20 -ErrorAction SilentlyContinue | ForEach-Object {
            Write-Host "         $_" -ForegroundColor Red
        }
        Remove-Item (Join-Path $LogDir "$Name.pid") -Force -ErrorAction SilentlyContinue
        exit 1
    }
    Write-Ok "$Name started (pid $($proc.Id)), log: $err"
}

if ($Stop) {
    Write-Step 'Stopping services'
    Stop-Service2 'llama-server'
    Stop-Service2 'whisper-server'
    Write-Ok 'stopped'
    exit 0
}

if ($Restart) {
    Write-Step 'Restarting services'
    Stop-Service2 'llama-server'
    Stop-Service2 'whisper-server'
}

# --- resolve config --------------------------------------------------------
$cfgJson = & $VenvPython -c @"
import json, sys
sys.path.insert(0, r'$ProjectRoot')
from src import config
c = config.load()
print(json.dumps({
    'whisper_exe':   str(c.whisper_exe),
    'whisper_model': str(c.whisper_model),
    'whisper_host':  c.get('whisper.host'),
    'whisper_port':  c.get('whisper.port'),
    'whisper_threads': c.get('whisper.threads'),
    'whisper_lang':  c.get('whisper.language'),
    'whisper_ac':    c.get('whisper.audio_ctx', 0),
    'llama_exe':     str(c.llama_exe),
    'llm_model':     str(c.llm_model),
    'llm_host':      c.get('llm.host'),
    'llm_port':      c.get('llm.port'),
    'ngl':           c.get('llm.n_gpu_layers'),
    'ctx':           c.get('llm.context_size'),
    'fa':            c.get('llm.flash_attn'),
    'ctk':           c.get('llm.cache_type_k'),
    'ctv':           c.get('llm.cache_type_v'),
    'lv':            c.get('llm.log_verbosity', 4),
    'thinking':      bool(c.get('llm.enable_thinking', False)),
}))
"@
if ($LASTEXITCODE -ne 0) { Fail "could not read config.yaml: $cfgJson" }
$C = $cfgJson | ConvertFrom-Json

if (-not ($CheckOnly -or $StaticOnly)) {
    Write-Step 'Starting services'

    $whisperArgs = @(
        '-m', $C.whisper_model,
        '--host', $C.whisper_host,
        '--port', "$($C.whisper_port)",
        '-t', "$($C.whisper_threads)",
        '-l', $C.whisper_lang
    )
    # 0 means "leave whisper's default 30s window alone"; see config.yaml.
    if ([int]$C.whisper_ac -gt 0) { $whisperArgs += @('-ac', "$($C.whisper_ac)") }
    Start-Service2 -Name 'whisper-server' -Exe $C.whisper_exe -Arguments $whisperArgs

    $llamaArgs = @(
        '-m', $C.llm_model,
        '-ngl', "$($C.ngl)",
        '-c', "$($C.ctx)",
        '--cache-type-k', $C.ctk,
        '--cache-type-v', $C.ctv,
        '--host', $C.llm_host,
        '--port', "$($C.llm_port)",
        # Without this the offload summary never reaches the log; see config.yaml.
        '-lv', "$($C.lv)",
        # Qwen3 thinks by default and streams it as reasoning_content, which
        # blows the TTFT budget before the first spoken word.
        '--reasoning', $(if ($C.thinking) { 'on' } else { 'off' })
    )
    # -fa is a valued argument in current builds: on|off|auto. Accept a YAML
    # boolean too, since the spec wrote it as a bare flag.
    if ($null -ne $C.fa) {
        $fa = "$($C.fa)".ToLower()
        if ($fa -eq 'true')  { $fa = 'on' }
        if ($fa -eq 'false') { $fa = 'off' }
        $llamaArgs += @('-fa', $fa)
    }
    Start-Service2 -Name 'llama-server' -Exe $C.llama_exe -Arguments $llamaArgs

    Write-Info 'loading a 5 GB model over Vulkan takes a moment on a cold cache'
}

# --- health check ----------------------------------------------------------
Write-Step 'Health check'
Push-Location $ProjectRoot
try {
    if ($StaticOnly) { & $VenvPython -m src.main --check-static }
    else             { & $VenvPython -m src.main --check }
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($code -ne 0) {
    Write-Host "Health check failed. Logs:" -ForegroundColor Red
    Write-Host "  $LogDir\llama-server.log" -ForegroundColor Red
    Write-Host "  $LogDir\whisper-server.log" -ForegroundColor Red
    exit $code
}

Write-Host "Servers are up and left running. Stop them with:  .\run.ps1 -Stop" -ForegroundColor White
exit 0
