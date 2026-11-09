<#
.SYNOPSIS
    One-shot environment setup for JARVIS (spec §4, §5).

.DESCRIPTION
    Creates .venv with Python 3.11, installs requirements.txt into it, then
    downloads the external binaries and models that are not pip-installable.

    Release asset names are resolved from the GitHub API at run time rather than
    hardcoded, per spec §5.

    Everything is idempotent: already-present files are skipped unless -Force.

.NOTES
    Deviation from spec §5, verified 2026-08-04: whisper.cpp publishes no Windows
    Vulkan release asset and never has (checked every release v1.5.0..v1.9.2).
    The only Windows options are CPU, CPU+OpenBLAS, and cuBLAS (NVIDIA). This
    script installs the OpenBLAS CPU build. If M1 misses the ~300ms STT budget,
    the escalation is a source build with -DGGML_VULKAN=ON.

.PARAMETER Force
    Re-download and re-extract everything even if already present.

.PARAMETER SkipPip
    Skip venv creation and pip install.

.PARAMETER SkipModels
    Skip the large model downloads (binaries only).

.PARAMETER LlamaTag
    Pin a specific llama.cpp release tag (e.g. b10269). Default: latest.

.PARAMETER WhisperTag
    Pin a specific whisper.cpp release tag (e.g. v1.9.2). Default: latest.
#>
[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$SkipPip,
    [switch]$SkipModels,
    [string]$LlamaTag,
    [string]$WhisperTag
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

function Write-Step   ($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Write-Ok     ($m) { Write-Host "  [ok]   $m" -ForegroundColor Green }
function Write-Info   ($m) { Write-Host "  [info] $m" -ForegroundColor Gray }
function Write-Warn2  ($m) { Write-Host "  [warn] $m" -ForegroundColor Yellow }
function Fail         ($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red; exit 1 }

function Get-GitHubRelease {
    param([string]$Repo, [string]$Tag)
    $url = if ($Tag) { "https://api.github.com/repos/$Repo/releases/tags/$Tag" }
           else      { "https://api.github.com/repos/$Repo/releases/latest" }
    $headers = @{ 'User-Agent' = 'jarvis-setup'; 'Accept' = 'application/vnd.github+json' }
    if ($env:GITHUB_TOKEN) { $headers['Authorization'] = "Bearer $env:GITHUB_TOKEN" }
    try { return Invoke-RestMethod -Uri $url -Headers $headers }
    catch { Fail "GitHub API call failed for $Repo ($url): $($_.Exception.Message)" }
}

function Get-File {
    param([string]$Url, [string]$Destination, [string]$Label)
    if ((Test-Path $Destination) -and -not $Force) {
        $mb = [math]::Round((Get-Item $Destination).Length / 1MB, 1)
        Write-Ok "$Label already downloaded ($mb MB) - skipping"
        return
    }
    Write-Info "downloading $Label"
    Write-Info "  from $Url"
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        # -C - resumes a partial download; -L follows the HF/GitHub CDN redirects.
        & $curl.Source -L --fail --retry 3 --retry-delay 2 -C - -o $Destination $Url
        if ($LASTEXITCODE -ne 0) {
            # 33 = server refused a range request. Anything else may also have left
            # a bad partial behind. Either way, start over from scratch once.
            Write-Warn2 "resume attempt failed (curl exit $LASTEXITCODE); restarting download"
            Remove-Item $Destination -Force -ErrorAction SilentlyContinue
            & $curl.Source -L --fail --retry 3 --retry-delay 2 -o $Destination $Url
            if ($LASTEXITCODE -ne 0) { Fail "download failed for $Label (curl exit $LASTEXITCODE)" }
        }
    } else {
        Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing
    }
    if (-not (Test-Path $Destination)) { Fail "download produced no file for $Label" }
    $mb = [math]::Round((Get-Item $Destination).Length / 1MB, 1)
    Write-Ok "$Label downloaded ($mb MB)"
}

function Expand-ToolZip {
    param([string]$Zip, [string]$TargetDir, [string]$MarkerExe, [string]$Label)
    $marker = Join-Path $TargetDir $MarkerExe
    if ((Test-Path $marker) -and -not $Force) {
        Write-Ok "$Label already extracted - skipping"
        return
    }
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("jarvis-x-" + [guid]::NewGuid().ToString('N').Substring(0,8))
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    try {
        Expand-Archive -Path $Zip -DestinationPath $tmp -Force
        # Release layouts vary (root, Release\, build\bin\). Find the exe and
        # take everything alongside it, so the DLLs travel with it.
        $found = Get-ChildItem -Path $tmp -Recurse -Filter $MarkerExe -File | Select-Object -First 1
        if (-not $found) { Fail "$MarkerExe not found inside $(Split-Path -Leaf $Zip)" }
        $srcDir = $found.Directory.FullName
        New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
        Copy-Item -Path (Join-Path $srcDir '*') -Destination $TargetDir -Recurse -Force
        Write-Ok "$Label extracted to $TargetDir"
    } finally {
        Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------
Write-Host "JARVIS setup" -ForegroundColor White
Write-Host "project root: $ProjectRoot" -ForegroundColor DarkGray

# --- 1. Python + venv ------------------------------------------------------
if (-not $SkipPip) {
    Write-Step '1/6  Python 3.11 virtual environment'

    $py = Get-Command py -ErrorAction SilentlyContinue
    if (-not $py) { Fail "the 'py' launcher was not found; install Python 3.11 from python.org" }
    & py -3.11 -c "import sys; print(sys.version)" > $null 2>&1
    if ($LASTEXITCODE -ne 0) { Fail "Python 3.11 not available via 'py -3.11'. Spec §4 requires 3.11." }
    Write-Ok "Python 3.11 found"

    if (-not (Test-Path $VenvPython)) {
        Write-Info "creating .venv"
        & py -3.11 -m venv (Join-Path $ProjectRoot '.venv')
        if ($LASTEXITCODE -ne 0) { Fail "venv creation failed" }
    }
    Write-Ok ".venv present"

    # Prove isolation before installing anything into it.
    $isolated = & $VenvPython -c "import sys; print(sys.prefix != sys.base_prefix)"
    if ($isolated.Trim() -ne 'True') { Fail ".venv is not isolated from the system Python - refusing to install" }
    Write-Ok ".venv is isolated from system Python"

    Write-Info "upgrading pip"
    & $VenvPython -m pip install --upgrade pip --disable-pip-version-check -q
    if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed" }

    Write-Info "installing requirements.txt (this takes a few minutes)"
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot 'requirements.txt') --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { Fail "pip install failed" }
    Write-Ok "Python dependencies installed"
} else {
    Write-Step '1/6  Python 3.11 virtual environment (skipped)'
}

if (-not (Test-Path $VenvPython)) { Fail ".venv\Scripts\python.exe missing - run without -SkipPip first" }

# --- 2. Resolve paths from config.yaml -------------------------------------
Write-Step '2/6  Resolving paths from config.yaml'
$pathsJson = & $VenvPython -c @"
import json, sys
sys.path.insert(0, r'$ProjectRoot')
from src import config
c = config.load()
print(json.dumps({
    'data_dir':      str(c.data_dir),
    'bin_dir':       str(c.bin_dir),
    'models_dir':    str(c.models_dir),
    'downloads_dir': str(c.downloads_dir),
    'whisper_exe':   str(c.whisper_exe),
    'whisper_model': str(c.whisper_model),
    'llama_exe':     str(c.llama_exe),
    'llm_model':     str(c.llm_model),
    'tts_model':     str(c.tts_model),
    'tts_voices':    str(c.tts_voices),
}))
"@
if ($LASTEXITCODE -ne 0) { Fail "could not read config.yaml: $pathsJson" }
$P = $pathsJson | ConvertFrom-Json
Write-Ok "data_dir = $($P.data_dir)"

foreach ($d in @($P.data_dir, $P.bin_dir, $P.models_dir, $P.downloads_dir,
                 (Split-Path -Parent $P.whisper_exe), (Split-Path -Parent $P.llama_exe),
                 (Split-Path -Parent $P.tts_model))) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

# --- 3. llama.cpp (Vulkan) -------------------------------------------------
Write-Step '3/6  llama.cpp - Vulkan Windows x64 build'
$rel = Get-GitHubRelease -Repo 'ggml-org/llama.cpp' -Tag $LlamaTag
$asset = $rel.assets | Where-Object { $_.name -like 'llama-*-bin-win-vulkan-x64.zip' } | Select-Object -First 1
if (-not $asset) {
    Write-Warn2 "release $($rel.tag_name) assets:"
    $rel.assets | ForEach-Object { Write-Warn2 "    $($_.name)" }
    Fail "no Vulkan Windows x64 asset in llama.cpp release $($rel.tag_name)"
}
Write-Ok "release $($rel.tag_name) -> $($asset.name)"
$llamaZip = Join-Path $P.downloads_dir $asset.name
Get-File -Url $asset.browser_download_url -Destination $llamaZip -Label $asset.name
Expand-ToolZip -Zip $llamaZip -TargetDir (Split-Path -Parent $P.llama_exe) -MarkerExe 'llama-server.exe' -Label 'llama.cpp'
Set-Content -Path (Join-Path (Split-Path -Parent $P.llama_exe) 'VERSION.txt') -Value "$($rel.tag_name)`n$($asset.name)" -Encoding utf8

# --- 4. whisper.cpp --------------------------------------------------------
Write-Step '4/6  whisper.cpp - Windows x64 build'
$wrel = Get-GitHubRelease -Repo 'ggml-org/whisper.cpp' -Tag $WhisperTag
$vulkanAsset = $wrel.assets | Where-Object { $_.name -match 'vulkan' } | Select-Object -First 1
if ($vulkanAsset) {
    # Upstream started shipping one: prefer it, the spec asked for Vulkan.
    Write-Ok "a Vulkan asset now exists upstream: $($vulkanAsset.name) - using it"
    $wasset = $vulkanAsset
} else {
    Write-Warn2 "whisper.cpp $($wrel.tag_name) ships no Windows Vulkan asset (expected; see .NOTES)"
    Write-Warn2 "falling back to the OpenBLAS CPU build - STT runs on CPU, not GPU"
    $wasset = $wrel.assets | Where-Object { $_.name -eq 'whisper-blas-bin-x64.zip' } | Select-Object -First 1
    if (-not $wasset) { $wasset = $wrel.assets | Where-Object { $_.name -eq 'whisper-bin-x64.zip' } | Select-Object -First 1 }
}
if (-not $wasset) {
    $wrel.assets | ForEach-Object { Write-Warn2 "    $($_.name)" }
    Fail "no usable Windows x64 asset in whisper.cpp release $($wrel.tag_name)"
}
Write-Ok "release $($wrel.tag_name) -> $($wasset.name)"
$whisperZip = Join-Path $P.downloads_dir $wasset.name
Get-File -Url $wasset.browser_download_url -Destination $whisperZip -Label $wasset.name
Expand-ToolZip -Zip $whisperZip -TargetDir (Split-Path -Parent $P.whisper_exe) -MarkerExe 'whisper-server.exe' -Label 'whisper.cpp'
Set-Content -Path (Join-Path (Split-Path -Parent $P.whisper_exe) 'VERSION.txt') -Value "$($wrel.tag_name)`n$($wasset.name)" -Encoding utf8

# --- 5. Models -------------------------------------------------------------
if (-not $SkipModels) {
    Write-Step '5/6  Models'
    Get-File -Url 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q5_1.bin?download=true' `
             -Destination $P.whisper_model -Label 'ggml-small.en-q5_1.bin (~190 MB)'
    Get-File -Url 'https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf?download=true' `
             -Destination $P.llm_model -Label 'Qwen3-8B-Q4_K_M.gguf (~5 GB)'
    Get-File -Url 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx' `
             -Destination $P.tts_model -Label 'kokoro-v1.0.onnx (~327 MB)'
    Get-File -Url 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin' `
             -Destination $P.tts_voices -Label 'voices-v1.0.bin (~27 MB)'

    # openWakeWord fetches its own models. Deliberately NOT into its package
    # default under .venv/Lib/site-packages, which a venv rebuild destroys.
    Write-Info "downloading openWakeWord models (hey_jarvis, ~5 MB)"
    & $VenvPython -c @"
import sys
sys.path.insert(0, r'$ProjectRoot')
from pathlib import Path
from src import config
from src.audio.wake_word import ensure_models
c = config.load()
target = c.models_dir / str(c.get('wake_word.models_dir', 'openwakeword'))
print(ensure_models(target, str(c.get('wake_word.model', 'hey_jarvis'))))
"@
    if ($LASTEXITCODE -ne 0) { Fail "openWakeWord model download failed" }
    Write-Ok "openWakeWord models ready"
} else {
    Write-Step '5/6  Models (skipped)'
}

# --- 6. Verify -------------------------------------------------------------
Write-Step '6/6  Verification'
$problems = @()
function Check-File($path, $label, $minMB) {
    if (-not (Test-Path $path)) { $script:problems += "$label MISSING at $path"; Write-Warn2 "$label MISSING"; return }
    $mb = [math]::Round((Get-Item $path).Length / 1MB, 1)
    if ($minMB -and $mb -lt $minMB) {
        $script:problems += "$label looks truncated ($mb MB, expected >= $minMB MB)"
        Write-Warn2 "$label truncated ($mb MB)"
        return
    }
    Write-Ok "$label ($mb MB)"
}
Check-File $P.llama_exe    'llama-server.exe'        0
Check-File $P.whisper_exe  'whisper-server.exe'      0
if (-not $SkipModels) {
    Check-File $P.whisper_model 'ggml-small.en-q5_1.bin' 150
    Check-File $P.llm_model     'Qwen3-8B-Q4_K_M.gguf'   4000
    Check-File $P.tts_model     'kokoro-v1.0.onnx'       250
    Check-File $P.tts_voices    'voices-v1.0.bin'        20
}

if ($problems.Count -gt 0) {
    Write-Host "`nSetup finished with problems:" -ForegroundColor Red
    $problems | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Next:  .\run.ps1        (starts both servers and health-checks them)" -ForegroundColor White
