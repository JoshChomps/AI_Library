# Kinesis AI Engine Build Script (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "          KINESIS AI ENGINE BUILD SCRIPT (PS)         " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$IncludeDir = Join-Path $ScriptDir "include"
$SrcDir = Join-Path $ScriptDir "src"
$TestsDir = Join-Path $ScriptDir "tests"
$BenchDir = Join-Path $ScriptDir "benchmarks"
$BinDir = Join-Path $ScriptDir "bin"

if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir | Out-Null
}

# Add WinGet Zig package location to PATH if present
$WinGetZig = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\zig.zig_Microsoft.Winget.Source_8wekyb3d8bbwe\zig-x86_64-windows-0.16.0"
if (Test-Path (Join-Path $WinGetZig "zig.exe")) {
    $env:PATH = "$WinGetZig;$env:PATH"
}

# Compiler discovery
$Compiler = $null
$CompilerExe = $null
$ExtraArgs = @()

if (Get-Command clang++ -ErrorAction SilentlyContinue) {
    $Compiler = "clang++"
    $CompilerExe = "clang++"
} elseif (Get-Command g++ -ErrorAction SilentlyContinue) {
    $Compiler = "g++"
    $CompilerExe = "g++"
} elseif (Get-Command zig -ErrorAction SilentlyContinue) {
    $Compiler = "zig c++"
    $CompilerExe = (Get-Command zig).Source
    $ExtraArgs = @("c++")
} elseif (Get-Command cl -ErrorAction SilentlyContinue) {
    $Compiler = "cl"
    $CompilerExe = "cl"
}

if (-not $Compiler) {
    Write-Host "[ERROR] No C++20 compiler found on PATH (clang++, g++, zig, or cl)." -ForegroundColor Red
    Write-Host "Please install Clang, GCC, Zig, or MSVC to compile Kinesis." -ForegroundColor Yellow
    exit 1
}

Write-Host "[INFO] Detected compiler: $Compiler" -ForegroundColor Green

$CoreSources = @(
    (Join-Path $SrcDir "core/arena_allocator.cpp"),
    (Join-Path $SrcDir "compute/gemm.cpp")
)

$TestSources = @(
    (Join-Path $TestsDir "test_main.cpp"),
    (Join-Path $TestsDir "test_arena_allocator.cpp"),
    (Join-Path $TestsDir "test_virtual_allocator.cpp"),
    (Join-Path $TestsDir "test_slab_allocator.cpp"),
    (Join-Path $TestsDir "test_memory_sandbox.cpp"),
    (Join-Path $TestsDir "test_hardware_info.cpp"),
    (Join-Path $TestsDir "test_tensor_view.cpp"),
    (Join-Path $TestsDir "test_gemm.cpp"),
    (Join-Path $TestsDir "test_dual.cpp"),
    (Join-Path $TestsDir "test_regressions.cpp")
) + $CoreSources

$TestOut = Join-Path $BinDir "kinesis_tests.exe"
$BenchOut = Join-Path $BinDir "kinesis_benchmarks.exe"

Write-Host "[BUILD] Compiling unit tests..." -ForegroundColor Cyan
if ($Compiler -eq "cl") {
    & cl /nologo /std:c++20 /O2 /W4 /arch:AVX2 /I "$IncludeDir" /I "$TestsDir" $TestSources /Fe"$TestOut"
} else {
    $compileArgs = $ExtraArgs + @("-std=c++20", "-O3", "-march=native", "-Wall", "-Wextra", "-Wno-nullability-completeness", "-I", $IncludeDir, "-I", $TestsDir) + $TestSources + @("-o", $TestOut)
    & $CompilerExe $compileArgs
}

if ($LASTEXITCODE -eq 0 -and (Test-Path $TestOut)) {
    Write-Host "[SUCCESS] Tests compiled: $TestOut" -ForegroundColor Green
    Write-Host "`nRunning Unit Tests..." -ForegroundColor Cyan
    & $TestOut

    Write-Host "`n[BUILD] Compiling benchmarks..." -ForegroundColor Cyan
    $BenchSources = @((Join-Path $BenchDir "bench_latency.cpp")) + $CoreSources
    if ($Compiler -eq "cl") {
        & cl /nologo /std:c++20 /O2 /W4 /arch:AVX2 /I "$IncludeDir" $BenchSources /Fe"$BenchOut"
    } else {
        $benchArgs = $ExtraArgs + @("-std=c++20", "-O3", "-march=native", "-DNDEBUG", "-Wall", "-Wextra", "-Wno-nullability-completeness", "-I", $IncludeDir) + $BenchSources + @("-o", $BenchOut)
        & $CompilerExe $benchArgs
    }

    if ($LASTEXITCODE -eq 0 -and (Test-Path $BenchOut)) {
        Write-Host "[SUCCESS] Benchmarks compiled: $BenchOut" -ForegroundColor Green
        Write-Host "`nRunning Latency & Throughput Benchmarks..." -ForegroundColor Cyan
        & $BenchOut
    }
} else {
    Write-Host "[ERROR] Compilation failed." -ForegroundColor Red
    exit 1
}
