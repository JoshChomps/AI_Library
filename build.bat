@echo off
setlocal enabledelayedexpansion

echo ======================================================
echo           KINESIS AI ENGINE BUILD SCRIPT
echo ======================================================

set SCRIPT_DIR=%~dp0
set INCLUDE_DIR=%SCRIPT_DIR%include
set SRC_DIR=%SCRIPT_DIR%src
set TESTS_DIR=%SCRIPT_DIR%tests
set BENCH_DIR=%SCRIPT_DIR%benchmarks
set BIN_DIR=%SCRIPT_DIR%bin

if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"

:: Add WinGet Zig package location to PATH if present
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages\zig.zig_Microsoft.Winget.Source_8wekyb3d8bbwe\zig-x86_64-windows-0.16.0" (
    set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\zig.zig_Microsoft.Winget.Source_8wekyb3d8bbwe\zig-x86_64-windows-0.16.0;!PATH!"
)

:: Compiler detection
set CXX=
where clang++ >nul 2>nul && set CXX=clang++
if "%CXX%"=="" (
    where g++ >nul 2>nul && set CXX=g++
)
if "%CXX%"=="" (
    where zig >nul 2>nul && set CXX=zig c++
)
if "%CXX%"=="" (
    where cl >nul 2>nul && set CXX=cl
)

if "%CXX%"=="" (
    echo [ERROR] No C++20 compiler found on PATH ^(clang++, g++, zig, or cl^).
    echo Please ensure Clang, GCC, Zig, or MSVC is installed.
    exit /b 1
)

echo [INFO] Using compiler: %CXX%

if "%CXX%"=="cl" (
    echo [BUILD] Compiling tests with MSVC...
    cl /nologo /std:c++20 /O2 /W4 /arch:AVX2 /I "%INCLUDE_DIR%" /I "%TESTS_DIR%" ^
       "%TESTS_DIR%\test_main.cpp" "%TESTS_DIR%\test_arena_allocator.cpp" ^
       "%TESTS_DIR%\test_virtual_allocator.cpp" "%TESTS_DIR%\test_slab_allocator.cpp" ^
       "%TESTS_DIR%\test_memory_sandbox.cpp" "%TESTS_DIR%\test_hardware_info.cpp" ^
       "%TESTS_DIR%\test_tensor_view.cpp" "%TESTS_DIR%\test_gemm.cpp" ^
       "%TESTS_DIR%\test_dual.cpp" "%TESTS_DIR%\test_regressions.cpp" ^
       "%SRC_DIR%\core\arena_allocator.cpp" "%SRC_DIR%\compute\gemm.cpp" ^
       /Fe"%BIN_DIR%\kinesis_tests.exe"
    
    echo [BUILD] Compiling benchmarks with MSVC...
    cl /nologo /std:c++20 /O2 /W4 /arch:AVX2 /I "%INCLUDE_DIR%" ^
       "%BENCH_DIR%\bench_latency.cpp" ^
       "%SRC_DIR%\core\arena_allocator.cpp" "%SRC_DIR%\compute\gemm.cpp" ^
       /Fe"%BIN_DIR%\kinesis_benchmarks.exe"
) else (
    echo [BUILD] Compiling tests with %CXX%...
    %CXX% -std=c++20 -O3 -march=native -Wall -Wextra -I "%INCLUDE_DIR%" -I "%TESTS_DIR%" ^
       "%TESTS_DIR%/test_main.cpp" "%TESTS_DIR%/test_arena_allocator.cpp" ^
       "%TESTS_DIR%/test_virtual_allocator.cpp" "%TESTS_DIR%/test_slab_allocator.cpp" ^
       "%TESTS_DIR%/test_memory_sandbox.cpp" "%TESTS_DIR%/test_hardware_info.cpp" ^
       "%TESTS_DIR%/test_tensor_view.cpp" "%TESTS_DIR%/test_gemm.cpp" ^
       "%TESTS_DIR%/test_dual.cpp" "%TESTS_DIR%/test_regressions.cpp" ^
       "%SRC_DIR%/core/arena_allocator.cpp" "%SRC_DIR%/compute/gemm.cpp" ^
       -o "%BIN_DIR%/kinesis_tests.exe"

    echo [BUILD] Compiling benchmarks with %CXX%...
    %CXX% -std=c++20 -O3 -march=native -DNDEBUG -Wall -Wextra -I "%INCLUDE_DIR%" ^
       "%BENCH_DIR%/bench_latency.cpp" ^
       "%SRC_DIR%/core/arena_allocator.cpp" "%SRC_DIR%/compute/gemm.cpp" ^
       -o "%BIN_DIR%/kinesis_benchmarks.exe"
)

if %ERRORLEVEL% equ 0 (
    echo.
    echo ======================================================
    echo [SUCCESS] Build completed successfully!
    echo Executable: %BIN_DIR%\kinesis_tests.exe
    echo ======================================================
    echo.
    echo Running unit tests...
    "%BIN_DIR%\kinesis_tests.exe"
) else (
    echo [ERROR] Build failed.
    exit /b %ERRORLEVEL%
)
