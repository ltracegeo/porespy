@echo off
setlocal enabledelayedexpansion

REM === Configuration ===
set SRC_DIR=src\porespy_c
set TARGET_DIR=src\porespy\beta

cd %SRC_DIR% || exit /b 1

echo Setting up build directory...
uv run meson setup build --buildtype=plain --vsenv || exit /b 1

echo Compiling C extension...
uv run meson compile -C build || exit /b 1

echo Moving compiled library into Python package...
if exist %TARGET_DIR%\build\walker.dll (
    move /Y %TARGET_DIR%\build\walker.dll %TARGET_DIR%\
    echo Moved walker.dll
) else (
    echo walker.dll not found.
    exit /b 1
)

echo Build complete.
