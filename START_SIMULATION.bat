@echo off
title Smart Traffic Light — RL Project
chcp 65001 >nul
if exist "C:\Python314\python.exe" (
    "C:\Python314\python.exe" -u run.py
) else (
    python -u run.py
)
if errorlevel 1 pause
