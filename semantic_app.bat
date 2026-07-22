@echo off
rem Launch the semantic notebook app (double-click me).
rem Project file + file browser root = this folder.
cd /d "%~dp0"
"C:\ProgramData\miniconda3\python.exe" semantic_render.py %*
pause
