@echo off
REM One-click Windows setup for a fresh PC (Git + Miniconda + clone + conda env).
REM Right-click -> Run, or from cmd: install-windows.cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $ErrorActionPreference='Stop'; $url='https://raw.githubusercontent.com/MosheVB/3D-Scanner/main/scripts/bootstrap_windows.ps1'; $dest='%USERPROFILE%\bootstrap_windows.ps1'; Invoke-WebRequest -Uri $url -OutFile $dest; & $dest }"
pause
