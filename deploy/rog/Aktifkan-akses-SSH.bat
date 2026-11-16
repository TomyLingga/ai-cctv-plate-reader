@echo off
rem Double-click on the inference PC. Asks for Administrator rights, then runs
rem enable-ssh-for-alpr.ps1 from the same folder. The window stays open to show the result.
setlocal
set "SCRIPT=%~dp0enable-ssh-for-alpr.ps1"
if not exist "%SCRIPT%" (
    echo File enable-ssh-for-alpr.ps1 tidak ada di folder yang sama dengan file ini.
    pause
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-NoExit','-File','\"%SCRIPT%\"'"
