@echo off
cd /d %~dp0
color 0a
conda activate soft & jupyter lab
if %errorlevel%==0 goto:EOF