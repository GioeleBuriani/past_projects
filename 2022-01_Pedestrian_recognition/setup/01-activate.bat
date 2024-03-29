@echo off
set THIS_DIR=%cd%

rem use 'source' for TA's (they have the 'source' dir) or 'release' (for students)
if exist %THIS_DIR%\source\ (
  set SOURCE_DIR=%THIS_DIR%\source
) else (
  set SOURCE_DIR=%THIS_DIR%\release
)

rem set variable to announce that we're properly sourced (to be checked in successive scripts)
set MP_IV_SOURCED=true

set PRACTICUM1_DATA_DIR=%THIS_DIR%\data
set PRACTICUM3MP_DATA_DIR=%THIS_DIR%\data

echo.
echo Adding %SOURCE_DIR%, each practicum dir and common dir to PYTHONPATH

set PYTHONPATH=%SOURCE_DIR%
set PYTHONPATH=%SOURCE_DIR%\practicum1;%PYTHONPATH%
set PYTHONPATH=%SOURCE_DIR%\practicum2;%PYTHONPATH%
set PYTHONPATH=%SOURCE_DIR%\practicum3_mp;%PYTHONPATH%
set PYTHONPATH=%SOURCE_DIR%\practicum3_iv;%PYTHONPATH%
set PYTHONPATH=%SOURCE_DIR%\assignment;%PYTHONPATH%
set PYTHONPATH=%SOURCE_DIR%\common;%PYTHONPATH%

rem activate conda environment mp-iv
conda activate mp-iv
