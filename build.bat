@echo off
setlocal

rem Define body type counts
set NUMPLANETS=10000
set NUMSTARS=30
set NUMSMALLBLACKHOLES=2
set NUMBLACKHOLES=1

rem Calculate total number of bodies
set /A TOTALBODIES=%NUMPLANETS% + %NUMSTARS% + %NUMSMALLBLACKHOLES% + %NUMBLACKHOLES%
set /A TOTALBODIES=%TOTALBODIES% * 2

rem Clean up old build files
echo Cleaning up old build files...
del /F /Q nbody.exe nbody.lib nbody.exp

rem Compile CUDA program
echo Compiling CUDA program...
nvcc -DNUMPLANETS=%NUMPLANETS% -DNUMSTARS=%NUMSTARS% -DNUMSMALLBLACKHOLES=%NUMSMALLBLACKHOLES% -DNUMBLACKHOLES=%NUMBLACKHOLES% -o nbody nbody.cu
if %ERRORLEVEL% neq 0 (
    echo Compilation failed!
    pause
    exit /b %ERRORLEVEL%
) else (
    echo Compilation succeeded!
)

rem Run the program
echo Running the program...
nbody.exe
if %ERRORLEVEL% neq 0 (
    echo Execution failed!
    pause
    exit /b %ERRORLEVEL%
) else (
    echo Execution succeeded!
)

rem Run visualizer script
echo Running visualizer script...
python visualizer2.py %TOTALBODIES%
if %ERRORLEVEL% neq 0 (
    echo Visualizer execution failed!
    pause
    exit /b %ERRORLEVEL%
) else (
    echo Visualizer execution succeeded!
)

echo All tasks completed successfully!
pause
