@echo off
setlocal

rem Define body type counts
set NUMPLANETS=20000
set NUMSTARS=2000
set NUMSMALLBLACKHOLES=200
set NUMBLACKHOLES=0
set TIME=500
set GALAXYR=150.0e9

rem Calculate total number of bodies
set /A TOTALBODIES=%NUMPLANETS% + %NUMSTARS% + %NUMSMALLBLACKHOLES% + %NUMBLACKHOLES%
set /A TOTALBODIES=%TOTALBODIES% * 2

rem Clean up old build files
echo Cleaning up old build files...
del /F /Q nbody.exe nbody.lib nbody.exp

rem Compile CUDA program
echo Compiling CUDA program...
nvcc -DTIME=%TIME% -DNUMPLANETS=%NUMPLANETS% -DNUMSTARS=%NUMSTARS% -DNUMSMALLBLACKHOLES=%NUMSMALLBLACKHOLES% -DNUMBLACKHOLES=%NUMBLACKHOLES% -DGALAXYR=%GALAXYR% -o nbody nbody.cu
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

rem Check if the output file exists
if exist nbody.out (
    echo nbody.out file found.
) else (
    echo nbody.out file not found. Exiting...
    pause
    exit /b 
)

rem Run visualizer script with galaxy radius and total number of bodies
echo Running visualizer script with galaxy radius %GALAXYR% and total bodies %TOTALBODIES%...
C:\Users\skyle\AppData\Local\Programs\Python\Python312\python.exe visualizer2.py %TOTALBODIES% %GALAXYR%
if %ERRORLEVEL% neq 0 (
    echo Visualizer execution failed!
    pause
    exit /b %ERRORLEVEL%
) else (
    echo Visualizer execution succeeded!
)

echo All tasks completed successfully!
pause
