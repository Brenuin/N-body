@echo off
setlocal enabledelayedexpansion

rem Define the values for NUMPLANETS and THREADS
set NUMPLANETS_VALUES=200 400 600 800 1000 1200 1400 1600 1800 2000
set THREADS_VALUES=32 64 128 256 512 1024

rem Set other simulation parameters
set NUMSTARS=50
set NUMSMALLBLACKHOLES=200
set NUMBLACKHOLES=100
set TIME=1000
set GALAXYR=150.0e12

rem Create or clear the performance.csv file and add the header
echo NUMPLANETS,THREADS,Performance > performance.csv

rem Loop over each value of NUMPLANETS
for %%p in (%NUMPLANETS_VALUES%) do (
  rem Loop over each value of THREADS
  for %%t in (%THREADS_VALUES%) do (
    rem Calculate total number of bodies
    set /A TOTALBODIES=%%p + %NUMSTARS% + %NUMSMALLBLACKHOLES% + %NUMBLACKHOLES%
    set /A TOTALBODIES=!TOTALBODIES! * 2

    rem Clean up old build files
    echo Cleaning up old build files...
    del /F /Q nbody.exe nbody.lib nbody.exp

    rem Compile CUDA program with the current NUMPLANETS and THREADS values
    echo Compiling CUDA program...
    nvcc -DTIME=%TIME% -DNUMPLANETS=%%p -DNUMSTARS=%NUMSTARS% -DNUMSMALLBLACKHOLES=%NUMSMALLBLACKHOLES% -DNUMBLACKHOLES=%NUMBLACKHOLES% -DGALAXYR=%GALAXYR% -DTHREADS=%%t -o nbody nbody.cu --ptxas-options=-v -maxrregcount=32 -rdc=true --gpu-architecture=compute_61 --gpu-code=sm_61 -lineinfo
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

    rem Read the performance result from the output file and append it to the performance.csv
    for /F "tokens=1,2,3 delims=," %%a in (performance_output.txt) do (
      echo %%p,%%t,%%c >> performance.csv
    )
  )
)

echo All tasks completed successfully!
pause
