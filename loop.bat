@echo off
setlocal enabledelayedexpansion

rem Define the values for NUMT
set NUMT_VALUES=1 2 4 6 8 10 12

rem Set other simulation parameters
set DT=0.01
set STEPS=100

rem Create or clear the performance.csv file and add the header
echo NUMT,NUMPLANETS,Performance > performance.csv

rem Loop over each value of NUMT
for %%t in (%NUMT_VALUES%) do (
  rem Loop over NUMPLANETS from 10 to 150
  for /l %%n in (200,200,2000) do (
    g++ nbody2.cpp -DNUMT=%%t -DNUMPLANETS=%%n -DDT=%DT% -DSTEPS=%STEPS% -o nbody_simulation -lm -fopenmp
    nbody_simulation.exe
  )
)
