@echo off
:: Extract the hour component and remove any leading space
set hh=%TIME:~0,2%
set hh=%hh: =%

:: Add a leading zero and ensure it's two digits
set hh=0%hh%
set hh=%hh:~-2%

:: Display the time with two-digit components
echo [%hh%%TIME:~2,6%]: %1
@echo on