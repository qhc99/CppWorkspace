git pull
.\_scripts\build-test-all.cmd > log.txt 2>&1

:wait_loop
set "last_size=0"
set "size=%~z1"

if %size% neq %last_size% (
    set "last_size=%size%"
    timeout /t 1 >nul
    goto wait_loop
)

findstr /i /n /l "warn" log.txt
findstr /i /n /l "error" log.txt
findstr /i /n /l "fail" log.txt
