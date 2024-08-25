git pull
.\_scripts\build-test-all.cmd > log.txt 2>&1
timeout /t 5
findstr /i /n /l "warn" log.txt
findstr /i /n /l "error" log.txt
findstr /i /n /l "fail" log.txt
