git pull
.\_scripts\build-test-all.sh > log.txt
grep -i -w -E "warn|error|fail" log.txt
