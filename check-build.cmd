git pull
call .\_scripts\build-test-all.cmd > log.txt 2>&1
timeout /t 5
rg -i --pcre2 "(warn|fail|(?<![\w_])error)" log.txt
