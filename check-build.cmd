git pull
.\_scripts\build-test-all.cmd > log.txt 2>&1
call rg -i --pcre2 '(warn|fail|(?<![\w_])error)' log.txt
