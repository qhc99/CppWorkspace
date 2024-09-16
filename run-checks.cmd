git pull
call .\_scripts\build-test-all.cmd > log.txt 2>&1
timeout /t 5
rg -i --pcre2 "(warn|fail|(?<![\w_])errors?\b|\[\d{2}:\d{2}:\d{2}\])" log.txt
