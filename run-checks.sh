git pull
./_scripts/build-test-all.sh > log.txt
rg -i --pcre2 "(warn|fail|(?<![\w_])errors?\b|\[\d{2}:\d{2}:\d{2}\])" log.txt
