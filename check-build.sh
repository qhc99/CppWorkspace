git pull
./_scripts/build-test-all.sh > log.txt
rg -i --pcre2 '(warn|fail|(?<![\w_])error)' log.txt
