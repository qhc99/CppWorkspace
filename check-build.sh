git pull
./_scripts/build-test-all.sh > log.txt
grep -iE "(warn|fail|[^a-zA-Z_]error)" log.txt
