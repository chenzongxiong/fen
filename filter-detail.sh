
cat $1 | grep "Epoch" | grep "Epoch" |awk '{print $8$11$14$16$18$28$30}' > ~/Desktop/$1.csv
