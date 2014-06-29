find $1 -type f -name "*.jpg"|while read line;do echo ${line#$1};done > $2
