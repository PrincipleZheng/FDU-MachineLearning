#!/bin/zsh
#阶乘
expr $1 + 2 >& /dev/null
flag=$?
function factorial()
{
    re=1
    if [ $1 -le 0 ]
    then
            echo "1"
    else
            for ((i=1;i<=$1;i=i+1))
            do
                    re=$[re*i]
            done
            echo "$re"
    fi
}
echo `factorial 4`