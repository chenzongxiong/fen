#!/bin/sh

lines=`ls -al *.py | awk '{print $9}' | xargs -I xxx cat xxx | wc -l`
echo "Total python code lines is ${lines}"
