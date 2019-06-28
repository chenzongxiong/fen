#!/bin/bash


ll *.py | awk '{print $9}' | xargs -I xxx cat xxx | wc -l
