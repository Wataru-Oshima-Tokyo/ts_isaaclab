#!/bin/bash
# Check if an argument is passed
if [ -z "$1" ]; then
    echo "please put the input mujoco file path"
    exit
fi
if [ -z "$2" ]; then
    echo "please put the output usd file path"
    exit
fi
./isaaclab.sh -p source/standalone/tools/convert_mjcf.py \
  $1 \
  $2 \
  --import-sites \
  --make-instanceable


echo "you should change the default prim path"
usdedit $2