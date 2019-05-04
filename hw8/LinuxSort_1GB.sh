#!/bin/bash

START_TIME=$SECONDS
LC_ALL=C sort -o ./Sorted-1GB ./data-1GB
ELAPSED_TIME=$(($SECONDS - $START_TIME))
./valsort ./Sorted-1GB
echo "Time taken by LinSort: $ELAPSED_TIME"