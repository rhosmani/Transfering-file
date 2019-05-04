#!/bin/bash

START_TIME=$SECONDS
LC_ALL=C sort -o ./Sorted-60GB ./data-60GB
ELAPSED_TIME=$(($SECONDS - $START_TIME))
./valsort ./Sorted-60GB
echo "Time taken by LinSort: $ELAPSED_TIME"