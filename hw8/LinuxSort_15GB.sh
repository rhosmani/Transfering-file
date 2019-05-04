#!/bin/bash

START_TIME=$SECONDS
LC_ALL=C sort -o ./Sorted-15GB ./data-15GB
ELAPSED_TIME=$(($SECONDS - $START_TIME))
./valsort ./Sorted-15GB
echo "Time taken by LinSort: $ELAPSED_TIME"