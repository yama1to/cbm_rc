#!/bin/bash

INCLUDE_DIR=include
SRC_DIR=src
TEST_DIR=test

if [ -z "$1" ]; then
    exit 1
fi

find ${SRC_DIR} ${TEST_DIR} -type d -exec echo "-y\"{}\"" \; \
    | xargs iverilog -g2012 -o "a.out" -I ${INCLUDE_DIR} "$1"


if [ $? -gt 0 ]; then
    exit $?
fi

vvp "a.out"
rm  "a.out"

exit 0
