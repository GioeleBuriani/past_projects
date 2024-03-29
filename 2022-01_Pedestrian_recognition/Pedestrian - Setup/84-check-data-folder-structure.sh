#!/usr/bin/env bash

# compare target folder structure with current tree output
# diff should only be on the right-hand-side (additional files, if at all)

# show missing files (diff lines starting with '<')
MISSING_FILES=$(diff 82-assemble-data-for-students.filelist <(find data -type f | sort) | grep '^<')
ZERO_ON_MISSING_FILES=$?

if [ $ZERO_ON_MISSING_FILES -eq 0 ]; then
    echo "Error! data folder not complete."
    echo "Missing files:"
    echo $MISSING_FILES
    exit 1
else
    echo "Great! data folder complete."
    exit 0
fi