#!/bin/bash
find . -type f -name "*Zone.Identifier" | while read -r file; do
    echo "deleting $file"
    rm -f "$file"
done
