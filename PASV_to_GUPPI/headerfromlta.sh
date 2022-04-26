#!/bin/bash

echo "Here are all the sources, please enter name of source to be used for header..."
cat sources.txt
read source
grep -A 2 $(grep $source sources.txt) ltaheaderinfo.txt | awk '{print $NF}' > sourceheaderinfo.txt; awk 'NR >=1 && NR <=2 {print $NF}' ltaheaderinfo.txt >> sourceheaderinfo.txt
