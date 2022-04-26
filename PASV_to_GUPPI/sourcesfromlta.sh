#!/bin/bash

echo "Gathering Metadata for all sources from the LTA file $1 ..."
egrep -wa 'CHANNELS|POLS|MJD_REF|OBJECT|^CODE' $1 > ltaheaderinfo.txt
echo "Done!"
echo " "
echo "Making list of sources..."
grep 'OBJECT' ltaheaderinfo.txt | awk '{print $NF}' > sources.txt
echo "Done!"
