#!/bin/bash

file="sourceheaderinfo.txt"

c=0

while read line; do
	((c=c+1))
	
	if [ $c -eq 1 ]
	then
		sed -i "s/^SRC_NAME=.*/SRC_NAME=$line/" main_header.txt
		echo "Updating SRC_NAME with sed -i 's/^SRC_NAME=.*/SRC_NAME=$line/' main_header.txt"
	elif [ $c -eq 2 ]
	then
		sed -i "s/^STT_IMJD=.*/STT_IMJD=$line/" main_header.txt
		echo "Updating STT_IMJD with sed -i 's/^STT_IMJD=.*/SRC_IMJD=$line/' main_header.txt"
		sed -i "s/^STT_SMJD=.*/STT_SMJD=$line/" main_header.txt
		echo "Updating STT_SMJD with sed -i 's/^STT_SMJD=.*/SRC_SMJD=$line/' main_header.txt"
	elif [ $c -eq 3 ]
	then
		sed -i "s/^PROJID=.*/PROJID=$line/" main_header.txt
		echo "Updating PROJID with sed -i 's/^PROJID=.*/PROJID=$line/' main_header.txt"
	elif [ $c -eq 4 ]
	then
		sed -i "s/^OBSNCHAN=.*/OBSNCHAN=$line/" main_header.txt
		echo "Updating OBSNCHAN with sed -i 's/^OBSNCHAN=.*/OBSNCHAN=$line/' main_header.txt"
	else
		sed -i "s/^NPOL=.*/NPOL=$((2*$line))/" main_header.txt
		echo "Updating OBSNCHAN with sed -i 's/^NPOL=.*/NPOL=$((2*$line))/' main_header.txt"
		

	fi 
done <$file
