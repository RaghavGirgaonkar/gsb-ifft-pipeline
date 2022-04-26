#!/bin/bash
#Author: Raghav Girgaonkar
#April 2022
#For any queries please send an email to raghav.girgaonkar@gmail.com

#set -x

#Usage function
function usage {
	echo "This is a bash script to convert PASV data from the uGMRT to GUPPI RAW format"
	echo " "
	echo "Usage: bash pasv2guppi.sh -l <LTA File path for header information> -u <Pol 1 file path> -d <Pol 2 file path> -o <Output file stem> -b <Bandwidth in MHz> -f <Starting Frequency in MHz> -t <Time duration in seconds to convert>"
	echo " "
	echo "Please make sure you have the following files in the same directory as you are running this script:"
	echo "1. main_header.txt"
	echo "2. pasv2guppi_multiplefiles.c, guppi_header.c and guppi_header.h"
	echo "3. Makefile"
	echo "5. headerfromlta.sh"
	echo "6. updateparams.sh"
	echo "7. updateheader.sh"
	echo " "
	echo "l: LTA File path for header info, optional if source list and other parameters already extracted"
	echo "u: File path of Polarisation 1 data"
	echo "d: File path of Polarisation 2 data"
	echo "o: Stem of Output file i.e the name without the extension"
	echo "b: Bandwidth of observation in MHz"
	echo "f: Starting Frequency of observation in MHz"
	echo "t: Time in seconds of data to convert to GUPPI RAW"
	echo " "
	exit 1
}

#Define list of arguments expected in the input 
optstring=":l:u:d:o:b:f:t:h"

while getopts ${optstring} arg; do
  case ${arg} in
    l) 
       LTA=${OPTARG}
       #echo "${LTA}" 
       echo "LTA file is $LTA"
       ;;
    u)
       P1=${OPTARG}
       echo "Pol 1 file is ${P1}" 
       ;;
    d)
       P2=${OPTARG}
       echo "Pol 2 file is ${P2}" 
       ;;
    o)
       o=${OPTARG}
       echo "Output file stem is ${o}" 
       ;;
    b) 
       bandwidth=${OPTARG}
       echo "Bandwidth is ${bandwidth} MHz" 
       ;;
    f)
       f=${OPTARG}
       echo "Starting frequency is ${f}" 
       ;;

    t) 
       t=${OPTARG}
       echo "Number of seconds to process is ${t}" 
       ;;
    h)
       usage
       ;; 
    :)
       echo "Invalid option: $OPTARG" 1>&2
       usage
       exit 1
       ;;

    \?)
      echo "Invalid option: -${OPTARG}."
      usage
      ;;
  esac
done

#Checking if all compulsory arguments have been passed
if [[ $P1 && $P2 && $bandwidth && $t && $o && $f ]]
then
	echo "All required arguments have been passed!"
else
	echo "Please enter all required arguments -u -d -o -b -f -t"
	exit 1
fi



#Extract header info if LTA File has been passed, if not then skip this step as it is assumed to have been extracted already
if [[ $LTA ]]
then
	if [ -f "$LTA" ]
	then
        	echo "LTA File exists!"
	else
        	echo "LTA File doesn't exist, please check name of file"
        	exit 1
	fi
	echo "Gathering Metadata for all sources from the LTA file $LTA ..."
	egrep -wa 'CHANNELS|POLS|MJD_REF|OBJECT|^CODE' $LTA > ltaheaderinfo.txt
	echo "Done!"
	echo " "
	echo "Making list of sources..."
	grep 'OBJECT' ltaheaderinfo.txt | awk '{print $NF}' > sources.txt
	echo "Done!"
else
	echo "Skipping step of extracting header parameters from LTA file"
fi


#Update header to source specified by user
if [[ -f "$PWD/ltaheaderinfo.txt" ]] && [[ -f "$PWD/sources.txt" ]]
then
	echo "Running bash updateheader.sh"
	bash updateheader.sh
else
	echo "Header params and sources not yet extracted, please extract them"
	exit 1
fi


#Make the C program
if [[ -f "$PWD/pasv2guppi_multiplefiles.c" ]] && [[ -f "$PWD/Makefile" ]] && [[ -f "$PWD/guppi_header.h" ]] && [[ -f "$PWD/guppi_header.c" ]]  
then
	make
	make clean
else
	echo "Either Makefile or pasv2guppi_multiples.c or guppi_header.c or guppi_header.h not in directory, please copy them to this directory"
	exit 1
fi

#Run pasv2guppi_multiplefile.c
if [[ -f "$P1" ]] && [[  -f "$P2" ]]
then
	echo "Running ./pasv2guppi_multiplefiles $P1 $P2 $o $b $f $t"
	./pasv2guppi_multiplefiles $P1 $P2 $o $b $f $t
else
	echo "File doesn't exist"
	exit 1

fi
