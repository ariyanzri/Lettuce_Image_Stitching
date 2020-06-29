#!/bin/bash

Home_dir=$1
Pass=$2
Scan_date=$3
Dest=$4
Code_loc=$5

# Coppying the iRods environment config file
cd ~

if test -d .irods; then
    cd .irods
else
    mkdir .irods
    cd .irods
fi

#if test -f /app/$Home_dir/.irods/.irodsEnv; then
#    cp /app/$Home_dir/.irods/.irodsEnv ~/.irods/.
#else
#    echo "iRods not initialized on the host. Please initialize iRods on the host first."
#    exit 1
#fi

cp /app/$Home_dir/.irods/.irodsEnv ~/.irods/.

# Initializing iRods
/app/usr/local/bin/iinit $Pass


# Creating directories
cd /app/$Dest
mkdir $Scan_date-rgb
cd $Scan_date-rgb
mkdir SIFT
mkdir logs


# Downloading files from iRods
/app/usr/local/bin/iget -rKVP /iplant/home/emmanuelgonzalez/gantry_test_outputs/2020-06-16_test_scans/$Scan_date\_coordinates.csv
/app/usr/local/bin/iget -rKVP /iplant/home/emmanuelgonzalez/gantry_test_outputs/2020-06-16_test_scans/$Scan_date.tar.gz
/app/usr/local/bin/iget -rKVP /iplant/home/ariyanzarei/metadata/lids.txt
/app/usr/local/bin/iget -rKVP /iplant/home/ariyanzarei/metadata/season10_ind_lettuce_2020-05-27.csv

# Untarring
tar -xvf $Scan_date.tar.gz
rm $Scan_date.tar.gz


# Running the Geocorrection script

python $Code_loc/Dockerized_GPS_Correction.py $Scan_date $Code_loc/geo_correction_config.txt $Dest

