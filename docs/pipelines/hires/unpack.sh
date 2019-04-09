#!/bin/bash

for i in HIRES*.tar
do
    echo unpacking $i
    tar -xf $i
done

for koa_folder in KOA_*
do

    for fname in $koa_folder/HIRES/raw/sci/*.fits.gz
    do
        mv $fname $koa_folder
    done

    for fname in $koa_folder/HIRES/raw/cal/*.fits.gz
    do
        mv $fname $koa_folder
    done

    rm -rf $koa_folder/HIRES/

    # unpack all .fits.gz files
    cd $koa_folder
    gzip -d *.fits.gz
    echo `ls *.fits | wc -l` fits files in $koa_folder

    # crete sub-folders with different dates
    for name in `for i in HI.*.fits; do echo ${i:7:4};done|sort|uniq`
    do
        mkdir $name
        mv HI.????$name.*.fits $name
        echo `ls $name| wc -l` fits files in $name
        # copy make log script into each sub-folder
    done
    cd ..
done

