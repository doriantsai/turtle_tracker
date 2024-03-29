#!/bin/bash

base_dir="/home/serena/Data/Turtles/"
to_do_dir=${base_dir}videos


for f in "$to_do_dir"/*
do 
    ext=${f##*.}
    fn="$(basename $f .$ext)"
    save_dir=${base_dir}bash_track_out/"$fn"
    echo "$f"
    echo "$save_dir"
    mkdir -p "$save_dir"
    python SMTrackingPipeline.py $f $save_dir
done