#!/bin/bash

base_dir="/home/serena/Data/Turtles/"
to_do_dir=${base_dir}videos


for f in "$to_do_dir"/*
do 
    ext=${f##*.}
    fn="$(basename $f .$ext)"
    save_dir=${base_dir}bash_track_out/"$fn"
    echo "Video file: $f"
    echo "Save directory: $save_dir"
    mkdir -p "$save_dir"
    python SMTrackingPipeline.py video_in_path:=$f output_path:=$save_dir
done