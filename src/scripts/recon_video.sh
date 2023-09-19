#!/bin/bash

log_dir="$1%d.png"

ffmpeg -framerate 10 -i $log_dir $2.mp4