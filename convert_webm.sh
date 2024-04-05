#!/bin/bash
ffmpeg -stream_loop 0 -i $1 -crf 32 -vf scale=640:-1 -sws_flags neighbor $2
