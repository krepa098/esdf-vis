#!/bin/bash
ffmpeg -stream_loop 0 -i $1 -crf 32 -vf scale=640:-1 -sws_flags neighbor $2
ffmpeg -i $2 -filter_complex "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=32[p];[s1][p]paletteuse=dither=bayer" $1.gif
