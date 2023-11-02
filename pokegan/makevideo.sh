ffmpeg -framerate 25 -i %03d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4

ffmpeg -framerate 15 -i %03d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4

ffmpeg -i %03d.png -vcodec libx264  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -r 24  -y -an video.mp4 