#!/usr/bin/env python

import cv2 as cv
import os
import argparse
from tqdm import tqdm

# Initialize parser
parser = argparse.ArgumentParser(description='Extracts frames from a specified video.')
parser.add_argument('vidname', type=str, help='Name of video (mp4 format).')

# Main function
def main():
    # Take all arguments from terminal
    args = parser.parse_args()
    print(f'Getting frames from ./videos/{args.vidname}.mp4')

    # Start the video to take the necessary frames
    vidcap = cv.VideoCapture('videos/'+args.vidname+'.mp4')
    total_frame_count = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    if total_frame_count == 0:
        # Since cv.VideoCapture can't force errors when no video is found, we do it manually.
        raise IndexError
    
    # Create output folder if it wasn't created yet
    if not os.path.exists('frames/'+args.vidname):
        os.mkdir('frames/'+args.vidname)
    
    # Start counters
    pbar = tqdm(desc='READING FRAMES', total=total_frame_count, unit=' frames', dynamic_ncols=True)
    frame_no = 0
    
    while(vidcap.isOpened()):
        frame_exists, curr_frame = vidcap.read()
        if frame_exists:
            cv.imwrite("frames/"+args.vidname+"/frame%d.jpg" % frame_no, curr_frame)
        else:
            pbar.close()
            print(f'All frames were saved in /sets/{args.vidname}')
            break
        frame_no += 1
        pbar.update(1)
    vidcap.release()

if __name__ == '__main__': main()
