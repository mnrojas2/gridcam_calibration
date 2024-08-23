import os
import cv2
import glob
import re
import argparse
import numpy as np
from skimage import measure, morphology
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Obtains the binarized image with only the laser trace visible. Script used to test parameters before being used in calibrate_grid.py.')
parser.add_argument('folder', type=str, help='Name of folder containing the frames.')
parser.add_argument('-s', '--save', action='store_true', default=False, help='Saves all processed frames inside a folder in /results/.')
parser.add_argument('-p', '--plot', action='store_true', default=False, help='Show plots from the process.')

# Get parse data
args = parser.parse_args()

line_threshold = 100
centroid_threshold = 200
mask_window = 150
small_object_threshold = 500


# Get images from directory
print(f"Searching images in {args.folder}/")
images = glob.glob(f'./frames/{args.folder}/*.jpg')
if len(images) == 0:
    images = glob.glob(f'./frames/{args.folder}/*.png')
images = sorted(images, key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

# Create output folder if it wasn't created yet
if args.save and not os.path.exists('results/bin/'):
    os.mkdir('results/bin/')

for fname in images:
    # Read the image
    img0 = cv2.imread(fname)

    # Convert image to LAB
    img_lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)

    # Get lightness channel
    img_labl = img_lab[:,:,0]

    # Get the center of the laser trace, first by thresholding the image around the brightest points
    _, img_labl_max = cv2.threshold(img_labl, centroid_threshold, 1, cv2.THRESH_BINARY)
    
    # Find the centroids of the laser traces in the image
    BW_cntrd_labels = measure.label(img_labl_max)
    BW_cntrd_props = sorted(measure.regionprops(BW_cntrd_labels), key=lambda r: r.area, reverse=True)

    # Choose the brightest centroid (the center of the laser trace)
    cntrd = BW_cntrd_props[0].centroid
    
    # Create a mask around the centroid found
    mask = np.zeros(img0.shape[:2], img0.dtype)
    mask[int(cntrd[0])-mask_window:int(cntrd[0])+mask_window, :] = 1

    # Change contrast and brightness of the image
    img_labl = cv2.convertScaleAbs(img_labl, alpha=10, beta=0)
    
    # Use an adaptative threshold to now get as much as possible of the line
    img_labl_BW = cv2.adaptiveThreshold(img_labl, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
    
    # Apply the filter around the centroid to only keep the line
    img_labl_BW = cv2.bitwise_and(img_labl_BW, img_labl_BW, mask = mask)
    
    # Remove small objects from the image, to only keep the laser trace
    img_open = cv2.morphologyEx(img_labl_BW, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 3)))
    img_erode = cv2.morphologyEx(img_open, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 5)))
    img_morph = (morphology.remove_small_objects(np.array(img_erode, dtype=bool), small_object_threshold)).astype(int)

    plt.figure()
    plt.imshow(img_morph)
    plt.plot(cntrd[1], cntrd[0], '-o')
    plt.title(fname.split('\\')[-1][:-4])
    
    if args.save:
        plt.savefig(f'results/bin/{fname.split('\\')[-1][:-4]}.png', bbox_inches='tight', dpi=300)
    
    if args.plot:
        plt.show()
    
    plt.close()


