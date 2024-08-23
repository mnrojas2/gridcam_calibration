import cv2
import glob
import re
import argparse
import numpy as np
from scipy import ndimage
from skimage import measure
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Obtains the relative error between the projected line and the calculated straight line.')
parser.add_argument('folder', type=str, help='Name of folder containing the frames.')
parser.add_argument('-p', '--plot', action='store_true', default=False, help='Show plots from the process.')


# Get parse data
args = parser.parse_args()

# Get relative angle and error values from a set of images inside a folder

# RX0-II Camera intrinsic parameters for calibration
# Camera matrix (vertical)
fx = 2568.584961
fy = 2569.605957
cx = 1087.135376
cy = 1881.565430

camera_matrix = np.array([[fx, 0., cx],
                        [0., fy, cy],
                        [0., 0., 1.]], dtype = "double")

# Radial distortion coefficients
k1 =  0.019473
k2 = -0.041976
k3 =  0.030603 

# Tangential distortion coefficients
p1 =  -0.000273
p2 =  -0.001083

dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3]))

# Get images from directory
print(f"Searching images in {args.folder}/")
images = glob.glob(f'./frames/{args.folder}/*.jpg')
if len(images) == 0:
    images = glob.glob(f'./frames/{args.folder}/*.png')
images = sorted(images, key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

images_list = []
relangles = []
stds = []

# Change this value to segment only the projected line from the laser
line_threshold = 9
"""
# [original]=5.1, [gridtest, gridtest_p2]=220, [C0015, C0019, C0020]=9
"""

# Change this value to segment only the center point of the polarized laser projection (ideally max brightness value of the picture)
center_threshold = 200

for fname in images:
    img0 = cv2.imread(fname)
    
    # If the image is not vertical, rotate it
    h, w, _ = img0.shape
    if w > h:
        img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    
    # Undistort image
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
    img_gray = cv2.undistort(img_gray, camera_matrix, dist_coeff, None, new_camera_matrix)
    img0 = cv2.undistort(img0, camera_matrix, dist_coeff, None, new_camera_matrix)
    
    # Get another binary image by thresholding it with a top value of brightness (looking to find the traces of the polarized laser)
    _, BW_cntrd = cv2.threshold(img_gray, center_threshold, 1, cv2.THRESH_BINARY)
    
    center = np.array([1710, 550])
    offset = np.array([150, 200])
    BW_cntrd[center[0]-offset[0]:center[0]+offset[0], center[1]-offset[1]:center[1]+offset[1]] = 0 # C0019
    
    # Find the centroids of the laser traces in the image
    BW_cntrd_labels = measure.label(BW_cntrd)
    BW_cntrd_props = sorted(measure.regionprops(BW_cntrd_labels), key=lambda r: r.area, reverse=True)

    # Choose the brightest centroid (most probably the center laser trace)
    cntrd = BW_cntrd_props[0].centroid
    
    # Get a binary image by thresholding it with a bottom value of brightness (to find the line)
    _, BW = cv2.threshold(img_gray, line_threshold, 1, cv2.THRESH_BINARY)
    
    # Remove the smaller areas of the binary image
    BW = cv2.morphologyEx(BW, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    
    # Apply an average filter of window length 8
    ws = 8
    BW = cv2.filter2D(BW, -1, kernel=np.ones((ws, ws)) / (ws**2), borderType=cv2.BORDER_CONSTANT)
    
    # Filter just a slice of the former thresholded to get the potential location of the polarized laser line
    BW[:, :int(np.ceil(cntrd[1])) - 100] = 0
    BW[:, int(np.ceil(cntrd[1])) + 100:] = 0
    
    # Get relative angle from file
    rel_angle = -1.0429820134301644 # Hardcoded
    
    # Determine the slope and offset to plot it in the image later
    m = np.tan(np.radians(-rel_angle))
    b = cntrd[1] - m * cntrd[0]
    
    # Make a line with the values got
    t = np.arange(img_gray.shape[0])
    ft = m * t + b
    
    cntrd_offset = 150
    search_radius = 1000

    row_centroids_top = [[x, float(ndimage.center_of_mass(BW[x,:])[0])] for x in range(int(cntrd[0]) - search_radius, int(cntrd[0]) - cntrd_offset) if not np.isnan(ndimage.center_of_mass(BW[x,:])[0])]
    row_centroids_bot = [[x, float(ndimage.center_of_mass(BW[x,:])[0])] for x in range(int(cntrd[0]) + cntrd_offset, int(cntrd[0]) + search_radius) if not np.isnan(ndimage.center_of_mass(BW[x,:])[0])]
    
    row_centroids_top = np.array(row_centroids_top)
    row_centroids_bot = np.array(row_centroids_bot)
    row_centroids = np.concatenate([row_centroids_top, row_centroids_bot])
    
    if args.plot:
        plt.figure()
        plt.imshow(BW)
        plt.plot(ft, t)
        plt.plot(cntrd[1], cntrd[0], 'o')
        plt.plot(row_centroids[:,1], row_centroids[:,0])
    
    dist_sum = 0
    for i in range(row_centroids.shape[0]):
        xi, yi = row_centroids[i, :]
        
        # Get closest point of the line to the point
        x_intersect = (m*(yi - cntrd[1]) + m**2 * cntrd[0] + xi) / (m**2 + 1)
        y_intersect = m * x_intersect + b
        
        # Get the distance (always positive, it's the absolute value)
        dist = np.linalg.norm(np.array((xi, yi)) - np.array((x_intersect, y_intersect)))
        
        # Plot the line
        if args.plot:
            plt.plot([yi, y_intersect], [xi, x_intersect])
        
        # Add this distance to get the average
        dist_sum += dist
    print(f"Error promedio entre linea y curva real: {dist_sum/row_centroids.shape[0]} pixeles")
    print(f"Error proporcional respecto a tamaño de la imagen: {dist_sum/BW.shape[1]/row_centroids.shape[0]}")
    
    if args.plot:
        plt.show()