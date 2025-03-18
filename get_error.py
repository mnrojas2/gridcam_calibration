#!/usr/bin/env python

import os
import argparse
import cv2 as cv
import glob
import re
import numpy as np
import camera
from scipy import ndimage
from skimage import measure, morphology
from matplotlib import pyplot as plt


def main():
    # Get relative angle and error values from a set of images inside a folder
    
    # Get images from directory
    frames_path = os.path.normpath(args.folder)
    print(f"Searching images in {frames_path}")
    
    # RX0-II Camera intrinsic parameters for calibration
    if args.calibfile:
        cam = camera.Camera(args.calibfile)
        camera_matrix = cam.cam_matrix()
        dist_coeff = cam.dist_coeff()
    else:
        # Camera matrix
        fx = 2568.584961
        fy = 2569.605957
        cx = 1087.135376
        cy = 1881.56543

        camera_matrix = np.array([[fx, 0., cx],
                                    [0., fy, cy],
                                    [0., 0., 1.]], dtype = "double")

        # Distortion coefficients
        k1 = 0.019473
        k2 = -0.041976
        p1 = -0.000273
        p2 = -0.001083
        k3 = 0.030603

        dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3]))

    # Get images from directory
    images = glob.glob(f'{frames_path}/*.jpg')
    if len(images) == 0:
        images = glob.glob(f'{frames_path}/*.png')
    images = sorted(images, key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

    images_list = []
    line_distance = []

    # Change this value to segment only the center point of the polarized laser projection (ideally max brightness value of the picture)
    centroid_threshold = 200

    # Change this value to increase or decrease the horizontal (vertical) range where the laser trace would be in the image
    mask_window = 150

    # Change this value to adjust the minimum size of objects in the binarized image while trying to get the best laser trace
    small_object_threshold = 500


    for fname in images:
        img0 = cv.imread(fname)
        
        # If the image is not vertical, rotate it
        h, w, _ = img0.shape
        if w > h:
            img0 = cv.rotate(img0, cv.ROTATE_90_COUNTERCLOCKWISE)
        
        # Undistort image
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
        img0 = cv.undistort(img0, camera_matrix, dist_coeff, None, new_camera_matrix)
        
        # Convert image to LAB
        img_lab = cv.cvtColor(img0, cv.COLOR_BGR2LAB)

        # Get lightness channel
        img_labl = img_lab[:,:,0]
        
        
        # Get a binary image by thresholding it with a top value of brightness (looking to find the traces of the polarized laser)
        _, img_labl_max = cv.threshold(img_labl, centroid_threshold, 1, cv.THRESH_BINARY)
        
        # Find the centroids of the laser traces in the image
        BW_cntrd_labels = measure.label(img_labl_max)
        BW_cntrd_props = sorted(measure.regionprops(BW_cntrd_labels), key=lambda r: r.area, reverse=True)

        # Choose the brightest centroid (the center of the laser trace)
        cntrd = BW_cntrd_props[0].centroid
        
        # Create a mask around the centroid found
        mask = np.zeros(img0.shape[:2], img0.dtype)
        mask[:, int(cntrd[1])-mask_window:int(cntrd[1])+mask_window] = 1
        
        
        # Change contrast and brightness of the image
        img_labl = cv.convertScaleAbs(img_labl, alpha=10, beta=0)
        
        # Use an adaptative threshold to now get as much as possible of the line
        img_labl_BW = cv.adaptiveThreshold(img_labl, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, 2)
        
        # Apply the filter around the centroid to only keep the line
        img_labl_BW = cv.bitwise_and(img_labl_BW, img_labl_BW, mask = mask)
        
        # Remove small objects from the image, to only keep the laser trace
        img_open = cv.morphologyEx(img_labl_BW, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 21)))
        img_erode = cv.morphologyEx(img_open, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 21)))
        BW = (morphology.remove_small_objects(np.array(img_erode, dtype=bool), small_object_threshold)).astype(int)
        
        # Get relative angle from file
        rel_angle = -0.5130709172843116 # Hardcoded #<<<<<<<<<<<<<<<<<<<<<
        
        # Determine the slope and offset to plot it in the image later
        m = np.tan(np.radians(-rel_angle))
        b = cntrd[1] - m * cntrd[0]
        
        # Make a line with the values got
        t = np.arange(img_labl.shape[0])
        ft = m * t + b
        
        cntrd_offset = 150
        search_radius = 1000

        # Get the list of centroids from each row where the line is visible
        row_centroids_top = [[x, float(ndimage.center_of_mass(BW[x,:])[0])] for x in range(int(cntrd[0]) - search_radius, int(cntrd[0]) - cntrd_offset) if not np.isnan(ndimage.center_of_mass(BW[x,:])[0])]
        row_centroids_bot = [[x, float(ndimage.center_of_mass(BW[x,:])[0])] for x in range(int(cntrd[0]) + cntrd_offset, int(cntrd[0]) + search_radius) if not np.isnan(ndimage.center_of_mass(BW[x,:])[0])]
        
        row_centroids_top = np.array(row_centroids_top)
        row_centroids_bot = np.array(row_centroids_bot)
        row_centroids = np.concatenate([row_centroids_top, row_centroids_bot])
        
        # Plot the straight line, the centroid of the laser trace and the line following the centroids of each row
        if args.plot:
            plt.figure()
            plt.imshow(BW)
            plt.plot(ft, t)
            plt.plot(cntrd[1], cntrd[0], 'o')
            plt.plot(row_centroids[:,1], row_centroids[:,0])
        
        # Calculate the average distance between the centroid line and the laser trace line
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
        
        line_distance.append(dist_sum/row_centroids.shape[0])
        
        final_str = f"{fname.split('\\')[-1]} average error between laser trace and straight line is: {dist_sum/row_centroids.shape[0]}"
        images_list.append(final_str)
        print(final_str)
        
        if args.plot:
            plt.show()

    line_distance = np.array(line_distance)

    meanstd = f"Average distance between laser trace and straight line is: {line_distance.mean()}, standard deviation is: {line_distance.std()}."
    print(meanstd)  
    
    with open(f"{frames_path}_disterror.txt", 'w') as f:
        for line in images_list:
            f.write(f"{line}\n")
        f.write(f"{meanstd}\n")
    f.close()


if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Obtains the relative error between the projected line and the calculated straight line.')
    parser.add_argument('folder', type=str, help='Name of folder containing the frames.')
    parser.add_argument('-cb', '--calibfile', type=str, metavar='file', help='Name of the file containing calibration results (*.txt), for point reprojection and/or initial guess during calibration.')
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='Show plots from the process.')

    # Get parse data
    args = parser.parse_args()
    
    # Main
    main()