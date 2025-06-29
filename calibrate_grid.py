#!/usr/bin/env python

"""
Author original code for MATLAB: fcarrero
Script translation and updates to Python by mnrojas2

* If source is a video, run get_frames.py to get all possible images. You can manually remove them if they don't fit for the calibration.
* Requires having all images inside a folder. If get_frames was run first, then the folder was created automatically.
"""

import os
import argparse
import cv2 as cv
import glob
import re
import numpy as np
import camera
from scipy import ndimage
from skimage import measure, morphology
from skimage import color as skc
from matplotlib import pyplot as plt
from tqdm import tqdm


def calculate_grid_angle(I, cntrd, cntrd_offset, plot):
    # Auxiliary function to calculate the relative angle between the grid and the camera in the frame

    # Get all row centroids and determine the distance between all centroids and the center of the line (cntrd).
    row_cntrd = center_of_mass_per_row(I, cntrd)
    
    # Filter all elements near the center (cntrd_offset) and get the maximum distance shared by both tails.
    row_filtered = row_cntrd[(row_cntrd[:, 2] < np.min([row_cntrd[0, 2], row_cntrd[-1, 2]])) & (row_cntrd[:, 2] >= cntrd_offset)]
    row_filtered = row_filtered[(row_filtered[:, 2] < np.min([row_filtered[0, 2], row_filtered[-1, 2]]))]
    
    # Calculate weights
    distances = np.linalg.norm(row_filtered[:, :2] - row_filtered[::-1, :2], axis=1)
    row_widths = row_filtered[:, 3]+row_filtered[::-1, 3]
    angle_weights = (distances/row_widths)[:len(distances)//2-1][::-1]
    angle_weights = angle_weights/np.sum(angle_weights)
    
    # Calculate angles
    angles_calc = ((row_filtered[:, 1]-row_filtered[::-1, 1])/(row_filtered[:, 0]-row_filtered[::-1, 0]))
    angle_vec = -np.degrees(np.arctan(angles_calc[:len(angles_calc)//2-1]))[::-1]

    # Calculate the relative angle and standard deviation of the angle
    relative_angle = np.average(angle_vec, weights=angle_weights)
    STD_final = np.std(angle_vec)
    
    # If plot is enabled
    if plot:
        # Create new lists of angle and standard deviation for plotting, where zero values are replaced by np.nan
        angle_plot = angle_vec.copy()
        angle_plot[angle_plot==0] = np.nan
        
        # Calculate incremental STD angles
        STD_plot = np.array([np.std(angle_vec[:i]) for i in range(len(angle_vec))])
        STD_plot[STD_plot==0] = np.nan
        
        # Create the x axis for the plot
        actual_radius = np.arange(1+cntrd_offset, 1+cntrd_offset+angle_plot.shape[-1])
        
        # Create subplots
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        # First subplot
        ax1.plot(actual_radius, angle_plot, 'r-', linewidth=0.5, label='Relative angle')
        ax1.set_xlabel('Distance from center (pix)')
        ax1.set_ylabel('Relative angle (°)', color='r')
        ax1.axhline(y=relative_angle, linewidth=0.5, linestyle='-', color='b', label='Mean relative angle')
        ax1.axhline(y=relative_angle+0.5, linewidth=0.5, linestyle='--', color='b')
        ax1.axhline(y=relative_angle-0.5, linewidth=0.5, linestyle='--', color='b')

        # Second subplot with its own y-axis
        ax2.plot(actual_radius, STD_plot, 'g-', linewidth=0.5, label='Accumulated Error (STD)')
        ax2.set_ylabel('Error (°)', color='g')
        
        # Combine the legend handles and labels from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles = handles + handles2
        all_labels = labels + labels2

        # Create a single legend, add a gridline, text and the title to the plot
        ax1.legend(all_handles, all_labels, loc='upper right')
        ax1.grid(visible=True, which='both')
        
        plt.annotate(f"Relative angle is: {np.round(relative_angle,3)}° ± {np.round(STD_final,3)}°", xy=(0.02, 0.95), xycoords='axes fraction', textcoords='axes fraction', ha='left')
        plt.title(f"Relative angle vs distance from center")
        
        # Plot an histogram of the obtained angle values
        plt.figure()
        plt.hist(angle_vec, bins=30, edgecolor='black')
        plt.title('Distribution of relative angles')
        plt.xlabel('Relative angle (deg)')
        plt.ylabel('Frequency')
        plt.show()

    return relative_angle, STD_final


def delta_E(image_1_rgb, color_target, sigma=2, dmax=1):
# Color filter function from photogrammetry software (Author: Federico Astori)
    
    # Convert image from RGB to CIE Lab
    Lab1 = cv.cvtColor((image_1_rgb/255).astype('float32'), cv.COLOR_RGB2LAB)
    
    # Rewrite the color_target vector from RGB to CIE Lab
    Lab2 = skc.rgb2lab(color_target.reshape(1, 1, 3))

    # Calculate the difference of color between the image and the color target value
    deltae1 = skc.deltaE_ciede2000(Lab1, Lab2)
    
    # Apply a gaussian filter
    deltae = ndimage.gaussian_filter(deltae1,3)
    
    # Determine minimum value of the previous result
    minDeltaE = np.min(deltae)

    # If the difference between the blurred image pixel and the color target is less than 60, then calculates fimage, otherwise return an array of zeros
    if minDeltaE < 60:
        fimage = dmax * np.exp(-(deltae-minDeltaE)**2/(2*(sigma**2)))
        fimage[fimage<0.65] = 0
    else:
        fimage = np.zeros_like(deltae) #np.nan

    return fimage #, minDeltaE, Lab1, Lab2, deltae1, deltae


def center_of_mass_per_row(image, centroid):
# Calculate the center of mass of an image per each row
    rows_centroid = []
    for i in range(len(image)):
        if np.sum(image[i]) != 0:
            com = ndimage.center_of_mass(image[i])[0]
            dst = np.linalg.norm(np.array((i,com))-centroid)
            rows_centroid.append([i, com, dst, np.sum(image[i])])
    rows_centroid = np.array(rows_centroid)
    return rows_centroid


# Main
def calibrate_grid_main(folder, calibfile=False, plot=False, fplot=False, std_show=False):
    # Get relative angle and error values from a set of images inside a folder

    # Get images from directory
    frames_path = os.path.normpath(folder)
    
    # RX0-II Camera intrinsic parameters for calibration
    # Camera matrix (x, y values are inverted since the image will be taken as vertical)
    if calibfile:
        cam = camera.Camera(calibfile)
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
    print(f"Searching for images in {frames_path}")
    images = glob.glob(f'{frames_path}/*.jpg')
    if len(images) == 0:
        images = glob.glob(f'{frames_path}/*.png')
    images = sorted(images, key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    
    relangles = []
    stds = []
    frames_results = np.empty((0,3), dtype=float)
    
    # Change this value to segment only the center point of the polarized laser projection (ideally max brightness value of the picture)
    centroid_threshold = 99 # 200
    
    # Change this value to increase or decrease the horizontal (vertical) range where the laser trace would be in the image
    mask_window = 150
    
    # Change this value to adjust the minimum size of objects in the binarized image while trying to get the best laser trace
    small_object_threshold = 500
    
    # Initialize the progress bar
    pbar = tqdm(desc='READING FRAMES', total=len(images), unit=' frames', dynamic_ncols=True, miniters=1)
    
    for fname in images:
        # Read the image
        img0 = cv.imread(fname)
        
        # Get height and width of the image
        h, w, _ = img0.shape
        
        # Undistort image
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
        img0 = cv.undistort(img0, camera_matrix, dist_coeff, None, new_camera_matrix)
        
        # If the image is not vertical, rotate it
        if w > h:
            img0 = cv.rotate(img0, cv.ROTATE_90_COUNTERCLOCKWISE)
        
        # Convert image to LAB
        img_lab = cv.cvtColor((img0/255).astype('float32'), cv.COLOR_BGR2LAB)

        # Get lightness channel
        img_labl = img_lab[:,:,0]
        
        # Get a binary image by thresholding it with a top value of brightness (looking to find the traces of the polarized laser)
        _, img_labl_max = cv.threshold(img_labl, centroid_threshold, 1, cv.THRESH_BINARY)
        
        if plot:
            plt.figure(0)
            plt.imshow(img_labl_max)
        
        # Find the centroids of the laser traces in the image
        BW_cntrd_labels = measure.label(img_labl_max)
        BW_cntrd_props = sorted(measure.regionprops(BW_cntrd_labels), key=lambda r: r.area, reverse=True)

        # Choose the brightest centroid (the center of the laser trace)
        cntrd = BW_cntrd_props[0].centroid
        
        # Create a mask around the centroid found
        mask = np.zeros(img0.shape[:2], img0.dtype)
        mask[:, int(cntrd[1])-mask_window:int(cntrd[1])+mask_window] = 1
        
        if plot:
            plt.figure(1)
            plt.imshow(img_labl_max)
            plt.axhline(y=cntrd[0], linestyle='--', linewidth=0.5, color='red')
            plt.axvline(x=cntrd[1], linestyle='--', linewidth=0.5, color='red')
             
        # Determine the color and get the distance with respect to that color in all the image (converted to LAB space)
        color_lab = skc.rgb2lab(np.array((255,0,0)).reshape(1, 1, 3))
        img_labl = skc.deltaE_ciede2000(img_lab, color_lab, kL=2.0, kC=1.0, kH=2.0)
        
        # Change contrast and brightness of the image
        img_labl = cv.convertScaleAbs(img_labl, alpha=10, beta=0)
        
        img_labl_BW = cv.adaptiveThreshold(img_labl, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, -5) # -5 for 209 / -25 for 206
        
        # Apply the filter around the centroid to only keep the line
        img_labl_BW = cv.bitwise_and(img_labl_BW, img_labl_BW, mask = mask)
        
        # Remove small objects from the image, to only keep the laser trace
        img_open = cv.morphologyEx(img_labl_BW, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 21)))
        img_erode = cv.morphologyEx(img_open, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 21)))
        img_close = cv.morphologyEx(img_erode, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 21)))
        BW = (morphology.remove_small_objects(np.array(img_close, dtype=bool), small_object_threshold)).astype(int)
                
        if plot:
            plt.figure(2)
            plt.imshow(BW)
        
        # Filter just a slice of the former thresholded to get the potential location of the polarized laser line
        BW[:, :int(np.ceil(cntrd[1])) - 100] = 0
        BW[:, int(np.ceil(cntrd[1])) + 100:] = 0
        
        if plot:
            plt.figure(3)
            plt.imshow(BW)

            plt.figure(4)
            plt.imshow(BW)
            plt.axhline(y=cntrd[0], linestyle='--', linewidth=0.5, color='red')
            plt.axvline(x=cntrd[1], linestyle='--', linewidth=0.5, color='red')

        # Set offset distance from line center to skip the brightest part of the line
        cntrd_offset = 150
        
        # Set expected line length from center
        trace_length = 1000

        # Execute the calculate_gridangle one more time to get the graphs of the angles and a final result for angle and angle standard deviation
        try:
            angle, error = calculate_grid_angle(BW, cntrd, cntrd_offset, std_show)
            
            fnumber = int(re.search(r'frame(\d+)', fname).group(1))           
            fresults = np.array([fnumber, angle, error])
            frames_results = np.vstack((frames_results, fresults))

            if plot:            
                plt.figure(4)
                x0 = [cntrd[0], cntrd[0] + -(cntrd_offset + trace_length) * np.cos(np.deg2rad(-angle))]
                y0 = [cntrd[1], cntrd[1] + -(cntrd_offset + trace_length) * np.sin(np.deg2rad(-angle))]

                plt.plot(y0, x0, linestyle='--', color='red')
                
                x1 = [cntrd[0], cntrd[0] + (cntrd_offset + trace_length) * np.cos(np.deg2rad(-angle))]
                y1 = [cntrd[1], cntrd[1] + (cntrd_offset + trace_length) * np.sin(np.deg2rad(-angle))]

                plt.plot(y1, x1, linestyle='--', color='red')
                plt.xlim(cntrd[1]-cntrd_offset - trace_length, cntrd[1]+cntrd_offset + trace_length)
                
                plt.show()
        except: pass
        
        # Update the progress bar
        pbar.update(1)
    pbar.close()
    
    # Remove outliers
    # Get percentile threshold
    per95 = np.percentile(frames_results[:,2], 95)
    frames_results = frames_results[frames_results[:,2] < per95]
    
    # Print statistics results
    meanstd = f"Average angle: {frames_results[:,1].mean()} deg, average error: {frames_results[:,2].mean()} deg, std angle: {frames_results[:,1].std()} deg."
    print(meanstd)           
    
    # Save the array in a txt file
    np.savetxt(f"{frames_path}_angle.txt", frames_results, delimiter=',', header='frame number, angle (deg), error (deg)', footer=meanstd)
    
    if fplot:
        # Get the angles from the array
        relangles = frames_results[:,1]
        
        # Create subplots
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        # First subplot
        ax1.plot(relangles, 'r-', linewidth=0.5, label='Relative angles')
        ax1.set_ylabel('Relative angle (°)', color='r')

        # Second subplot with its own y-axis
        ax2.plot(stds, 'g-', linewidth=0.5, label='Angle error (STD)')
        ax2.set_ylabel('Error (°)', color='g')
        
        # Combine the legend handles and labels from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles = handles + handles2
        all_labels = labels + labels2

        # Create a single legend, add a gridline, text and the title to the plot
        ax1.legend(all_handles, all_labels, loc='upper right')
        ax1.grid(visible=True, which='both')
        
        plt.title(f"Relative angle and Angle Error")
        
        # Plot histogram
        plt.figure()
        plt.hist(relangles, bins=30, edgecolor='black')
        plt.title('Distribution of relative angles')
        plt.xlabel('Relative angle (deg)')
        plt.ylabel('Frequency')

        # Plot image with the expected angled line
        plt.figure()
        plt.imshow(cv.cvtColor(img0, cv.COLOR_BGR2RGB))
        
        x0 = [cntrd[0], cntrd[0] + -(cntrd_offset + trace_length) * np.cos(np.deg2rad(-relangles.mean()))]
        y0 = [cntrd[1], cntrd[1] + -(cntrd_offset + trace_length) * np.sin(np.deg2rad(-relangles.mean()))]

        plt.plot(y0, x0, linestyle='--', color='blue')
        
        x1 = [cntrd[0], cntrd[0] + (cntrd_offset + trace_length) * np.cos(np.deg2rad(-relangles.mean()))]
        y1 = [cntrd[1], cntrd[1] + (cntrd_offset + trace_length) * np.sin(np.deg2rad(-relangles.mean()))]

        plt.plot(y1, x1, linestyle='--', color='blue')
        plt.xlim(cntrd[1]-cntrd_offset - trace_length, cntrd[1]+cntrd_offset + trace_length)
        
        plt.show()

            
if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Obtains the relative angle between the camera and polarized wave grid from a series of pictures.')
    parser.add_argument('folder', type=str, help='Name of folder containing the frames.')
    parser.add_argument('-cb', '--calibfile', type=str, metavar='file', help='Name of the file containing calibration results (*.txt), for point reprojection and/or initial guess during calibration.')
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='Show plots from each image showing some of the steps of the procedure to find the center, extension and angle of the projected laser trace.')
    parser.add_argument('-fp', '--fplot', action='store_true', default=False, help='Show plots of accumulated results, including an histogram and a line with the average relative angle over the last frame.')
    parser.add_argument('-sd', '--std_show', action='store_true', default=False, help='Show a plot of the standard deviation of the angle for each image.')
    
    # Get parse data
    args = parser.parse_args()
    
    # Main
    calibrate_grid_main(args.folder, args.calibfile, args.plot, args.fplot, args.std_show)