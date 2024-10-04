"""
Author original code for MATLAB: fcarrero
Script translation and updates to Python by mnrojas2

* If source is a video, run get_frames.py to get all possible images. You can manually remove them if they don't fit for the calibration.
* Requires having all images inside a folder. If get_frames was run first, then the folder was created automatically.
"""

import cv2
import argparse
import glob
import re
import numpy as np
import scipy.optimize
from scipy import ndimage
from skimage import measure, morphology
from skimage import color as skc
from matplotlib import pyplot as plt


def calculate_gridangle(cntrd, I, cntrd_offset, searchradius, plot):
# Auxiliary function to calculate the relative angle between the grid and the camera in the frame

    # Create lists to save values for angle and standard deviation of the angle
    angle_vec = []
    STD_vec = []

    for i in range(searchradius):
        # Get slices of the image (rows), at a certain distance over and under the center of the line.
        top = I[int(np.floor(cntrd[0]-cntrd_offset-i)), :]
        bot = I[int(np.floor(cntrd[0]+cntrd_offset+i)), :]
        
        if sum(top) != 0 and sum(bot) != 0:            
            # Get the centroid location of both rows
            top_cntrd = ndimage.center_of_mass(top)[0]
            bot_cntrd = ndimage.center_of_mass(bot)[0]
            
            # Get the angles between the top and bottom centroids with respect from the center
            angle_top = np.degrees(np.arctan((top_cntrd - cntrd[1])/(cntrd_offset + i)))
            angle_bot = np.degrees(np.arctan((bot_cntrd - cntrd[1])/(cntrd_offset + i)))
                       
            # Get the difference between the top and bottom angles to get the average angle of the line
            angle_diff = (angle_top - angle_bot)/2 # angle_top is defined as -angle, compared to angle_bot

            angle_vec.append(angle_diff)
            STD_vec.append(np.std(angle_vec))
    
    # Convert the lists to numpy ndarray format
    angle_vec = np.array(angle_vec)
    STD_vec = np.array(STD_vec)
    
    # Create new lists of angle and standard deviation for plotting, where zero values are replaced by np.nan
    angle_plot = angle_vec.copy()
    angle_plot[angle_plot==0] = np.nan
    
    STD_plot = STD_vec.copy()
    STD_plot[STD_plot==0] = np.nan

    # Calculate the relative angle and standard deviation of the angle
    relative_angle = np.mean(angle_vec)
    STD_final = np.std(angle_vec)
    
    # If plot is enabled
    if plot:
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

# Main
def calibrate_grid():
    # Get relative angle and error values from a set of images inside a folder

    # RX0-II Camera intrinsic parameters for calibration
    # Camera matrix (x, y values are inverted since the image will be taken as vertical)
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
    
    # Change this value to segment only the center point of the polarized laser projection (ideally max brightness value of the picture)
    centroid_threshold = 200
    
    # Change this value to increase or decrease the horizontal (vertical) range where the laser trace would be in the image
    mask_window = 150
    
    # Change this value to adjust the minimum size of objects in the binarized image while trying to get the best laser trace
    small_object_threshold = 500
    
    for fname in images:
        # Read the image
        img0 = cv2.imread(fname)
        
        # If the image is not vertical, rotate it
        h, w, _ = img0.shape
        if w > h:
            img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Undistort image
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
        img0 = cv2.undistort(img0, camera_matrix, dist_coeff, None, new_camera_matrix)
        
        # Convert image to LAB
        img_lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)

        # Get lightness channel
        img_labl = img_lab[:,:,0]
        
        
        # Get a binary image by thresholding it with a top value of brightness (looking to find the traces of the polarized laser)
        _, img_labl_max = cv2.threshold(img_labl, centroid_threshold, 1, cv2.THRESH_BINARY)
        
        if args.plot:
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
        
        if args.plot:
            plt.figure(1)
            plt.imshow(img_labl_max)
            plt.axhline(y=cntrd[0], linestyle='--', linewidth=0.5, color='red')
            plt.axvline(x=cntrd[1], linestyle='--', linewidth=0.5, color='red')
        
        
        # Change contrast and brightness of the image
        img_labl = cv2.convertScaleAbs(img_labl, alpha=10, beta=0)
        
        # Use an adaptative threshold to now get as much as possible of the line
        img_labl_BW = cv2.adaptiveThreshold(img_labl, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
        
        # Apply the filter around the centroid to only keep the line
        img_labl_BW = cv2.bitwise_and(img_labl_BW, img_labl_BW, mask = mask)
        
        # Remove small objects from the image, to only keep the laser trace
        img_open = cv2.morphologyEx(img_labl_BW, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 21)))
        img_erode = cv2.morphologyEx(img_open, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 21)))
        BW = (morphology.remove_small_objects(np.array(img_erode, dtype=bool), small_object_threshold)).astype(int)
        
        if args.plot:
            plt.figure(2)
            plt.imshow(BW)
        
        
        # Filter just a slice of the former thresholded to get the potential location of the polarized laser line
        BW[:, :int(np.ceil(cntrd[1])) - 100] = 0
        BW[:, int(np.ceil(cntrd[1])) + 100:] = 0
        
        if args.plot:
            plt.figure(3)
            plt.imshow(BW)

            plt.figure(4)
            plt.imshow(BW)
            plt.axhline(y=cntrd[0], linestyle='--', linewidth=0.5, color='red')
            plt.axvline(x=cntrd[1], linestyle='--', linewidth=0.5, color='red')


        # Calculate_gridangle function parameters
        cntrd_offset = 150
        search_radius = 1000

        # Execute the calculate_gridangle one more time to get the graphs of the angles and a final result for angle and angle standard deviation
        angle, error = calculate_gridangle(cntrd, BW, cntrd_offset, search_radius, args.std_show)
        
        relangles.append(angle)
        stds.append(error)
        
        final_str = f"{fname.split('\\')[-1]} relative angle is: {str(np.round(angle, 3))} deg +/- {str(np.round(error, 3))} deg"
        images_list.append(final_str)
        print(final_str)

        if args.plot:
            plt.figure(4)
            x0 = [cntrd[0], cntrd[0] + -(cntrd_offset + search_radius) * np.cos(np.deg2rad(-angle))]
            y0 = [cntrd[1], cntrd[1] + -(cntrd_offset + search_radius) * np.sin(np.deg2rad(-angle))]

            plt.plot(y0, x0, linestyle='--', color='red')
            
            x1 = [cntrd[0], cntrd[0] + (cntrd_offset + search_radius) * np.cos(np.deg2rad(-angle))]
            y1 = [cntrd[1], cntrd[1] + (cntrd_offset + search_radius) * np.sin(np.deg2rad(-angle))]

            plt.plot(y1, x1, linestyle='--', color='red')
            plt.xlim(cntrd[1]-cntrd_offset - search_radius, cntrd[1]+cntrd_offset + search_radius)
            
            # plt.savefig(f'results/bin/{fname.split('\\')[-1][:-4]}-4.png', bbox_inches='tight', dpi=300)
            plt.show()
            
    relangles = np.array(relangles)
    stds = np.array(stds)
    
    meanstd = f"Average image relative angle is: {relangles.mean()} deg, average image error is: {stds.mean()} deg, standard deviation from images relative angles is: {relangles.std()} deg."
    print(meanstd)           
    
    with open(f"{args.folder}_output_python.txt", 'w') as f:
        for line in images_list:
            f.write(f"{line}\n")
        f.write(f"{meanstd}\n")
    f.close()
    
    if args.fplot:
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
        plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        
        x0 = [cntrd[0], cntrd[0] + -(cntrd_offset + search_radius) * np.cos(np.deg2rad(-relangles.mean()))]
        y0 = [cntrd[1], cntrd[1] + -(cntrd_offset + search_radius) * np.sin(np.deg2rad(-relangles.mean()))]

        plt.plot(y0, x0, linestyle='--', color='blue')
        
        x1 = [cntrd[0], cntrd[0] + (cntrd_offset + search_radius) * np.cos(np.deg2rad(-relangles.mean()))]
        y1 = [cntrd[1], cntrd[1] + (cntrd_offset + search_radius) * np.sin(np.deg2rad(-relangles.mean()))]

        plt.plot(y1, x1, linestyle='--', color='blue')
        plt.xlim(cntrd[1]-cntrd_offset - search_radius, cntrd[1]+cntrd_offset + search_radius)
        
        plt.show()
        
        
def delta_E(image_1_rgb, color_target, sigma=2, dmax=1):
# Color filter function from photogrammetry software (Author: Federico Astori)
    
    # Convert image from RGB to CIE Lab
    Lab1 = cv2.cvtColor((image_1_rgb/255).astype('float32'), cv2.COLOR_RGB2LAB)
    
    # Rewrite the color_target vector from RGB to CIE Lab
    Lab2 = skc.rgb2lab(color_target.reshape(1, 1, 3))

    # Calculate the difference of color between the image and the color target value
    deltae1 = skc.deltaE_ciede2000(Lab1, Lab2)
    
    # Apply a gaussian filter
    deltae = scipy.ndimage.gaussian_filter(deltae1,3)
    
    # Determine minimum value of the previous result
    minDeltaE = np.min(deltae)

    # If the difference between the blurred image pixel and the color target is less than 60, then calculates fimage, otherwise return an array of zeros
    if minDeltaE < 60:
        fimage = dmax * np.exp(-(deltae-minDeltaE)**2/(2*(sigma**2)))
        fimage[fimage<0.65] = 0
    else:
        fimage = np.zeros_like(deltae) #np.nan

    return fimage #, minDeltaE, Lab1, Lab2, deltae1, deltae

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Obtains the relative angle between the camera and polarized wave grid from a series of pictures.')
    parser.add_argument('folder', type=str, help='Name of folder containing the frames.')
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='Show plots from each image showing some of the steps of the procedure to find the center, extension and angle of the projected laser trace.')
    parser.add_argument('-fp', '--fplot', action='store_true', default=False, help='Show plots of accumulated results, including an histogram and a line with the average relative angle over the last frame.')
    parser.add_argument('-sd', '--std_show', action='store_true', default=False, help='Show a plot of the standard deviation of the angle for each image.')
    
    # Get parse data
    args = parser.parse_args()
    calibrate_grid()