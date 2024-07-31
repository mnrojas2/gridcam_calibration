"""
Author original code for MATLAB: fcarrero
Code translation to Python by mnrojas2

* If source is a video, run get_frames.py to get all possible images. You can manually remove them if they don't fit for the calibration.
* Requires having all images inside a folder. If get_frames was run first, then the folder was created automatically.
"""

import cv2
import argparse
import glob
import re
import numpy as np
import scipy.optimize
import scipy.ndimage
from matplotlib import pyplot as plt
from skimage import measure
from skimage import color as skc


def calculate_gridangle(cntrd, I, cntrd_offset, searchradius, plot, skip, skipstart, skiplength):
# Auxiliary function to calculate the relative angle between the grid and the camera in the frame

    # print(cntrd, I.shape, cntrd_offset, searchradius, plot, skip, skipstart, skiplength)
    loopvar = np.arange(searchradius)
    
    # If skip flag is enabled, the list will skip a certain range, defined by skipstart, skiplength and the offset center
    if skip == 1:
        skipstart = skipstart - cntrd_offset
        loopvar = loopvar[((loopvar <= skipstart) | (loopvar > skipstart + skiplength))]
        
    # Create lists to save values for angle and standard deviation of the angle
    angle_vec = []
    valid_angle_diff = []
    STD_vec = []

    for i in loopvar:
        top = I[int(np.floor(cntrd[0]-cntrd_offset-(i+1))), :]
        bot = I[int(np.floor(cntrd[0]+cntrd_offset+(i+1))), :]
        
        if sum(top) != 0 and sum(bot) != 0:
            top_weights = np.arange(1, 1+top.shape[-1]) * top
            top_cntrd = np.sum(top_weights) / np.sum(top)
            
            bot_weights = np.arange(1, 1+bot.shape[-1]) * bot
            bot_cntrd = np.sum(bot_weights) / np.sum(bot)
            
            angle_top = np.degrees(np.arctan((top_cntrd - cntrd[1])/(cntrd_offset + (i+1))))
            angle_bot = np.degrees(np.arctan((bot_cntrd - cntrd[1])/(cntrd_offset + (i+1))))
            
            angle_diff = angle_top - angle_bot
            radius_vector = cntrd_offset + (i+1)
            
            if angle_diff != 0:
                valid_angle_diff.append(angle_diff)

            angle_vec.append(angle_diff)  
            STD_vec.append(np.std(valid_angle_diff))
    
    # Convert the lists to numpy ndarray format
    angle_vec = np.array(angle_vec)
    STD_vec = np.array(STD_vec)
    
    # Create new lists of angle and standard deviation for plotting, where zero values are replaced by np.nan
    angle_plot = angle_vec.copy()
    angle_plot[angle_plot==0] = np.nan
    
    STD_plot = STD_vec.copy()
    STD_plot[STD_plot==0] = np.nan

    # Calculate the relative angle and standard deviation of the angle
    relative_angle = np.mean(valid_angle_diff)
    STD_final = np.std(valid_angle_diff)
    
    # If plot is enabled
    if plot == 1:
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
        plt.hist(valid_angle_diff, bins=30, edgecolor='black')
        plt.title('Distribution of relative angles')
        plt.xlabel('Relative angle (deg)')
        plt.ylabel('Frequency')
        plt.show()

    return STD_final, relative_angle

# Main
def calibrate_grid():
    # Get relative angle and error values from a set of images inside a folder

    # RX0-II Camera intrinsic parameters for calibration
    # Camera matrix
    # fx = 2569.605957 # 2684.27089
    # fy = 2568.584961 # 1125.05137
    # cx = 1881.565430 # 2315.60138
    # cy = 1087.135376 # 1147.69833
    fx = 2568.584961 # 1125.05137
    fy = 2569.605957 # 2684.27089
    cx = 1087.135376 # 1147.69833
    cy = 1881.565430 # 2315.60138

    camera_matrix = np.array([[fx, 0., cx],
                            [0., fy, cy],
                            [0., 0., 1.]], dtype = "double")

    # Radial distortion coefficients
    k1 =  0.019473 #  0.88010421
    k2 = -0.041976 # -2.66361019
    k3 =  0.030603 #  2.08744759 

    # Tangential distortion coefficients
    p1 =  -0.000273 # 0.04255226
    p2 =  -0.001083 # 0.00462325

    dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3]))

    # Get images from directory
    print(f"Searching images in {args.folder}/")
    images = sorted(glob.glob(f'./frames/{args.folder}/*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    images_list = []
    relangles = []
    stds = []
    
    # Change this value to segment only the projected line from the laser
    line_threshold = 9
    """
    # [original]=5.1, [gridtest, gridtest_p2]=220, [C0015, C0019, C0020]=9
    """
    
    # Change this value to segment only the center point of the polarized laser projection (ideally max brightness value of the picture)
    center_threshold = 253.725
    
    for fname in images:
        # Read the image
        img0 = cv2.imread(fname)
        
        # If the image is not vertical, rotate it
        h, w, _ = img0.shape
        if w > h:
            img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        # Convert the image to grayscale
        img_gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        
        # Change contrast and brightness
        # img_gray = cv2.convertScaleAbs(img_gray, alpha=1, beta=0) # Not useful for now
        
        # Undistort image
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
        img_gray = cv2.undistort(img_gray, camera_matrix, dist_coeff, None, new_camera_matrix)
        img0 = cv2.undistort(img0, camera_matrix, dist_coeff, None, new_camera_matrix)
        
        # Get a binary image by thresholding it with a bottom value of brightness (to find the line)
        _, BW = cv2.threshold(img_gray, line_threshold, 1, cv2.THRESH_BINARY)
        
        # Remove the smaller areas of the binary image
        BW = cv2.morphologyEx(BW, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        
        # # Erode away the boundaries of foreground object
        # erode_ker = np.array([[0, 0, 1, 0, 0],[0, 1, 1, 1, 0],[1, 1, 1, 1, 1],[0, 1, 1, 1, 0],[0, 0, 1, 0, 0]], dtype=np.uint8)
        # BW = cv2.erode(BW, kernel=erode_ker, borderType=cv2.BORDER_CONSTANT)
        
        # Apply an average filter of window length 8
        ws = 8
        BW = cv2.filter2D(BW, -1, kernel=np.ones((ws, ws)) / (ws**2), borderType=cv2.BORDER_CONSTANT)
        
        if args.plot:
            plt.figure(0)
            plt.imshow(BW)
        
        # Get another binary image by thresholding it with a top value of brightness (looking to find the traces of the polarized laser)
        _, BW_cntrd = cv2.threshold(img_gray, center_threshold, 1, cv2.THRESH_BINARY)
        # BW_cntrd[1650:1750, 550:700] = 0 # C0017
        
        center = np.array([1710, 550])
        offset = np.array([150, 200])
        BW_cntrd[center[0]-offset[0]:center[0]+offset[0], center[1]-offset[1]:center[1]+offset[1]] = 0 # C0019
        
        if args.plot:
            plt.figure(1)
            plt.imshow(BW_cntrd)
        
        # Find the centroids of the laser traces in the image
        BW_cntrd_labels = measure.label(BW_cntrd)
        BW_cntrd_props = sorted(measure.regionprops(BW_cntrd_labels), key=lambda r: r.area, reverse=True)

        # Choose the brightest centroid (most probably the center laser trace)
        cntrd = BW_cntrd_props[0].centroid
        
        if args.plot:
            plt.figure(2)
            plt.imshow(BW_cntrd)
            plt.axhline(y=cntrd[0], linestyle='--', linewidth=0.5, color='red')
            plt.axvline(x=cntrd[1], linestyle='--', linewidth=0.5, color='red')
        
        # Filter just a slice of the former thresholded to get the potential location of the polarized laser line
        BW[:, :int(np.ceil(cntrd[1])) - 60] = 0
        BW[:, int(np.ceil(cntrd[1])) + 60:] = 0
        
        if args.plot:
            plt.figure(3)
            plt.imshow(BW)
        
        # Find the optimized center point of the image
        # Calculate_gridangle function parameters
        cntrd_offset = 150#300
        search_radius = 1000
        
        skip=0
        skipstart=550
        skiplength=100
        
        verbose=0

        # Optimization parameters
        # initial_guess (x0) = cntrd
        
        # Get bounds
        cntrd_search_vert_m=0
        cntrd_search_vert_p=0
        cntrd_search_horz_m=0
        cntrd_search_horz_p=0
        cntrd_bounds = [(cntrd[0]-cntrd_search_vert_m, cntrd[0]+cntrd_search_vert_p), (cntrd[1]-cntrd_search_horz_m, cntrd[1]+cntrd_search_horz_p)]

        # Optimizate the position of the center of the laser trace    
        res = scipy.optimize.minimize(calculate_gridangle, cntrd, bounds=cntrd_bounds, method='L-BFGS-B', 
                                        args=(BW, cntrd_offset, search_radius, verbose, skip, skipstart, skiplength), 
                                        options={'maxiter': 10})
        
        # Get the resultant cntrd vector, now known as c
        c = res.x

        if args.plot:
            plt.figure(4)
            plt.imshow(BW)
            plt.axhline(y=cntrd[0], linestyle='--', linewidth=0.5, color='red')
            plt.axvline(x=cntrd[1], linestyle='--', linewidth=0.5, color='red')

        # Execute the calculate_gridangle one more time to get the graphs of the angles and a final result for angle and angle standard deviation
        error, angle = calculate_gridangle(c, BW, cntrd_offset, search_radius, args.std_show, skip, skipstart, skiplength)
        
        relangles.append(angle)
        stds.append(error)
        
        final_str = f"{fname.split('\\')[-1]} relative angle is: {str(np.round(angle, 3))} deg +/- {str(np.round(error, 3))} deg"
        images_list.append(final_str)
        print(final_str)

        if args.plot:
            x0 = [c[0], c[0] + -(cntrd_offset + search_radius) * np.cos(np.deg2rad(-angle))]
            y0 = [c[1], c[1] + -(cntrd_offset + search_radius) * np.sin(np.deg2rad(-angle))]

            plt.plot(y0, x0, linestyle='--', color='red')
            
            x1 = [c[0], c[0] + (cntrd_offset + search_radius) * np.cos(np.deg2rad(-angle))]
            y1 = [c[1], c[1] + (cntrd_offset + search_radius) * np.sin(np.deg2rad(-angle))]

            plt.plot(y1, x1, linestyle='--', color='red')
            # plt.ylim(c[0]-cntrd_offset - search_radius, c[0]+cntrd_offset + search_radius)
            plt.xlim(c[1]-cntrd_offset - search_radius, c[1]+cntrd_offset + search_radius)
            
            plt.show()
            
    relangles = np.array(relangles)
    stds = np.array(stds)
    
    meanstd = f"Average relative angle is: {relangles.mean()}, average error is: {stds.mean()}, standard deviation from relative angles is: {relangles.std()}."
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


        plt.figure()
        # plt.imshow(img_gray, cmap='gray')
        plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        
        x0 = [c[0], c[0] + -(cntrd_offset + search_radius) * np.cos(np.deg2rad(-relangles.mean()))]
        y0 = [c[1], c[1] + -(cntrd_offset + search_radius) * np.sin(np.deg2rad(-relangles.mean()))]

        plt.plot(y0, x0, linestyle='--', color='blue')
        
        x1 = [c[0], c[0] + (cntrd_offset + search_radius) * np.cos(np.deg2rad(-relangles.mean()))]
        y1 = [c[1], c[1] + (cntrd_offset + search_radius) * np.sin(np.deg2rad(-relangles.mean()))]

        plt.plot(y1, x1, linestyle='--', color='blue')
        # plt.ylim(c[0]-cntrd_offset - search_radius, c[0]+cntrd_offset + search_radius)
        plt.xlim(c[1]-cntrd_offset - search_radius, c[1]+cntrd_offset + search_radius)
        
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
    parser = argparse.ArgumentParser(description='Obtains the relative angle between the camera in the RF source frame and the grid from a series from a given picture or pictures.')
    parser.add_argument('folder', type=str, help='Name of folder containing the frames.')
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='Show plots from the process.')
    parser.add_argument('-fp', '--fplot', action='store_true', default=False, help='Show plots from the process.')
    parser.add_argument('-sd', '--std_show', action='store_true', default=False, help='Show plots of the standard deviation.')
    
    # Get parse data
    args = parser.parse_args()
    calibrate_grid()