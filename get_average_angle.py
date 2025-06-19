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
from concurrent.futures import ThreadPoolExecutor


def calculate_grid_angle(I, cntrd, cntrd_offset, plot):
    # Auxiliary function to calculate the relative angle between the grid and the camera in the frame

    # Get all row centroids and determine the distance between all centroids and the center of the line (cntrd).
    row_cntrd = center_of_mass_per_row(I, cntrd)
    
    # Filter all elements near the center (cntrd_offset) and get the maximum distance shared by both tails.
    row_filtered = row_cntrd[(row_cntrd[:, 2] < np.min([row_cntrd[0, 2], row_cntrd[-1, 2]])) & (row_cntrd[:, 2] >= cntrd_offset)]
    row_filtered = row_filtered[(row_filtered[:, 2] < np.min([row_filtered[0, 2], row_filtered[-1, 2]]))]
    
    # Define x, y from the filtered list of points
    x = row_filtered[:,0]
    y = row_filtered[:,1]
    
    # Fit a straight line to the list of points
    fit = np.polyfit(x, y, deg=1)
    m, b = fit
    
    # Generate fit line
    x_fit = np.linspace(x.min(), x.max(), int(x.max()-x.min()))
    y_fit = m * x_fit + b    
    
    # Get expected angle from the fit's calculated slope
    expected_angle_rad = np.arctan2(m, 1)
    expected_angle_deg = np.degrees(expected_angle_rad)
    
    # Determine the orthogonal euclidean distance from each point to the fitted line
    numerator = np.abs(m*x+b - y)
    denominator = np.sqrt(1+m**2)
    orthogonal_residuals = numerator / denominator
    
    # Get standard deviation of the orthogonal residuals
    std_orthogonal = np.std(orthogonal_residuals)
    
    # Approximate angular deviation: arc tangent of spread over baseline (span)
    dx = x.max() - x.min()  # horizontal span in pixels
    dy = m * dx
    length = np.sqrt(dx**2 + dy**2)

    # Get the angular spread
    angle_spread_rad = np.arctan(std_orthogonal/length)
    angle_spread_deg = np.degrees(angle_spread_rad)
    
    if plot:
        plt.figure()
        plt.imshow(I)
        plt.scatter(row_filtered[:,1], row_filtered[:,0])
        plt.scatter(y_fit, x_fit)
        plt.show()
    
    return expected_angle_deg, angle_spread_deg


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


def load_image(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    # img0 = cv.cvtColor((img/255).astype('float32'), cv.COLOR_BGR2LAB)
    return img


def displayImage(img, width=1280, height=720, name='Picture'):
    # Small simple function to display images without needing to add the auxiliar functions. By default it reduces the size of the image to 1280x720.
    cv.imshow(name, cv.resize(img, (width, height)))
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_frames_from_video(filename):
    # Open the video
    cap = cv.VideoCapture(filename)
    if not cap.isOpened():
        print("Error: Cannot open video.")
    
    total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    images = []
    for frame_idx in range(total_frame_count):
        ret, frame = cap.read()
        if ret:
            images.append(frame)
        else:
            print(f"Failed to read frame {frame_idx}")
    return images


# Main
def get_wiregrid_angle(folder, calibfile=False, plot=False, fplot=False, std_show=False):
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
    
    # Load images from folder and save them in a list
    image = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_image, path): path for path in images}
        for future in tqdm(futures, desc="Loading Images", unit="img"):
            image.append(future.result())
    
    # Average every 100 images and filter the result to find the laser trace
    for i in range(len(image)//100):
        # Convert images to numpy arrays
        image_batch = np.array(image[100*i:100*(i+1)], dtype=np.float32)
        img0 = np.mean(image_batch, axis=0)
        img0_std = np.std(image_batch, axis=0)
            
        # Get height and width of the image
        h, w = img0.shape
        
        # Undistort image
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
        img0 = cv.undistort(img0, camera_matrix, dist_coeff, None, new_camera_matrix)
        img0_std = cv.undistort(img0_std, camera_matrix, dist_coeff, None, new_camera_matrix)
        
        # If the image is not vertical, rotate it
        if w > h:
            img0 = cv.rotate(img0, cv.ROTATE_90_COUNTERCLOCKWISE)
            img0_std = cv.rotate(img0_std, cv.ROTATE_90_COUNTERCLOCKWISE)
        
        # Get a binary image by thresholding it with a top value of brightness (looking to find the traces of the polarized laser)
        _, img0_center = cv.threshold(img0, 254, 1, cv.THRESH_BINARY)
        _, img0_max = cv.threshold(img0, 3, 1, cv.THRESH_BINARY)
        
        if plot:
            plt.figure(0)
            plt.imshow(img0)
            
            plt.figure(1)
            plt.imshow(img0_std)
            
            plt.figure(2)
            plt.imshow(img0_center)
            
            plt.figure(3)
            plt.imshow(img0_max)
        
        # Find the centroids of the laser traces in the image
        BW_cntrd_labels = measure.label(img0_center)
        BW_cntrd_props = sorted(measure.regionprops(BW_cntrd_labels), key=lambda r: r.area, reverse=True)

        # Choose the brightest centroid (the center of the laser trace)
        cntrd = BW_cntrd_props[0].centroid
        
        # Change this value to increase or decrease the horizontal (vertical) range where the laser trace would be in the image
        mask_window = 150
        
        # Create a mask around the centroid found
        mask = np.zeros(img0.shape[:2], img0.dtype)
        mask[:, int(cntrd[1])-mask_window:int(cntrd[1])+mask_window] = 1
        
        if plot:
            plt.figure(4)
            plt.imshow(img0_center)
            plt.axhline(y=cntrd[0], linestyle='--', linewidth=0.5, color='red')
            plt.axvline(x=cntrd[1], linestyle='--', linewidth=0.5, color='red')

        BW = (morphology.remove_small_objects(np.array(img0_max, dtype=bool), 20)).astype(int)
                
        if plot:
            plt.figure(5)
            plt.imshow(BW)
        
        # Filter just a slice of the former thresholded to get the potential location of the polarized laser line
        tail = 1000
        BW[:int(np.ceil(cntrd[0])) - tail,:] = 0
        BW[int(np.ceil(cntrd[0])) + tail:,:] = 0
        BW[:, :int(np.ceil(cntrd[1])) - 100] = 0
        BW[:, int(np.ceil(cntrd[1])) + 100:] = 0
        
        if plot:
            plt.figure(6)
            plt.imshow(BW)

            plt.figure(7)
            plt.imshow(BW)
            plt.axhline(y=cntrd[0], linestyle='--', linewidth=0.5, color='red')
            plt.axvline(x=cntrd[1], linestyle='--', linewidth=0.5, color='red')
            # plt.show()
            plt.close()

        # Set offset distance from line center to skip the brightest part of the line
        cntrd_offset = 100

        # Find the angle and error from the binarized image, and the found center of the trace
        angle, error = calculate_grid_angle(BW, cntrd, cntrd_offset, plot=std_show)
        print(f"Batch {i}: Average angle: {angle} deg, average error: {error} deg")

            
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
    videofile = "D:\\Logs\\calibration_tests\\calibration_tests_20250520\\Grid_calibration_4\\C0206.MP4"
    get_wiregrid_angle(args.folder, args.calibfile, args.plot, args.fplot, args.std_show)