import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

file = sys.argv[1]

# Read the image
img = cv2.imread(f'./frames/multiple/C0026-sk{file}.png')
img_og = cv2.imread(f'./frames/multiple/C0026-f{file}.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_og = cv2.GaussianBlur(img_og, (9,9), 2)
lab = cv2.cvtColor(img_og, cv2.COLOR_BGR2LAB)

# Find centers
_, lab_l = cv2.threshold(lab[:,:,0], 254, 1, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(lab_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

plt.figure()
plt.imshow(img_og)
plt.figure()
plt.imshow(lab_l)
plt.show()

centers = []
for c in contours:
    # calculate moments for each contour
    M = cv2.moments(c)

    # calculate x,y coordinate of center
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers.append([cX, cY])
        cv2.circle(img_og, (cX, cY), 5, (0, 255, 0), -1)
    except:
        pass
centers = np.array(centers)
print(f'centers={centers}')

# Plot image as background
plt.figure()
plt.imshow(cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB))

# Produce mask for every line and save them in a list
mask_r = cv2.inRange(hsv, (0, 25, 25), (10, 255, 255))
mask_y = cv2.inRange(hsv, (11, 25, 25), (35, 255, 255))
mask_g = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
mask_b = cv2.inRange(hsv, (71, 25, 25), (150, 255, 255))

mask_list = [mask_r, mask_y, mask_g, mask_b]

# Get list of colors to plot
colors = ['red', 'yellow', 'green', 'cyan']

for i in range(len(mask_list)):
    mask = mask_list[i]
    
    # Slice the images
    imask = mask>0
    
    # Get the line as the max values of the binarized image per column 
    max = np.argmax(imask, axis=0)
    max_nz = np.array([[j, max[j]] for j in range(len(max)) if imask[max[j], j] != 0])
    
    # If the resulting array is not empty, continue
    if max_nz.shape != (0,):
        # Get the closest centroid point to line (brightest point of the line)
        closest = np.argmin(np.abs(centers[:,1] - np.mean(max_nz[:,1])*np.ones(centers.shape[0])))
        centroid = centers[closest,:]
        
        # Cut the line to have both limits at the same distance from center
        line_center = np.argwhere(max_nz[:,0] == centroid[0])[0,0]
        
        if (line_center - 0) <= (max_nz.shape[0] - line_center):
            max_dist = line_center
        else:
            max_dist = max_nz.shape[0] - line_center
        
        max_centered = max_nz[line_center-max_dist:line_center+max_dist+1,:]
        
        # Get the polynomial fit     
        polyline = np.polynomial.polynomial.polyfit(max_centered[:,0], max_centered[:,1], 2)
        
        # print(f'{colors[i]} line fit values are {polyline}')
        fn_line = np.polynomial.Polynomial(polyline)
        x_poly = fn_line(max_centered[:,0])
        
        slopes = []
        slopes2center = []
        
        cntrd = max_centered[int(max_centered.shape[0]/2),:]
        
        for k in range(int(0.5*(max_centered.shape[0]-1))):
            # Get slope from extreme points
            slope = np.degrees(np.arctan2((max_centered[-(k+1),1] - max_centered[k,1]), (max_centered[-(k+1),0] - max_centered[k,0])))
            
            # Get slopes from extreme points to center then substract both
            slope_top = np.degrees(np.arctan2((max_centered[-(k+1),1] - cntrd[1]), (max_centered[-(k+1),0] - cntrd[0])))
            slope_bottom = np.degrees(np.arctan2((cntrd[1] - max_centered[k,1] ), (cntrd[0] - max_centered[k,0])))
            # slope_tb = (slope_top + slope_bottom)/2
            slope_tb = slope_top - slope_bottom
            
            slopes.append(slope)
            slopes2center.append(slope_tb)
            
        slopes = np.array(slopes)
        slopes2center = np.array(slopes2center)
        
        print(f'Average angle 1: {slopes.mean()} ± {slopes.std()}. Average angle 2: {slopes2center.mean()} ± {slopes2center.std()}')

        # plt.plot(max_nz_centered[:,0], max_nz_centered[:,1], color=colors[i])
        plt.plot(max_centered[:,0], x_poly, color=colors[i])
        
plt.show()