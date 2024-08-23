import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

file = sys.argv[1]

line_threshold = 100

# Read the image
img0 = cv2.imread(f'./frames/multiple/C0026-f{file}.png')

mask = np.zeros(img0.shape[:2], img0.dtype)
mask[1000:1350, 1100:] = 1

# img0 = cv2.bitwise_and(img0, img0, mask = mask)
# img0 = img0[1000:1350, 1100:]

# Convert image to properly RGB
img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

# Get red, green and blue channels
img_r = img_rgb[:,:,0]
img_g = img_rgb[:,:,1]
img_b = img_rgb[:,:,2]

# Convert image to grayscale
img_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# Convert image to HSV
img_hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)

# Get hue, saturation and value channels
img_h = img_hsv[:,:,0]
img_s = img_hsv[:,:,1]
img_v = img_hsv[:,:,2]

# Convert image to LAB
img_lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)

# Get lightness, red/green and yellow/blue channels
img_labl = img_lab[:,:,0]
img_laba = img_lab[:,:,1]
img_labb = img_lab[:,:,2]

# Convert image to HLS
img_hls = cv2.cvtColor(img0, cv2.COLOR_BGR2HLS)

# Get hue, lightness and saturation channels
img_hlsh = img_hls[:,:,0]
img_hlsl = img_hls[:,:,1]
img_hlss = img_hls[:,:,2]

# plt.clf()
# plt.figure()
# plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
# plt.title("Image RGB")

# plt.figure()
# plt.imshow(img_r)
# plt.title("Red channel")

# plt.figure()
# plt.imshow(img_g)
# plt.title("Green channel")

# plt.figure()
# plt.imshow(img_b)
# plt.title("Blue channel")

# plt.figure()
# plt.imshow(img_hsv)
# plt.title("Image HSV")

# plt.figure()
# plt.imshow(img_h)
# plt.title("HSV Hue channel")

# plt.figure()
# plt.imshow(img_s)
# plt.title("HSV Saturation channel")

# plt.figure()
# plt.imshow(img_v)
# plt.title("HSV Value channel")

# plt.figure()
# plt.imshow(img_lab)
# plt.title("Image LAB")

# plt.figure()
# plt.imshow(img_labl)
# plt.title("LAB Lightness channel")

# plt.figure()
# plt.imshow(img_laba)
# plt.title("LAB Red/Green channel")

# plt.figure()
# plt.imshow(img_labb)
# plt.title("LAB Yellow/Blue channel")

# plt.figure()
# plt.imshow(img_hls)
# plt.title("Image HLS")

# plt.figure()
# plt.imshow(img_hlsh)
# plt.title("HLS Hue channel")

# plt.figure()
# plt.imshow(img_hlsl)
# plt.title("HLS Lightness channel")

# plt.figure()
# plt.imshow(img_hlss)
# plt.title("HLS Saturation channel")

# plt.show()

#################################################################################################

# _, img_labl_BW = cv2.threshold(img_labl, 90, 1, cv2.THRESH_BINARY)
# Change contrast and brightness
_, img_labl_max = cv2.threshold(img_labl, 200, 1, cv2.THRESH_BINARY)

img_labl = cv2.convertScaleAbs(img_labl, alpha=10, beta=0)
img_labl_BW = cv2.adaptiveThreshold(img_labl, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
img_open = cv2.morphologyEx(img_labl_BW, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 3)))
img_close = cv2.morphologyEx(img_open, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 3)))

img_section = cv2.bitwise_and(img_close, img_close, mask = mask)

plt.figure()
plt.imshow(img_labl_max)
plt.figure()
plt.imshow(img_labl_BW)
plt.figure()
plt.imshow(img_open)
plt.figure()
plt.imshow(img_close)
plt.figure()
plt.imshow(img_section)
# plt.show()

img_laplac = cv2.Laplacian(img_labl, cv2.CV_64F)
img_laplac_blur = cv2.GaussianBlur(img_laplac, (5,5), 0) 

# Change contrast and brightness
img_laplac_sca = cv2.convertScaleAbs(img_laplac_blur, alpha=10, beta=0)

# Binarize 
_, img_laplac_BW = cv2.threshold(img_laplac_sca, 90, 1, cv2.THRESH_BINARY)
img_laplac_clo = cv2.morphologyEx(img_laplac_BW, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
img_laplac_clonope = cv2.morphologyEx(img_laplac_clo, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

# find contours in the binary image
contours, hierarchy = cv2.findContours(img_labl_max, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

point_list = []
for c in contours:
    # calculate moments for each contour
    M = cv2.moments(c)

    # calculate x,y coordinate of center
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        point_list.append([cX, cY])
        cv2.circle(img0, (cX, cY), 5, (0, 255, 0), -1)
    except:
        pass
point_list = np.array(point_list)
polyline = np.polyfit(point_list[:, 0], point_list[:, 1], 2)
print(polyline)
fn_line = np.poly1d(polyline)

tf = np.arange(0, 3840)
xf = fn_line(tf)

# plt.figure()
# plt.imshow(img_laplac)

# plt.figure()
# plt.imshow(img_laplac_BW)

# plt.figure()
# plt.imshow(img_laplac_clo)

# plt.figure()
# plt.imshow(img_laplac_clonope)

plt.figure()
plt.imshow(img0)
plt.plot(tf, xf)
plt.show()

# # Apply an average filter of window length 8
# ws = 8
# BW = cv2.filter2D(BW, -1, kernel=np.ones((ws, ws)) / (ws**2), borderType=cv2.BORDER_CONSTANT)


