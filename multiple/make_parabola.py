import sys
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
from scipy.ndimage import rotate

image_dim = [3840, 2160]

img = np.zeros(image_dim)

# y = a2*x**2 + a1*x + a0 
a2 = 0.00001
a1 = -a2*image_dim[0]
a0 = image_dim[1]/2 + a1**2/(4*a2)

p_coef = [a0, a1, a2]
# Get the polynomial fit     
# polyline = np.polynomial.polynomial.polyfit(max_centered[:,0], max_centered[:,1], 2)

# print(f'{colors[i]} line fit values are {polyline}')
fn_line = np.polynomial.polynomial.Polynomial(p_coef)
t_poly = np.arange(image_dim[0], dtype=np.int32)
x_poly = fn_line(t_poly)
x_poly = np.array(x_poly, dtype=np.int32)

xt_poly = np.column_stack((t_poly, x_poly))

# Rotate vector
# rotation_center = xt_poly[545, :]
# angle_c = 45
# rotated_x_poly = rotate(xt_poly, angle=angle_c, reshape=False)

# translated_x_poly = rotated_x_poly - rotated_x_poly[0,:] + rotation_center*np.ones(rotated_x_poly.shape[0])

# Write parabola in image
for i in range(len(xt_poly)):
    img[xt_poly[i,0], xt_poly[i,1]-1:xt_poly[i,1]+1] = 250

img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.circle(img, center=(int(image_dim[0]/2), int(image_dim[1]/2)), radius=9, color=(255, 255, 255), thickness=-1)

img_color = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
img_color[:,:,1:] = 0 # Only red channel is on, the rest is em

cv2.circle(img_color, center=(int(image_dim[0]/2), int(image_dim[1]/2)), radius=7, color=(255, 255, 255), thickness=-1)

cv2.imwrite('frames/multiple/C0026-fparabola.png', img=img_color)
cv2.imwrite('frames/multiple/C0026-skparabola.png', img=img_color)

plt.figure()
plt.imshow(img_color)
plt.show()