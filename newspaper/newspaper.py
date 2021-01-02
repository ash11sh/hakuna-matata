import cv2
import numpy as np
import sys


# Read source image.
im_src = cv2.imread("office.jpg")
im_src = cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)

# Read destination image newspaper
im_dst = cv2.imread("news.jpg")


size = im_src.shape

# Create a vector of source points.
pts_src = np.array(
    [[0, 0], [size[1] - 1, 0], [size[1] - 1, size[0] - 1], [0, size[0] - 1]],
    dtype=float,
)


# location of destination photo
pts_dst = np.array([[383.0, 194.0], [574.0, 198.0], [516.0, 344.0], [328.0, 349.0]])


# Calculate Homography between source and destination points
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image
im_temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
im_temp = cv2.cvtColor(im_temp, cv2.COLOR_GRAY2BGR)

# Black out polygonal area in destination image.
cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)

# Add warped source image to destination image.
# im_dst = im_dst + im_temp
im_dst = cv2.addWeighted(im_dst, 0.9, im_temp, 0.4, 0)

# Display image.
cv2.imshow("Image", im_dst)
# cv2.imwrite("o.jpg", im_dst)
cv2.waitKey(0)
