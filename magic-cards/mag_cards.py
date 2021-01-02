import numpy as np
import cv2


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


#source image
image = cv2.imread("cat.jpg")
size = image.shape


#for image rotation
angle = 180
rimage = rotate_image(image, angle)
r_size = rimage.shape


# Create a vector of source points.
pts_src1 = np.array(
    [[0, 0], [size[1] - 1, 0], [size[1] - 1, size[0] - 1], [0, size[0] - 1]],
    dtype=float,
)

pts_src2 = np.array(
    [[0, 0], [r_size[1] - 1, 0], [r_size[1] - 1, r_size[0] - 1], [0, r_size[0] - 1]],
    dtype=float,
)


# Read destination image
im_dst = cv2.imread("cards.jpg")

# location of destination photo

pts_dst1 = np.array([[184.0, 330.0], [305.0, 270.0], [342.0, 352.0], [235.0, 437.0]])
pts_dst2 = np.array([[235.0, 436.0], [342.0, 354.0], [390.0, 461.0], [274.0, 516.0]])


# Calculate Homography between source and destination points
h, status = cv2.findHomography(pts_src1, pts_dst1)

# Warp source image
im_temp1 = cv2.warpPerspective(image, h, (im_dst.shape[1], im_dst.shape[0]),flags=cv2.INTER_CUBIC)


# Black out polygonal area in destination image.
cv2.fillConvexPoly(im_dst, pts_dst1.astype(int), 0, 16)

# Add warped source image to destination image.
im_dst = im_dst + im_temp1


# for rotated image

# Calculate Homography between source and destination points
h, status = cv2.findHomography(pts_src2, pts_dst2)

# Warp source image
im_temp2 = cv2.warpPerspective(rimage, h, (im_dst.shape[1], im_dst.shape[0]),flags=cv2.INTER_CUBIC)


# Black out polygonal area in destination image.
cv2.fillConvexPoly(im_dst, pts_dst2.astype(int), 0, 16)

# Add warped source image to destination image.
im_dst = im_dst + im_temp2


cv2.imshow("blackout", im_dst)
cv2.imwrite("catm.jpg", im_dst)
cv2.waitKey(0)
