import cv2
import numpy as np
from PIL import Image


src = cv2.imread("wick.jpg")


scale = 2
delta = 0
ddepth = cv2.CV_16S

src = cv2.GaussianBlur(src, (3, 3), 0)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


# apply sobel filter
grad_x = cv2.Sobel(
    gray, ddepth, 1, 0, ksize=1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT
)

# Gradient-Y
# grad_y = cv.Scharr(gray,ddepth,0,1)
grad_y = cv2.Sobel(
    gray, ddepth, 0, 1, ksize=1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT
)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
src1 = cv2.filter2D(grad, -1, kernel)

# dilate
# kernel = np.ones((1,1),np.uint8)
# src1 = cv2.dilate(src1,kernel,iterations = 5)

# convert single channel to multi channel
src1 = cv2.cvtColor(src1, cv2.COLOR_GRAY2BGR)

# resize blackboard -  image
src2 = cv2.imread("blackboard.jpg")
src2 = cv2.resize(src2, src1.shape[1::-1])


# alpha blending
dst = cv2.addWeighted(src1, 0.4, src2, 0.8, 30)



cv2.imshow("chalk-effect", dst)
cv2.imshow('org',src)
# cv2.imwrite('chalk-effect.jpg',dst)
cv2.waitKey(0)
