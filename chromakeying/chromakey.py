import numpy as np
import cv2

# Read source image.
im_src = cv2.imread("friends.jpg")

# Read background image
im_dst = cv2.imread("john.jpg")

# make a copy of background image
img = im_dst.copy()


# four points of location to be replaced
pts_dst = np.array([[154.0, 148.0], [316.0, 118.0], [316.0, 377.0], [133.0, 377.0]])


def warp_img(im_src, im_dst, pts_dst):

    # Create a vector of source points.
    size = im_src.shape

    pts_src = np.array(
        [[0, 0], [size[1] - 1, 0], [size[1] - 1, size[0] - 1], [0, size[0] - 1]],
        dtype=float,
    )

    # Calculate Homography between source and background points
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image
    im_temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
    im_dst = cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)
    im_dst = im_dst + im_temp

    return im_dst


# Apply homographic alignment on source image to match with the location in  background image.
im_dst = warp_img(im_src, im_dst, pts_dst)


# blending source and background images
alpha = 0.9
beta = 1.0 - alpha
im_dst = cv2.addWeighted(im_dst, alpha, img, beta, 0)


# polygon points of location to be replaced.
points = [
    [190.9310201104511, 145.84557674273205],
    [271.4499322705011, 128.12128790246948],
    [271.95634052308003, 159.5185995623632],
    [316.3884710065644, 152.54567833698022],
    [319.5587162654995, 376.7677399187245],
    [299.8087944149212, 366.1331666145669],
    [292.21267062623724, 367.6523913723037],
    [284.1101385849744, 363.0947170990933],
    [284.1101385849744, 356.0050015629883],
    [281.5780973220797, 352.9665520475147],
    [280.5652808169218, 317.01156611441064],
    [277.0204230488693, 319.5436073773053],
    [277.0204230488693, 358.53704282588296],
    [268.91789100760644, 357.5242263207251],
    [267.90507450244854, 358.03063457330404],
    [266.3858497447118, 309.9218505783056],
    [263.3474002292382, 310.42825883088454],
    [263.3474002292382, 357.0178180681462],
    [260.8153589663435, 345.37042825883077],
    [251.7000104199228, 338.28071272272575],
    [252.2064186725017, 328.15254767114715],
    [245.1167031363967, 319.5436073773053],
    [233.97572157966022, 318.0243826195685],
    [171.18109825987278, 324.6076899030946],
    [140.29019485255802, 310.93466708346347],
    [150.41835990413662, 186.3582369490465],
    [187.8925705949775, 176.73648015004682],
]


# converting polygon points to int type from float
poly_points = []
for i in points:
    poly_points.append([int(i[0]), int(i[1])])

contours = np.array(poly_points)

# filling the area to be replaced with green colour
green_screen = cv2.fillPoly(img, pts=[contours], color=(0, 100, 0))


# chroma key conversion
green_range = np.array([0, 100, 0])  # rgb
mask = cv2.inRange(green_screen, green_range, green_range)
res = cv2.bitwise_and(green_screen, green_screen, mask=mask)
fu = green_screen - res
fu = np.where(fu == 0, im_dst, fu)


# view final image
cv2.imshow("res", fu)
cv2.waitKey(0)
