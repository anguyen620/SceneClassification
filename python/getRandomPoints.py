import numpy as np
import cv2
import random


def get_random_points(img, alpha):

    random.seed()

    if len(img.shape) == 3 and img.shape[2] == 3:
        # should be OK in standard BGR format
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

    # -----fill in your implementation here --------

    points = []
    for entry in range(alpha):
        xRand = random.randint(0, img.shape[0] - 1)
        yRand = random.randint(0, img.shape[1] - 1)
        points.append([xRand, yRand])

    # ----------------------------------------------
    
    return points


# start of some code for testing get_random_points()
if __name__ == '__main__':
    a = 500

    img = cv2.imread("../data/campus/sun_abslhphpiejdjmpz.jpg")
    points = get_random_points (img, a)
    
    # map on image points
    for coords in points:
        img = cv2.circle(img, (coords[1], coords[0]), radius=2, color=(255, 0, 0), thickness=-1)

    cv2.imshow(str(a)+" Random points", img)
    cv2.waitKey(0)
    cv2. destroyAllWindows()

