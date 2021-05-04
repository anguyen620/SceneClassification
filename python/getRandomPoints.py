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
        xRand = random.randint(0, img.shape[1])
        yRand = random.randint(0, img.shape[0])
        points.append([xRand, yRand])

    # ----------------------------------------------
    
    return points


# start of some code for testing get_random_points()
if __name__ == "__main__":
    img = cv2.imread ("../data/bedroom/sun_aiydcpbgjhphuafw.jpg")
    print (get_random_points (img, 50))

