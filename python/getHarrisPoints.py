import numpy as np
import cv2


def get_harris_points (img, alpha, k):

    if len(img.shape) == 3 and img.shape[2] == 3:
        # should be OK in standard BGR format
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

    # -----fill in your implementation here --------



    # ----------------------------------------------
    
    return points


# start of some code for testing get_harris_points()
if __name__ == "__main__":
    img = cv2.imread ("../data/bedroom/sun_aiydcpbgjhphuafw.jpg")
    print (get_harris_points (img, 50, 0.04))

