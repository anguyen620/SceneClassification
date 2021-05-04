import numpy as np
import cv2


def get_harris_points(img, alpha, k=0.04):

    if len(img.shape) == 3 and img.shape[2] == 3:
        # should be OK in standard BGR format
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

    # -----fill in your implementation here --------

    # Get keypoints
    gray = np.float32(img)
    kps = cv2.cornerHarris(gray, 2, 5, k)

    # Make value array, which consists of [val, x, y]
    valueArr = []
    for row in range(kps.shape[0]):
        for col in range(kps.shape[1]):
            valueArr.append([kps[row][col], row, col])

    # Shitty code
    sortedValues = sorted(valueArr, key = lambda x: x[0], reverse = True) # Sort the keypoints in descending order
    sortedValues = [elems[1:] for elems in sortedValues]
    points = sortedValues[:alpha]

    # ----------------------------------------------
    
    return points


# start of some code for testing get_harris_points()
if __name__ == "__main__":
    img = cv2.imread ("../data/bedroom/sun_aiydcpbgjhphuafw.jpg")
    print (get_harris_points (img, 50, 0.04))

