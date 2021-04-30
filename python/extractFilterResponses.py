import cv2
import numpy as np
from utils import *
from createFilterBank import create_filterbank


def extract_filter_responses (img, filterBank):

    if len(img.shape) == 2:
        img = cv2.merge ([img, img, img])

    img = cv2.cvtColor (img, cv2.COLOR_BGR2Lab)

    # -----fill in your implementation here --------
    filterResponses = np.ndarray((img.shape[0], img.shape[1], img.shape[2]*20))
    # filterResponses[0, 0, 0] = (img, -1, filterBank[0])
    for i in range(len(filterBank)):
        filter = cv2.filter2D(img, -1, filterBank[i])
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                for z in range(3):
                    filterResponses[x, y, 3*i+z] = filter[x, y, z]
    # ----------------------------------------------

    return filterResponses

# start of some code for testing extract_filter_responses()
if __name__ == "__main__":
    fb = create_filterbank ()
    img = cv2.imread ("../data/bedroom/sun_aiydcpbgjhphuafw.jpg")

#    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
#    print (extract_filter_responses (gray, fb))

    responses = extract_filter_responses (img, fb)

    print(img.shape)
    print(responses[0].shape)

    cv2.imshow('something',responses[0])

    cv2.waitKey(0)

    cv2.destroyAllWindows()


