import cv2
import numpy as np
from random import randint
from utils import *
from createFilterBank import create_filterbank

def extract_filter_responses(img, filterBank):

    if len(img.shape) == 2:
        img = cv2.merge([img, img, img])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # -----fill in your implementation here --------
    l,a,b = cv2.split(img)  # Split the color space
    colorSpaces = [l,a,b]

    filterResponses = np.ndarray((img.shape[0], img.shape[1], 3*len(filterBank)))
    filterResponses = filterResponses.astype(np.uint8) 
    
    for filterNumber in range(len(filterBank)):
        for colorSpaceIndex in range(len(colorSpaces)):
            dst = cv2.filter2D(colorSpaces[colorSpaceIndex], -1, filterBank[filterNumber])    # Use the filter for each of the 3 color spaces

            # There might be a better way to do this, but put every pixel into the correct location for that filterNumber
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    filterResponses[x, y, 3*filterNumber + colorSpaceIndex] = dst[x,y]

    # ----------------------------------------------

    return filterResponses


# start of some code for testing extract_filter_responses()
if __name__ == "__main__":
    fb = create_filterbank()
    img = cv2.imread("../data/desert/sun_adpbjcrpyetqykvt.jpg")

    responses = extract_filter_responses(img, fb)
    
    # For testing, build the sample image from each of the response indices
    for i in range(responses.shape[2]):
        filtered_image = np.zeros((img.shape[0], img.shape[1]))
        filtered_image = filtered_image.astype(np.uint8) 
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                filtered_image[x,y] = responses[x,y, i]
        cv2.imshow(str(i),filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

