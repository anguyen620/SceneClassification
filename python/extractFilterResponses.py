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
    # filterResponses = []

    # Lightness
    light = img[:,:,0]

    # Red/Green
    redGreen = img[:,:,1]

    # Blue/Green
    blueGreen = img[:,:,2]

    colorSpaces = [light, redGreen, blueGreen]
    # for colorSpace in colorSpaces:
    #     for j in range(len(filterBank)):
    #         dst = cv2.filter2D(colorSpace, -1, filterBank[j])
    #         filterResponses.append(dst)


    filterResponses = np.ndarray((img.shape[0], img.shape[1], 3*len(filterBank)))
    for filterNumber in range(len(filterBank)):
        for colorSpaceIndex in range(len(colorSpaces)):
            dst = cv2.filter2D(colorSpaces[colorSpaceIndex], -1, filterBank[filterNumber])    # Use the filter for each of the 3 color spaces

            # There might be a better way to do this, but put every pixel into the correct location for that filterNumber
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    filterResponses[x, y, 3*filterNumber + colorSpaceIndex]

    # ----------------------------------------------

    return filterResponses

def randPts(img, alpha):
    """ Get random points """
    ret = []
    for entry in range(alpha):
        xRand = randint(0, img.shape[1])
        yRand = randint(0, img.shape[0])
        ret.append([xRand, yRand])
    return ret

def getKps(img, alpha, k = 0.04):
    """ Get top alpha keypoints """
    # Get keypoints
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    kps = cv2.cornerHarris(gray, 2, 5, k)

    # Make value array, which consists of [val, x, y]
    valueArr = []
    for row in range(kps.shape[0]):
        for col in range(kps.shape[1]):
            valueArr.append([kps[row][col], row, col])

    # Shitty code
    sortedValues = sorted(valueArr, key = lambda x: x[0], reverse = True) # Sort the keypoints in descending order
    sortedValues = [elems[1:] for elems in sortedValues]
    top50 = sortedValues[:alpha]
    return top50

def getDict(imgPaths, alpha, K, method):
    fb = create_filterbank()
    for fp in imgPaths:
        # For each image, get the filter responses?
        img = cv2.imread(fp)
        responses = extract_filter_responses(img, fb)

        # Get alpha points
        if method == "Random":
            pts = randPts(img, alpha)
        elif method == "Harris":
            pts = getKps(img, alpha)
        
    # return

# start of some code for testing extract_filter_responses()
if __name__ == "__main__":
    fb = create_filterbank()
    img = cv2.imread("../data/desert/sun_adpbjcrpyetqykvt.jpg")

    #gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    #print (extract_filter_responses (gray, fb))
    print(img.shape)

    responses = extract_filter_responses(img, fb)
    print(responses.shape)

    #randomPts = randPts(img, 5)
    #print(randomPts)

    # topKps = getKps(img, 5, 0.05)
    # print(topKps)
    
    # for i in range(len(responses)):
    #     cv2.imshow(str(i),responses[i])
    #     cv2.waitKey(0)

    #     cv2.destroyAllWindows()
    # cv2.imshow('1',responses[0])
    # cv2.imshow('2',responses[20])
    # cv2.imshow('3',responses[40])

    #cv2.imshow("merged", cv2.merge([responses[0], responses[20], responses[40]]))

    # cv2.waitKey(0)

    # cv2.destroyAllWindows()



