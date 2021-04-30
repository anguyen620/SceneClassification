import numpy as np
import cv2
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from sklearn.cluster import KMeans


def get_dictionary (imgPaths, alpha, K, method):

    filterBank = create_filterbank()

    pixelResponses = np.zeros ((alpha * len(imgPaths), 3 * len(filterBank)))

    for i, path in enumerate(imgPaths):
        print ('-- processing %d/%d' % (i, len(imgPaths)))
        image = cv2.imread ('../data/%s' % path)
        # should be OK in standard BGR format
#        image = cv2.cvtColor (image, cv2.COLOR_BGR2RGB)  # convert the image from bgr to rgb
        
        # -----fill in your implementation here --------





        # ----------------------------------------------

    # can use either of these K-Means approaches...  (i.e. delete the other one)
    # OpenCV K-Means
#    pixelResponses = np.float32 (pixelResponses)
#    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#    ret,label,dictionary=cv2.kmeans(pixelResponses,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # sklearn K-Means
    dictionary = KMeans (n_clusters=K, random_state=0).fit(pixelResponses).cluster_centers_
    return dictionary

