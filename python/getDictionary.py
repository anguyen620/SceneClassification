import numpy as np
import cv2
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from sklearn.cluster import KMeans


def get_dictionary(imgPaths, alpha, K, method):

    filterBank = create_filterbank()

    # Array of alpha*len(images) number of arrays, each with 3*len(filterBank) elements
    pixelResponses = np.zeros ((alpha * len(imgPaths), 3 * len(filterBank)))

    for i, path in enumerate(imgPaths):
        print ('-- processing %d/%d' % (i, len(imgPaths)))
        image = cv2.imread ('../data/%s' % path)
        # should be OK in standard BGR format
#        image = cv2.cvtColor (image, cv2.COLOR_BGR2RGB)  # convert the image from bgr to rgb
        
        # -----fill in your implementation here --------

        # Apply filter bank to image
        filter_responses = extract_filter_responses(image, filterBank)

        # Get alpha points for a given image filter
        if method == "Random":
            points = get_random_points(image, alpha)
        elif method == "Harris":
            points = get_harris_points(image, alpha, 0.04)


        # Each row corresponds to a single pixel
        for row in range(alpha):

            # Store the whole list of that specific pixel from the filter_responses
            pixelResponses[(i*alpha)+row][:] = filter_responses[points[row][0], points[row][1]]


        # ----------------------------------------------

    # can use either of these K-Means approaches...  (i.e. delete the other one)
    # OpenCV K-Means
#    pixelResponses = np.float32 (pixelResponses)
#    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#    ret,label,dictionary=cv2.kmeans(pixelResponses,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # sklearn K-Means
    print("pixelResponses shape: ", pixelResponses.shape)
    dictionary = KMeans (n_clusters=K, random_state=0).fit(pixelResponses).cluster_centers_
    return dictionary


if __name__ == "__main__":
    # soloImage = ["../data/desert/sun_adpbjcrpyetqykvt.jpg"]
    fourImages = ["../data/bedroom/sun_aacyfyrluprisdrx.jpg","../data/bedroom/sun_aakejyolaigkvisc.jpg", "../data/bedroom/sun_abelvucjanbnkioi.jpg", "../data/bedroom/sun_azocvgpraiggyssd.jpg"]
    

    testDict = get_dictionary(fourImages, 50, 100, "Harris")
    print(testDict)
