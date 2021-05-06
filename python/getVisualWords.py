import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses


def get_visual_words (img, dictionary, filterBank):

    # -----fill in your implementation here --------
    wordMap = np.zeros ((img.shape[0], img.shape[1]))
    filterResponses = extract_filter_responses(img, filterBank)

    for x in range(len(img.shape[0])):
        for y in range(len(img.shape[1])):
            distances = cdist(img[x,y], wordMap)
            minDistance = min(distances)

    # ----------------------------------------------

    return wordMap

