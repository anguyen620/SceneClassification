import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses
import pickle
from createFilterBank import create_filterbank
import cv2


def get_visual_words (img, dictionary, filterBank):

    # -----fill in your implementation here --------
    wordMap = np.zeros ((img.shape[0], img.shape[1]))
    filterResponses = extract_filter_responses(img, filterBank)
    responses2D = np.zeros ((m*n, 3*len(filterBank)))

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            distances = cdist(responses2D, dictionary)
            minDistance = min(distances)
            index = np.where(distances == minDistance)
            label = np.where(dictionary[index])

            wordMap[x, y] = label


    # ----------------------------------------------

    return wordMap

#Test
if __name__ == '__main__':
    img = cv2.imread('../data/airport/sun_aesovualhburmfhn.jpg')
    dict = pickle.load(open( "dictionaryHarris.pkl", "rb" ))
    filterBank = create_filterbank()

    wordMap = get_visual_words(img, dict, filterBank)

    print(wordMap)
