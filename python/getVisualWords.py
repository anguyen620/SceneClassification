import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses
import pickle
from createFilterBank import create_filterbank
import cv2
from skimage.color import label2rgb


def get_visual_words (img, dictionary, filterBank):

    # -----fill in your implementation here --------
    wordMap = np.zeros ((img.shape[0], img.shape[1]))
    filterResponses = extract_filter_responses(img, filterBank)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            distances = cdist(dictionary, filterResponses[x][y].reshape(1,60))
            wordMap[x, y] = np.argmin(distances)

    # ----------------------------------------------

    return wordMap

#Test
if __name__ == '__main__':
    img = cv2.imread('../data/airport/sun_aesovualhburmfhn.jpg')
    rand_dict = pickle.load(open( "dictionaryRandom.pkl", "rb" ))
    harris_dict = pickle.load(open( "dictionaryHarris.pkl", "rb" ))
    filterBank = create_filterbank()

    randomWordMap = get_visual_words(img, rand_dict, filterBank)
    harrisWordMap = get_visual_words(img, harris_dict, filterBank)

    visualize_random = label2rgb(randomWordMap, img, bg_label=0)
    visualize_harris = label2rgb(harrisWordMap, img, bg_label=0)
    cv2.imshow(str("Random Word Map in RGB"), visualize_random)
    cv2.imshow(str("Harris Word Mapp in RGB"), visualize_harris)

    visualize_random = cv2.cvtColor(np.float32(visualize_random), cv2.COLOR_RGB2BGR)
    visualize_harris = cv2.cvtColor(np.float32(visualize_harris), cv2.COLOR_RGB2BGR)
    cv2.imshow(str("Random Word Map in BGR?"), visualize_random)
    cv2.imshow(str("Harris Word Mapp in BGR?"), visualize_harris)

    cv2.waitKey(0)
    cv2. destroyAllWindows()
