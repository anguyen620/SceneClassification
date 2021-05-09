import numpy as np


def get_image_features (wordMap, dictionarySize):

    # -----fill in your implementation here --------
    h = np.zeros (dictionarySize)

    for x in range(wordMap.shape[0]):
        for y in range(wordMap.shape[1]):
            word = int(wordMap[x,y])
            h[word]+=1

    h = h/np.sum(h)
    # h = h / np.linalg.norm(h)

    # ----------------------------------------------
    
    return h
