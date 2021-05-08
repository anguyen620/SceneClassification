import numpy as np


def get_image_features (wordMap, dictionarySize):

    # -----fill in your implementation here --------
    h = np.zeros ((1, dictionarySize))

    for i in range(len(wordMap.shape[0])):
        for j in range(len(wordMap.shape[1])):
            for k in range(dictionarySize):
                if wordMap[i,j]==k:
                    h[k]=h[k]+1

    h = h/np.sum(h)

    # ----------------------------------------------
    
    return h
