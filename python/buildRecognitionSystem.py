import cv2
import pickle
import numpy as np

from createFilterBank import create_filterbank
from getImageFeatures import get_image_features
from getVisualWords import get_visual_words


def build_system (inputDict, outputDict):
    '''
    inputDict: String filepath to previously calcuated dictionary of word
    outputDict: String filepath for where to save the new dictionary
    '''

    # Known info
    fb = create_filterbank()
    prev_dict = pickle.load(open( inputDict + ".pkl", "rb" ))
    k = len(prev_dict) # prev_dict.shape[0]

    meta = pickle.load (open('../data/traintest.pkl', 'rb'))
    train_imagenames = meta['train_imagenames']
    train_labels = meta['train_labels']
    t = len(train_imagenames)


    # Generate histogram data
    trainFeatures = np.ndarray((t,k))
    imgPaths = ["../data/" + path for path in train_imagenames]
    for i, path in enumerate(imgPaths):
        print('Processing image %d/%d\r'%(i,t), end="")
        img = cv2.imread(path)
        wordMap = get_visual_words(img, prev_dict, fb)
        hist = get_image_features(wordMap, k)

        trainFeatures[i] = hist
    print("\nDone")


    # Assign new dictionary
    vision = {
        "dictionary": prev_dict,
        "filterBank": fb,
        "trainFeatures" : trainFeatures,
        "trainLabels" : train_labels
    }

    with open(outputDict + ".pkl", 'wb') as handle:
        pickle.dump(vision, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return vision


if __name__ == '__main__':
    print("Building recog system for Random (This will take awhile)...")
    build_system("dictionaryRandom", "visionRandom")

    print("Building recog system for Harris (This will take awhile)...")
    build_system("dictionaryHarris", "visionHarris")