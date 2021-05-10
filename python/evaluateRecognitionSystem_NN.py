import cv2
import pickle
import numpy as np
from getDictionary import get_dictionary
from getImageDistance import get_image_distance
from getImageFeatures import get_image_features
from getVisualWords import get_visual_words


meta = pickle.load (open('../data/traintest.pkl', 'rb'))

test_imagenames = meta['test_imagenames']


# -----fill in your implementation here --------
test_labels = meta['test_labels']
dictionary_options = ["Random","Harris"]
distance_options = ["euclidean", "chi2"]
imgPaths = ["../data/" + path for path in test_imagenames]

for dict_option in dictionary_options:  
    for dist_option in distance_options:
        # We need to run the same confusion matrix generation for all 4 combos
        print('#'*5,dict_option,"+",dist_option,'#'*5)

        # The previously calculated recog system will have most of what we need
        recog_system = pickle.load (open('vision%s.pkl'%dict_option, 'rb'))
        labels = recog_system["trainLabels"]
        dictionary = recog_system["dictionary"]
        k = dictionary.shape[0]

        # Begin processing each of the test images
        confusion_matrix = np.zeros((8,8))
        for i, path in enumerate(imgPaths):
                print('Processing image %d/%d (%s)\r'%(i,len(imgPaths),path[8:]), end="")
                img = cv2.imread(path)
                wordMap = get_visual_words(img, dictionary, recog_system["filterBank"])
                test_hist = get_image_features(wordMap, len(dictionary))

                # Now we compare each histogram to all of the trianing histograms to find a match
                nearest = 999
                match_index = 0
                for hist_index, train_img_hist in enumerate(recog_system["trainFeatures"]):
                    dist = get_image_distance(test_hist,train_img_hist,dist_option)

                    if dist < nearest:
                        nearest = dist
                        match_index = hist_index

                # Map to confusion matrix requires -1 since labels go 1-8 but python goes 0-7
                nearest_label = int(labels[match_index]) - 1 
                actual_label = int(test_labels[i]) - 1
                confusion_matrix[actual_label, nearest_label]+=1

        print("\nResulting confusion matrix: ")
        print(confusion_matrix)       
        print("\n")         
        
# ----------------------------------------------



