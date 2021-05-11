import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from getDictionary import get_dictionary
from getImageDistance import get_image_distance
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
from sklearn.neighbors import KNeighborsClassifier
from utils import chi2dist

meta = pickle.load(open('../data/traintest.pkl', 'rb'))

test_imagenames = meta['test_imagenames']

def getChiSquaredDistance(hist1, hist2):
    dist = chi2dist(hist1, hist2)
    return dist

# -----fill in your implementation here --------
test_labels = meta['test_labels']
method = 'Random'
imgPaths = ["../data/" + path for path in test_imagenames]

recog_system = pickle.load(open('vision%s.pkl'%method, 'rb'))

train_features = recog_system['trainFeatures']
train_labels = recog_system['trainLabels']
train_num, cluster_num = train_features.shape

accuracy = np.zeros(41)
best_confusion = np.zeros((8,8))
dictionary = recog_system["dictionary"]

for i in range(1,41):
    print(f'Current i: {i}')
    neighbors = KNeighborsClassifier(n_neighbors = i, metric = getChiSquaredDistance)
    train_labels = np.ravel(train_labels)
    neighbors.fit(train_features, train_labels)
    confusion = np.zeros((8,8))

    for j, path in enumerate(imgPaths):
        img = cv2.imread(path)
        wordMap = get_visual_words(img, dictionary, recog_system["filterBank"])
        features = get_image_features(wordMap, cluster_num)
        
        actual = int(test_labels[i])
        classified = int(neighbors.predict(features.reshape(1, -1))[0])
        print(classified, actual)

        confusion[actual-1, classified-1] = confusion[actual-1, classified-1] + 1

    accuracy[i] = np.trace(confusion)/np.sum(confusion)

    if i == np.argmax(accuracy):
        best_confusion = confusion

best_k = np.argmax(accuracy)
print("Best k number of nearest neighbors was {} with accuracy {}".format(best_k,accuracy[best_k]))
print(best_confusion)

plt.plot(accuracy)
plt.ylabel('Accuracy')
plt.xlabel('K Nearest Neighbors')
plt.title('Accuracies')

# ----------------------------------------------



