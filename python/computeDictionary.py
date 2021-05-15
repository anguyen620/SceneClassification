import pickle
from getDictionary import get_dictionary


meta = pickle.load (open('../data/traintest.pkl', 'rb'))

train_imagenames = meta['train_imagenames']

# -----fill in your implementation here --------

# Adjust these values for potentially more accurate results
a = 200
k = 500

imgPaths = ["../data/" + path for path in train_imagenames]

print("Creating dictionary of words for random points...")
random_words_dictionary = get_dictionary(imgPaths, a, k, "Random")

print("Creating dictionary of words for top Harris points...")
harris_words_dictionary = get_dictionary(imgPaths, a, k, "Harris")

with open("dictionaryRandom.pkl", 'wb') as handle:
    pickle.dump(random_words_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("dictionaryHarris.pkl", 'wb') as handle:
    pickle.dump(harris_words_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ----------------------------------------------



