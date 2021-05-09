from utils import chi2dist
from scipy.spatial.distance import cdist

def get_image_distance(hist1, hist2, method):
    if method == "euclidean":
        dist = cdist(hist1, hist2, method)
    elif method == "chi2":
        dist = chi2dist(hist1, hist2)
    return dist

