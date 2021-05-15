from utils import chi2dist
from scipy.spatial.distance import cdist
import numpy as np

def get_image_distance(hist1, hist2, method):
    if method == "euclidean":
        dist = np.linalg.norm(hist1 - hist2)
    elif method == "chi2":
        dist = chi2dist(hist1, hist2)
    return dist

