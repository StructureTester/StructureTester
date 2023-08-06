import string

import numpy as np
import spacy
from sklearn.cluster import KMeans


def get_central_word(sentence, i):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    vectors = np.array([word.vector for word in doc])
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(vectors)

    central_vector = kmeans.cluster_centers_[0]
    central_word = ''
    min_distance = float('inf')
    for word in doc:
        distance = np.linalg.norm(word.vector - central_vector)
    if distance < min_distance:
        central_word = word.text

    return central_word
