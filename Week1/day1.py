import numpy as np

# Cosine similarity is a measure of similarity between two non-zero vectors in an inner product space. It is defined as the cosine of the angle between them, which ranges from -1 to 1. A value of 1 indicates that the vectors are identical, while a value of -1 indicates that they are opposite. A value of 0 indicates that the vectors are orthogonal (i.e., they have no similarity).
a = np.array([1, 1])
b = np.array([10, 10])
d = np.array([1, 0])

def cosine_similarity(x, y):
    dot = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return dot / (norm_x * norm_y)

print("cosine(a, b):", cosine_similarity(a, b))
print("cosine(a, d):", cosine_similarity(a, d))

# In the context of word embeddings, we can represent words as vectors in a high-dimensional space. The cosine similarity between these vectors can indicate how similar the meanings of the words are. For example, if we have vectors for the words "good", "great", and "bad", we can compute their cosine similarities to see how closely they are related in meaning.
good = np.array([0.9, 0.8, 0.1])
great = np.array([0.95, 0.85, 0.1])
bad = np.array([0.1, 0.2, 0.9])

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)

print("cosine(good, great):", cosine_similarity(good, great))
print("cosine(good, bad):", cosine_similarity(good, bad))