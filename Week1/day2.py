import numpy as np

# In the context of word embeddings, we can represent words as vectors in a high-dimensional space. Each word is represented as a point in this space, and the coordinates of the point are determined by the values in the embedding vector. For example, if we have a 4-dimensional embedding space, we can represent the words "I", "like", and "cats" as follows:
X = np.array([
    [0.2, 0.1, 0.7, 0.3],  # "I"
    [0.6, 0.8, 0.1, 0.4],  # "like"
    [0.9, 0.2, 0.3, 0.5],  # "cats"
])

print(X.shape)

# The shape of the matrix X is (3, 4), which means it has 3 rows and 4 columns. Each row corresponds to a word ("I", "like", "cats"), and each column corresponds to a dimension in the embedding space. The values in the matrix represent the coordinates of each word in the embedding space.
# X = np.array([
#     [0.1, 0.2, 0.3],
#     [0.4, 0.5],
# ])

# print(X)
# print(X.shape)