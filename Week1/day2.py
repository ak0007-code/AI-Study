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

# We can also perform operations on these embeddings. For example, we can multiply the embedding matrix X by a weight matrix W to get a new representation Y. The weight matrix W can be thought of as a transformation that maps the original embedding space to a new space. For example:
X = np.array([
    [0.2, 0.1, 0.7, 0.3],  # "I"
    [0.6, 0.8, 0.1, 0.4],  # "like"
    [0.9, 0.2, 0.3, 0.5],  # "cats"
])

W = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, -0.5],
])

Y = X @ W

print("X shape:", X.shape)
print("W shape:", W.shape)
print("Y shape:", Y.shape)
print(Y)

# We can also add a bias term to the output Y. The bias term is a vector that is added to each row of Y to shift the values. This can help the model learn more complex relationships between the input and output. For example:
print("--------------")
Y = np.array([
    [1.0, 2.0, 3.0],
    [0.5, 1.5, 2.5],
    [2.0, 0.0, 1.0],
    [3.0, 1.0, 0.5],
    [0.2, 0.4, 0.6],
])

b = np.array([0.1, -0.2, 0.5])

Y_with_bias = Y + b

print("Y shape:", Y.shape)
print("b shape:", b.shape)
print("Y_with_bias shape:", Y_with_bias.shape)
print(Y_with_bias)

# We can also combine the weight matrix W and the bias vector b into a single operation. This is often done in neural networks, where the weights and biases are learned during training. For example:
print("--------------")
X = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
])

W = np.array([
    [1.0, 0.0, 0.5, 0.2],
    [0.0, 1.0, 0.5, 0.3],
    [1.0, 1.0, 0.0, 0.4],
])

b = np.array([0.1, -0.1, 0.2, 0.0])

Y = X @ W + b

print("X shape:", X.shape)
print("W shape:", W.shape)
print("b shape:", b.shape)
print("Y shape:", Y.shape)
print(Y)