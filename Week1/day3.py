import numpy as np

logits = np.array([2.0, 1.0, -1.0])

print("logits:", logits)
print("shape:", logits.shape)

print("--------------")
h_last = np.array([0.4, 0.9, 0.2])  # current context vector

W_vocab = np.array([
    [1.0, 0.5, -0.5, 0.2, 0.1],
    [0.3, 0.8, -1.0, 0.4, 0.0],
    [0.2, 0.1, 0.3, 0.5, -0.2],
])

b = np.array([0.1, 0.0, 0.2, -0.1, 0.0])

logits = h_last @ W_vocab + b

print("h_last shape:", h_last.shape)
print("W_vocab shape:", W_vocab.shape)
print("b shape:", b.shape)
print("logits:", logits)
print("logits shape:", logits.shape)

print("--------------")
logits = np.array([2.0, 1.0, -1.0])

exp_values = np.exp(logits)
probabilities = exp_values / np.sum(exp_values)

print("logits:", logits)
print("logits shape:", logits.shape)

print("exp values:", exp_values)
print("exp values shape:", exp_values.shape)

print("probabilities:", probabilities)
print("probabilities shape:", probabilities.shape)

print("sum:", np.sum(probabilities))

print("--------------")
tokens = ["cats", "dogs", "pizza"]
logits = np.array([2.0, 1.0, -1.0])

exp_values = np.exp(logits)
print("exp values:", exp_values)

probabilities = exp_values / np.sum(exp_values)
print("probabilities:", probabilities)

best_token_index = np.argmax(probabilities)
best_token = tokens[best_token_index]
print("best token index:", best_token_index)
print("best token:", best_token)