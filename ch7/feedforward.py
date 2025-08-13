import numpy as np
import matplotlib.pyplot as plt

# Utility Functions 
def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(q, k, v):
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(d_k)
    weights = softmax(scores)
    output = np.matmul(weights, v)
    return output, weights

def conceptual_feed_forward_network(x, w1, b1, w2, b2):
    hidden = np.dot(x, w1) + b1
    relu = np.maximum(0, hidden)
    output = np.dot(relu, w2) + b2
    return output, relu  

np.random.seed(42)
batch_size = 1
seq_len = 4
d_model = 8
d_ff = 4 * d_model

x = np.random.randn(batch_size, seq_len, d_model)

# Attention weights
W_q = np.random.randn(d_model, d_model) * 0.01
W_k = np.random.randn(d_model, d_model) * 0.01
W_v = np.random.randn(d_model, d_model) * 0.01

# FFN weights
W1 = np.random.randn(d_model, d_ff) * 0.01
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.01
b2 = np.zeros(d_model)

# Attention Block 
q = np.dot(x, W_q)
k = np.dot(x, W_k)
v = np.dot(x, W_v)

attn_output, attn_weights = scaled_dot_product_attention(q, k, v)
attn_residual = layer_norm(x + attn_output)

# Feed-Forward Block 
ffn_output, relu_activations = conceptual_feed_forward_network(attn_residual, W1, b1, W2, b2)
ffn_residual = layer_norm(attn_residual + ffn_output)

# Attention Heatmap
plt.figure(figsize=(5, 4))
plt.title("Self-Attention Weights (Token → Token)")
plt.imshow(attn_weights[0], cmap="plasma")
plt.colorbar(label="Attention Strength")
plt.xlabel("Key Token Index")
plt.ylabel("Query Token Index")
plt.tight_layout()
plt.show()

# ReLU Activation Plot (First Token)
plt.figure(figsize=(6, 3))
plt.title("FFN ReLU Activations (First Token)")
plt.bar(range(d_ff), relu_activations[0, 0])
plt.xlabel("Hidden Unit Index")
plt.ylabel("Activation")
plt.tight_layout()
plt.show()

print("Final Output Shape:", ffn_residual.shape)
print("First Token Vector After FFN + LayerNorm:\n", ffn_residual[0, 0])




