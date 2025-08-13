import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Cross-Attention: Machine Translation ---")

# The English Sentence (Source)
source_tokens = ["The", "dog", "barks", "loudly", "."]
embedding_dim = 8

# Embeddings for each English word 
source_embeddings_map = {
    "The":    torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=torch.float32),
    "dog":    torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], dtype=torch.float32),
    "barks":  torch.tensor([0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7], dtype=torch.float32),
    "loudly": torch.tensor([0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4], dtype=torch.float32),
    ".":      torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
}

## Convert to tensor sequence
encoder_output_sequence = torch.stack([
    source_embeddings_map[token] for token in source_tokens
]).unsqueeze(0)

encoder_output_length = encoder_output_sequence.shape[1]

print(
    f"Encoder Output (Source: '{' '.join(source_tokens)}') shape: "
    f"{encoder_output_sequence.shape}"
)

# The French Sentence (Target)
target_embeddings_map = {
    "Le": torch.tensor(
        [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4], dtype=torch.float32
    ),
    "chien": torch.tensor(
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float32
    ),
    "aboyer": torch.tensor(
        [0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8], dtype=torch.float32
    ),
    ".": torch.tensor([0.05] * 8, dtype=torch.float32),
}

decoder_sequence = ["Le", "chien", "aboyer", "."]

decoder_queries = [
    target_embeddings_map.get(tok, torch.zeros(embedding_dim))
    for tok in decoder_sequence
]

decoder_queries_tensor = torch.stack(decoder_queries).unsqueeze(0)  


# Dimensions and layers
d_k = embedding_dim
key_layer_cross = nn.Linear(embedding_dim, d_k, bias=False)
value_layer_cross = nn.Linear(embedding_dim, d_k, bias=False)
query_layer_cross = nn.Linear(embedding_dim, d_k, bias=False)

# Compute Q, K, V - Cross Attention
K_cross = key_layer_cross(encoder_output_sequence)
V_cross = value_layer_cross(encoder_output_sequence)
Q_cross_multi = query_layer_cross(decoder_queries_tensor)  

# Attention scores
attention_scores_multi = torch.matmul(Q_cross_multi, K_cross.transpose(-2, -1))  
scaled_scores_multi = attention_scores_multi / (d_k ** 0.5)
attention_weights_multi = F.softmax(scaled_scores_multi, dim=-1)

# Convert to NumPy 
attention_matrix = attention_weights_multi.squeeze(0).detach().numpy()

# Plot heatmap of attention
plt.figure(figsize=(8, 4))
sns.set(style="white")

ax = sns.heatmap(attention_matrix, annot=True, cmap="YlGnBu",
                 xticklabels=source_tokens, yticklabels=decoder_sequence,
                 cbar_kws={"label": "Attention Weight"}, vmin=0.0, vmax=1.0)

ax.set_title("Cross-Attention: Decoder Tokens Attending to Source Tokens", fontsize=12)
ax.set_xlabel("Source Tokens (English)")
ax.set_ylabel("Decoder Tokens (French)")

plt.tight_layout()
plt.show()

# Single-step attention 
decoder_query_token = "aboyer"
decoder_input_query = target_embeddings_map[decoder_query_token].unsqueeze(0).unsqueeze(0)

Q_cross = query_layer_cross(decoder_input_query)
K_cross = key_layer_cross(encoder_output_sequence)
V_cross = value_layer_cross(encoder_output_sequence)

attention_scores_cross = torch.matmul(Q_cross, K_cross.transpose(-2, -1))
scaled_attention_scores_cross = attention_scores_cross / (d_k ** 0.5)
attention_weights_cross = F.softmax(scaled_attention_scores_cross, dim=-1)

print(f"\nAttention weights for '{decoder_query_token}':")
for i, token in enumerate(source_tokens):
    weight = attention_weights_cross[0, 0, i].item()
    print(f"  {decoder_query_token} → {token}: {weight:.4f}")

# Context vector = weighted sum of value vectors
output_cross_attention = torch.matmul(attention_weights_cross, V_cross)
print(f"\nContext vector shape: {output_cross_attention.shape}")
print(f"First two numbers of context vector: {output_cross_attention[0, 0, :2]}\n")
print("\n--- Summary of Cross-Attention in Machine Translation ---")
print(
    "Imagine you're trying to say something in French by looking at "
    "an English sentence for help."
)
print(
    "At each step, the model (decoder) decides which English words "
    "matter most for the next French word."
)
print(
    "It does this by comparing the current word it's working on "
    "(called the 'query') to all the English words (called 'keys')."
)
print(
    "Then it figures out how important each English word is — "
    "these are called 'attention weights'."
)
print(
    "Finally, it builds a smart summary (a 'context vector') using "
    "those important English words,"
)
print(
    "and uses that to help guess the next French word."
)
print(
    "This way, the model doesn't treat all words equally — it focuses "
    "on the parts that matter most."
)





