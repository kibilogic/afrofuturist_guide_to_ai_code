import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt

class SingleHeadAttentionTranslator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32))
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)
        return self.out(attended), weights

class MultiHeadAttentionTranslator(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        out, weights = self.mha(x, x, x, need_weights=True, average_attn_weights=False)
        return out, weights

# Embedded tokens 
embed_dim = 8
tokens = ["She", "gave", "birth", "to", "a", "healthy", "boy"]
x = torch.rand((1, len(tokens), embed_dim))  # batch_size=1

# Single-head
single = SingleHeadAttentionTranslator(embed_dim)
single_output, single_weights = single(x)

# Multi-head (4 heads)
multi = MultiHeadAttentionTranslator(embed_dim, num_heads=4)
multi_output, multi_weights = multi(x)

def show_attention(weights, title, tokens):
    plt.figure(figsize=(6, 5))
    plt.imshow(weights.detach().numpy(), cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Single-head
show_attention(single_weights[0], "Single-Head Attention", tokens)

head_0_matrix = multi_weights[0, 0] 

# Multi-head
show_attention(head_0_matrix, "Multi-Head Attention (Head 1)", tokens)


