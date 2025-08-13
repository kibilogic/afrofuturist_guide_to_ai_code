# Install required packages (if not already)
# !pip install transformers torch scikit-learn matplotlib

import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Function to extract contextual embedding for a word
def get_contextual_embedding(word, sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state.squeeze(0)

    tokens = tokenizer.tokenize(sentence)
    word_tokens = tokenizer.tokenize(word)

    for i in range(len(tokens)):
        if tokens[i:i+len(word_tokens)] == word_tokens:
            return last_hidden[i:i+len(word_tokens)].mean(dim=0)
    return None

# Define contexts for "bank"
contexts = {
    "river_bank": "He sat near the river bank and watched the water.",
    "money_bank": "She went to the bank to deposit her check.",
    "blood_bank": "The hospital relies on the blood bank for emergencies.",
}

# Compute embeddings
contextual_embeddings = {
    label: get_contextual_embedding("bank", sentence)
    for label, sentence in contexts.items()
}

# Visualize using PCA
def plot_embeddings(embedding_dict, title):
    vecs = torch.stack(list(embedding_dict.values()))
    labels = list(embedding_dict.keys())
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vecs.numpy())

    plt.figure(figsize=(6, 4))
    for i, label in enumerate(labels):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, label, fontsize=12)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()

# Run the visualization
plot_embeddings(contextual_embeddings, "Contextual Embeddings of 'bank'")


