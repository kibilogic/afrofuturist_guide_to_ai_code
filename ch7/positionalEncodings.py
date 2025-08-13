from transformers import BertTokenizer, BertModel
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Load BERT model 
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True, attn_implementation="eager")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Wolof, How are you?
sentence = "Nanga def?"  
inputs = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Extract QKV from first layer
layer = model.encoder.layer[0].attention.self
query_weight = layer.query.weight
key_weight = layer.key.weight
value_weight = layer.value.weight

print("Query shape:", query_weight.shape)  
print("Key shape:", key_weight.shape)      
print("Value shape:", value_weight.shape)  

# Visualize learned positional encodings
embedding_layer = model.embeddings
position_embeddings = embedding_layer.position_embeddings.weight[:inputs["input_ids"].shape[1]]

plt.figure(figsize=(12, 5))
sns.heatmap(position_embeddings.detach().cpu().numpy(), cmap="viridis")
plt.title("Positional Encoding (BERT) – First N Positions")
plt.xlabel("Embedding Dimensions")
plt.ylabel("Position Index")
plt.show()

