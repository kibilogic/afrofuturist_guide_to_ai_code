from transformers import AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text
text = "Ubuntu philosophy emphasizes our interconnectedness"
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# Convert to numerical IDs
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)

# Convert back to text
decoded_text = tokenizer.decode(token_ids)
print("Decoded:", decoded_text)
