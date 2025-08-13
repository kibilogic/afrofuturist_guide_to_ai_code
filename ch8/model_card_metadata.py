from huggingface_hub import ModelCard

# Load the model card
card = ModelCard.load('facebook/bart-large-cnn')

# Access the metadata (YAML header)
metadata = card.data.to_dict()

print("––– Model Card Metadata –––")
for key, value in metadata.items():
    print(f"{key}: {value}")

print("\n––– Model Card Content –––")
print(card.text)
