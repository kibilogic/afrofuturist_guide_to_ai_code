from transformers import pipeline

# Load pre-trained BERT model for fill-mask tasks
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Masked word
text = "The [MASK] walked into the forest."

# Model predicts the missing word
predictions = fill_mask(text)

print(f"Original sentence: {text}")
print("BERT's predictions for the masked word:")

# Show top 5 predictions
for p in predictions[:5]:
    print(f"- {p['token_str']} (score: {p['score']:.4f})")

