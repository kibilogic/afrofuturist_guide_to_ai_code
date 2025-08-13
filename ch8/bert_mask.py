from transformers import pipeline

# Create a fill-mask pipeline using BERT
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Predict missing words based on context
result = fill_mask("The wisdom of [MASK] ancestors guides us today")
print(result)
