# Load the fine-tuned model for evaluation
from transformers import pipeline

# Create a classification pipeline with saved "fine-tuned" model
classifier = pipeline(
    "text-classification",
    model='./fine_tuned_model',
    tokenizer='./fine_tuned_model'
)

# Test with new examples
test_texts = [
    "Scientists develop new drought-resistant crops for African farmers",
    "Cultural festival showcases traditional dance and music",
    "Government announces new healthcare initiatives for rural areas"
]

# Get predictions
for text in test_texts:
    result = classifier(text)
    print(f"Text: {text}")
    print(f"Prediction: {result}")
    print()
