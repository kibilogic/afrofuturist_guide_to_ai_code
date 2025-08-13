# Exploring different types of models
from transformers import pipeline

# Text classification model
classifier = pipeline("sentiment-analysis",
                      model="nlptown/bert-base-multilingual-uncased-sentiment")

# Test with text that might reflect context
sample_texts = [
    "The new community health program is helping many families",
    "Traffic congestion makes daily commuting very stressful",
    "Traditional music festivals bring joy to our neighborhood"
]

print("Sentiment Analysis Results:")
for text in sample_texts:
    result = classifier(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result[0]['label']}, Confidence: {result[0]['score']:.3f}\n")
