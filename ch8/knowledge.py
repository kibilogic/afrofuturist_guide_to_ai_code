from transformers import pipeline

# Create a translation pipeline using T5
translator = pipeline("translation_en_to_fr", model="t5-small")

# Translate English to French
result = translator("Knowledge is the foundation of wisdom")
print(result[0]['translation_text'])
