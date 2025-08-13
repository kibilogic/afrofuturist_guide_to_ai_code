from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model and tokenizer from Hugging Face Hub
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Input sentence
sentence = "Ubuntu teaches us we are stronger together."

# Tokenize and run through the model
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)

# Convert logits to probabilities
probs = F.softmax(outputs.logits, dim=1)
pred_index = torch.argmax(probs, dim=1).item()
score = probs[0][pred_index].item()

labels = model.config.id2label
predicted_label = labels[pred_index]

result = [{'label': predicted_label, 'score': round(score, 4)}]
print(result)

# Adaptive summary
print("\nSummary:")
print(f"The model analyzed the sentence:\n  \"{sentence}\"")
print(f"It predicts the sentiment is **{predicted_label}** with a confidence of {round(score * 100, 2)}%.")

# Interpretation
if predicted_label == "POSITIVE":
    print("This suggests the sentence expresses a tone of encouragement, approval, or satisfaction.")
elif predicted_label == "NEGATIVE":
    print("This suggests the sentence expresses disapproval, frustration, or dissatisfaction.")
else:
    print("The sentiment appears neutral or ambiguous.")

