from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Choose a model based on your needs
model_name = 'distilbert-base-uncased'

# Load model for classification (specify number of labels)
num_labels = 5
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# Load corresponding tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Model loaded: {model_name}")
print(f"Number of labels: {num_labels}")
