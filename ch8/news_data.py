from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd

news_data = {
    'text': [
        "The new solar farm in Ghana will provide clean energy to rural communities",
        "Local farmers report increased crop yields using traditional intercropping methods",
        "The African Union summit addresses economic cooperation across the continent",
        "Community health workers in Rwanda expand vaccination programs",
        "Traditional music festival celebrates cultural heritage in Senegal"
    ],
    'labels': [0, 1, 2, 3, 4],  # 0: energy, 1: agriculture, 2: politics, 3: health, 4: culture
    'label_names': ['energy', 'agriculture', 'politics', 'health', 'culture']
}

# Create dataset
dataset = Dataset.from_dict({
    'text': news_data['text'],
    'labels': news_data['labels']
})

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
print("Data prepared for training:", tokenized_dataset)

