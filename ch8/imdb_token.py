from transformers import AutoTokenizer
from datasets import load_dataset

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("imdb", split="train[:100]")

# Create a function to tokenize text
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512
    )

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Compare original and tokenized data
original_text = dataset[0]["text"][:100]
tokenized_text = tokenizer.convert_ids_to_tokens(tokenized_dataset[0]["input_ids"][:20])

print("Original text:", original_text)
print("First 20 tokens:", tokenized_text)
