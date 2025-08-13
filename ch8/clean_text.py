# Basic data cleaning
def clean_text(examples):
    # Remove extra whitespace and convert to lowercase
    cleaned_texts = []
    for text in examples["text"]:
        cleaned = text.strip().lower()
        # Remove very short or very long examples
        if 10 <= len(cleaned) <= 1000:
            cleaned_texts.append(cleaned)
        else:
            cleaned_texts.append("")
    examples["text"] = cleaned_texts
    return examples

# Apply cleaning function
dataset = load_dataset("imdb", split="train[:100]")
cleaned_dataset = dataset.map(clean_text, batched=True)

# Filter out empty examples
cleaned_dataset = cleaned_dataset.filter(lambda x: len(x["text"]) > 0)
print("Original size:", len(dataset))
print("After cleaning:", len(cleaned_dataset))
