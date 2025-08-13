# Load language modeling dataset
lm_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10]")

# Look at text examples
for i, example in enumerate(lm_dataset):
    if example["text"].strip():  # Skip empty lines
        print(f"Example {i}: {example['text'][:150]}...")
        if i >= 2:  # Only show a few examples
            break
