# Load a multilingual dataset (PAWS-X has limited language options)
english_dataset = load_dataset("paws-x", "en")  # English version
french_dataset = load_dataset("paws-x", "fr")   # French version

# Compare examples
print("English PAWS-X example:", english_dataset["train"][0])
print("French PAWS-X example:", french_dataset["train"][0])
