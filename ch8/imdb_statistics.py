import matplotlib.pyplot as plt
from collections import Counter

# Analyze text lengths in a dataset
dataset = load_dataset("imdb", split="train[:1000]")

# Calculate text lengths
text_lengths = [len(example["text"]) for example in dataset]

# Create basic statistics
print("Average text length:", sum(text_lengths) / len(text_lengths))
print("Shortest text:", min(text_lengths))
print("Longest text:", max(text_lengths))

# Count labels
label_counts = Counter([example["label"] for example in dataset])
print("Label distribution:", label_counts)
