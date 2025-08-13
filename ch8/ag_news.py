from datasets import load_dataset

# Explore specific datasets
try:
    news_dataset = load_dataset("ag_news", split="train[:10]")
    print("\nSample from AG News dataset:")
    for i, example in enumerate(news_dataset):
        if i < 3:
            categories = ["World", "Sports", "Business", "Technology"]
            category = categories[example["label"]]
            print(f"Category: {category}")
            print(f"Text: {example['text'][:100]}...")
            print()
except Exception as e:
    print(f"Could not load news dataset: {e}")

try:
    imdb_dataset = load_dataset("imdb", split="train[:5]")
    print("IMDB Movie Review Dataset Sample:")
    for example in imdb_dataset:
        sentiment = "Positive" if example["label"] == 1 else "Negative"
        print(f"Review: {example['text'][:100]}...")
        print(f"Sentiment: {sentiment}\n")
except Exception as e:
    print(f"Could not load IMDB dataset: {e}")

# Find datasets by browsing the Hub website at:
print("To explore more datasets, visit: https://huggingface.co/datasets")
