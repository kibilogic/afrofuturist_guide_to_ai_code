from datasets import load_dataset

# Load dataset 
dataset = load_dataset("imdb")

# Examine the structure
print("Dataset structure:", dataset)
print("Training examples:", len(dataset["train"]))
print("Test examples:", len(dataset["test"]))

# View examples
first_example = dataset["train"][0]
print("Text preview:", first_example["text"][:200] + "...")
print("Label:", first_example["label"])
