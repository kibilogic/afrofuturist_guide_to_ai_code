# Explore a text classification dataset
dataset = load_dataset("emotion")

# See what emotions are included
print("Available emotions:", dataset["train"].features["label"].names)

# Look at examples of different emotions
for i in range(5):
    example = dataset["train"][i]
    emotion = dataset["train"].features["label"].names[example["label"]]
    print(f"Text: {example['text']}")
    print(f"Emotion: {emotion}\n")
