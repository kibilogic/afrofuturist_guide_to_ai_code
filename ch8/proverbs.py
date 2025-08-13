from datasets import Dataset

# Create a simple dataset of traditional proverbs with themes
proverbs = [
    "When spider webs unite, they can tie up a lion",
    "The child who is not embraced by the village will burn it down to feel its warmth",
    "If you want to go fast, go alone. If you want to go far, go together",
    "A tree cannot make a forest",
    "The best time to plant a tree was 20 years ago. The second best time is now"
]

themes = ["unity", "belonging", "cooperation", "community", "action"]

# Create the dataset
custom_dataset = Dataset.from_dict({
    "text": proverbs,
    "theme": themes
})

print("Custom dataset:", custom_dataset)
print("First example:", custom_dataset[0])
