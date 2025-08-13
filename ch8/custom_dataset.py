# Save dataset locally
custom_dataset.save_to_disk("./my_custom_dataset")

# Load it
from datasets import load_from_disk
loaded_dataset = load_from_disk("./my_custom_dataset")

# Can also save in different formats
custom_dataset.to_json("my_dataset.json")
custom_dataset.to_csv("my_dataset.csv")
