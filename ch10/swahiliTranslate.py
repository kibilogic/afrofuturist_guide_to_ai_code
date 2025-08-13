# ========== put this code block in a new cell ==========
#Load Pretrained Tokenizer and Masked Language Model for Swahili
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load the lightweight African-focused XLM-R model
model_name = "Davlan/afro-xlmr-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

print("Model loaded successfully")
print(f"Vocabulary size: {tokenizer.vocab_size}")

# ========== put this code block in a new cell ==========
# Define Custom Swahili Training Sentences with <mask> Tokens
texts = [
    "Ubuntu ni falsafa ya <mask> kati ya binadamu.",
    "Watu wa Afrika wana lugha <mask>.",
    "Watoto hucheza kwa <mask> kila jioni.",
    "Maisha ya kijijini ni <mask> kuliko mjini."
]

# ========== put this code block in a new cell ==========
# Convert the List of Sentences into a Hugging Face Dataset
from datasets import Dataset

# Create a Dataset from the Swahili text examples
dataset = Dataset.from_dict({"text": texts})

# ========== put this code block in a new cell ==========
# Tokenize the Dataset 
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

# Apply tokenization to all dataset entries
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ========== put this code block in a new cell ==========
# Define the Data Collator for MLM (Masked Language Modeling)
from transformers import DataCollatorForLanguageModeling

# Randomly masks 15% of tokens 
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# ========== put this code block in a new cell ==========
# Set Up Training Configuration and Initialize Trainer
from transformers import TrainingArguments, Trainer

# Define how the model should be trained and saved
training_args = TrainingArguments(
    output_dir="./swahili-mlm-model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=10,
    report_to="none"  # Disable logging to WandB 
)

# Create the Trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ========== put this code block in a new cell ==========
# Train the Model on the Swahili Text
trainer.train()

# ========== put this code block in a new cell ==========
# Save the Fine-Tuned Model and Tokenizer Locally
trainer.save_model("./swahili-mlm-model")
tokenizer.save_pretrained("./swahili-mlm-model")

# ========== put this code block in a new cell ==========
# Run Inference Using the Fine-Tuned Model
from transformers import pipeline

# Load the trained model and run a fill-mask test
fill_mask = pipeline("fill-mask", model="./swahili-mlm-model", tokenizer="./swahili-mlm-model")

# Provide masked Swahili sentence for prediction
results = fill_mask("Maisha ya kijijini ni <mask> kuliko mjini.")

# Print top predicted completions
for r in results:
    print(f"{r['sequence']}  (score={r['score']:.4f})")





