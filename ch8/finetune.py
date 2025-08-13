# Install the datasets library if not already installed
#!pip install -U transformers datasets

# Disable wandb logging (removes the API key prompt)
import os
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset, Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 1. Load a custom proverbs dataset 
proverbs = {
    'text': [
        "Wisdom is like a baobab tree; no one individual can embrace it.",
        "It takes a village to raise a child.",
        "The brave man is not he who does not feel afraid, but he who conquers that fear.",
        "A family tie is like a tree, it can bend but it cannot break.",
        "When spider webs unite, they can tie up a lion.",
        "A child belongs to everyone in the village.",
        "Courage is not the absence of fear, but mastery of it.",
        "Blood is thicker than water, but love is thicker than blood."
    ],
    'label': [0, 3, 2, 1, 3, 3, 2, 1]  # 0: wisdom, 1: family, 2: courage, 3: community
}

dataset = Dataset.from_dict(proverbs).train_test_split(test_size=0.5, seed=42)

# 2. Tokenize the text
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding=True, max_length=128)

tokenized = dataset.map(tokenize, batched=True)

# 3. Load pre-trained model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# 4. Training configuration 
args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  
    logging_dir="./logs",
    num_train_epochs=3,  
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    warmup_steps=10,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    seed=42,
    report_to=[]  # Disable wandb and other reporting
)

# 5. Metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# 6. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer  
)

print("Starting training...")
print(f"Training samples: {len(tokenized['train'])}")
print(f"Evaluation samples: {len(tokenized['test'])}")

# 7. Fine-tune
trainer.train()

# 8. Evaluate the model
eval_results = trainer.evaluate()
print(f"\nEvaluation Results:")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")

# 9. Test on new proverbs
label_names = ['Wisdom', 'Family', 'Courage', 'Community']

test_proverbs = [
    "A single bracelet does not jingle.",
    "The child who is not embraced by the village will burn it down.",
    "Even the mightiest eagle comes down to the treetops to rest.",
    "Unity is strength, division is weakness."
]

print(f"\n" + "="*50)
print("TESTING ON NEW PROVERBS")
print("="*50)

for text in test_proverbs:
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).max().item()
    
    print(f"\nProverb: \"{text}\"")
    print(f"Predicted theme: {label_names[predicted_class]} (confidence: {confidence:.3f})")
    
    # Show all probabilities
    probabilities = torch.softmax(logits, dim=1)[0]
    print("All probabilities:")
    for i, (label, prob) in enumerate(zip(label_names, probabilities)):
        print(f"  {label}: {prob:.3f}")

print(f"\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)

