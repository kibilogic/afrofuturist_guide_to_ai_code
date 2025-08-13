# Split data for training and evaluation
train_dataset = tokenized_dataset.select(range(4))  # First 4 examples for training
eval_dataset = tokenized_dataset.select([4])        # Last example for evaluation

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to=[],  # Disable wandb
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Start training
print("Beginning training...")
trainer.train()

# Save model
trainer.save_model('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
print("Training completed and model saved!")

