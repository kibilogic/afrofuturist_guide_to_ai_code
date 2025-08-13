from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',               # where to save model checkpoints
    num_train_epochs=3,                   # number of training epochs
    per_device_train_batch_size=16,       # batch size for training
    per_device_eval_batch_size=64,        # batch size for evaluation
    warmup_steps=500,                     # number of warmup steps
    weight_decay=0.01,                    # strength of weight decay
    logging_dir='./logs',                 # directory for storing logs
    logging_steps=10,                     # log every N steps
    eval_strategy="epoch",                # evaluate at the end of each epoch
    save_strategy="epoch",                # save model at the end of each epoch
    load_best_model_at_end=True,          # load the best model when finished
)

# Define metrics for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

print("Training configuration set up successfully")
