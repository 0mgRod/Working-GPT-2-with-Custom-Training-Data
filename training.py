import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the custom training data
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize the text
input_ids = tokenizer.encode(text, return_tensors='pt')

# Train the model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    save_total_limit=2,
    learning_rate=1e-5,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=input_ids,
    data_collator=lambda data: {'input_ids': torch.stack(data)},
)

trainer.train()
