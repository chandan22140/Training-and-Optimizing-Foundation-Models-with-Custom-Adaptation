import os
import yaml
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_from_disk

def train_model(config_path='src/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    dataset = load_from_disk('data/processed/dataset')
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_model'])
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['pretrained_model'],
        num_labels=dataset['train'].features['label'].num_classes
    )
    
    training_args = TrainingArguments(
        output_dir=config['model']['output_dir'],
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='logs',
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(config['model']['output_dir'])

if __name__ == "__main__":
    train_model()
