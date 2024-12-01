import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import yaml

def load_and_preprocess_data(config_path='src/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    dataset_name = config['dataset']['name']
    dataset_config = config['dataset'].get('config', None)
    task = config['dataset']['task']
    
    # Load dataset
    if dataset_config:
        raw_datasets = load_dataset(dataset_name, dataset_config)
    else:
        raw_datasets = load_dataset(dataset_name)
    
    print(raw_datasets)
    print(raw_datasets['text'])

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_model'])
    
    def preprocess_function(examples):
        # Ensure that tokenizer works with batched input
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    def preprocess_function(examples):
        # Flatten the nested structure of examples['text'] into a single list of strings
        flat_text = [text for sublist in examples['text'] for text in sublist]
        return tokenizer(flat_text, truncation=True, padding='max_length', max_length=128)

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    tokenized_datasets.save_to_disk('data/processed/dataset')

if __name__ == "__main__":
    load_and_preprocess_data()

