import yaml
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk, load_metric

def evaluate_model(config_path='src/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    dataset = load_from_disk('data/processed/dataset')
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_model'])
    model = AutoModelForSequenceClassification.from_pretrained(config['model']['output_dir'])
    
    metric = load_metric("accuracy") if config['dataset']['task'] == "classification" else load_metric("f1")
    
    training_args = TrainingArguments(
        output_dir='results',
        per_device_eval_batch_size=64,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        compute_metrics=lambda p: metric.compute(predictions=p.predictions.argmax(axis=-1), references=p.label_ids)
    )
    
    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    evaluate_model()
