import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    GPT2Config,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class WeaponDataset(Dataset):
    """Custom dataset for weapon generator fine-tuning"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', 
                                  max_length=max_length, return_tensors='pt')
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item
    
    def __len__(self):
        return len(self.encodings.input_ids)

def prepare_data():
    
    with open("data/weapons_training.json", "r") as f:
        training_data = json.load(f)
    
    texts = []
    
    for item in training_data:
        text = f"<|prompt|>{item['prompt']}<|completion|>{item['completion']}<|endoftext|>"
        texts.append(text)
    
    train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
    
    print(f"Training examples: {len(train_texts)}")
    print(f"Validation examples: {len(val_texts)}")
    
    return train_texts, val_texts

def finetune_model(train_texts, val_texts, output_dir="models/gpt2-fantasy-weapons"):
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = "distilgpt2"  
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    special_tokens = {
        'additional_special_tokens': ['<|prompt|>', '<|completion|>', '<|endoftext|>']
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    train_dataset = WeaponDataset(train_texts, tokenizer)
    val_dataset = WeaponDataset(val_texts, tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_steps=100,
        save_steps=100,
        warmup_steps=100,
        evaluation_strategy="steps",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  
        report_to="none",  
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    print("Starting training...")
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    return model, tokenizer

def generate_samples(model, tokenizer, prompts, output_dir):
    model.eval()
    model.to(device)
    
    results = []
    
    for prompt in tqdm(prompts):
        formatted_prompt = f"<|prompt|>{prompt}<|completion|>"
        
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
        
        output = model.generate(
            input_ids,
            max_length=200,
            temperature=0.8,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
        
        if "<|completion|>" in generated_text and "<|endoftext|>" in generated_text:
            completion = generated_text.split("<|completion|>")[1].split("<|endoftext|>")[0]
        else:
            completion = generated_text.split("<|completion|>")[1] if "<|completion|>" in generated_text else generated_text
        
        results.append({
            "prompt": prompt,
            "completion": completion.strip()
        })
    
    with open(f"{output_dir}/sample_generations.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def compare_with_baseline():
    fine_tuned_model = GPT2LMHeadModel.from_pretrained("models/gpt2-fantasy-weapons")
    fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2-fantasy-weapons")
    
    baseline_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    baseline_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    
    test_prompts = [
        "Generate a name for a fire sword with the following properties: enchanted, ancient",
        "Write a description for a weapon called 'Frostbite'",
        "Generate a fantasy weapon of type: bow",
        "Create a legendary ice dagger"
    ]
    
    fine_tuned_model.to(device)
    baseline_model.to(device)
    
    comparison = []
    
    for prompt in test_prompts:
        formatted_prompt = f"<|prompt|>{prompt}<|completion|>"
        input_ids = fine_tuned_tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
        
        fine_tuned_output = fine_tuned_model.generate(
            input_ids,
            max_length=200,
            temperature=0.8,
            num_return_sequences=1,
            pad_token_id=fine_tuned_tokenizer.eos_token_id
        )
        
        fine_tuned_text = fine_tuned_tokenizer.decode(fine_tuned_output[0], skip_special_tokens=False)
        if "<|completion|>" in fine_tuned_text:
            fine_tuned_text = fine_tuned_text.split("<|completion|>")[1].split("<|endoftext|>")[0] if "<|endoftext|>" in fine_tuned_text else fine_tuned_text.split("<|completion|>")[1]
        
        input_ids = baseline_tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        baseline_output = baseline_model.generate(
            input_ids,
            max_length=200,
            temperature=0.8,
            num_return_sequences=1,
            pad_token_id=baseline_tokenizer.eos_token_id
        )
        
        baseline_text = baseline_tokenizer.decode(baseline_output[0], skip_special_tokens=True)
        baseline_text = baseline_text[len(prompt):].strip()
        
        comparison.append({
            "prompt": prompt,
            "fine_tuned": fine_tuned_text.strip(),
            "baseline": baseline_text.strip()
        })
    
    with open("models/comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print("Comparison completed and saved!")
    
    return comparison

def main():
    os.makedirs("models", exist_ok=True)
    
    train_texts, val_texts = prepare_data()
    
    model, tokenizer = finetune_model(train_texts, val_texts)
    
    test_prompts = [
        "Generate a name for a fire sword with the following properties: enchanted, ancient",
        "Write a description for a weapon called 'Dragonbane'",
        "Generate a fantasy weapon of type: axe",
        "Create a legendary bow with ice damage",
        "Name a dagger that steals souls"
    ]
    
    results = generate_samples(model, tokenizer, test_prompts, "models")
    
    comparison = compare_with_baseline()
    
    print("Fine-tuning and evaluation complete!")

if __name__ == "__main__":
    main()