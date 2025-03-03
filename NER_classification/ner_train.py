import os
import ast
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

def create_animal_ner_dataset():
    '''
    Creates a synthetic dataset for NER with animal names

    Returns:
        pandas.DataFrame: dataset with synthetic examples
    '''
    animals = ["bear", "beaver", "chimpanzee", "fox", "kangaroo", "lion", "otter", "porcupine", "raccoon", "wolf"]
    
    templates = [
        "There is a {animal} in the picture.",
        "I can see a {animal} in this image.",
        "The picture shows a {animal}.",
        "Look at this {animal}.",
        "Can you spot the {animal} in this photo?",
        "This is a photo of a {animal}.",
        "The {animal} is clearly visible in the image.",
        "There's a {animal} in the foreground.",
        "I believe this image contains a {animal}.",
        "Is there a {animal} in this picture?",
    ]
    
    examples = []
    
    for animal in animals:
        for template in templates:
            text = template.format(animal=animal)
            start = text.find(animal)
            end = start + len(animal)
            examples.append({"text": text, "entities": [(start, end, "ANIMAL")]})
    
    negative_templates = [
        "This is a beautiful landscape.",
        "I can see a car in this image.",
        "The picture shows a building.",
        "There is a tree in the foreground.",
        "This is a photo of the beach.",
    ]
    
    for template in negative_templates:
        examples.append({"text": template, "entities": []})
    
    return pd.DataFrame(examples)

class NERDataset(Dataset):
    '''
    Dataset class for named entity recognition

    Attributes:
        encodings (dict): tokenized inputs and labels
    '''
    def __init__(self, encodings):
        '''
        Constructor for NERDataset class

        Args:
            encodings (dict): tokenized inputs and labels
        '''
        self.encodings = encodings

    def __len__(self):
        '''
        Returns the number of examples in the dataset

        Returns:
            int: number of examples
        '''
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        '''
        Returns the item at the specified index

        Args:
            idx (int): index of the item

        Returns:
            dict: dictionary with tokenized inputs and labels
        '''
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

def tokenize_and_align_labels(tokenizer, texts, all_entities):
    '''
    Tokenizes the input texts and aligns the labels with the tokenized inputs

    Args:
        tokenizer (transformers.AutoTokenizer): tokenizer for the model
        texts (list): list of input texts
        all_entities (list): list of entity annotations
    Returns:
        dict: dictionary with tokenized inputs and labels
    '''
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    labels = []
    for i, (text, entities) in enumerate(zip(texts, all_entities)):
        label = [0] * len(tokenized_inputs["input_ids"][i])  # "O" for all tokens
        
        if isinstance(entities, str):
            try:
                entities = ast.literal_eval(entities)
            except (ValueError, SyntaxError):
                entities = []
        
        if not entities:
            labels.append(label)
            continue
        
        for entity in entities:
            if isinstance(entity, (list, tuple)) and len(entity) == 3:
                start, end, tag = entity
                token_start, token_end = None, None
                
                for j, offset_map in enumerate(tokenizer(text, return_offsets_mapping=True)["offset_mapping"]):
                    if offset_map[0] <= start < offset_map[1]:
                        token_start = j
                    if offset_map[0] < end <= offset_map[1]:
                        token_end = j
                        break
                
                if token_start is not None and token_end is not None:
                    for j in range(token_start, token_end + 1):
                        if j < len(label):
                            label[j] = 1
        
        labels.append(label)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main(args):
    '''
    Main function for named entity recognition

    Args:
        args (argparse.Namespace): command-line arguments
    '''
    if args.dataset_path and os.path.exists(args.dataset_path):
        df = pd.read_csv(args.dataset_path)
    else:
        df = create_animal_ner_dataset()
        if args.dataset_path:
            df.to_csv(args.dataset_path, index=False)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_encodings = tokenize_and_align_labels(
        tokenizer, 
        train_df['text'].tolist(), 
        train_df['entities'].tolist()
    )
    val_encodings = tokenize_and_align_labels(
        tokenizer, 
        val_df['text'].tolist(), 
        val_df['entities'].tolist()
    )
    
    train_dataset = NERDataset(train_encodings)
    val_dataset = NERDataset(val_encodings)
    
    id2label = {0: "O", 1: "ANIMAL"}
    label2id = {"O": 0, "ANIMAL": 1}
    
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
    )
    
    def compute_metrics(pred):
        predictions = pred.predictions.argmax(-1)
        labels = pred.label_ids
        
        active_preds = predictions[labels != -100]
        active_labels = labels[labels != -100]
        
        correct = (active_preds == active_labels).sum().item()
        total = len(active_labels)
        
        return {
            "accuracy": correct / total if total > 0 else 0
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-uncased", type=str)
    parser.add_argument("--dataset_path", default="animal_ner_dataset.csv", type=str)
    parser.add_argument("--output_dir", default="./animal-ner-model", type=str)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    
    args = parser.parse_args()
    main(args)