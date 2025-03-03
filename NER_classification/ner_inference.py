import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class AnimalNER:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
        self.animals = ["bear", "beaver", "chimpanzee", "fox", "kangaroo", 
                       "lion", "otter", "porcupine", "raccoon", "wolf"]
    
    def extract_animals(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, padding=True)
        offset_mapping = inputs.pop("offset_mapping")[0]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0]
        
        animal_tokens = []
        for idx, pred in enumerate(predictions):
            if pred == 1:  # 1 відповідає класу "ANIMAL"
                start, end = offset_mapping[idx]
                token = text[start:end]
                if token.strip():
                    animal_tokens.append(token.lower())
        
        found_animals = []
        for animal in self.animals:
            animal_parts = animal.split()
            
            if len(animal_parts) > 1:
                full_match = all(part.lower() in ' '.join(animal_tokens) for part in animal_parts)
                if full_match:
                    found_animals.append(animal)
            else:
                # Для простих назв (наприклад, "lion")
                for token in animal_tokens:
                    if animal in token:
                        found_animals.append(animal)
                        break
        
        return list(set(found_animals))

def main(args):
    ner = AnimalNER(args.model_path)
    
    if args.text:
        animals = ner.extract_animals(args.text)
        print(f"Text: {args.text}")
        print(f"Extracted animals: {animals}")
    
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text:
                    animals = ner.extract_animals(text)
                    print(f"Text: {text}")
                    print(f"Extracted animals: {animals}")
                    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to the trained NER model")
    parser.add_argument("--text", type=str, help="Text to extract animal names from")
    parser.add_argument("--input_file", type=str, help="File with texts to process")
    
    args = parser.parse_args()
    main(args)