import argparse
import torch
from ner_inference import AnimalNER  
from image_classification_inference import AnimalClassifier 

def verify_animal(text, image_path, ner_model, classifier):
    '''
    Check if the animal extracted from the text matches the predicted animal from the image.

    Args:
        text (str): Text input describing the image
        image_path (str): Path to the image file
        ner_model (AnimalNER): Named Entity Recognition model
        classifier (AnimalClassifier): Image classifier model
    '''
    extracted_animals = ner_model.extract_animals(text)  
    predicted_animal, confidence = classifier.predict(image_path)  
    
    print(f"Extracted from text: {extracted_animals}")
    print(f"Predicted from image: {predicted_animal} (confidence: {confidence:.4f})")
    
    return any(animal.lower() == predicted_animal.lower() for animal in extracted_animals)

def main(args):
    '''
    Start the verification process.

    Args:
        args: Command line arguments
    '''
    ner_model = AnimalNER(args.ner_model_path) 
    classifier = AnimalClassifier(args.classifier_model_path)
    
    result = verify_animal(args.text, args.image_path, ner_model, classifier)
    print(f"Result: {result}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, type=str, help="Text input describing the image")
    parser.add_argument("--image_path", required=True, type=str, help="Path to the image file")
    parser.add_argument("--ner_model_path", required=True, type=str, help="Path to the trained NER model")
    parser.add_argument("--classifier_model_path", required=True, type=str, help="Path to the trained image classifier model")
    
    args = parser.parse_args()
    main(args)
