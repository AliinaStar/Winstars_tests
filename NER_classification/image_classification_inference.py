import argparse
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

class AnimalClassifier:
    '''
    Class for image classification
    
    Attributes:
        model_path (str): path to the trained classifier model
        class_names (list): list of class names
        model (torch.nn.Module): trained model
        transform (torchvision.transforms.Compose): image transformation
    '''
    def __init__(self, model_path):
        '''
        Constructor for AnimalClassifier class
        
        Args:
            model_path (str): path to the trained classifier model
        '''
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        self.class_names = checkpoint['class_names']
        
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.class_names))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        '''
        Predicts the class of the image

        Args:
            image_path (str): path to the image file
        '''
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, predicted = torch.max(probabilities, 1)
        
        class_name = self.class_names[predicted.item()]
        confidence = conf.item()
        
        return class_name, confidence

def main(args):
    '''
    Main function for image classification

    Args:
        args (argparse.Namespace): command-line arguments
    '''
    classifier = AnimalClassifier(args.model_path)
    
    if args.image_path:
        class_name, confidence = classifier.predict(args.image_path)
        print(f"Image: {args.image_path}")
        print(f"Predicted class: {class_name} (confidence: {confidence:.4f})")
    
    if args.image_dir:
        results = {}
        for filename in os.listdir(args.image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.image_dir, filename)
                class_name, confidence = classifier.predict(image_path)
                results[filename] = {"class": class_name, "confidence": float(confidence)}
                print(f"Image: {filename}")
                print(f"Predicted class: {class_name} (confidence: {confidence:.4f})")
                print("-" * 50)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to the trained classifier model")
    parser.add_argument("--image_path", type=str, help="Path to image file to classify")
    parser.add_argument("--image_dir", type=str, help="Directory with images to classify")
    parser.add_argument("--output_file", type=str, help="Output JSON file for batch classification")
    
    args = parser.parse_args()
    main(args)