# Animal Verification Pipeline

## Description
This project verifies the match between a textual description of an animal and its image. It utilizes two models:
- **AnimalNER** — a Named Entity Recognition (NER) model that extracts animal names from text.
- **AnimalClassifier** — an image classification model that identifies the animal in an image.

## File Structure
- `pipeline.py` — the main script that coordinates the verification process.
- `ner_inference.py` — contains the `AnimalNER` class for extracting animal names from text.
- `image_classification_inference.py` — contains the `AnimalClassifier` class for image classification.
- `demo2.ipynb` — Jupyter Notebook for code demonstration.

## Requirements
The project requires Python 3 and the following libraries:
- `torch`
- `transformers`
- `torchvision`
- `PIL`
- `argparse`
- `json`

Install dependencies with:
```sh
pip install torch torchvision transformers pillow
```

## Usage

### Running `pipeline.py`
```sh
python pipeline.py --text "A fox in the forest" --image_path "image.jpg" --ner_model_path "path_to_ner_model" --classifier_model_path "path_to_classifier_model"
```
Parameters:
- `--text` — text description of the image.
- `--image_path` — path to the image file.
- `--ner_model_path` — path to the pre-trained NER model.
- `--classifier_model_path` — path to the image classification model.

### Running `ner_inference.py`
```sh
python ner_inference.py --model_path "path_to_ner_model" --text "A lion is running."
```
or for a text file:
```sh
python ner_inference.py --model_path "path_to_ner_model" --input_file "texts.txt"
```

### Running `image_classification_inference.py`
```sh
python image_classification_inference.py --model_path "path_to_classifier_model" --image_path "image.jpg"
```
or for a directory of images:
```sh
python image_classification_inference.py --model_path "path_to_classifier_model" --image_dir "images/" --output_file "results.json"
```

### Using `demo2.ipynb`
Open `demo2.ipynb` in Jupyter Notebook and execute the cells to interactively test the code.


