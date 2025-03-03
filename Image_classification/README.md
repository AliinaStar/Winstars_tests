# MNIST Classification

This project implements several models for classifying images from the MNIST dataset.

## Description

The project includes three different classification models:

- **Random Forest (RF)** — uses `sklearn.ensemble.RandomForestClassifier`.
- **Feedforward Neural Network (FNN)** — implemented with `torch.nn`.
- **Convolutional Neural Network (CNN)** — also implemented using `torch.nn`.

## Installation

Before running the code, install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Loading Data

The `load_data()` function loads the MNIST dataset and splits it into training and test sets.

### Choosing a Model

You can select one of the three models by passing the appropriate argument when creating an instance of `MnistClassifier`:

```python
from train_model import MnistClassifier, load_data

X_train, y_train, X_test, y_test = load_data()

# Choose a model ('rf' - Random Forest, 'nn' - Feedforward Neural Network, 'cnn' - Convolutional Neural Network)
classifier = MnistClassifier(algorithm='cnn')
classifier.classifier.train_model(X_train, y_train)
predictions = classifier.classifier.predict_model(X_test)
```

## File Description

- `train_model.py` — contains the implementation of data loading and classification models.
- `demo.ipynb` — Jupyter notebook demonstrating the models in action.


