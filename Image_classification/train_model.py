import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier as SklearnRF

def load_data():
    '''
    Laod MNIST dataset and separate to train and test data

    Returns:
        X_train (np.array): Training images
        y_train (np.array): Training labels
        X_test (np.array): Test images
        y_test (np.array): Test labels
    '''
    training_transforms = transforms.Compose([
        transforms.RandomResizedCrop(28),
        transforms.RandomRotation(45),
        transforms.ToTensor()
    ])

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=training_transforms
    )

    predict_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    X_train = training_data.data.numpy()
    y_train = training_data.targets.numpy()
    X_test = predict_data.data.numpy()
    y_test = predict_data.targets.numpy()

    return X_train, y_train, X_test, y_test


class MnistClassifierInterface(ABC):
    '''
    Interface for MNIST classifiers
    '''
    @abstractmethod
    def train_model(self, X_train, y_train):
        '''
        Train the model on the training data

        Args:
            X_train (np.array): Training images
            y_train (np.array): Training labels
        '''
        pass
    
    @abstractmethod
    def predict_model(self, X_test):
        '''
        Predict the labels for the test data

        Args:
            X_test (np.array): Test images
        '''
        pass


class RandomForestClassifier(MnistClassifierInterface):
    """
    Random Forest classifier for MNIST

    Attributes:
        model (SklearnRF): Random Forest model
    """
    def __init__(self, model=None):
        '''
        Initialize the Random Forest classifier

        Args:
            model (SklearnRF): Random Forest model
        '''
        self.model = model
        
    def train_model(self, X_train, y_train):
        '''
        Train the Random Forest model

        Args:
            X_train (np.array): Training images
            y_train (np.array): Training labels
        '''
        X_train = X_train.reshape(-1, 28*28)
        self.model = SklearnRF()
        self.model.fit(X_train, y_train)
        return self.model

    def predict_model(self, X_test):
        '''
        Predict the labels for the test data

        Args:
            X_test (np.array): Test images
        '''
        X_test = X_test.reshape(-1, 28*28)
        return self.model.predict(X_test)


class FeedforwardNeuralNetworkClassifier(nn.Module, MnistClassifierInterface):
    """
    Neural Network classifier for MNIST

    Attributes:
        fc1 (nn.Linear): First fully connected layer
        fc2 (nn.Linear): Second fully connected layer
        fc3 (nn.Linear): Third fully connected layer
        activation (nn.ReLU): Activation function
        loss_function (nn.CrossEntropyLoss): Loss function
        optimizer (optim.Adam): Optimizer
    """
    def __init__(self, input_size=784, hidden_sizes=[256, 128], output_size=10):
        '''
        Initialize the Feedforward Neural Network classifier

        Args:
            input_size (int): Input size
            hidden_sizes (list): List of hidden layer sizes
            output_size (int): Output size
        '''
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.activation = nn.ReLU()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, x):
        '''
        Forward pass of the neural network

        Args:
            x (torch.Tensor): Input tensor
        '''
        x = x.view(-1, 784)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_model(self, X_train, y_train, epochs=5, batch_size=64):
        '''
        Train the Feedforward Neural Network model

        Args:
            X_train (np.array): Training images
            y_train (np.array): Training labels
            epochs (int): Number of epochs
            batch_size (int): Batch size
        '''
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for images, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
        return self

    def predict_model(self, X_test):
        '''
        Predict the labels for the test data

        Args:
            X_test (np.array): Test images
        '''
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
        with torch.no_grad():
            outputs = self(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            predictions = predicted.cpu().numpy()
        return predictions


class CnnClassifier(nn.Module, MnistClassifierInterface):
    """
    CNN classifier for MNIST using PyTorch

    Attributes:
        model (nn.Sequential): CNN model
        loss_function (nn.CrossEntropyLoss): Loss function
        optimizer (optim.Adam): Optimizer
    """
    def __init__(self):
        '''
        Initialize the CNN classifier
        '''
        super(CnnClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # Convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Pooling layer
            nn.Conv2d(32, 64, kernel_size=3),  # Convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Pooling layer
            nn.Flatten(),  # Flatten layer
            nn.Linear(64 * 5 * 5, 64),  # Fully connected layer
            nn.ReLU(),
            nn.Linear(64, 10),  # Output layer
            nn.Softmax(dim=1)
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self, X_train, y_train, epochs=5, batch_size=64):
        '''
        Train the CNN model

        Args:
            X_train (np.array): Training images
            y_train (np.array): Training labels
            epochs (int): Number of epochs
            batch_size (int): Batch size
        '''
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for images, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    def predict_model(self, X_test):
        '''
        Predict the labels for the test data

        Args:
            X_test (np.array): Test images
        '''
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            predictions = predicted.cpu().numpy()
        return predictions


class MnistClassifier:
    '''
    Class to initialize the classifier based on the selected algorithm

    Attributes:
        classifier (MnistClassifierInterface): Classifier based on the selected algorithm
    '''
    def __init__(self, algorithm):
        '''
        Initialize the classifier based on the selected algorithm

        Args:
            algorithm (str): Algorithm to use for classification
        '''
        if algorithm == 'rf':
            self.classifier = RandomForestClassifier()
        elif algorithm == 'nn':
            self.classifier = FeedforwardNeuralNetworkClassifier()
        elif algorithm == 'cnn':
            self.classifier = CnnClassifier()
        else:
            raise ValueError(f"Wrong algorithm: {algorithm}. Use 'cnn', 'rf', or 'nn'")