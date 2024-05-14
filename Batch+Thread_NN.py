import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Manager

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_and_preprocess():
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test

def train_model(model, dataset, criterion, optimizer, start, end):
    # Subsetting the dataset
    inputs, labels = dataset.tensors
    inputs, labels = inputs[start:end], labels[start:end]
    dataset_subset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset_subset, batch_size=16)

    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model.state_dict()

def main():
    with ThreadPoolExecutor(max_workers=4) as executor:
        future = executor.submit(load_and_preprocess)
        X_train, X_test, y_train, y_test = future.result()

    train_dataset = TensorDataset(X_train, y_train)
    num_processes = 4
    dataset_size = len(train_dataset)
    chunk_size = dataset_size // num_processes

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    processes = []
    manager = Manager()
    model_states = manager.list()

    start_time = time.time()

    for i in range(num_processes):
        start_index = i * chunk_size
        end_index = start_index + chunk_size if i < num_processes - 1 else dataset_size
        p = Process(target=train_model, args=(model, train_dataset, criterion, optimizer, start_index, end_index))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed_time = time.time() - start_time
    print(f"Parallel training completed in {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
