import flwr as fl
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd


# Define the ANN model with dynamic input size
class ANN(nn.Module):
    def __init__(self, input_size):
        super(ANN, self).__init__()
        self.input_size = input_size  # Store input size dynamically
        self.fc1 = nn.Linear(input_size, 64)  # Input layer adapts to dataset
        self.fc2_class = nn.Linear(64, 10)  # Output for 'class' target
        self.fc2_state = nn.Linear(64, 8)  # Output for 'state' target (assuming 3 states)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        class_out = self.fc2_class(x)  # Output for 'class'
        state_out = self.fc2_state(x)  # Output for 'state'
        return class_out, state_out


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, cid):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cid = cid
        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_state = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def get_parameters(self, config=None):
        # Skip input layer and return parameters
        params = [val.detach().cpu().numpy() for i, val in enumerate(self.model.parameters()) if i > 0]
        return params

    def set_parameters(self, parameters):
        # Load parameters, but keep input layer unchanged
        with torch.no_grad():
            for i, param in enumerate(self.model.parameters()):
                if i > 0:  # Do not overwrite the input layer
                    param.copy_(torch.tensor(parameters[i - 1]))

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(epochs=1)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    def train(self, epochs):
        self.model.train()
        for _ in range(epochs):
            for batch in self.train_loader:
                features, class_labels, state_labels = (
                    batch["features"],
                    batch["class"],
                    batch["state"],
                )

                self.optimizer.zero_grad()
                class_out, state_out = self.model(features)

                loss_class = self.criterion_class(class_out, class_labels)
                loss_state = self.criterion_state(state_out, state_labels)
                loss = loss_class + loss_state

                loss.backward()
                self.optimizer.step()

    def test(self):
        self.model.eval()
        total_class_correct = 0
        total_state_correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for batch in self.test_loader:
                features, class_labels, state_labels = (
                    batch["features"],
                    batch["class"],
                    batch["state"],
                )
                class_out, state_out = self.model(features)

                loss_class = self.criterion_class(class_out, class_labels)
                loss_state = self.criterion_state(state_out, state_labels)
                loss += loss_class.item() + loss_state.item()

                _, class_predicted = torch.max(class_out.data, 1)
                _, state_predicted = torch.max(state_out.data, 1)

                total_class_correct += (class_predicted == class_labels).sum().item()
                total_state_correct += (state_predicted == state_labels).sum().item()
                total += class_labels.size(0)

        class_accuracy = total_class_correct / total
        state_accuracy = total_state_correct / total
        overall_accuracy = (class_accuracy + state_accuracy) / 2
        return loss / total, overall_accuracy


# Custom Dataset Loader
class CustomCSVLoader(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.num_features = self.data.shape[1] - 2  # Assuming first two columns are labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx, 2:].values.astype("float32")  # Features
        class_label = self.data.iloc[idx, 0]  # 'class'
        state_label = self.data.iloc[idx, 1]  # 'state'
        return {
            "features": torch.tensor(features),
            "class": torch.tensor(class_label, dtype=torch.long),
            "state": torch.tensor(state_label, dtype=torch.long),
        }


def load_data(csv_path):
    dataset = CustomCSVLoader(csv_file=csv_path)
    num_features = dataset.num_features  # Store feature count BEFORE making it a Subset

    # subset_size = min(100000, len(dataset))  # Ensure it doesn't exceed the dataset size
    # subset_indices = list(range(subset_size))
    # dataset_subset = torch.utils.data.Subset(dataset, subset_indices)

    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, num_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID (unique for each client)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to client CSV data file")
    args = parser.parse_args()

    # Load dataset and get feature count
    train_loader, test_loader, input_size = load_data(args.data_path)

    # Create model dynamically based on feature size
    model = ANN(input_size=input_size)

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(model, train_loader, test_loader, cid=args.cid),
    )
