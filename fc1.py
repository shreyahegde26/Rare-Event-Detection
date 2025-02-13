import flwr as fl
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch.nn.functional as F

# =============================================================================
# Define the ANN model with dynamic input size and dynamic output sizes
# =============================================================================
class ANN(nn.Module):
    def __init__(self, input_size, num_class, num_state):
        super(ANN, self).__init__()
        self.input_size = input_size  # dynamic input size

        # Shared layers (including input layer)
        self.fc1 = nn.Linear(input_size, 128)  # fc1.weight: (128, input_size)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)           # fc2.weight: (256, 128)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)           # fc3.weight: (128, 256)
        self.dropout = nn.Dropout(0.3)

        # Output layers: using dynamic numbers for classification and state tasks
        self.fc2_class = nn.Linear(128, num_class)  # classification head
        self.fc2_state = nn.Linear(128, num_state)    # state head

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))  # shared input layer
        x = F.relu(self.bn2(self.fc2(x)))  # shared hidden layer 1
        x = self.dropout(x)
        x = F.relu(self.fc3(x))            # shared hidden layer 2
        class_out = self.fc2_class(x)
        state_out = self.fc2_state(x)
        return class_out, state_out

# =============================================================================
# Define the Flower client
# =============================================================================
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
        """Return all model parameters as a list of NumPy arrays."""
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        """Set the model parameters from a list of NumPy arrays."""
        model_params = list(self.model.parameters())
        with torch.no_grad():
            for param, new_param in zip(model_params, parameters):
                param.copy_(torch.tensor(new_param, dtype=param.dtype, device=param.device))
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(epochs=1)
        self.save_model()  # Save the model after training
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    def train(self, epochs):
        self.model.train()
        for _ in range(epochs):
            for batch in self.train_loader:
                features = batch["features"]
                class_labels = batch["class"]
                state_labels = batch["state"]

                self.optimizer.zero_grad()
                class_out, state_out = self.model(features)
                loss_class = self.criterion_class(class_out, class_labels)
                loss_state = self.criterion_state(state_out, state_labels)
                loss = loss_class + loss_state

                loss.backward()
                self.optimizer.step()

    def test(self):
        self.model.eval()
        total_loss = 0.0
        total_class_correct = 0
        total_state_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.test_loader:
                features = batch["features"]
                class_labels = batch["class"]
                state_labels = batch["state"]

                class_out, state_out = self.model(features)
                loss_class = self.criterion_class(class_out, class_labels)
                loss_state = self.criterion_state(state_out, state_labels)
                total_loss += loss_class.item() + loss_state.item()

                _, class_predicted = torch.max(class_out, 1)
                _, state_predicted = torch.max(state_out, 1)
                total_class_correct += (class_predicted == class_labels).sum().item()
                total_state_correct += (state_predicted == state_labels).sum().item()
                total_samples += class_labels.size(0)

        class_accuracy = total_class_correct / total_samples
        state_accuracy = total_state_correct / total_samples
        overall_accuracy = (class_accuracy + state_accuracy) / 2
        avg_loss = total_loss / total_samples
        return avg_loss, overall_accuracy

    def save_model(self):
        """Save the trained model (whole model, not only weights)."""
        model_path = f"flower_client_{self.cid}_model.pt"
        torch.save(self.model, model_path)
        print(f"Whole model saved to {model_path}")

# =============================================================================
# Custom Dataset Loader
# =============================================================================
class CustomCSVLoader(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # The new dataset: features are all columns except the last two.
        self.num_features = self.data.shape[1] - 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        num_columns = self.data.shape[1]
        # Features are all columns except the last two.
        features = self.data.iloc[idx, :num_columns-2].values.astype("float32")
        # The second-to-last column is the class label.
        class_label = int(self.data.iloc[idx, num_columns-2])
        # The last column is the state label.
        state_label = int(self.data.iloc[idx, num_columns-1])
        return {
            "features": torch.tensor(features),
            "class": torch.tensor(class_label, dtype=torch.long),
            "state": torch.tensor(state_label, dtype=torch.long),
        }

def load_data(csv_path):
    dataset = CustomCSVLoader(csv_file=csv_path)

    num_columns = dataset.data.shape[1]
    num_features = dataset.num_features
    # Compute the number of classes and states based on the last two columns.
    num_class = int(dataset.data.iloc[:, num_columns-2].max()) + 1
    num_state = int(dataset.data.iloc[:, num_columns-1].max()) + 1


    subset_size = min(10000, len(dataset))  # Ensure it doesn't exceed the dataset size
    subset_indices = list(range(subset_size))
    dataset_subset = torch.utils.data.Subset(dataset, subset_indices)

    train_size = int(0.8 * len(dataset_subset))
    test_size = len(dataset_subset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_subset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, num_features, num_class, num_state

# =============================================================================
# Main: Start the Flower client
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID (unique for each client)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to client CSV data file")
    args = parser.parse_args()

    train_loader, test_loader, input_size, num_class, num_state = load_data(args.data_path)
    print(f"Using {num_class} classes and {num_state} states. Input size: {input_size}")
    model = ANN(input_size=input_size, num_class=num_class, num_state=num_state)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(model, train_loader, test_loader, cid=args.cid),
    )
