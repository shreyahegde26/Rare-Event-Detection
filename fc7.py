import flwr as fl
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd


# Define the ANN model
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(17, 64)  # Input size matches dataset features, one hidden layer
        self.fc2_class = nn.Linear(64, 10)  # Output for 'class' target (binary classification)
        self.fc2_state = nn.Linear(64, 1)  # Output for 'state' target (assuming 3 states)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        class_out = self.fc2_class(x)  # Output for 'class'
        state_out = self.fc2_state(x)  # Output for 'state'
        return class_out, state_out


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader,cid):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cid = cid
        self.criterion_class = nn.CrossEntropyLoss()  # For 'class' target
        self.criterion_state = nn.CrossEntropyLoss()  # For 'state' target
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def get_parameters(self, config=None):
        # Return model parameters as NumPy arrays
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        # Load parameters into the model
        for param, val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(val)

    def fit(self, parameters, config):
        # Update model parameters
        self.set_parameters(parameters)
        # Train the model for one epoch
        self.train(epochs=1)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Update model parameters
        self.set_parameters(parameters)
        # Evaluate the model
        loss, accuracy = self.test()
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    # def train(self, epochs):
    #     self.model.train()
    #     for _ in range(epochs):
    #         for batch in self.train_loader:
    #             features, class_labels, state_labels = (
    #                 batch["features"],
    #                 batch["class"],
    #                 batch["state"],
    #             )
    #             self.optimizer.zero_grad()
    #             class_out, state_out = self.model(features)
    #             loss_class = self.criterion_class(class_out, class_labels)
    #             loss_state = self.criterion_state(state_out, state_labels)
    #             loss = loss_class + loss_state
    #             loss.backward()
    #             self.optimizer.step()
    def train(self, epochs):
        self.model.train()
        for _ in range(epochs):
            for batch in self.train_loader:
                features, class_labels, state_labels = (
                    batch["features"],
                    batch["class"],
                    batch["state"],
                )

                # Debug: Print unique label values in the batch

                if max(class_labels.unique() > 2): 
                    print(f"Batch Class Labels: {class_labels.unique()}")
                    print(f"Batch State Labels: {state_labels.unique()}")

                self.optimizer.zero_grad()
                class_out, state_out = self.model(features)

                # Debug: Print shape of model outputs
                # print(f"class_out shape: {class_out.shape}")
                # print(f"state_out shape: {state_out.shape}")

            # Compute loss
            try:
                loss_class = self.criterion_class(class_out, class_labels)
                loss_state = self.criterion_state(state_out, state_labels)
                loss = loss_class + loss_state
                loss.backward()
                self.optimizer.step()
            except IndexError as e:
                print(f"🔥 Error occurred! {e}")
                print(f"🔥 Problematic class labels: {class_labels}")
                print(f"🔥 Problematic state labels: {state_labels}")
                exit(1)  # Stop execution for debugging

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


# Custom Dataset
class CustomCSVLoader(Dataset):
    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)
        # Map labels if needed
        # self.data['class'] = self.data['class'].map({0.0: 0, 2.0: 2})

        print("Unique class labels:", self.data["class"].unique())
        print("Unique state labels:", self.data["state"].unique())

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


def load_data():
    csv_path = "/Users/shreyahegde/Desktop/COLLEGE/SEM 6/TDL/MINI PROJECT /combined files _cleaned/7_combined_cleaned.csv"
    dataset = CustomCSVLoader(csv_file=csv_path)

    # Limit the dataset to the first 10,000 instances for verification
    # subset_size = min(100000, len(dataset))  # Ensure it doesn't exceed the dataset size
    # dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))

    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID (unique for each client)")
    args = parser.parse_args()

    # Load data
    train_loader, test_loader = load_data()

    # Create model
    model = ANN()

    # Start Flower client with the given client ID
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(model, train_loader, test_loader,cid = args.cid),
    )
