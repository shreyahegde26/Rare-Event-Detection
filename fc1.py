import flwr as fl
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Define the ANN Model (DNN)
# -------------------------------
class DNN(nn.Module):
    def __init__(self, input_size, num_class, num_state):
        super(DNN, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.LayerNorm(512)
        
        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.LayerNorm(1024)
        
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.LayerNorm(512)
        
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.LayerNorm(256)

        self.dropout = nn.Dropout(p=0.3)

        self.fc2_class = nn.Linear(256, num_class)
        self.fc2_state = nn.Linear(256, num_state)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.fc4(x)))

        class_out = self.fc2_class(x)  # Logits for class prediction
        state_out = self.fc2_state(x)
        return class_out, state_out

# -------------------------------
# 2. Flower Client Definition
# -------------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, cid):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cid = cid

        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_state = nn.CrossEntropyLoss()
        # Consider lowering the learning rate for stability
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        # Set device (CPU by default, or GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        model_params = list(self.model.parameters())
        with torch.no_grad():
            for param, new_param in zip(model_params, parameters):
                param.copy_(torch.tensor(new_param, dtype=param.dtype, device=param.device))
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(epochs=1)
        self.save_model()  # Save model after training (optional)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in self.train_loader:
                # Move data to device
                features = batch["features"].to(self.device)
                class_labels = batch["class"].to(self.device)
                state_labels = batch["state"].to(self.device)

                self.optimizer.zero_grad()
                class_out, state_out = self.model(features)
                loss_class = self.criterion_class(class_out, class_labels)
                loss_state = self.criterion_state(state_out, state_labels)
                loss = loss_class + loss_state

                # Check if loss is NaN
                if torch.isnan(loss):
                    print("Warning: NaN loss encountered during training!")
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Client {self.cid}: Epoch {epoch+1}, Loss: {epoch_loss/len(self.train_loader):.4f}")

    def test(self):
        self.model.eval()
        total_loss = 0.0
        total_class_correct = 0
        total_state_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.test_loader:
                # Move data to device
                features = batch["features"].to(self.device)
                class_labels = batch["class"].to(self.device)
                state_labels = batch["state"].to(self.device)

                class_out, state_out = self.model(features)
                loss_class = self.criterion_class(class_out, class_labels)
                loss_state = self.criterion_state(state_out, state_labels)
                batch_loss = loss_class.item() + loss_state.item()
                if np.isnan(batch_loss):
                    print("Warning: NaN loss encountered in test batch!")
                total_loss += batch_loss

                _, class_predicted = torch.max(class_out, 1)
                _, state_predicted = torch.max(state_out, 1)
                total_class_correct += (class_predicted == class_labels).sum().item()
                total_state_correct += (state_predicted == state_labels).sum().item()
                total_samples += class_labels.size(0)

        if total_samples == 0:
            print("Warning: No test samples found!")
            return float('nan'), float('nan')

        class_accuracy = total_class_correct / total_samples
        state_accuracy = total_state_correct / total_samples
        overall_accuracy = (class_accuracy + state_accuracy) / 2
        avg_loss = total_loss / total_samples
        print(f"Client {self.cid}: Test samples: {total_samples}, Class Acc: {class_accuracy:.4f}, State Acc: {state_accuracy:.4f}")
        return avg_loss, overall_accuracy

    def save_model(self):
        model_path = f"/Users/saahil/Desktop/College/Sem 6/TDL_PROJ/FL/models/flower_client_{self.cid}_model.pt"
        torch.save(self.model, model_path)
        print(f"Whole model saved to {model_path}")


# -------------------------------
# 3. Custom CSV Dataset Loader
# -------------------------------
class CustomCSVLoader(Dataset):
    def __init__(self, csv_file, scaler=None, fit_scaler=False):
        self.data = pd.read_csv(csv_file)
        self.num_features = self.data.shape[1] - 2  # Last two columns are labels

        # Extract features
        features = self.data.iloc[:, :self.num_features].values

        # Debug: Print max/min values
        print("Max value in dataset:", np.max(features))
        print("Min value in dataset:", np.min(features))

        # Handle inf/nan values
        if np.isinf(features).any() or np.isnan(features).any():
            print("Warning: Dataset contains inf/nan values. Replacing with column means.")
            features = np.where(np.isfinite(features), features, np.nan)
            col_means = np.nanmean(features, axis=0)
            indices = np.where(np.isnan(features))
            features[indices] = np.take(col_means, indices[1])
        
        # Clip extreme values and convert to float32
        features = np.clip(features, -1e6, 1e6).astype("float32")

        # Scale features
        if fit_scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(features)
        else:
            self.scaler = scaler
        self.features = self.scaler.transform(features)

        # Load labels
        self.class_labels = self.data.iloc[:, self.num_features].values.astype(int)
        self.state_labels = self.data.iloc[:, self.num_features + 1].values.astype(int)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "class": torch.tensor(self.class_labels[idx], dtype=torch.long),
            "state": torch.tensor(self.state_labels[idx], dtype=torch.long),
        }

def load_data(csv_path):
    full_dataset = CustomCSVLoader(csv_path, fit_scaler=True)
    scaler = full_dataset.scaler  # Save scaler for later use

    # Use a subset for memory efficiency, if desired.
    subset_size = min(1000000, len(full_dataset))
    subset_indices = list(range(subset_size))
    dataset_subset = Subset(full_dataset, subset_indices)

    # Split into training and test sets from the same subset
    train_size = int(0.8 * len(dataset_subset))
    test_size = len(dataset_subset) - train_size
    train_dataset, test_dataset = random_split(dataset_subset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_features = full_dataset.num_features
    num_class = int(full_dataset.class_labels.max()) + 1
    num_state = int(full_dataset.state_labels.max()) + 1

    return train_loader, test_loader, num_features, num_class, num_state

# -------------------------------
# 4. Main: Start the Flower Client
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID (unique for each client)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to client CSV data file")
    args = parser.parse_args()

    train_loader, test_loader, input_size, num_class, num_state = load_data(args.data_path)
    print(f"Using {num_class} classes and {num_state} states. Input size: {input_size}")
    model = DNN(input_size=input_size, num_class=num_class, num_state=num_state)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(model, train_loader, test_loader, cid=args.cid),
    )
