import flwr as fl
from typing import List, Tuple, Dict, Optional
import numpy as np

class CustomAdaptiveAggregation(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[fl.common.Parameters, Dict[str, float]]]:
        """
        Aggregates all layers (including the input layer) using equal weighting for IID data.
        For the classification layer (assumed to be the last layer), extra weight is applied to
        the classes 1, 5, and 6.
        """
        if not results:
            print("No clients returned results.")
            return fl.common.ndarrays_to_parameters([]), {}
        
        # With IID clients, all clients get equal weight.
        num_clients = len(results)
        norm_weights = [1.0 / num_clients] * num_clients
        
        # Determine the number of layers from the first client.
        first_client_params = fl.common.parameters_to_ndarrays(results[0][1].parameters)
        num_layers = len(first_client_params)
        # We assume the layers are named "layer_0", "layer_1", ... "layer_{num_layers-1}"
        # The classification layer is assumed to be the last layer.
        classification_layer_key = f"layer_{num_layers-1}"
        # Classes for which we wish to assign extra priority
        priority_classes = [1, 5, 6]
        
        # Dictionaries to accumulate the weighted parameters and corresponding weights.
        aggregated_parameters: Dict[str, np.ndarray] = {}
        layer_counts: Dict[str, np.ndarray or float] = {}
        
        # Loop over the clients’ updates.
        for client_weight, (client_proxy, fit_res) in zip(norm_weights, results):
            client_id = str(client_proxy.cid)
            client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            client_layers = {f"layer_{i}": param for i, param in enumerate(client_params)}
            
            # Debug: Print the layers received from each client.
            print(f"Client {client_id} Layers: {list(client_layers.keys())}")
            
            # Iterate over ALL layers (including the input layer, i.e. "layer_0")
            for layer_name, param in client_layers.items():
                # If this is the classification layer, apply extra per-class weighting.
                if layer_name == classification_layer_key:
                    # Handle weight matrices (2D) and bias vectors (1D) separately.
                    if param.ndim == 2:
                        num_classes = param.shape[0]
                        # Create per-class multipliers (default 1.0, but 3.0 for classes 1, 5, and 6)
                        multipliers = np.ones(num_classes)
                        for cls in priority_classes:
                            if cls < num_classes:
                                multipliers[cls] = 3.0
                        # Multiply the client weight with the per-class multipliers.
                        effective_multiplier = client_weight * multipliers  # shape: (num_classes,)
                        effective_multiplier = effective_multiplier[:, np.newaxis]  # reshape for broadcasting
                        weighted_update = param * effective_multiplier
                        if layer_name not in aggregated_parameters:
                            aggregated_parameters[layer_name] = weighted_update
                            layer_counts[layer_name] = effective_multiplier.squeeze()
                        else:
                            aggregated_parameters[layer_name] += weighted_update
                            layer_counts[layer_name] += effective_multiplier.squeeze()
                    elif param.ndim == 1:
                        num_classes = param.shape[0]
                        multipliers = np.ones(num_classes)
                        for cls in priority_classes:
                            if cls < num_classes:
                                multipliers[cls] = 3.0
                        effective_multiplier = client_weight * multipliers
                        weighted_update = param * effective_multiplier
                        if layer_name not in aggregated_parameters:
                            aggregated_parameters[layer_name] = weighted_update
                            layer_counts[layer_name] = effective_multiplier
                        else:
                            aggregated_parameters[layer_name] += weighted_update
                            layer_counts[layer_name] += effective_multiplier
                    else:
                        # Fallback: if not 1D or 2D, perform simple weighted update.
                        if layer_name not in aggregated_parameters:
                            aggregated_parameters[layer_name] = client_weight * param
                            layer_counts[layer_name] = client_weight
                        else:
                            aggregated_parameters[layer_name] += client_weight * param
                            layer_counts[layer_name] += client_weight
                else:
                    # For all other layers, use uniform averaging.
                    if layer_name not in aggregated_parameters:
                        aggregated_parameters[layer_name] = client_weight * param
                        layer_counts[layer_name] = client_weight
                    else:
                        aggregated_parameters[layer_name] += client_weight * param
                        layer_counts[layer_name] += client_weight
        
        # Normalize the aggregated parameters by dividing by the accumulated weights.
        for layer_name, agg_param in aggregated_parameters.items():
            if layer_name == classification_layer_key:
                # Handle element-wise normalization for the classification layer.
                if agg_param.ndim == 2:
                    counts = layer_counts[layer_name][:, np.newaxis]
                    aggregated_parameters[layer_name] = agg_param / counts
                elif agg_param.ndim == 1:
                    aggregated_parameters[layer_name] = agg_param / layer_counts[layer_name]
                else:
                    aggregated_parameters[layer_name] = agg_param / layer_counts[layer_name]
            else:
                aggregated_parameters[layer_name] = agg_param / layer_counts[layer_name]
        
        # (Optional) Aggregate metrics using the same weighting.
        aggregated_metrics: Dict[str, float] = {}
        for metric_name in results[0][1].metrics.keys():
            aggregated_metrics[metric_name] = sum(
                client_weight * result[1].metrics[metric_name]
                for client_weight, result in zip(norm_weights, results)
            )
        
        # Convert the aggregated parameters (a dictionary) back into a list ordered by layer index.
        aggregated_parameters_list = [
            aggregated_parameters[f"layer_{i}"] for i in range(num_layers)
        ]
        
        return fl.common.ndarrays_to_parameters(aggregated_parameters_list), aggregated_metrics

if __name__ == "__main__":
    # Start the Flower server with the custom strategy.
    strategy = CustomAdaptiveAggregation(min_fit_clients=2, min_available_clients=2)

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
