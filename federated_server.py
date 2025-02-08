# import flwr as fl
# from typing import List, Tuple, Dict, Optional
# import numpy as np

# class CustomAdaptiveAggregation(fl.server.strategy.FedAvg):
#     def aggregate_fit(self, rnd: int,results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],failures: List[BaseException],) -> Optional[Tuple[fl.common.Parameters, Dict[str, float]]]:
#         """
#         Aggregate model parameters while handling different input dimensions.
#         Only aggregate shared layers, ignoring input layers.
#         """
#         if not results:
#             return fl.common.ndarrays_to_parameters([]), {}  # Ensure function always returns a tuple

#         # Assign weights to clients (higher weight to clients 1, 5, 6)
#         client_weights = []
#         for client_proxy, fit_res in results:
#             client_id = client_proxy.cid  # Assuming client IDs are strings
#             if client_id in ["1", "5", "6"]:
#                 client_weights.append(3.0)  # Higher weight
#             else:
#                 client_weights.append(1.0)  # Default weight

#         # Normalize weights
#         total_weight = sum(client_weights)
#         normalized_weights = [w / total_weight for w in client_weights]

#         # Initialize aggregated parameters
#         aggregated_parameters = None

#         for normalized_weight, (_, fit_res) in zip(normalized_weights, results):
#             # Convert parameters to NumPy arrays
#             client_parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)

#             # Skip first layer (input layer) since input dimensions vary
#             shared_parameters = client_parameters[1:]

#             if aggregated_parameters is None:
#                 aggregated_parameters = [
#                     normalized_weight * param for param in shared_parameters
#                 ]
#             else:
#                 aggregated_parameters = [
#                     agg + normalized_weight * param
#                     for agg, param in zip(aggregated_parameters, shared_parameters)
#                 ]

#         # Aggregate metrics (e.g., accuracy or loss)
#         aggregated_metrics = {}
#         for metric_name in results[0][1].metrics.keys():
#             aggregated_metrics[metric_name] = sum(
#                 normalized_weight * result[1].metrics[metric_name]
#                 for normalized_weight, result in zip(normalized_weights, results)
#             )

#         # Convert aggregated parameters back to Flower Parameters (excluding input layer)
#         aggregated_parameters.insert(0, np.zeros((1,)))  # Dummy placeholder for input layer
#         return fl.common.ndarrays_to_parameters(aggregated_parameters), aggregated_metrics


# if __name__ == "__main__":
#     # Start the Flower server with the custom adaptive strategy
#     strategy = CustomAdaptiveAggregation(min_fit_clients=2, min_available_clients=2)

#     fl.server.start_server(
#         server_address="localhost:8080",
#         config=fl.server.ServerConfig(num_rounds=3),
#         strategy=strategy,
#     )



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
        Aggregates only shared layers, excluding input layers, to handle heterogeneous input dimensions.
        """
        if not results:
            return fl.common.ndarrays_to_parameters([]), {}  # Ensure function always returns a tuple

        # Assign weights to clients (higher weight to clients 1, 5, 6)
        client_weights = []
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            if client_id in ["1", "5", "6"]:
                client_weights.append(3.0)  # Higher weight
            else:
                client_weights.append(1.0)  # Default weight

        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # Initialize aggregated parameters storage
        aggregated_parameters = {}
        layer_counts = {}

        for normalized_weight, (_, fit_res) in zip(normalized_weights, results):
            client_layers = {
                f"layer_{i}": param
                for i, param in enumerate(fl.common.parameters_to_ndarrays(fit_res.parameters))
            }

            # Exclude the first layer (input layer) dynamically
            layer_names = list(client_layers.keys())[1:]

            for layer_name in layer_names:
                if layer_name not in aggregated_parameters:
                    aggregated_parameters[layer_name] = normalized_weight * client_layers[layer_name]
                    layer_counts[layer_name] = normalized_weight
                else:
                    aggregated_parameters[layer_name] += normalized_weight * client_layers[layer_name]
                    layer_counts[layer_name] += normalized_weight

        # Normalize by weight
        for layer_name in aggregated_parameters.keys():
            aggregated_parameters[layer_name] /= layer_counts[layer_name]

        # Aggregate metrics (e.g., accuracy or loss)
        aggregated_metrics = {}
        for metric_name in results[0][1].metrics.keys():
            aggregated_metrics[metric_name] = sum(
                normalized_weight * result[1].metrics[metric_name]
                for normalized_weight, result in zip(normalized_weights, results)
            )

        # Convert aggregated parameters back to Flower Parameters (EXCLUDING INPUT LAYER)
        aggregated_parameters_list = [
            aggregated_parameters[layer_name]
            for layer_name in sorted(aggregated_parameters.keys())
        ]
        
        # Insert a dummy first layer to ensure proper serialization
        aggregated_parameters_list.insert(0, np.zeros((1,)))  # Dummy placeholder for input layer

        return fl.common.ndarrays_to_parameters(aggregated_parameters_list), aggregated_metrics


if __name__ == "__main__":
    # Start the Flower server with the custom adaptive strategy
    strategy = CustomAdaptiveAggregation(min_fit_clients=9, min_available_clients=9)

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
