import flwr as fl
from typing import List, Tuple, Dict, Optional
import numpy as np

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[fl.common.Parameters, Dict[str, float]]]:
        """
        Aggregates all layers using equal weighting across clients.
        """
        if not results:
            print("No clients returned results.")
            return fl.common.ndarrays_to_parameters([]), {}

        num_clients = len(results)
        client_weight = 1.0 / num_clients  # Equal weight for all clients
        
        # Determine number of layers from the first client
        first_client_params = fl.common.parameters_to_ndarrays(results[0][1].parameters)
        num_layers = len(first_client_params)

        # Accumulators for sum of weights and counts
        aggregated_parameters = [np.zeros_like(param) for param in first_client_params]
        
        # Aggregate client updates
        for _, fit_res in results:
            client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            for i, param in enumerate(client_params):
                aggregated_parameters[i] += client_weight * param  # Equal weighting
        
        # Convert back to Flower's parameter format
        return fl.common.ndarrays_to_parameters(aggregated_parameters), {}

if __name__ == "__main__":
    strategy = CustomFedAvg(min_fit_clients=2, min_available_clients=2)

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
