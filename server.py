import numpy as np


class Server:
    def __init__(self, clients):
        self.clients = clients

    def update_clients(self, clients):
        self.clients = clients

    def start(self, clients):
        self.update_clients(clients)
        return self.FedAvg()

    def FedAvg(self):
        models = []
        for i in range(len(self.clients)):
            models.append(self.clients[i].get_model())
        weights = [model.get_weights() for model in models]
        new_weights = list()
        for weights_list_tuple in zip(*weights):
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
        return new_weights