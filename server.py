import numpy as np


class Server:
    def __init__(self, clients):
        self.clients = clients
        self.current_weights = None

    def update_clients(self, clients):
        self.clients = clients

    def start(self, clients):
        self.update_clients(clients)
        self.update_weights()
        self.send_weights()

    def update_weights(self):
        weights = [client.get_parameters() for client in self.clients]

        new_weights = list()
        for weights_list_tuple in zip(*weights):
            new_weights.append(
                np.array([np.array(weights_).mean(axis=0)
                          for weights_ in zip(*weights_list_tuple)])
            )

        self.current_weights = new_weights

    def send_weights(self):
        for client in self.clients:
            client.set_parameters(self.current_weights)
