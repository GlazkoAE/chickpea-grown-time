from client import Client
from server import Server

path_client_1 = 'models/chickpea_model_1.h5'
path_client_2 = 'models/chickpea_model_2.h5'
path_client_3 = 'models/chickpea_model_3.h5'
path_client_4 = 'models/chickpea_model_4.h5'
path_client_5 = 'models/chickpea_model_5.h5'
path_client_6 = 'models/chickpea_model_6.h5'
dataset_1 = 'datasets/chickpea/response1.csv'
dataset_2 = 'datasets/chickpea/response2.csv'
clients = []

client1 = Client(path_client_1, dataset_1)
client2 = Client(path_client_2, dataset_1)
client3 = Client(path_client_3, dataset_1)
client4 = Client(path_client_4, dataset_2)
client5 = Client(path_client_5, dataset_2)
client6 = Client(path_client_6, dataset_2)

clients.append(client1)
clients.append(client2)
clients.append(client3)
clients.append(client4)
clients.append(client5)
clients.append(client6)

server = Server(clients)

num_of_rounds = 10

for num in range(num_of_rounds):
    for i in range(len(clients)):
        clients[i].fit(clients[i].get_parameters())
    mean_weights = server.start(clients)
    clients[i].set_parameters(mean_weights)