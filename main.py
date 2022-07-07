from client import Client
from server import Server
import os
from progress.bar import Bar


def main():
    num_of_rounds = 10
    model_dir = 'models'
    model_names = ['vigna_model_1.h5', 'vigna_model_2.h5', 'vigna_model_3.h5']
    dataset_dir = os.path.join('datasets', 'chickpea')
    dataset_names = ['response1.csv', 'response2.csv']
    images_path = os.path.join(dataset_dir, 'images')

    models = []
    for model in model_names:
        models.append(os.path.join(model_dir, model))

    datasets = []
    for file in dataset_names:
        datasets.append(os.path.join(dataset_dir, file))

    with Bar('Clients initialization', max=(len(datasets)) * (len(models))) as bar:
        clients = []
        client_num = 0
        for dataset in datasets:
            for model in models:
                client_num += 1
                clients.append(Client(path_to_model=model,
                                      csv_file=dataset,
                                      images_dir=images_path,
                                      name='client_' + str(client_num),
                                      wandb_group='chickpea_clients',
                                      model_save_path='models/client_' + str(client_num) + '.h5'
                                      ))
                bar.next()

    print('Server initialization')
    server = Server(clients)
    print('Done')

    for round_num in range(num_of_rounds):
        print(f'Start of round {round_num+1} of federated learning')

        # Simulate training each client's model
        with Bar('Clients training simulation', max=len(clients)) as bar:
            for client in clients:
                client.train()
                bar.next()

        # Calculate average weights
        print('  Calculating average weights and sending them to all clients...')
        server.start(clients)
        print('  Done')
        print()


if __name__ == "__main__":
    main()
