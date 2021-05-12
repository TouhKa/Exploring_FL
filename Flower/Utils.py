import numpy as np
import logging
class Utils:
    def __init__(self) -> None:
        logging.basicConfig(filename='../logs/Utils.log', level=logging.DEBUG)
    def map_ids(self, x):
        new_ids = [float(i) for i in range(871 + 1)]
        old_ids = np.unique(x[:, 0])
        for idx, id in enumerate(old_ids):
            new_id = new_ids[idx]
            indicees = x[:, 0] == id
            x[indicees, 0] = new_id
        return x

    def create_clients(self, x, num_clients, strict = False):
        idx_per_client = int(len(x) / num_clients)
        overhead = len(x) % num_clients

        if strict:
            #TODO
            pass

        else:
            rng = np.random.default_rng()
            rng.shuffle(x, axis=0)
            client_id_index = np.array([])
            for i in range(0, num_clients-1):
                this_id = [i] * idx_per_client
                client_id_index = np.concatenate( (client_id_index, np.array(this_id)), axis = 0)
            this_id = [num_clients-1] * (idx_per_client + overhead)
            client_id_index = np.concatenate( (client_id_index, np.array(this_id)), axis = 0)

            if len(x) == len(client_id_index):
                x_new = np.insert(x, 0, client_id_index, axis=1)
                return x_new
            else:
                logging.debug(f"Data length {len(x)} \nIndicees length: {len(client_id_index)}")
                logging.info("Formatation error: Could not generate Clients")