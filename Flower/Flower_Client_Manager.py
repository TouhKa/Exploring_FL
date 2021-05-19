import random
import sys
sys.path.append("../")
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Reader import Reader
from Utils import Utils 

class Client_Manager:
    def __init__(self, file_name, out_folder, num_clients, just_IDs) -> None:
        self.__data_dir = "../data/"
        self.__file_name = file_name
        self.__out_folder = out_folder
        self.__num_clients = num_clients
        self.__just_IDs = just_IDs
        self.___create_flower_datasets()

    def ___create_flower_datasets(self):
        client_ids = None
        scaler = StandardScaler()
        utils = Utils()
        reader = Reader(self.__data_dir, self.__file_name)
        data = reader.get_data()
        
        if ("IID" in self.__file_name): 
            data = utils.create_clients(data, self.__num_clients, strict = False)
            reader.set_features(reader.get_features() + 1)
            client_ids =  [i for i in range(0, self.__num_clients)]
        cols = [i for i in range(0, reader.get_features())]
        del cols[0]
        del cols[2]
    
        features = reader.get_features()
        scaler.fit(data[:, cols])
        if ((self.__file_name == "App_usage_trace.txt") or (self.__file_name == "top_90_apps.csv")): 
            data  = utils.map_ids(data.copy())
            num_of_users = int((np.amax(data[:, 0]) + 1))
            client_ids = list(range(0, num_of_users))
            logging.info(len(client_ids))
            random.shuffle(client_ids)
            client_ids = client_ids[:self.__num_clients]

        if not (self.__just_IDs):
            for id in client_ids:
                indicees = data[:, 0] == id
                former_shape = data[indicees, 1:features].shape
                #delete index 0 and 3, containing the label and the user id
                client_x = np.delete( data[indicees, 1:features], 2, 1 ).reshape(former_shape[0], former_shape[1]-1)
                #scale 
                client_x = scaler.transform(client_x)
                client_y = data[indicees, 3].reshape(-1, 1)
                if len(client_x) > 1:
                    X_train, X_test, y_train, y_test = train_test_split(client_x, client_y, test_size=0.2, random_state=42)    
                    self.__write_data_to_file(X_train, "X_train", id)
                    self.__write_data_to_file(X_test, "X_test", id)
                    self.__write_data_to_file(y_train, "y_train", id)
                    self.__write_data_to_file(y_test, "y_test", id)
                    logging.info("Client {}: Created Flower dataset".format(id))
                else:
                    logging.warning("Could not generate datasets for client {} as there is just one entry in X_train".format(id))
                    client_ids.remove(id)

            self.__write_ids_to_file(client_ids)

    def __write_data_to_file(self, data, modus, client_id):
        df = pd.DataFrame(data)
        df.to_csv(  
                    self.__data_dir + self.__out_folder + "/" + modus + "_" + str(client_id) + ".csv", 
                    encoding="utf8")

    def __write_ids_to_file(self, ids):
        # write 10, and 100, to files
        with open(self.__data_dir + self.__out_folder + "/" + str(self.__num_clients) + "_clients.txt", "w") as fout:
            for id in ids:
                fout.write(str(id) + "\n")

if __name__ == "__main__":
    logging.basicConfig(filename='../logs/Flower_Client_Manager.log', level=logging.DEBUG)
    if len(sys.argv) == 5:
        file_name = sys.argv[1]
        out_folder = sys.argv[2]
        num_clients = int(sys.argv[3])
        just_IDs = bool(sys.arv[4])
        logging.info("Starting Client Manager")
        Client_Manager(file_name, out_folder, num_clients, just_IDs=just_IDs)
    else:
        logging.warning("Invalid Parameters!\nNeccessary Parameters are: Input file: String, Output folder: String, number of clients to generate: Number, just_IDs: Bool ")
        exit()