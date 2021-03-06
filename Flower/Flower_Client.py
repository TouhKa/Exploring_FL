import sys
sys.path.append("../")
import flwr as fl
import tensorflow as tf
import json
from  model.FLModel import FLModel
import numpy as np
import pandas as pd
import logging
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define Flower client
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, batch_size, epochs):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.steps = epochs
        self.batch_size = batch_size
        self.bn_weights = None
        self.start = 2
        self.end = 6

    def get_parameters(self):
        weights = self.model.get_weights()
        if len(weights) > 7:
            dense_weights = [weights[0], weights[1], weights[6], weights[7]]
            return dense_weights
        return weights

    def set_model_weights(self, parameters):
        new_weights = []
        for weight in parameters[:self.start]:
            new_weights.append(weight)
        for weight in self.bn_weights:
            new_weights.append(weight)
        for weight in parameters[self.start:]:
            new_weights.append(weight)
        return new_weights

    def fit(self, parameters, config):
        if len(parameters) == 4 :
            self.model.set_weights(self.set_model_weights(parameters))
        else:
            self.model.set_weights(parameters)
        with tf.device('/gpu:0'):
            history = self.model.fit(
                self.X_train,
                self.y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=0.2
                )

        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["sparse_categorical_accuracy"][0],
            "top_10_accuracy":  history.history["sparse_top_k_categorical_accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_sparse_categorical_accuracy"][0],
            "val_top_5_accuracy":  history.history["val_sparse_top_k_categorical_accuracy"][0],
        }
        temp_weights = self.model.get_weights()
        self.bn_weights = temp_weights[self.start:self.end]
        dense_weights =  [temp_weights[0], temp_weights[1], temp_weights[6], temp_weights[7]]
        return dense_weights, len(self.X_train), results

    def evaluate(self, parameters, config):
        self.model.set_weights(self.set_model_weights(parameters))
        with tf.device('/gpu:0'):
            loss, accuracy, top_k_acc= self.model.evaluate(
                                                self.X_test,
                                                self.y_test,
                                                batch_size=self.batch_size,
                                                steps=self.steps)
        return loss, len(self.X_test), {"accuracy": accuracy, "top-10-acc": top_k_acc}

def read_data(file_path):

    df = pd.read_csv(file_path)
    data = df.to_numpy(dtype=float)
    #drop index column
    data = np.delete(data, 0, 1)
    return data

def create_dataset(x, y):
    return tf.data.Dataset.from_tensor_slices((x, x)) \
                                                .shuffle(config["SHUFFLE_BUFFER"]) \
                                                .batch( config["BATCH_SIZE"],
                                                        drop_remainder=True
                                                        )

def read_config():
    config_file = "../config/config.json"
    config = None
    with open(config_file) as json_file:
        config = json.loads(json_file.read())
    return config

if __name__ == "__main__":

    try:
        client_id = sys.argv[1]
        folder = "Flower/" + sys.argv[2] + "/"
        port = sys.argv[3]
        logging.info("\nStarting Client {}\n".format(client_id))
    except:
        logging.warning("No valid arguments for client_id, folder, port")
        exit()

    config = read_config()
    X_train = read_data("../" \
                + config["DATA_DIR"] \
                + folder
                + "X_train_" \
                +  client_id \
                + ".csv")
    y_train = read_data("../" \
                + config["DATA_DIR"] \
                + folder
                + "y_train_" \
                +  client_id \
                + ".csv")

    X_test = read_data("../" \
                + config["DATA_DIR"] \
                + folder
                + "X_test_" \
                + client_id \
                + ".csv")
    y_test = read_data("../" \
                + config["DATA_DIR"] \
                + folder
                + "y_test_" \
                +  client_id \
                + ".csv")
    model = FLModel(config["NUM_CLASSES"])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.5, momentum=0.5, nesterov=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [ tf.keras.metrics.SparseCategoricalAccuracy(),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10)]
    model.compile(optimizer, loss, metrics = [metrics])

    client = fl.client.start_numpy_client("localhost:" + port,
                                        client=FLClient(model,
                                                        X_train,
                                                        y_train,
                                                        X_test,
                                                        y_test,
                                                        config["BATCH_SIZE"],
                                                        config["EPOCHS"]
                                                        )
                                        )