# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import flwr as fl
import tensorflow as tf
import json
from FLModel import FLModel

def read_config():
    config_file = "../config/config.json"
    config = None
    with open(config_file) as json_file:
        config = json.loads(json_file.read())
    return config

# def get_eval_fn(model):
#     """Return an evaluation function for server-side evaluation."""
#     # The `evaluate` function will be called after every round
#     def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
#         model.set_weights(weights)  # Update model with the latest parameters
#         loss, accuracy = model.evaluate(x_val, y_val)
#         return loss, accuracy
#     return evaluate

def main() -> None:
  config = read_config()
  model = FLModel(config["NUM_CLASSES"])
  server_optimizer_fn = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.1, nesterov=True)
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
  model.compile(server_optimizer_fn, loss, metrics = [metrics])

#Overwrite number of clients for fitting and evaluation
#TODO get_eval_fn(model)
  strategy = fl.server.strategy.FedAvg(
      min_fit_clients=10,
      min_eval_clients=10,
      min_available_clients=10
    #   eval_fn=get_eval_fn(model),
  )
  fl.server.start_server(
                          "localhost:8080",
                          config={"num_rounds": 10},
                          strategy=strategy
                        )

if __name__ == "__main__":
  main()