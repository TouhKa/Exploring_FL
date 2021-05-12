import numpy as np
import pandas as pd
import logging
class Reader():
  def __init__(self, file_path:str, file_name:str):
    self.__data = None
    self.__data_dir = file_path
    self.__file_name = file_name
    self.__num_of_user:int = 1
    self.__num_of_classes = 0
    seperator = file_name.index(".")
    file_type = file_name[(seperator+1) : len(file_name)]

    if file_type == "txt":
      logging.info("Opening txt file")
      self.__data = self.__read_traces()

    elif file_type == "csv":
      logging.info("csv file")
      self.__data = self.__read_csv()

    else:
      logging.info("No valid file")

    if (self.__data is not None):
      self.__features = self.__data.shape[1]
      self.__num_of_user = (np.amax(self.__data[:, 0]) + 1)

  def get_data(self):
    return self.__data

  def get_number_of_users(self):
    return self.__num_of_user

  def get_number_of_classes(self):
    return self.__num_of_classes

  def set_features(self, features):
    self.__features = features

  def set_number_of_users(self, nou):
    self.__num_of_user = nou

  def set_number_of_classes(self, noc):
    self.__num_of_classes = noc


  def __read_traces(self):
    x =[]

    try:
      with open(self.__data_dir + self.__file_name) as fin:
          for idx, line in enumerate(fin):
              splitLine = line.rstrip().split()
              if idx != 0:
                  splitLine = np.array([int(i) for i in splitLine])
                  x.append(splitLine)
    except:
      logging.info("Non existing txt file")
      return

    x = np.array(x, dtype="float")
    return x

  def __read_csv(self):
    try:
      data = pd.read_csv((self.__data_dir + self.__file_name), encoding="utf8", delimiter=",")
      logging.debug(data.info)
      data = data.to_numpy(dtype=float)
      #drop index column
      data = np.delete(data, 0, 1 )

    except:
      logging.info("Non existing csv file")
      return
    return data

