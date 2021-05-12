# Exploring_FL
First Steps in Federated Learning with [Tensorflow Federated](https://www.tensorflow.org/federated) and [Flower](https://flower.dev/).

The dataset is from the paper [App Usage Behavior Modeling and Prediction](http://fi.ee.tsinghua.edu.cn/appusage/), containing four files:
- App_usage_trace
- App2Category
- base_poi
- Categorys

Used 3 of the total 21 possible features:
- Timestamp
- Basestation Id
- Traffic in bytes
- 
# Modified Datasets
TODO Description

# Flower
1. With respect to finite resource limitations, Flower is trained with a choice of clients. These are chosen as follows:
- original dataset (app_usage_trace.txt):   N random Client_IDs
- top_apps_non_iid.csv :                    N random client_IDs
- top_apps_iid.csv :                        generates N clients over all entries of the dataset. 
2. To create this subsets of Clients, run [Flower_Client_Manager.py]() with these parameters:
  * dataset name e.g. data.txt, data.csv
  * output folder
  * numer of clients to sample/create

3. To run a Federated Training with Flower run [run.bat]() with these parameters:
  * number of clients: equal to the output folder from step 2
  * folder containing the id_file created in step 2


# TFF
