# Exploring_FL
First Steps in Federated Learning with [Tensorflow Federated](https://www.tensorflow.org/federated) and [Flower](https://flower.dev/).
## Dataset 1
The dataset is from the paper [App Usage Behavior Modeling and Prediction](http://fi.ee.tsinghua.edu.cn/appusage/), containing four files:
- App_usage_trace
- App2Category
- base_poi
- Categorys

Used 3 of the total 21 possible features to predict the corresponding 'App_ID':
- Timestamp
- Basestation Id
- Traffic in bytes

#### Modified Datasets
Run [IID_Dataset_Generator](https://github.com/TouhKa/Exploring_FL/blob/main/data/IID_Dataset_Generator.py) to create an IID and a non-IID dataset of the 90% quantile of the most common apps. Hereafter called `top_apps_non_iid.csv` and `top_apps_iid.csv`.

## Dataset 2
The second dataset used is [CoSphere(Communication Context for Adaptive Mobile Applications) dataset](https://crawdad.org/novay/cosphere/20090501/) containing thenetwork traces on the personal mobile devices of 11 trial participants over a period of approximately one month in the February/March 2007 time frame.
Used 3 of all features to predict 'BSSID':
- Timestamp
- CopID
- LAC

## Dataset 3
Run [Dataset_Generator](https://github.com/TouhKa/Exploring_FL/blob/main/data/Dataset_Generator.ipynb) to create a dataset that generates nominal person data and infection status from a determinable normal distribution. Hereafter called `Infected.csv` and `Infected_shuffled.csv`.

# Flower
1. With respect to finite resource limitations, Flower is trained with a choice of clients. These are chosen as follows:
- original dataset (`app_usage_trace.txt`):   N random Client_IDs
- `top_apps_non_iid.csv`:                    N random client_IDs
- `top_apps_iid.csv`:                        generates N clients over all entries of the dataset. 
2. To create this subsets of Clients, run [Flower_Client_Manager.py](https://github.com/TouhKa/Exploring_FL/blob/main/Flower/Flower_Client_Manager.py) with these parameters:
  * `dataset name` e.g. data.txt, data.csv
  * `output folder`
  * `numer of clients` to sample/create

3. To run a Federated Training with Flower run [run.bat](https://github.com/TouhKa/Exploring_FL/blob/main/Flower/run.bat) with these parameters:
  * `number of clients`: equal to the output folder from step 2
  * `folder containing the id_file` created in step 2  <break>
  
 **Note**: If you want to use a lots of clients remove **/k** from [run.bat](https://github.com/TouhKa/Exploring_FL/blob/main/Flower/run.bat) <break> 
 
 **Note**: Check your path of the python.exe and adjust it if necessary

# TFF
 
Comparison of TFF and normal TF for dataset 1, 2 and 3. Please update the name and number of label classes for the desired record according to the [config](https://github.com/TouhKa/Exploring_FL/blob/main/config/config.json)
