# Exploring_FL
First Steps in Federated Learning with [Tensorflow Federated](https://www.tensorflow.org/federated) and [Flower](https://flower.dev/).
The main part was implemented with TFF. A first approach with Flower is included, see explanation below, but was not pursued further.

## TFF (Main part)
This contains a Jupyter notebook [see here]() to compare different setups. The main goal is to 

1. analyze the impact of non--IID--data (see dataset description 1 and 3) on certain metrics
2. analyze the impact of lossy compression with quantization on certain metrics. 

The first part is done by comparing naive FedAVG and custom TFF-FedBN.
The second part is implemented by quantization. Here, two different libraries are used because a custom TFF lifecycle does not support built-in unified quantization encoders. Therefore, the quantization and dequantization methods of tf.core are used.
The setup is as follows:

* Execute centralized learning.
* Run FedAVG without compression 
* Execute FedBN without compression
* Loop with different compression thresholds:
    * Run FedAVG with compression and a given threshold.
    * Run FedBN with compression and a specific threshold.

# Flower

**Note**: This setup runs all clients as a seperat process, which does not scale by default. Please consider strategies like virtual RAM or restricting Tensorflow RAM ressources 
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

### Dataset 1
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

### Dataset 2
The second dataset used is [CoSphere(Communication Context for Adaptive Mobile Applications) dataset](https://crawdad.org/novay/cosphere/20090501/) containing thenetwork traces on the personal mobile devices of 11 trial participants over a period of approximately one month in the February/March 2007 time frame.
Used 3 of all features to predict 'BSSID':
- Timestamp
- CopID
- LAC

### Dataset 3
Run [Dataset_Generator](https://github.com/TouhKa/Exploring_FL/blob/main/data/Dataset_Generator.ipynb) to create a dataset that generates nominal person data and infection status from a determinable normal distribution. Hereafter called `Infected.csv` and `Infected_shuffled.csv`.
