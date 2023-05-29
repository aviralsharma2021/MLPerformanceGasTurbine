Introduction

In this blog post, we will explore the journey of developing an ML-based performance deck for off-design steady-state performance prediction using altitude, Mach number, and HPC (High-Pressure Compressor) spool speed as input parameters. We will delve into the key steps involved in this process, including the generation of synthetic data, cleaning the data, model development, and the deployment of a REST API using Django REST Framework. Furthermore, we will see how Dockerization streamlines the deployment process, ensuring easy accessibility and eliminating the need for users to install additional dependencies.

By leveraging the power of ML algorithms, we aim to create a model that can accurately predict crucial gas turbine performance metrics such as thrust, SFC (Specific Fuel Consumption), exhaust gas temperature, and more. The REST API built using Django REST Framework will enable users to conveniently interact with the ML model and obtain predictions in a simple and standardized manner. The Dockerization of the microservice provides a self-contained and portable environment, eliminating compatibility issues and simplifying deployment on different systems.

Synthetic Data Generation

To train and validate the ML model, the first step involved generating synthetic data. Synthetic data allows us to create a diverse dataset with known ground truth values, which is crucial for training and evaluating the accuracy of the model's predictions.

Process

Sobol Sampling (Every 25th point shown)

Sobol Sampling (All 8142 points shown)



Defining Input Ranges: The input parameters for the model, namely Altitude, Mach number, and HPC spool speed, were defined within specific ranges. Altitude ranged from 0 to 6000 meters, Mach number from 0 to 0.7, and HPC spool speed from 0.6 to 1.

SOBOL Sampling: To ensure a representative distribution of data points across the design space, SOBOL sampling was employed. SOBOL sampling is a quasi-random method that generates a sequence of points with better coverage than traditional random sampling methods. It helps ensure that the dataset is well-distributed and avoids clustering in certain regions of the design space. 

# importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc

# sampling the design space using scipy Quasi-Monte Carlo SOBOL Scheme
# setting dimensions to 3
sampler = qmc.Sobol(d=3, scramble=False, seed=42)
sample = sampler.random_base2(m=13)

# plotting the sample space, every 25 points
ax = plt.axes(projection='3d')
ar = np.arange(0,8142,25)
ax.scatter3D(sample[ar,0], sample[ar,1], sample[ar, 2])

# defining lower and upper bounds for 3 inputs
l_bounds = [0, 0, 0.6]
u_bounds = [6000, 0.7, 1]

# scaling the sample to bounds
data = qmc.scale(sample, l_bounds, u_bounds)

# converting to pandas dataframe
df = pd.DataFrame(data)
df.columns = ["alt", "XM", "ZXN_HPC"]

# writing to file
file = open("inputs.txt", "w")
for i in range(0,df.shape[0]):
    L = ["alt = " + str(df["alt"][i]) + "\n", "XM = " + str(df["XM"][i]) + "\n", "ZXN_HPC = " + str(df["ZXN_HPC"][i]) + "\n"]

    file.write("[Single Data]\n")
    file.writelines(L)
    file.write("[Calculate] "+str(i+1)  + "\n")
file.close()

Data Generation with GasTurb Software: The generated list of input data points was utilized with the GasTurb software. GasTurb is a powerful tool used in the gas turbine industry for performance analysis. By providing the input parameters (Altitude, Mach number, and HPC spool speed), GasTurb simulated the gas turbine performance and provided the corresponding output values.

Output Variables:

The GasTurb software provided the following output variables for each data point:

Thrust

Specific Fuel Consumption (SFC)

Exhaust Gas Temperature

Inlet Total Temperature

Compressor Exit Total Temperature

Inlet Total Pressure

Compressor Exit Total Pressure

Fuel Flow Rate

Data Size and Quantity

To ensure a sufficiently large and diverse dataset for training and evaluation, a total of 8142 data points were generated. This size allows the ML model to capture the relationships and patterns between the input parameters and the corresponding gas turbine performance metrics effectively.

Data Cleaning and Model Development

Once the synthetic data was generated, the next step involved cleaning the data and developing a Deep and Wide Neural Network model using the Keras Subclassing API. This section will describe how the data was cleaned and provide an overview of the model architecture and training process.

Data Preparation

To ensure the quality and reliability of the data, a data cleaning step was performed. In this step, any data points with negative thrust values were removed from the dataset. Negative thrust values are physically implausible and may indicate erroneous or invalid data. By eliminating such data points, we maintained the integrity of the dataset.

# subsetting data where Thrust (FN) < 0
data3 = data2[data2["FN"]<0]
removal = list(data3.index)       # converting indices of all -ve Thrust datapoints to list
gtdata = data2.drop(removal)      # dropping rows with indices in removal

# dividing features and targets 
X_gt = gtdata[["alt", "XM", "ZXN_HPC"]]
y_gt = gtdata.drop(["alt", "XM", "ZXN_HPC"], axis = 1)

from sklearn.model_selection import train_test_split

# splitting data into train set and test set, with test size 20%
xgt_train, xgt_test, ygt_train, ygt_test = train_test_split(X_gt, y_gt, test_size=0.2)

Model Architecture

The Deep and Wide Neural Network model was chosen for its ability to capture both low- and high-level interactions between features. The model consists of two main parts: the deep part and the wide part.

The deep part incorporates two hidden layers, which allows the model to learn complex nonlinear relationships between the input features and the output variables. Each hidden layer employs the Rectified Linear Unit (ReLU) activation function, a popular choice for deep neural networks. ReLU introduces nonlinearity and helps the model to learn more intricate patterns in the data.

The wide part, on the other hand, does not include any hidden layers. It directly connects the input features to the output layer, enabling the model to capture simple and linear relationships between the features.

Wide and Deep Model Architecture

Before reaching the output layer, the outputs from the deep and wide parts are concatenated to combine their respective strengths. This fusion of deep and wide components allows the model to leverage both the expressive power of deep learning and the memorization capability of shallow learning.

import tensorflow as tf

# defining class WideandDeepModel, inheriting from tf.keras.Model 
class WideandDeepModel(tf.keras.Model):
    def __init__(self, units=256, activation="relu", **kwargs):
        super().__init__(**kwargs)
        
        # normalizing inputs to be between 0 & 1
        self.norm_layer_wide = tf.keras.layers.Normalization()
        self.norm_layer_deep = tf.keras.layers.Normalization()
        
        # adding fully-connected layers
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        
        # output layer has 9 outputs
        self.outputs = tf.keras.layers.Dense(9)
        
    def call(self, inputs):
        input_ = inputs
        norm_wide = self.norm_layer_wide(input_)
        hidden1 = self.hidden1(norm_wide)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])
        outputs = self.outputs(concat)
        return outputs  

Model Training

To train the model, the Adam optimizer was used, which is well-suited for optimizing deep neural networks. The Root Mean Squared Error (RMSE) was chosen as the loss function, as it provides a measure of the model's prediction accuracy.

# instantiating WideandDeepModel
model = WideandDeepModel(256, activation="relu", name="model1")
model.call(xgt_train)

# compiling the model with Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])

# adapting the normalization layer
model.norm_layer_wide.adapt(xgt_train)

To monitor the training progress and performance, TensorBoard, a visualization tool in TensorFlow, was utilized. TensorBoard helps track various metrics, including loss and accuracy, enabling effective monitoring and analysis during the training process.

# setting log path for tensorboard
import os
from time import strftime

# joining path with current time leads to logs saved time wise in 'logs' folder
def get_run_logdir(root_logdir=os.path.join('..', 'logs')):
    return os.path.join(root_logdir, strftime("run_%Y_%m_%d_%H_%M_%S"))

run_logdir = get_run_logdir()
run_logdir

Additionally, early stopping was implemented to prevent overfitting and improve generalization. Early stopping stops the training process if the validation loss does not improve over a specified number of epochs.

tf.random.set_seed(20)

# setting callbacks for keras model training
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)
models_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("perf_deck_2", save_weights_only=True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir, profile_batch=(100,200))
callbacks = [early_stopping_cb, models_checkpoint_cb, tensorboard_cb]

Once the model training was completed, the trained model was saved in TensorFlow format, allowing for easy reusability and deployment in the future.

# training the model
# setting 10% as validation split
model.fit(xgt_train, ygt_train, epochs=2500, batch_size=16, validation_split=0.1, callbacks=callbacks)

# saving the model
model.save("final_model_2", save_format="tf")

Tensorboard could be run in a terminal as follows:

tensorboard --log_dir=$PATH_TO_LOGS_FOLDER

The learning rate was reduced after around 620 epochs.

RestAPI Development and Dockerization

In this step, a REST API was developed using Django REST Framework to facilitate the inference function. The API accepts input parameters, including altitude, Mach number, and HPC spool speed, in JSON format and returns the output in JSON format as well. Additionally, the microservice was wrapped in a Docker container to simplify the deployment process and handle the necessary dependencies.

RestAPI Development

To provide a user-friendly interface for the ML model, a REST API was developed using Django REST Framework. This framework allows for the quick and efficient creation of APIs with built-in functionality for handling requests and responses. The API was designed to receive input parameters in JSON format, containing altitude, Mach number, and HPC spool speed. It processes these inputs and provides the corresponding output metrics in JSON format. This approach makes it easier for users to interact with the model without needing to understand the underlying implementation details.

Dockerization

To ensure a seamless and consistent deployment experience, the microservice was encapsulated in a Docker container. Docker allows for the creation of lightweight, portable, and isolated environments that include all the necessary dependencies. By packaging the ML model and its dependencies within a Docker container, users are relieved of the burden of manually installing libraries such as TensorFlow, NumPy, and Pandas. This ensures a consistent and reliable execution environment across different systems.

Deployment with docker-compose

To simplify the deployment process and manage multiple containers, Docker Compose was utilized. Docker Compose is a tool that enables the orchestration of multi-container applications. It uses a YAML file, commonly named docker-compose.yml, to define the services and their configurations. By specifying the necessary configurations in the docker-compose.yml file, the microservice can be deployed on a development server with ease. This approach streamlines the deployment process and allows for scalability and management of the microservices.

API Usage

To access the API and obtain predictions for gas turbine performance metrics, users can send a POST request to the specified endpoint: 192.168.1.32:3030/v1/api/perf/. The input parameters, including altitude, Mach number, and HPC spool speed, should be provided in JSON format as part of the request body. The API will process the request and return the predicted performance metrics in JSON format as the response. This straightforward approach allows users to interact with the ML model and obtain predictions conveniently.

There are two ways to send a POST request to the API:

Method 1: Through Browser

Go to http://192.168.1.32:3030/v1/api/perf/

You will be met by the following screen. Enter the output in a JSON format as shown below.



Result:



Method 2: Though the cURL command

You can send a POST request using the following cURL command:

curl -d "alt=1000&mach=0.5&zxn=0.8" http://192.168.1.32:3030/v1/api/perf/



The github repository for this project is at . 

Summary

In this blog post, we embarked on a journey to revolutionize gas turbine performance prediction using machine

learning (ML) and RESTful API technologies. We began by generating synthetic data using SOBOL sampling and the GasTurb software to capture the relationship between altitude, Mach number, HPC spool speed, and gas turbine performance metrics. With 8000 data points in hand, we proceeded to clean the data, removing negative thrust values and ensuring data integrity.

Next, we developed a deep and wide neural network using the Keras Subclassing API. This architecture combined the power of deep layers for capturing complex nonlinear relationships and wide layers for incorporating simpler linear relationships. By compiling the model with the Adam optimizer, using ReLU activation functions for hidden layers, and RMSE as the loss function, we created a robust model for gas turbine performance prediction. The training process was carefully monitored using TensorBoard, and early stopping was employed to prevent overfitting. Once trained, the model was saved in TensorFlow format for future use.

To make the ML model easily accessible, we built a RESTful API using the Django REST Framework. This API accepts JSON-formatted inputs for altitude, Mach number, and HPC spool speed and returns the predicted gas turbine performance metrics in JSON format. We then containerized the microservice using Docker, ensuring seamless deployment and eliminating the need for users to install additional dependencies like TensorFlow, NumPy, or Pandas.

Finally, we deployed the Docker container on a development server using a docker-compose.yml file. With the REST API up and running, users can send a POST request with JSON input to 192.168.1.32:3030/v1/api/perf/ and receive the predicted performance metrics in JSON format.

By harnessing the power of ML and RESTful APIs, we have paved the way for accurate and efficient gas turbine performance prediction. This innovative approach simplifies access to predictions, enhances deployment flexibility, and empowers aerospace engineers to optimize gas turbine operations. With the steps outlined in this blog post, you can embark on your own journey to revolutionize performance prediction in the aerospace industry.
