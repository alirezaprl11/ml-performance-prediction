# Predicting the Computational Cost of Deep Learning Models
## Updated version for Tensorflow 2.x - Required Libraries 
pandas                   2.2.2

numpy                    1.26.4

tensorflow               2.17.0 (Use pip install tensorflow[and-cuda] for TF GPU-Enabled)

matplotlib               3.9.1

scikit-learn             1.5.1


## Look through the following Notebooks to run the prediction model
[Notebook 1](https://github.com/alirezaprl11/ml-performance-prediction/blob/master/prediction_model/notebooks/model_compTime_V2.ipynb): Small model to predict compute time from parameters of dense/convolutional layer 
## performance-prediction
Code related to the paper **Predicting the Computational Cost of Deep Learning Models**. This code allows to train a machine learning model that can predict the execution time for commonly used layers within deep neural networks - and, by combining these, for the full network.

A python package that utilises this model for inference can be found at https://github.com/CDECatapult/mlpredict.

This work is intended as starting point for an open source machine learning tool, capable of accurately predicting the time that is required to train any neural network on any given hardware. As such, it is easy for everyone to add additional hardware, model layers, or input features, or optimise the prediction model itself.

The folder *benchmark* contains code for benchmarking deep neural networks as well as single layers within these.

The folder *prediction_model* contains code to generate training data for the model described in the above paper, a data preparation pipeline, and the model training procedures. This folder also contains the training data and the existing tensorflow models.



