# cs221_project

This repository contains the Tensorflow codebase of our implementation to tackle the task Facial Expression Recognition and this is the project we have done for Stanford course CS221: Artificial Intelligence. We train MLP (Multiple-layer Perceptron) and CNN (Convolutional Neural Network) models to tackle the classification task. We use the training data from [Kaggle Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and our best model achieves 69.0% of testing accuracy, approaching the state of the art method as stated in the [Kaggle Leadboard](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/leaderboard).

## Data Download and Pre-processing

* All data can be downloaded from the official Kaggle challenge webpage: [Kaggle Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

* We convert the data into HDF5 file format by running the script 
            
            write_hdf5.py
            
* We use the following script to compute the mean and scale of the training images in order to perform whitening during training and testing.
 
            compute_mean_scale.py
            
## Code Organization

* All the codes are in the `exp` folder.
* `exp/cnn` contains the code for our CNN experiments
* `exp/fcn` contains the code for our MLP experiments
* The other python scripts under `exp` folder are used to draw the training curves, confusion matrix and other analysis and visualization that are presented in the final report.
