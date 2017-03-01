# cs221_project

This repository contains the Tensorflow codebase of our implementation to tackle the task Facial Expression Recognition and this is the project we have done for Stanford course CS221: Artificial Intelligence. We train MLP (Multiple-layer Perceptron) and CNN (Convolutional Neural Network) models to tackle the classification task. We use the training data from [Kaggle Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and our best model achieves 69.0% of testing accuracy, approaching the state of the art method as stated in the [Kaggle Leadboard](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/leaderboard).

## Detailed Project Report:

http://www.cs.stanford.edu/~kaichun/resume/cs221_project_report.pdf

## Data Download and Pre-processing

* All data can be downloaded from the official Kaggle challenge webpage: [Kaggle Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

* We convert the data into HDF5 file format by running the script 
            
            python write_hdf5.py
            
* We use the following script to compute the mean and scale of the training images in order to perform whitening during training and testing.
 
            python compute_mean_scale.py
            
## Code Organization

* All the codes are in the `exp` folder.
* `exp/cnn` contains the code for our CNN experiments
* `exp/fcn` contains the code for our MLP experiments
* The other Python scripts under `exp` folder are used to draw the training curves, confusion matrix and other analysis and visualization that are presented in the final report.
* Insides each of `exp/cnn` and `exp/mlp` folders, there are three main Python scripts: `train.py`, `model.py`, `eval.py`. Different version of the three files exist in both folders. They are different variants of training scripts and network architectures that we experimented.
* To train a model, you should run `train.py` like the following
            
            python train.py [python_model_file_prefix] --gpu [gpu_id] --batch [batch_size] 
            --epoch [training_epochs] --describe [other_descriptive_comments_for_this experiments] --wd [weight_decay]

Run the following command to get the detailed usage for `train.py`.

            python train.py -h

* Network architecture are defined in `model.py`.
* After training, we use `eval.py` script to test the model on testing split by running the following command.

            python eval.py [python_model_file_prefix] [path_to_pretrained_model_checkpoint] [output_file] --gpu [gpu_id] --batch [batch_size]

`[output_file]` stores the logs of this evaluation.

## Contact
Webpage: http://www.cs.stanford.edu/~kaichun/ 

E-mail: kaichun [at] cs.stanford.edu
