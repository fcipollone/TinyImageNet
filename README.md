# TinyImageNet Competition

### A Question Answering Model for the Stanford TinyImageNet Competition
Welcome to the Stanford CS231n Project of Tyler Romero, Frank Cipollone, and Zach Barnes

The project has several dependencies that have to be satisfied before running the code. You can install them using your preferred method -- we list here the names of the packages using `pip`.

## Requirements
The code provided pressuposes a working installation of Python 3.6, as well as TensorFlow 1.0.

It should also install all needed dependnecies through
`pip install -r requirements.txt`.

## Data and Preprocessing

You can get started by downloading the datasets and doing dome basic preprocessing :

`$ code/get_started.sh`

Note that you will always want to run your code from the root directory of this repo. Not the code directory.
This ensures that any files created in the process don't pollute the code directoy.

This preprocessing step also creates Train, Val, and Test sets. The Test set answers are held secret by the competiton organizers.

## Training the Model

Once the data is downloaded and preprocessed, training can begin:

`$ python code/train.py`

You can use the flag `--help` to see potential arguements for training a model
While training, occasionally the model will give sample accuracies for both the Train and Val sets.

## Evaluating the Model

Evaluation is done on the Dev set.

First, generate answers for the test set questions:

`$ python code/ti_answer.py`

Then submit to the TinyImageNet competition.

## Acknowledgements






