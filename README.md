# TinyImageNet Competition

### A Question Answering Model for the Stanford TinyImageNet Competition
Welcome to the Stanford CS231n Project of Tyler Romero, Frank Cipollone, and Zach Barnes

The project has several dependencies that have to be satisfied before running the code. You can install them using your preferred method -- we list here the names of the packages using `pip`.

## Requirements
The code provided pressuposes a working installation of Python 2.7, as well as TensorFlow 0.12.1.

It should also install all needed dependnecies through
`pip install -r requirements.txt`.

## Data and Preprocessing

You can get started by downloading the datasets and doing dome basic preprocessing :

`$ code/get_started.sh`

`$ python code/qa_data.py`

Note that you will always want to run your code from the root directory of this repo. Not the code directory.
This ensures that any files created in the process don't pollute the code directoy.

This preprocessing step also creates Train, Val, and Dev sets. The Test set is held secret by SQUAD.

## Training the Model

Once the data is downloaded and preprocessed, training can begin:

`$ python code/train.py`

You can use the flag `--help` to see potential arguements for training a model
While training, occasionally the model will give a sample F1 and EM score based on the Val set.

## Evaluating the Model

Evaluation is done on the Dev set.

First, generate answers for the dev set questions:

`$ python code/qa_answer.py`

Then, compare the model's answers to the ground thruth:

`$ python code/evaluate.py`

This will spit out a final F1 and EM score.

## Acknowledgements

Our baseline model is based on [Machine Comprehension Using Match-LSTM and Answer Pointer](https://arxiv.org/abs/1608.07905)




