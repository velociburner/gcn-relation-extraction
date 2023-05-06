# Contextualized Graph Convolutional Networks for Entity Relation Extraction
Here is the code for my final project for COSI 137b Information Extraction. I chose to do a methods focused project,
which has an emphasis on implementation over experimentation. The model architecture is heavily based on
[this paper](https://aclanthology.org/D18-1244/) from EMNLP 2018.

## Instructions
Requires Python 3.9+

### Using iPython notebook

The easiest way to run the code is using the provided notebook `run.ipynb`. This can either be as a Jupyter notebook or
from Google Colab. Using Colab involves uploading the project folder to your Google Drive, in order to be able to run
the provided scripts. Follow the instructions in the notebook for installing the requirements and running experiments.

### Running locally

Create a virtual environment, then install the requirements:
```sh
pip install -r requirements.txt
```
See the [PyTorch website](https://pytorch.org/get-started/locally/) for more information on installing torch with cuda
for GPU training.

Then, run the main script with the following command:
```sh
python src/main.py
```
There are a number of command line arguments for training the model. Add the `--help` flag to see the different
options.
