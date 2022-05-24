# LSTM-UNet

The code in this repository is suplamentary to our paper "Dual-Task ConvLSTM-UNet for Instance Segmentation of Weakly Annotated Microscopy Videos" published in IEEE Transactions on Medical Imaging (Early Access).
If this code is used please cite the paper:

@ARTICLE{9717246,  author={Arbelle, Assaf and Cohen, Shaked and Raviv, Tammy Riklin},  journal={IEEE Transactions on Medical Imaging},   title={Dual-Task ConvLSTM-UNet for Instance Segmentation of Weakly Annotated Microscopy Videos},   year={2022},  volume={},  number={},  pages={1-1},  doi={10.1109/TMI.2022.3152927}}

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites


This project is writen in Python 3 and makes use of tensorflow 2.0.0a0. 
Please see the requierments.txt file for all prerequisits. 

### Installing

In order to get the code, either clone the project, or download a zip from GitHub:
```
git clone https://gitlab.com/shaked0/lstmUnet.git
```

Install all python requierments

```
pip3 install -r <PATH TO REPOSITORY>/requirements.txt 
```

This should do it!
### Data
#### Get The Data
The training script was tailored for the Cell Tracking Benchmarck
If you do not have the training data and wish to train on the challenge data please contact the organizers through the website: www.celltrackingchallenge.net
Once you have the data, untar the file metadata_file.tar.gz into the direcroty of the training data: 

```
cd <PATH TO CELLTRACKINGCHALLENGE DATA>/Training
tar -xzvf  <PATH TO REPOSITORY>/metadata_files.tar.gz
```
#### Create Metadata File 
Create your using dataset metadata files using the create_sequence_metadata.py script.
Make sure that metadata_01.pickle and metadata_02.pickle are located in each dataset directory (Only of 2D datasets)

## Training
### Modify Parameters

Open the Params.py file and change the paths for ROOT_DATA_DIR and ROOT_SAVE_DIR. 
ROOT_DATA_DIR should point to the directory of the cell tracking challenge training data: <PATH TO CELLTRACKINGCHALLENGE DATA>/Training and ROOT_SAVE_DIR should point to whichever directory you would like to save the checkpoints and tensorboard logs.

  

### Run Training Script:
In order to set the parameters for training you could either change the parameters if Params.py file under class CTCParams
or input them through command line.
You are encourged to go over the parameters in CTCParams to see the avialable options
The training script is train2D.py

```
python3 train2D.py
```
### Training on Private Data:
Since there are many formats for data and many was to store annotations, we could note come up with a generic data reader.
So if one would like to train on private data we recommend one of the following:
1. Save the data in the format of the cell tracking challenge and create the corresponding metadata_<sequenceNumber>.pickle file. 
2. Write you own Data reader with similar api to ours. See the data reader in DataHandling.py

## Inference 
### Modify Parameters

Open the Params.py file and change the paths for ROOT_TEST_DATA_DIR. 
ROOT_TEST_DATA_DIR should point to the directory of the cell tracking challenge training data: <PATH TO CELLTRACKINGCHALLENGE DATA>/Test.

### Download Pre-trained models for Cell Segmentation Benchmark
If you would like to run the pretrained models for the cell segmentation benchmark datasets, you could download the models from:



### Run Inference Script:
In order to set the parameters for training you could either change the parameters if Params.py file under class CTCInferenceParams
or input them through command line.
You are encourged to go over the parameters in CTCInferenceParams to see the avialable options
The training script is train2D.py

```
python3 Inference2D.py
```

## Authors

Assaf Arbelle (arbellea@post.bgu.ac.il), Shaked Cohen (shaked0@post.bgu.ac.il) and Tammy Riklin Raviv (rrtammy@ee.bgu.ac.il)
Ben Gurion University of the Negev
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
