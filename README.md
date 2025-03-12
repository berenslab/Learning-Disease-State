# Learning Disease State from Noisy Ordinal Disease Progression Labels
This repository contains the code for the MICCAI Submission "Learning Disease State from Noisy Ordinal Disease Progression Labels". 

# Installation

Set up a python environment with a python version `3.13`. Then, download the repository,
activate the environment and install all other dependencies with
```bash
cd Learning-Disease-State
pip install --editable . 
```

This installs the code in `src` as an editable package and all the dependencies in
[requirements.txt](requirements.txt).

# Organization of the repo
* [configs](./configs/): Configuration files for both mario and internal experiments.
* [src](./src/): Main source code to run the experiments.
* [train_mario.py](./src/train_mario.py): Training on the Mario challenge dataset. 
* [train_internal.py](./src/train_internal.py): Running the trained models on out-of-domain dataset.
* [loss.py](./src/loss.py): Contains the loss function.
* [dataset.py](./src/dataset.py): Contains mario dataset.

# Running the Model

## Mario Challenge Dataset
To train the model on the **Mario Challenge** dataset:  

1. Update the dataset path in the `train_mario.yaml` config file.  
2. Run the following command:  

    ```bash
    python src/train_mario.py
    ```

## Custom Dataset
To train the model on a **different dataset**:
1. Create your own pytorch dataset.
2. Update the dataset path and pretrained model path in the `train_internal.yaml` config file.  
3. Run the following command:  

    ```bash
    python src/train_internal.py
    ```

# Cite
If you find our code or paper useful, please consider citing this work:
```bibtex

``` 
