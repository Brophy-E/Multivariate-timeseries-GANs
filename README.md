# Multivariate-timeseries-GANs
This repository is an artifact for the paper under review "Multivariate Generative Adversarial Networks and their Loss Functions for Synthesis of Multichannel ECGs" submitted to IEEE Pattern Recognition and Machine Intelligence 2020.  The code is available as Jupyter Notebooks `.ipynb`, `.py` files will be added shortly.

----------------
## Downloading the Data
Run through the Notebooks `Arr_Data_Prep.ipynb` and `Nsr_Data_Prep.ipynb` to download and preprocess the datasets

----------------
## Loading and Processing the Data

See above

 ---------------
## Training the model
`train.py` 

Trains the GAN model defined in `model.py` and saves each generator and discriminator after every training epoch as `.pt` files.

 ---------------

## Running Evaluation Metrics

The evaluation metrics are comptued for a batch of data from one of the generators of your choice. You can choose to generate a batch of data and use it for evaluation. Alternatively, we have provided a batch of generated data in `Data/generated_data_epoch_14_mbd_3.pt` which has been included in the directories for you to use.

`mmd_eval.py`

Computes MMD 3 Sample Test from `mmd.py` (taken from [eugenium's Github](https://github.com/eugenium/MMD)), returns a test statistic for whether the training data is closer to the generated data then a random noise sample.

    
`dtw_eval.py`

Calculates the dependent multivariate Dynamic Time Warping values between the training and generated data.

`membership_inference_attack.py`
 
 Computes a simple membership inference attack on the generated data from the model. It uses the euclidean distance for the multivariate signals. You can use the *DTW_d* as the distance in `mia_dtw.py`, this takes significantly longer to execute.
 
 
 ---------------

## Plotting Results

The training losses and generated results are plotted using the Notebook `plot_gen.ipynb`

