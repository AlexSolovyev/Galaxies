### load models and functions
from train_and_test_functions import *
#### Import libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from random import sample
import os
import pandas as pd
import sklearn.preprocessing as preprocessing
plt.switch_backend('agg')
tfd = tf.contrib.distributions

### load data

Data = pd.read_csv("../Data/DATA.csv")
Mask = Data.notnull()

for col in list(Data.columns):
    Data[col] = Data[col].fillna(Data[col].mean())

Data = Data.as_matrix()
missing_mask = np.array(Mask).astype(np.float64)


### data preprocess
max_Data, min_Data = 1, 0
Data_std = (Data - Data.min(axis=0)) / (Data.max(axis=0) - Data.min(axis=0))
Data = Data_std * (max_Data - min_Data) + min_Data

Data_train = Data_test = Data
mask_train = missing_mask

print('number of 0s (missing):', np.sum(1-mask_train))
print('number of 1s (observed) ', np.sum(mask_train))

### Train the model and save the trained model.
vae = train_p_vae(Data_train, mask_train, 3000, 10, 100, 0.5, 20, -1)

tf.reset_default_graph()
Data_ground_truth = Data_test
#mask_obs = np.array([bernoulli.rvs(1 - 0.3, size=Data_ground_truth.shape[1]*Data_ground_truth.shape[0])]) # manually create missing data on test set
#mask_obs = mask_obs.reshape(Data_ground_truth.shape)
mask_obs= missing_mask
Data_observed = Data_ground_truth*mask_obs

mask_target = 1-mask_obs # During test time, we use 1 to indicate missingness for imputing target.
# This line below is optional. Turn on this line means that we use the new comming testset to continue update the imputing model. Turn off this linea means that we only use the pre-trained model to impute without futher updating the model.
#vae = train_p_vae(Data_ground_truth, mask_obs, 3000, 10, 100, 0, 20, -1)
tf.reset_default_graph()
# Note that by default, the model calculate RMSE averaged over different imputing samples from partial vae.
RMSE,X_fill_mean_eddi = impute_p_vae(Data_observed,mask_obs,Data_ground_truth,mask_target,10,100,20,50)
np.save('X_fill_mean_eddi.npy', np.array(X_fill_mean_eddi)) 
# Alternatively, you can also first compute the mean of partial vae imputing samples, then calculated RMSE.
Diff_eddi=X_fill_mean_eddi*mask_target - Data*mask_target
print('test impute RMSE eddi (estimation 2):', np.sqrt(np.sum(Diff_eddi**2)/np.sum(mask_target)))
