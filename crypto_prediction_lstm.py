# Import
import csv
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Import data
data = pd.read_csv('data/consolidated_data.csv')

# Drop date variable
data = data.drop(['Unix Date'], 1)

y_data = "BTC Px"
print "Using " + y_data + " as the dataset we want to predict"

#how many hours into the future will we predict
y_offset_predict = 1

data["y_axis"] = data[y_data]
data["y_axis"] = data["y_axis"].shift(-y_offset_predict)

# Dimensions of dataset
n = data.shape[0] #should be same as n above
p = data.shape[1]

print "Copying forward data for our offset.  Keep this in mind, it means the end of our data set is dummy data"
for x in range (0, y_offset_predict):
#    print str(n - x - 1) + " becomes " + str( n - y_offset_predict - 1 )
    data["y_axis"].at[n - x - 1] =  data["y_axis"].at[ n - y_offset_predict -1 ]    

#from here on, the code assumes the first pandas column is our y_axis.  Let's make that so:
cols = list(data)
cols = cols[-1:] + cols[:-1]
data = data[cols]

# Training and test data
train_start = 0
train_end = int(np.floor(0.80*n))
test_start = train_end + 1
test_end = n
data_train = data[train_start:train_end].copy()
data_test  = data[test_start :test_end ].copy()
test_dps = test_end - test_start

print "Test data ends at " + str(train_end)

# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# Number of stocks in training data
n_features = X_train.shape[1]

# Time Period (hours)
n_periods = 24

#construct our batches to input to the NN
#training data
y_train_clipped = y_train[n_periods:]
a = y_train_clipped[np.newaxis]
y_batches = np.transpose( a )

trainable_dps = y_train_clipped.shape[0]

norm_cols = [coin+metric for coin in ['BTC ', 'ETH ', 'LTC ', 'XRP '] for metric in ['Px','Volume']]

X_batches = []
for i in range ( trainable_dps ):
    temp_set = X_train[i:(i+n_periods)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    X_batches.append(temp_set)













    
# test/validation data sets
y_test_clipped = y_test[n_periods:]
a = y_test_clipped[np.newaxis]
y_test_batch = np.transpose( a )

test_dps = y_test_clipped.shape[0]
X_test_batch = np.empty( [ test_dps, n_periods * n_features] )
for j in range( test_dps ):
    X_test_batch[j] = X_test[j:j+n_periods,:].reshape( 1, n_periods * n_features )

# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

# random seed for reproducibility
np.random.seed(202)

# initialise model architecture
model = build_model(X_batches, output_size=1, neurons = 20)

# train model on data
# note: history contains information on the training error per epoch
history = model.fit(X_batches, y_batches, epochs=50, batch_size=1, verbose=2, shuffle=True)
