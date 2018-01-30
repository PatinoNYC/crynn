import datetime
import time
import csv
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Import data
data = pd.read_csv('data/consolidated_data.csv')

# Drop date variable
dates = np.array ( data['Unix Date'].copy() )
dates_np = np.asarray( dates, dtype='datetime64[s]')
dates_pd = pd.to_datetime( dates_np )
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
data_train = data[train_start:train_end]
data_test  = data[test_start :test_end ]

print "Test data ends at " + str(train_end)

# Build X and y
X_train = data_train[ cols[1:] ]
y_train = data_train[ y_data ]
X_test  = data_test[ cols[1:] ]
y_test  = data_test[ y_data ]

# Number of stocks in training data
n_features = X_train.shape[1]

# Time params
n_periods = 10 #periods of input
pred_range = 5 #periods of output

norm_cols = [coin+metric for coin in ['BTC ', 'ETH ', 'LTC ', 'XRP '] for metric in ['Px','Volume']]

np_y_train = np.array ( y_train )
y_train_outputs = []
for i in range( n_periods, len( np_y_train ) - pred_range ):
        this_y_range = np_y_train[i:i+pred_range]/np_y_train[i-n_periods] - 1
        y_train_outputs.append( this_y_range )

y_train_outputs = np.array( y_train_outputs )
y_train_outputs = y_train_outputs[0:500] #HAX
train_dps = y_train_outputs.shape[0]

X_train_batches = []
for i in range ( train_dps ):
    if i == 500: break #HAX
    temp_set = X_train[i:(i+n_periods)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    temp_set = temp_set.fillna(0)
    temp_set = temp_set.replace([np.inf, -np.inf], 99)
    X_train_batches.append(temp_set)

np_y_test = np.array ( y_test )
y_test_outputs = []
for i in range( n_periods, len( np_y_test ) - pred_range ):
        this_y_range = np_y_test[i:i+pred_range]/np_y_test[i-n_periods] - 1
        y_test_outputs.append( this_y_range )

y_test_outputs = np.array( y_test_outputs )
test_dps = y_test_outputs.shape[0]

X_test_batches = []
for i in range ( test_dps ):
    temp_set = X_test[i:(i+n_periods)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    temp_set = temp_set.fillna(0)
    temp_set = temp_set.replace([np.inf, -np.inf], 99)
    X_test_batches.append(temp_set)


X_train_batches = [np.array(X_train_batches) for X_train_batches in X_train_batches]
X_train_batches = np.array(X_train_batches)

X_test_batches = [np.array(X_test_batch) for X_test_batch in X_test_batches]
X_test_batches = np.array(X_test_batches)

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
model = build_model(X_train_batches, output_size=pred_range, neurons = 20)

history = model.fit(X_train_batches, y_train_outputs, epochs=50, batch_size=1, verbose=2, shuffle=True)

# we may use dates on the X-axis.  we may not!
datetime_list = dates_str.tolist()

# plot error over epochs
fig, ax1 = plt.subplots(1,1)

ax1.plot(history.epoch, history.history['loss'])
ax1.set_title('Training Error')

if model.loss == 'mae':
    ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
# just in case you decided to change the model loss calculation
else:
    ax1.set_ylabel('Model Loss',fontsize=12)
ax1.set_xlabel('# Epochs',fontsize=12)
plt.show()

x = 1+model.predict( X_test_batches )[::-pred_range]
rounder = int( np.ceil( test_dps/float(pred_range)))
y = np_y_test[n_periods:n_periods+test_dps][::pred_range].reshape( rounder,1 )
predictions = x * y

#plotting Act vs Pred from training data
fig, ax1 = plt.subplots(1,1)

ax1.set_xticks( dates )
ax1.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,5,9]])

ax1.plot(dates_pd[n_periods:n_periods+train_dps], y_train[n_periods:n_periods+train_dps], label='Actual')

pred_colors = ["#FF69B4", "#5D6D7E", "#F4D03F","#A569BD","#45B39D"]

for i, (pred) in enumerate(zip ( predictions ) ) :
    ax1.plot( dates_pd[i*pred_range:i*pred_range+pred_range], pred[0], color=pred_colors[i%5] )

ax1.set_title('Training Set: Single Timepoint Prediction')
ax1.set_ylabel(y_data, fontsize=12)
ax1.legend(bbox_to_anchor=(0.15, 1), loc=2, borderaxespad=0., prop={'size': 14})

plt.show()
