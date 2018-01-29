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

# Time Period (hours)
n_periods = 10

#construct our batches to input to the NN
#training data
y_train = np.array ( y_train )
# normalize
y_train_clipped =  y_train[n_periods:]/y_train[n_periods-1:-1]

# trainable Y data points 
trainable_dps = len ( y_train_clipped )

norm_cols = [coin+metric for coin in ['BTC ', 'ETH ', 'LTC ', 'XRP '] for metric in ['Px','Volume']]

X_train_batches = []
for i in range ( trainable_dps ):
    if ( i > 500 ): break #HAX
    temp_set = X_train[i:(i+n_periods)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    temp_set = temp_set.fillna(0)
    temp_set = temp_set.replace([np.inf, -np.inf], 99)
    X_train_batches.append(temp_set)

# test/validation data sets
y_test = np.array ( y_test )
y_test_clipped = y_test[n_periods:]/y_test[n_periods-1:-1]
test_dps = len ( y_test_clipped )

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
model = build_model(X_train_batches, output_size=1, neurons = 10)

# train model on data
# note: history contains information on the training error per epoch

y_train_clipped2 = y_train_clipped[0:501]
history = model.fit(X_train_batches, y_train_clipped2, epochs=50, batch_size=1, verbose=2, shuffle=True)

# plot error over epoch
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


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

fig, ax1 = plt.subplots(1,1)
ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,5,9]])
ax1.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,5,9]])

ax1.plot(data[n_periods:train_end]['Date'][n_periods:].astype(datetime.datetime),
         data[y_data][n_periods:train_end], label='Actual')
ax1.plot(data[n_periods:train_end]['Date'][n_periods:].astype(datetime.datetime),

         np.transpose( model.predict( X_train_batches ) ) + 1 * 

         ((np.transpose(model.predict())+1) * training_set['eth_Close'].values[:-window_len])[0], 
         label='Predicted')
ax1.set_title('Training Set: Single Timepoint Prediction')
ax1.set_ylabel('Ethereum Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.15, 1), loc=2, borderaxespad=0., prop={'size': 14})
ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(eth_model.predict(LSTM_training_inputs))+1)-            (training_set['eth_Close'].values[window_len:])/(training_set['eth_Close'].values[:-window_len]))), 
             xy=(0.75, 0.9),  xycoords='axes fraction',
            xytext=(0.75, 0.9), textcoords='axes fraction')
