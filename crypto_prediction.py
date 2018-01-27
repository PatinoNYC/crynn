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

data = data[ ["y_axis", "BTC Px"] ] #HAX
    
# Make data a np.array
np_data = data.values

# Training and test data
train_start = 0
train_end = int(np.floor(0.80*n))
test_start = train_end + 1
test_end = n
data_train = np_data[np.arange(train_start, train_end), :] 
data_test = np_data[np.arange(test_start, test_end), :]
test_dps = test_end - test_start

print "Test data ends at " + str(train_end)

# Scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# Number of stocks in training data
n_features = X_train.shape[1]

# Neurons
n_periods = 24
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

# Session
net = tf.InteractiveSession()

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_periods*n_features])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Hidden weights
# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_periods*n_features, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output weights
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# RELU approach (generally better and more widely adopted
# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out) )

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
net.run(tf.global_variables_initializer())

#construct our batches to input to the NN
#training data
y_train_clipped = y_train[n_periods:]
a = y_train_clipped[np.newaxis]
y_batches = np.transpose( a )

trainable_dps = y_train_clipped.shape[0]
X_batches = np.empty( [ trainable_dps, n_periods * n_features] )
for i in range ( trainable_dps ):
    X_batches[i] = X_train[i:i+n_periods,:].reshape( 1, n_periods * n_features )

# test/validation data sets
y_test_clipped = y_test[n_periods:]
a = y_test_clipped[np.newaxis]
y_test_batch = np.transpose( a )

test_dps = y_test_clipped.shape[0]
X_test_batch = np.empty( [ test_dps, n_periods * n_features] )
for j in range( test_dps ):
    X_test_batch[j] = X_test[j:j+n_periods,:].reshape( 1, n_periods * n_features )

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_autoscale_on(True)

axes = plt.gca()
axes.set_ylim([-5,15])

line1, = ax1.plot(y_test_clipped)
line2, = ax1.plot(y_test_clipped * 0.5)
plt.show()

# Fit neural net
mse_train = []
mse_test = []

# Run
epochs = 100
n_batches = 256
    
for e in range(epochs):
    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange( trainable_dps ) )
    X_batches = X_batches[shuffle_indices]
    y_batches = y_batches[shuffle_indices]

    # Minibatch training
    for i in range(0, int ( trainable_dps / n_batches ) ):
        start_index = i*n_batches
        this_X_batch = X_batches[start_index:start_index+n_batches]
        this_y_batch = y_batches[start_index:start_index+n_batches]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: this_X_batch, Y: this_y_batch})

        # Show progress
        if (i == 0) or ( i == ( int ( trainable_dps / n_batches ) - 1 ) ):
            # Prediction
            pred = net.run(out, feed_dict={X: X_test_batch })
            pred_array = pred[0][:]
            line2.set_ydata( pred_array )
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plt.savefig('images/epoch' + str(e) + "-" + str(i) + '.png')

            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: this_X_batch, Y: this_y_batch }))
            mse_test.append(net.run(mse, feed_dict={X: X_test_batch, Y: y_test_batch }))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])

data_test_orig = scaler.inverse_transform( data_test )
data_test[:,0] = pred_array[0]
data_test_pred = scaler.inverse_transform( data_test )

#assumes 1st 2 columns are BTC px data
a = zip( data_test_orig[:,0], data_test_pred[:,0] ) 
np.savetxt('prediction.csv', a, newline="\n", fmt="%1.2f", delimiter=",")
