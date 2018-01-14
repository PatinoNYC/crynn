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

# y_data is set above, it can be BTC, ETH, LTC, etc..
data["y_axis"] = data[y_data]

# now we move the data UP in time, we'll plug it back into our matrix later but this will be our _target_
data["y_axis"] = data["y_axis"].shift(-y_offset_predict)

# Dimensions of dataset
n = data.shape[0] #rows
p = data.shape[1] #columns

print "Copying forward data for our offset.  Keep this in mind, it means the end of our data set is dummy data"
for x in range (0, y_offset_predict):
#    print str(n - x - 1) + " becomes " + str( n - y_offset_predict - 1 )
    data["y_axis"].at[n - x - 1] =  data["y_axis"].at[ n - y_offset_predict -1 ]    

#from here on, the code assumes the first pandas column is our y_axis.  Let's make that so:
cols = list(data)
cols = cols[-1:] + cols[:-1]
data = data[cols]
    
# Make data a np.array
np_data = data.values

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
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
num_features = X_train.shape[1]

# Session
net = tf.InteractiveSession()
output = 1 # our prediction is just a vector
hidden = 100 #number of neurons we will recursively work through 

# Placeholder
num_periods = 720
X = tf.placeholder(dtype=tf.float32, shape=[None, num_periods, num_features])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1, output])

num_batches = train_end - num_periods
num_batches = 1000
X_batches = np.empty( [ num_batches, num_periods, num_features] )
y_batches = np.empty( [ num_batches, output, 1 ] )
for i in range ( num_batches ):
    X_batches[i] = X_train[ i:i+num_periods, :]
    y_batches[i] = y_train[ i+num_periods ]

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])          #shape of results
 
#cost function
mse = tf.reduce_sum( tf.square ( outputs - y ) )

# Optimizer using gradient descent
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer( learning_rate = learning_rate )

#train to cost function
training_op = optimizer.minimize( mse )

#init
init = tf.global_variables_initializer()

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
plt.show()

# Fit neural net
mse_train = []
mse_test = []

# Run
pred = 0
epochs = 100

print X_batches.shape
print y_batches.shape
with tf.Session() as sess:
    init.run()
    for ep in range (epochs):
        print ep
        sess.run ( training_op, feed_dict={X: X_batches, y: y_batches } )
        if ep % 10 == 0:
            mse_val = mse.eval( feed_dict={X: X_batches, y: y_batches } )
            print(ep, "\tMSE:", mse_val )

    for i in range( test_dps ):
        print "i: " + i
        sample_test_data = X_test[i:num_periods+i,:].reshape(1,num_periods,num_features)
        pred[i] = sess.run(outputs, feed_dict={X: sample_test_data } )

    line2.set_ydata(pred)
    plt.title('Epoch ' + str(ep) )
    plt.pause(0.01)
    plt.savefig('images/epoch' + str(ep) + '.png')
            

data_test_orig = scaler.inverse_transform( data_test )
data_test[:,0] = pred[0]
data_test_pred = scaler.inverse_transform( data_test )

#assumes 1st 2 columns are BTC px data
a = zip( data_test_orig[:,0], data_test_pred[:,0] ) 
np.savetxt('prediction.csv', a, newline="\n", fmt="%1.2f", delimiter=",")
