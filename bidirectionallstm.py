from __future__ import print_function
from random import randint
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import dataset
from sklearn.metrics import confusion_matrix,classification_report,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
training_steps = 5000
batch_size = 20
display_step = 200

# Network Parameters
num_input = 15  # MNIST data input (img shape: 28*28)
timesteps = 32  # timesteps
num_hidden = 50  # hidden layer num of features
num_classes = 16  # MNIST total classes (0-9 digits)

data_train, data_test, label_train, label_test = dataset.get_dataset()
print('{} {}'.format(data_train.shape, label_train.shape))

def get_next_batch(data,labels,batchsize):
    limit = data.shape[0]
    flag = 0
    while(flag==0):
        ind = randint(0,limit)
        if ind + batchsize < limit:
            batch_x = data[ind:ind+batchsize]
            batch_y = labels[ind:ind+batchsize]
            flag = 1
    return batch_x,batch_y

X = tf.placeholder(tf.float32, shape=[None, timesteps , num_input])
Y = tf.placeholder("float", [None, timesteps, num_classes])
y_true = tf.reshape(tf.stack(Y), [-1, num_classes])

weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2 * 12, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def BiRNN(x):
    x=tf.unstack(tf.transpose(x, perm=[1, 0, 2]))

    def rnn_cell():
        cell = tf.nn.rnn_cell.LSTMCell(12, forget_bias=1,state_is_tuple=True)
        return cell

    fw_cell=rnn_cell()
    bw_cell=rnn_cell()
    output,_, _ = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell,x, dtype=tf.float32)
    weight = weights['out']
    bias = biases['out']
    output = tf.reshape(tf.transpose(tf.stack(output), perm=[1, 0, 2]), [-1, 2 * 12])
    return (tf.matmul(output, weight) + bias)



logits = BiRNN(X)
#logits = tf.reshape(tf.stack(logits), [-1, timesteps,num_classes])
prediction_out = tf.nn.softmax(logits)
prediction = tf.reshape(prediction_out, [-1, timesteps, num_classes])

# Define loss and optimizer
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

pred = tf.argmax(prediction, 2)
actual = tf.argmax(Y, 2)
# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 2), tf.argmax(Y, 2))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

losses = list()
accuracies = list()
iter = list()
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps + 1):
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x,batch_y = get_next_batch(data_train,label_train,20)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc, predict,true = sess.run([loss_op, accuracy, pred, actual], feed_dict={X: batch_x,Y: batch_y})
            losses.append(loss)
            accuracies.append(acc)
            iter.append(step)
            print("Step " + str(step) + ", Minibatch Loss= " + "{}".format(loss) + ", Training Accuracy= " + "{}".format(acc))
            #print("predict {} true {}".format(predict,true))


    # Calculate accuracy for 128 mnist test images
    #test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    #test_label = mnist.test.labels[:test_len]
    test_data = data_test
    test_label = label_test
    acc, pred, true = sess.run([accuracy,pred,actual] ,feed_dict={X: test_data, Y: test_label})
    pred = np.reshape(pred,newshape=(pred.shape[0]*pred.shape[1],1))
    true = np.reshape(true,newshape=(true.shape[0]*true.shape[1],1))
    print('{} {}'.format(pred.shape,true.shape))
    #precision = precision_score(true,pred,average='micro')
    #recall = recall_score(true,pred,a)
    #f_1 = f1_score(true,pred)
    #print('{} {} {}'.format(precision,recall,f_1))
    cmat = classification_report(true,pred)

    print("Testing Accuracy:{} {} {}".format(acc,pred,true))
    print("Confusion matrix {}".format(cmat))

    plt.plot(losses, 'r', label="Loss", linewidth=2)
    plt.plot(accuracies, 'g', label="Accuracy", linewidth=2)
    #plt.plot(timestamp3, scienceData, 'y', label="Science", linewidth=2)

    plt.legend()
    plt.grid(True, color='k')
    plt.show()

