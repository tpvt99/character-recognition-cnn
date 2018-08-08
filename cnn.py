import math
import numpy as np
import tensorflow as tf
import scipy
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

np.random.seed(1)

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_C0), name='X')
    Y = tf.placeholder(tf.float32, shape = (None, n_y), name = 'Y')
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    return X,Y,keep_prob

def initialize_parameters():

    # adjust this parameter
    W1 = tf.get_variable("W1", [4,4,1,8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [2,2,8,16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {"W1": W1,
                "W2": W2}

    return parameters

def forward_propagation(X, keep_prob):
    tf.set_random_seed(1)

    # CONV 1
    W1 = tf.get_variable("W1", [4,4,1,64], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b1 = tf.get_variable("b1", [64,1], initializer = tf.zeros_initializer())

    Z1 = tf.nn.conv2d(input = X, filter = W1, strides = [1,1,1,1], padding = "SAME")
    A1 = tf.nn.relu(Z1)
    print(A1)
    # MAX POOL 1
    P1 = tf.nn.max_pool(A1, ksize = [1,4,4,1], strides = [1,4,4,1], padding = "SAME")
    print(P1)

    # CONV 2
    W2 = tf.get_variable("W2", [2,2,64,128], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b2 = tf.get_variable("b2", [128,1], initializer = tf.zeros_initializer())

    Z2 = tf.nn.conv2d(input = P1, filter = W2, strides = [1,1,1,1], padding = "SAME")
    A2 = tf.nn.relu(Z2)
    print(A2)
    # MAX POOL 2
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = "SAME")
    print(P2)
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    print(P2)
    # FULL CONNECT 1
    W3 = tf.get_variable('W3', [512, P2.shape[1:2].num_elements()], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    b3 = tf.get_variable('b3', [512,1], initializer = tf.zeros_initializer())
    Z3 = tf.add(tf.matmul(W3,tf.matrix_transpose(P2)), b3)
    A3 = tf.nn.relu(Z3)
    print(A3)
    # FULL CONNECT 2
    W4 = tf.get_variable('W4', [256, 512], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    b4 = tf.get_variable('b4', [256,1], initializer = tf.zeros_initializer())
    A4_drop = tf.nn.dropout(A3, keep_prob)
    Z4 = tf.add(tf.matmul(W4,A4_drop), b4)
    A4 = tf.nn.relu(Z4)
    print(A4)
    # FULL CONNECT 3
    W5 = tf.get_variable('W5', [63,256], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    b5 = tf.get_variable('b5', [63,1], initializer = tf.zeros_initializer())
    A5_drop = tf.nn.dropout(A4, keep_prob)
    Z5 = tf.add(tf.matmul(W5,A5_drop), b5)
    print(Z5)

    Z5 = tf.matrix_transpose(Z5)

    return Z5

def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z3, labels = Y))

    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001, keep_prob_prob = 1,
        num_epochs = 20, minibatch_size = 64, print_cost = True):

	ops.reset_default_graph()
	tf.set_random_seed(1)
	seed = 3
	m, n_H0, n_W0, n_C0 = X_train.shape
	n_y = Y_train.shape[1]
	costs = []

	X, Y, keep_prob = create_placeholders(n_H0, n_W0, n_C0, n_y)

	Z3 = forward_propagation(X, keep_prob)

	cost = compute_cost(Z3, Y)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	# This is for accuracy testing
	y_pred = tf.nn.softmax(Z3)
	y_pred_class = tf.argmax(y_pred, axis = 1)
	y_true_class = tf.argmax(Y, axis = 1)

	correct_prediction = tf.equal(y_pred_class, y_true_class)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:

		sess.run(init)

		for epoch in range(num_epochs):

			minibatch_cost = 0
			num_minibatches = int(m / minibatch_size)
			seed = seed + 1
			minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

			for minibatch in minibatches:

				minibatch_X, minibatch_Y = minibatch

				_, temp_cost = sess.run([optimizer, cost], {X:minibatch_X, Y:minibatch_Y, keep_prob :keep_prob_prob})
				minibatch_cost += temp_cost / num_minibatches


			if print_cost == True:
				print("Cost after epoch %i %f" %(epoch, minibatch_cost))
				costs.append(minibatch_cost)

			acc = sess.run(accuracy, feed_dict = {X: X_test, Y: Y_test, keep_prob : keep_prob_prob})
			print("Accuracy on test dataset after epoch %i: %f" %(epoch, acc))

		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate = " + str(learning_rate))
		plt.show()
		saver = tf.train.Saver()
		saver.save(sess, 'my-model.ckpt')

		predict_op = tf.argmax(Z3, 1)
		correct_prediction = tf.equal(predict_op, tf.argmax(Y,1))

		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print(accuracy)
		train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob: keep_prob_prob})
		test_accuracy = accuracy.eval({X: X_test, Y: Y_test, keep_prob: keep_prob_prob})
		print("Train Accuracy:", train_accuracy)
		print("Test Accuracy:", test_accuracy)


		return None

if __name__ == "__main__":
	train_inputs, train_outputs , test_inputs, test_outputs = np.load('training_inputs_0_9.npy'), np.load('training_outputs_0_9.npy'), np.load('test_inputs_0_9.npy'), np.load('test_outputs_0_9.npy')

	X = np.array(train_inputs)
	Y = np.array(train_outputs)
	Y = Y.reshape(Y.shape[0],Y.shape[1])
	X_test = np.array(test_inputs)
	Y_test = np.array(test_outputs)
	Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1])
	print(X.shape)
	print(Y.shape)
	print(X_test.shape)
	print(Y_test.shape)

	parameters = model(X_train = X, Y_train = Y, X_test = X_test, Y_test = Y_test)
	print(parameters)
