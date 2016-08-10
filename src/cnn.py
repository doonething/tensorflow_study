#!/usr/bin/python

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("down/", one_hot=True)

def weight_var( shape ):
	init = tf.truncated_normal( shape , stddev=.1)
	return tf.Variable(init)

def bias_var(shape ) :
	init = tf.constant(.1, shape = shape)
	return tf.Variable(init)

def conv2d (x, W) :
	return tf.nn.conv2d( x, W, strides=[1,1,1,1], padding='SAME' )

def max_pool_2x2(x) :
	return tf.nn.max_pool(x, ksize =[1,2,2,1]
				,strides=[1,2,2,1]
				,padding='SAME' )

# target
y_ = tf.placeholder ( tf.float32, [None , 10])

# input placeholder 28 * 28 = 784  
x = tf.placeholder ( tf.float32, [None, 784] ) 
x_image = tf.reshape(x, [-1,28,28,1])

w1 = weight_var([5,5,1,32])
b1 = bias_var([32])
h1 = tf.nn.relu( conv2d(x_image, w1) + b1)
pool = max_pool_2x2(h1)

w2 = weight_var([5,5,32,64])
b2 = bias_var([64])
h2 = tf.nn.relu( conv2d(pool, w2)      + b2)
pool2= max_pool_2x2(h2)

# densely connected layer
W_fc1 = weight_var([7 * 7 * 64, 1024])
b_fc1 = bias_var([1024])

# using pool
h_pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# densely connected layer : dropout 
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
# densely connected layer : weight and biase
W = weight_var( [1024,10])
b = weight_var([10])

y = tf.nn.softmax(tf.matmul( h_fc1_drop, W) + b)

cross_entropy = tf.reduce_mean (tf.reduce_sum ( y_ * -tf.log(y), reduction_indices=[1] ) )
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
y_max = tf.argmax(y,1)
y__max= tf.argmax(y_,1)
correct_prediction = tf.equal(y_max, y__max ) # tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as s:
	s.run( tf.initialize_all_variables() )
	for i in range(10000):
		b = mnist.train.next_batch(100)
		fd = {x:b[0], y_:b[1], keep_prob:.5}
		train_step.run( feed_dict=fd)
		if i % 10 == 0 : 
			print i, s.run(accuracy, feed_dict=fd)
		
