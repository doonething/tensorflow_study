#!/usr/bin/python

import tensorflow as tf

# input data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("down/", one_hot=True)


# input placeholder
x = tf.placeholder ( tf.float32, [None, 784] )

# weights
W = tf.Variable ( tf.random_normal( [784, 10], stddev = .1 ) )
# bias
b = tf.Variable ( tf.zeros( [10] )  ) 

# output
y = tf.nn.softmax ( tf.matmul ( x, W ) + b )

#target place holder
y_ = tf.placeholder ( tf.float32, [None, 10] )

cross_entry = tf.reduce_mean (
		tf.reduce_sum ( y_ * -tf.log(y), reduction_indices=[1] ) 
              )
train_step = tf.train.GradientDescentOptimizer(0.5). minimize ( cross_entry )

delta = tf.equal( tf.argmax(y,1), tf.argmax(y_,1) )
accuracy = tf.reduce_mean(tf.cast(delta, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as s:
	# initialize variables
	s.run(init)
	for i in range(1000) :
		xs, ys = mnist.train.next_batch(100)
		fd = {x:xs, y_:ys}
		s.run ( train_step, feed_dict=fd )
		
	print 'same rate' ,s.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
