'''
Created on 2016. 10. 26.

@author: bi
'''

import math

import numpy as np
import tensorflow as tf

class Model:
    def __init__(self):
        self.current = None
        self.entropy_func = self.euclid_distance
        self.Ws = []
        self.layers = []
        self.entropy  = None #tf.Variable([1], name='dum_my')
        self.loop_cnt = 3000
        self.sess     = tf.Session()
        # self.sess     = tf.InteractiveSession()
        self.is_sess_init = False
        self.unsupervised = False
        self.verbose = False
        
        self.feed = {}
        self.target = None
        self.output = None
        
        self.is_stop = False
    
    def add_layer(self, neuron_num=-1, name=None, act_func=tf.nn.softplus
                  ,shape=None):
        if self.current is None :
            self.init_input_layer(neuron_num, shape, name)
            return
        in_num = self.get_input_num ( self.current )
        W = tf.Variable ( tf.random_normal( [in_num, neuron_num ], stddev = .1 ) , name=name)
        
        # assgin() make copy, origin( caller of assgin() ) is not changed 
        # W = W.assign ( np.random.rand(in_num, neuron_num) )
        
        self.Ws.append(W)
        # bias
        #b = tf.Variable ( tf.zeros( [neuron_num] )  )
        b = tf.Variable ( tf.random_normal( [neuron_num], stddev=.3 )  )
        
        current = self.current
        current = self.if_4_rank_then_2_rank(current)
        current = tf.matmul (current, W) + b
        #         self.current = tf.nn.softmax ( self.current)
        #         self.current = tf.nn.relu(self.current)
        #         self.current = tf.nn.softplus(current)
        self.current = act_func ( current)
        
        self.layers.append(self.current)
        return self
    
    def init_input_layer(self, neuron_num=-1, shape=None, name=None):
        if self.current is not None : return
        if shape is None : shape = [None,neuron_num]
        self.current = tf.placeholder ( tf.float32, shape, name=name )
        self.input = self.current
        self.layers.append(self.current)
    
    def if_4_rank_then_2_rank (self, tensor):
        list = tensor.get_shape().as_list()
        if list.__len__() == 4 :
            return tf.reshape( tensor, [-1,list[1]*list[2]*list[3]])
        return tensor 
    
    def get_input_num (self, tensor):
        list = tensor.get_shape().as_list()
        if list.__len__() == 4 : # conv 
            return list[1]*list[2]*list[3]
        return list[1]

    def euclid_distance(self, y):
        self.entropy = tf.reduce_mean( .5 *  np.power(self.output - y , 2) )

    def entropy_log (self, y):
        self.entropy = tf.reduce_mean (tf.reduce_sum ( self.target * -tf.log(self.output), reduction_indices=[1] ) )

    def set_entropy_func(self, func):
        self.entropy_func = func
        
    def optimize(self):
        #self.train_step = tf.train.GradientDescentOptimizer(0.5). minimize ( self.entropy )
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        self.grads_and_vars = optimizer.compute_gradients ( self.entropy)
        return optimizer.apply_gradients (self.grads_and_vars, global_step=None, name=None)
    
    def learn(self, x=None,y=None):
        self.output = self.current
        if y is None : y = self.target
        self.entropy_func(y)
        if not self.unsupervised :
            self.train_step = self.optimize()
        self.do_tf_session(x)

    def do_tf_session(self, x):
#         with tf.Session() as s:
#         self.sess.run(tf.initialize_all_variables())
        self.init_session()
#         feed = {self.input:x}
        feed = self.get_feed_before_loop(x)
        
#         print
#         for key in feed :
#             print type(key)
        self.loop(self.sess,feed)
        
    def get_feed_before_loop(self,x):
#         return {self.input:x}
        return self.feed

    
    def get_feed_each_one_step(self, feed):
        return feed
        
    def init_session(self):
        if not self.is_sess_init :
            self.sess.run(tf.initialize_all_variables())
            self.is_sess_init = True
    
    def loop (self, s, feed):
        self.save_first_entropy(s,feed)
        for i in range(self.loop_cnt) :
            feed = self.get_feed_each_one_step (feed)
            self.one_step( i, s, feed)
            if self.is_stop : break
            
    def save_first_entropy(self, sess, feed):
        self.one_step(-1, sess, feed)
        self.first_entropy = sess.run( self.entropy, feed )
        

    def before_one_step(self, loop_index, sess, feed):
        pass
    
    
    def after_one_step(self, loop_index, sess, feed):
        pass
    
    
    def one_step(self, loop_index, sess, feed):
        self.before_one_step(loop_index, sess, feed)
        sess.run ( self.train_step, feed_dict= feed )
        self.after_one_step(loop_index,sess, feed)

    
    def test(self, x):
        if self.output is None :
            print ' do test()  after learn()'
            return
        feed = {self.input:x}
#         with tf.Session() as s:
#             s.run(tf.initialize_all_variables())
        ret = self.test_sess(self.sess,feed)
        return ret
    
    def test_sess(self,sess, feed):
        return sess.run(self.output , feed_dict = feed)
    
    
    '''----------- helper func'''
    
    def run(self, tensor , feed ={}):
        self.init_session()
        return self.sess.run(tensor, feed)
    
    def get_W(self, index):
        return self.run( self.Ws[index])
    
    def get_entropy (self, x=None):
        return self.run ( self.entropy, {self.input:x} )
    
    def set_input(self, *input):
        self.feed[self.input] = input
    
    def set_target(self, *target):
        ''' after this function call, add layer make problem ?'''
#         if self.target is None :
#             if self.output is None :
#                 self.output = self.current 
#             self.target = tf.placeholder ( tf.float32, [None , self.output.get_shape().as_list()[1]] , name = 'target')
        self.init_target_placeholder() 
        self.feed[self.target] = target
        
    def init_target_placeholder(self):
        if self.output is None :
            self.output = self.current 
        if self.target is None :
            self.target = tf.placeholder ( tf.float32, [None , self.output.get_shape().as_list()[1]] , name = 'target')

    def stop(self):
        self.is_stop = True
    
class Saver:
    def __init__(self):
        self.saver = tf.train.Saver()
        self.file_name = 'save.ckpt'
    def save (self, sess ):
        self.saver.save(sess, self.file_name)
    
    def load(self, sess):
        self.saver.restore(sess, self.file_name)

      
class Cnn(Model):
    def add_cnn(self, out_channel=32, filter_num=5, input=None 
                , input_channel_size = 1):
        if input is None : input = self.current
        input_channel_size, width = self.get_width_input_channel_size (input, input_channel_size=input_channel_size)
        if filter_num > width : filter_num = width - 1
        
        w = self.weight_var ( filter_num,filter_num,input_channel_size, out_channel )
        
        b = self.bias_var( out_channel)
        reshape = tf.reshape (input,[ -1, width,width , input_channel_size ])
        self.current = tf.nn.relu( tf.nn.conv2d( reshape,w,strides=[1,1,1,1], padding='SAME' ) +b)
        
    def get_width_input_channel_size(self, input, input_channel_size=1):
        list = input.get_shape().as_list()
        rank = list.__len__()
        if rank == 3 :  # input channel exist
            input_channel_size = list[2]
            width = int ( math.sqrt( float( list[1] ) ) )
        if rank == 2 :   # normal layer
            square = self.check_input_channel_size(list[1] , input_channel_size )
            width = int ( math.sqrt( float( square ) ) )
        if rank == 4 :   # connecting to cnn
            width = list[1]   # if and only if input width == height
            input_channel_size = list[3]
        return input_channel_size, width
    
    def check_input_channel_size(self, square, input_channel_size):
        if square % input_channel_size != 0 :
            raise Exception( 
                'square % input_channel_size != 0 : square:%d input_channel_size=%d'%(square,input_channel_size))
        return square/ input_channel_size

    def add_cnn_pool(self, out_channel=32, filter_num=5, input=None
                     ,input_channel_size=1 ):
        self.add_cnn(out_channel,filter_num, input, input_channel_size)
        self.add_pool()

    def optimize(self):
        return tf.train.AdamOptimizer(1e-4).minimize(self.entropy)
#             return Model.optimize(self)
    
    def weight_var( self, *shape ):
        init = tf.truncated_normal( shape , stddev=.1)
        return tf.Variable(init)
    
    def bias_var( self, *shape ) :
        init = tf.constant(.1, shape = shape)
        return tf.Variable(init)
    
    def add_pool(self, size=2):
        self.current = tf.nn.max_pool ( self.current, ksize =[1,size,size,1]
            ,strides=[1,size,size,1]
            ,padding='SAME' )
        
    def learn_back (self, x=None, y= None):
        self.output = self.current
        y_max = tf.argmax(self.output,1)
        self.y = tf.placeholder ( tf.float32, [None , 10])
        y__max= tf.argmax(self.y,1)
        correct_prediction = tf.equal(y_max, y__max ) # tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        Model.learn(self, x, self.y)
         
    def learn(self, x=None, y= None):
        self.init_target_placeholder()
        out = tf.argmax(self.output,1)
        target = tf.argmax(self.target,1)
        correct_prediction = tf.equal(out, target ) # tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        Model.learn(self,x,y)
         
    
    def loop(self, s, feed):
        Model.loop(self, s, feed)
        self.last_acc = self.run(self.accuracy, feed)

    
    
