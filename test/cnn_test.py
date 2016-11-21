'''
Created on 2016. 11. 14.

@author: bi
'''

import math

import unittest

import tensorflow as tf

import traceback

import sys
sys.path.append('/media/sf_share/pylib/tensorflow_util')
import board

from deep.simple import Model, Cnn

class Test(unittest.TestCase):

    def setUp(self):
        self.mock = Cnn()


    def tearDown(self):
        pass


        

    def test_math_root(self):
        self.assertEquals(5, math.sqrt(25))
        self.assertEquals(6, math.sqrt(36))
        self.assertEquals(7, math.sqrt(49))
             
    def test_one_cnn(self):
        self.mock.add_layer(49)
        self.mock.add_cnn()
        
        
    def test_pool(self):
        self.mock.add_layer(49)
        self.mock.add_cnn()
        self.mock.add_pool()
        
    def test_normal_rank_size(self):
        self.mock.add_layer(3)
        self.assertEquals( 2, self.mock.current.get_shape().as_list().__len__() )
        
    def test_cnn_rank_size(self):
        self.mock.add_layer(50)
        self.mock.add_cnn()
        self.assertEquals( 4, self.mock.current.get_shape().as_list().__len__() )

    def test_get_width_input_channel_size(self):
        self.mock.add_layer(4)
        input_channel_size, width = self.mock.get_width_input_channel_size(self.mock.current)
        self.assertEqual(width , 2 )
        self.assertEqual(input_channel_size , 1 )
        
    def test_get_width_input_channel_size_when_connect_to_cnn(self):
        self.mock.add_layer(49)
        self.mock.add_cnn ()
        self.assertEqual(7, self.mock.current.get_shape().as_list()[1])
        input_channel_size, width = self.mock.get_width_input_channel_size(self.mock.current)
        self.assertEqual(width , 7 )
        self.assertEqual(input_channel_size , 32 )
        
    def test_get_width_input_channel_size_when_connect_to_pool_of_cnn(self):
        self.mock.add_layer(49)
        self.mock.add_cnn ()
        self.mock.add_pool()
        self.assertEqual(4, self.mock.current.get_shape().as_list()[1])
        input_channel_size, width = self.mock.get_width_input_channel_size(self.mock.current)
        self.assertEqual(width , 4 )
        self.assertEqual(input_channel_size , 32 )
        
    def test_get_width_input_channel_size_when_multi_input_channel(self):
        self.mock.add_layer (300)
        in_channel, width = self.mock.get_width_input_channel_size( self.mock.current, input_channel_size=3)
        self.assertEqual(3, in_channel)
        self.assertEqual(10, width)
        
    def test_add_cnn_multi_input_channel(self):
        self.mock.add_layer(300)
        self.mock.add_cnn( input_channel_size= 3)
        shape= self.mock.current.get_shape().as_list()
        self.assertEquals([None, 10,10,32], shape)
        
    def test_two_cnn(self):
        self.mock.add_layer(49)
        self.mock.add_cnn()
        self.mock.add_pool()
        self.mock.add_cnn()
        
    def test_add_pool(self):
        self.mock.add_layer(300)
        self.mock.add_cnn_pool(input_channel_size=3)
        
    def test_input_layer_by_shape(self):
        self.mock.add_layer(shape = [100,3])
        self.assertEquals([100,3], self.mock.current.get_shape().as_list())
        
    def test_input_layer_by_shape_and_add_normal_layer(self):
        self.mock.add_layer(shape = [100,3])
        self.mock.add_layer(30)
        

class LayerTest(unittest.TestCase):
    def test_ConvolutionLayer(self):
        model = Cnn()
        model.add_layer(shape=[None, 300*300, 3])
        in_channel, width = model.get_width_input_channel_size(model.current)
        self.assertEqual(3, in_channel)
        self.assertEqual(300, width)
        
        
class SlowTest(unittest.TestCase):
    def setUp(self):
        self.mock = Cnn()
        
    def test_fully_connected(self):
#         self.mock.verbose = True
        self.mock.loop_cnt = 65
        self.mock.add_layer(784)
        self.mock.add_cnn()
        self.mock.add_pool()
        self.mock.add_cnn()
        self.mock.add_pool()
        self.mock.add_layer(1024, act_func=tf.nn.relu)
        self.mock.add_layer(10, act_func= tf.nn.softmax)
        self.mock.set_entropy_func(self.mock.entropy_log)
        
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("down/", one_hot=True)

        def get_feed(x=None):
            b = mnist.train.next_batch(100)
            feed = {self.mock.input:b[0], self.mock.target:b[1]}
            return feed
            
        self.mock.get_feed_before_loop = get_feed
        self.mock.get_feed_each_one_step = get_feed
        
#         def print_entropy (i, sess, feed):
#             print  sess.run( self.mock.entropy , feed)
#         self.mock.after_one_step = print_entropy
            
        self.mock.learn()
        self.assertTrue(0.5 <self.mock.last_acc , 'less 0.5 acc %2.3f'%self.mock.last_acc )
        
    def test_set_target(self):
#         self.mock.loop_cnt = 100
        self.mock.add_layer(3)
        self.mock.add_layer(2)
        self.mock.set_input ([1,2,3], [2,3,4])
        self.mock.set_target([1,1], [1,0])
        
#         def one_step(loop_index, sess, feed):
#             print self.mock.run(self.mock.entropy, feed)
#             sess.run ( self.mock.train_step, feed_dict= feed )
#         self.mock.one_step = one_step
        
        self.mock.learn()
#         print self.mock.test([[1,2,3]])
#         print self.mock.test([[0,0,3]])
#         print self.mock.test([[2,3,4]])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    #unittest.main()
    
    suite = unittest.TestSuite()

    # adding a test case
    suite.addTest(unittest.makeSuite(Test))
    suite.addTest(unittest.makeSuite(LayerTest))
    suite.addTest(unittest.makeSuite(SlowTest))
    
    unittest.TextTestRunner().run(suite)