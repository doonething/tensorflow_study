'''
Created on 2016. 10. 26.

@author: bi
'''
import unittest

import sys
import random

sys.path.append('..')
sys.path.append('deep_net')

import tensorflow as tf

from deep import simple, p
from deep.simple import Saver

def nothing(*args, **kwargs):
    pass

class Decorator :
    def common(self, ori, before=nothing, after=nothing):
        def deco  (*args, **kwargs):
            before(*args, **kwargs)
            ori   (*args, **kwargs)
            after (*args, **kwargs)
        return deco
    
class ModelDecorator(Decorator) :
    def __init__(self, ori):
        self.ori = ori
        self.ori_one_step = ori.one_step
        self.ori_loop = ori.loop
        self.ori_test_sess = ori.test_sess
        
    def do_tf_session(self):
        return self.common( self.ori_do_tf_session)
    
    def loop(self):
        return self.common( self.ori_loop, before=self.loop_before
                            , after=self.loop_after)
        
    def test_sess(self, *args, **kwargs):
        return self.common( self.ori_test_sess
                            , before=self.test_sess_before
                            , after =self.test_sess_after
                            )
    
    def loop_before (self,*args, **kwargs):
        self.before_w = self.weight(args[0], args[1])
    
    def loop_after (self, *args, **kwargs):
        sess = args[0]
        feed = args[1]
        after_w = self.weight(args[0], args[1])
        print 'before w ', self.before_w
        print 'after  w ', p.p(after_w)
        print 'out      ', p.p(self.output (args[0], args[1]) )
#         self.saver = Saver()
#         self.saver.save(sess)
        
    def test_sess_before(self, *args, **kwargs):
        sess = args[0]
#         self.saver.load(sess)
        self.before_w = self.weight(args[0], args[1])
        print 'test before w ', p.p(self.before_w)
        
    def test_sess_after (self, *args, **kwargs):
        print 'test out      ', p.p(self.output (args[0], args[1]) )
        pass
        
    def weight(self, sess, feed):
        return sess.run(self.ori.Ws[0], feed_dict = feed).tolist()
    
    def output(self, sess, feed):
        return sess.run(self.ori.output, feed_dict= feed).tolist()
        
    def one_step (self, *args, **kwargs) :
        return self.common ( self.ori_one_step, after=self.one_step_after  )
    
    def one_step_after (self, *args, **kwargs):
#         self.entropy ( *args, **kwargs )
#         g = self.gradient( args[1], args[2])
        pass
        
    def gradient(self, sess, feed):
        print
        print type ( self.ori.grads_and_vars )
#         print self.ori.grads_and_vars[0][1]
        
    def entropy(self,*args, **kwargs):
        i = args[0]
        s = args[1]
        feed = args[2]
        entropy_ = s.run ( self.ori.entropy, feed_dict = feed)
        if i%120 == 0 : 
            p.p( i, entropy_ )
            

    
class Test(unittest.TestCase):
    
    def setUp(self):
        self.mock = simple.Model()

    '''
#         def one_step_backup(self): 
#             def deco(*args, **kwargs):
#                 self.ori_one_step(*args, **kwargs)
#                 i = args[0]
#                 s = args[1]
#                 feed = args[2]
#                 entropy_ = s.run ( self.simple_instance.entropy, feed_dict = feed)
#                 if i%120 == 0 : 
#                     p.p( i, entropy_ )
#             return deco
    '''
                
    @staticmethod
    def decorate_entropy(f, instance):
        def dec(*args, **kwargs):
            f(*args, **kwargs)
            print args[0]
        return dec
        
    def make_net(self):
        self.mock.add_layer(3)
        self.mock.add_layer(2)
        self.mock.add_layer(1)
        
    def test_learn_test(self):
        if self.mock.Ws.__len__() < 1 :
            self.make_net()
#         decorator = Test.PrintDecorator(self.mock) 
#         self.mock.loop = Test.decorate_entropy(self.mock.loop, self.mock)
#         self.mock.one_step = decorator.one_step()
#         self.mock.loop  = decorator.loop()

        self.mock.loop_cnt =900
#         deco = ModelDecorator ( self.mock)
#         self.mock.one_step = deco.one_step()
#         self.mock.loop     = deco.loop()
#         self.mock.test_sess= deco.test_sess()

        beforeW0       = self.mock.get_W(0).tolist()
        
        x = [[.1,.2,.3],[.2,.3,.4]]
        y = [[2.],[5.]]
#         before_entropy = self.mock.get_entropy(x)
                  
        self.mock.learn( x  , y)
        
        afterW0 = self.mock.get_W(0).tolist()
        after_entropy  = self.mock.get_entropy(x)
        
        self.assertNotEquals (beforeW0, afterW0)
        self.assertTrue( ( after_entropy < .1 ) 
                         or
                         ( self.mock.first_entropy >  after_entropy )
                         , msg =" fail entropy samller \n"
                              "   %f -> %f"
                              %( self.mock.first_entropy, after_entropy)
                       )
        
        out = self.mock.test ( [[.1,.2,.3],[.2,.3,.4]] ).tolist()
        
        # check learned set
        self.compare(out, y, self.assertAlmostEqual
                           , msg= "  entropy ( %f -> %f ) \n"
                                  "  before weights %s \n"   
                                  "  after  weights %s \n"
                                      %( self.mock.first_entropy, after_entropy
                                        ,beforeW0.__str__()
                                        ,afterW0 .__str__()
                                        )
                     )
            
    def compare(self, y, out, assert_func, msg = ''):
        for i in range(y[0].__len__() ):
            expected = y[0][i]
            real     = out[0][i]
            delta    = expected*.2
            return   assert_func (expected, real, delta=delta
                               ,msg = "fail %s \n"
                                      "  out=%f   , expected=%f  delta=%f\n"
                                      "  %s"
                                      %( assert_func.__name__, real, expected, delta, msg )
                               )
            
    def test_compare_fail_assertAlmostEqual(self):
        y = [[1.]]
        out = [[.5]]
        try :
            self.compare(y, out, self.assertAlmostEqual )
            self.fail('do not here')
        except AssertionError:
            pass
    
    def test_other_input(self):
        def randomx():
            return [random.random()
                       ,random.random()
                       ,random.random()]
        self.make_net()
        x = [randomx(), randomx()]
        y = [[1.2],  [1.5]]
        self.mock.learn(x, y)
        other_input = [randomx()]
        other_out   = self.mock.test( other_input ).tolist()
        self.compare(y,other_out, self.assertNotAlmostEqual
                     , msg = " random x = %s \n"
                             " other  x = %s " 
                             %( x, other_input )
                    )

    def test_loop(self):
        for i in range(20) :
            self.test_learn_test()
        

class Overriding(simple.Model):
    def loop(self, sess, feed):
        simple.Model.loop(self,sess, feed)
        
class TestSave(unittest.TestCase):
    def test(self):
        x = Overriding()
        x.add_layer(2)
        x.add_layer(1)
        xx = [[1,0]] 
        y = [1]
        x.learn(xx, y)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()