'''
Created on 2016. 11. 15.

@author: bi
'''

import os.path

import Image as im

import unittest

import tensorflow as tf

from wrapper.simple import Model, Cnn, Saver

class Test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.dir   = '/media/sf_pic/y'
        self.dir_y = '/media/sf_pic/yuna'
        
    def setUp(self):
        self.mock = Cnn()
        
    def tearDown(self):
        pass

    def test_jpg_open(self):
        jpgfile = im.open(self.dir + "/m.jpg")
#         print jpgfile.bits, jpgfile.size, jpgfile.format
#         jpgfile.rotate(30).save(self.dir + "/m2.jpg")
        
    def test_get_data(self):
        i = im.open(self.dir + "/m.jpg")
        l = list( i.getdata() )
        self.assertEquals ( list, type(l))
        
    def test_resize(self):
        i = im.open(self.dir + "/m.jpg")
        resized = i.resize( (300,300) ) #,filter=im.NEAREST)
        resized.save( self.dir + '/resized.jpg')
        
    def get_list_of_image(self, image_file_name, dir='', resize=(300,200)):
        i = im.open ( dir + image_file_name )
        ts= list( i.resize( resize).getdata() )
        return self.normalize(ts)
    
    def normalize (self, x ):
        new_x = []
        for i in x:
            new_t = []
            for j in i :
                new_t.append( float(i[0])/256 )
            new_x.append ( new_t )
        return new_x
        
    def bb(self, list):
        str = ''
        for i in list[0] :
            str += '%2.5f '% i
        return str
    
    def test_assert_diff_list(self):
        pass #self.assertNotAlmostEqual([1.1], [1.2], delta= .05)
    
    def test_learn(self):
        self.save_file = 'saved_net'
        if os.path.exists(self.save_file) == False:
            self.learn()
        else:
            self.make_net()
            self.mock.output = self.mock.current
            self.mock.init_session()
            self.saver.load(self.mock.sess)
            
        x = self.get_list_of_image ( '/1.jpg', dir=self.dir_y, resize=(300,300))
        print 
        print 'test with yuna'
        print self.bb(self.mock.test( [x]))
        x     = self.get_list_of_image ( '/resized.jpg', dir=self.dir, resize=(300,300))
        print self.bb(self.mock.test( [x]))
        
    def make_net(self):
        self.mock.loop_cnt = 300
        self.mock.add_layer(shape =[None,300*300, 3])
        self.mock.add_cnn_pool(input_channel_size=3)
        self.mock.add_cnn_pool()
        self.mock.add_cnn_pool()
        self.mock.add_layer(50 )
        self.mock.add_layer(50 )
        self.mock.add_layer(5 )
        self.saver = Saver()
        self.saver.file_name = self.save_file
        
    def learn(self):
        self. make_net()
        x     = self.get_list_of_image ( '/resized.jpg', dir=self.dir, resize=(300,300))
        
        y = [0,0,0,0,1]
        self.mock.set_input(x)
        self.mock.set_target(y)
        
        self.small_count = 0
        def print_entropy (i, sess, feed):
            entropy = sess.run( self.mock.entropy , feed)
            print '%3.5f : %s'%( entropy \
                  ,self.bb ( sess.run( self.mock.output , feed).tolist() )
                  )
            if entropy < .0001 : self.small_count+=1
            if self.small_count > 5 : self.mock.stop()
            
        self.mock.after_one_step = print_entropy
        self.mock.set_entropy_func(self.mock.euclid_distance)
        self.mock.learn()
        
        if  self.small_count > 5 :
            self.saver.save(self.mock.sess)
        else : raise Exception ( 'fail to learn')
        
        
#         print self.mock.run(self.mock.output, self.mock.feed)
    
def resize_save(from_, to, size=(300,300), dir='./'):
    if dir[-1] != '/' : dir += '/'
    im.open(dir + from_)\
    .resize(size)\
    .save( dir + to)
    
class Prepare (unittest.TestCase):
    dir_ = '/media/sf_pic/yuna'
    def test_end_char(self):
        s = 'ss3'
        self.assertEqual('3', s[-1])
    def test(self):
        resize_save ( 'yn1.jpg', '1.jpg', dir=Prepare.dir_)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
#     unittest.main()
    
    suit = unittest.TestSuite()
    suit.addTest( unittest.makeSuite(Test) )
#     suit.addTest( unittest.makeSuite(Prepare) )
    unittest.TextTestRunner().run(suit)