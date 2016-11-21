'''
Created on 2016. 11. 14.

@author: bi
'''

#!/usr/bin/python

import unittest as u
import sys

import simple_test
import cnn_test


# creating a new test suite
suite = u.TestSuite()
 

# adding a test case
suite.addTest(u.makeSuite(simple_test.Test))
suite.addTest(u.makeSuite(cnn_test.Test))


u.TextTestRunner().run(suite)