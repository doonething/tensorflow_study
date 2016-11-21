import tensorflow as tf

merged = None
s = None
train_writer = None

def scalar(name, val):
	with tf.name_scope(name):
		tf.scalar_summary(name, val )

def histo( name, val):
	with tf.name_scope(name):
		tf.histogram_summary(name, val )

def make_writer( dir ) :
	global merged, s, train_writer
	merged = tf.merge_all_summaries()
	train_writer =  tf.train.SummaryWriter(dir, s.graph)
	return train_writer

def add( i, fd ) :
	global merged, s, train_writer
	summary = s.run ( merged , feed_dict = fd ) 
	train_writer.add_summary(summary, i)

def scalars_of_2d( name_prefix, matrix):
	shape = matrix.get_shape()
	for i in range(shape[0]):
		for j in range(shape[1]):
			scalar('%s_%d_%d'%(name_prefix,i,j), matrix[i,j])
