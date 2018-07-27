import tensorflow as tf

class HistoryPool(object):
	def __init__(self, shape, max_size=128, dtype=tf.float32):
		self.shape = shape
		self.max_size = max_size

		self.counter = tf.Variable(0, dtype=tf.int32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
		self.buf = tf.Variable(tf.zeros([self.max_size] + list(self.shape)), dtype=dtype, collections=[tf.GraphKeys.LOCAL_VARIABLES])

	def query(self, x):

		if self.max_size == 0:
			return x


		count = tf.assign_add(self.counter, tf.shape(x)[0])

		# if count < max_size
		# add to buffer

		count = tf.assign_add(self.counter, tf.shape(x)[0])

		idx_sequential = tf.range(count - tf.shape(x)[0], count, dtype=tf.int32) % self.max_size
		batch = tf.gather(tf.scatter_update(self.buf, idx_sequential, x), idx_sequential)

		# else

		# select random elements
		# select random indices
		# overwrite unique indices with first slots of x
		
		# Prevent overwriting old data before the buffer is full by setting choice to zeros while not full
		choice = tf.cond(count < self.max_size, lambda: tf.zeros((tf.shape(x)[0],), dtype=tf.int32), lambda: tf.random_uniform((tf.shape(x)[0],), 0, 2, dtype=tf.int32))

		# select random slots to overwrite
		idx_random = tf.random_uniform((tf.shape(x)[0],), 0, self.max_size, dtype=tf.int32)
		unique_idx, _ = tf.unique(tf.boolean_mask(idx_random, choice))
		update = tf.scatter_update(self.buf, unique_idx, x[:tf.shape(unique_idx)[0]])
		random_batch = tf.gather(update, idx_random)

		return tf.cond(count < self.max_size, lambda: batch, lambda: random_batch)