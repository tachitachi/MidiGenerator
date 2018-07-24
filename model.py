import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell


class SequenceGenerator(object):
	def __init__(self, vocab_size, embedding_size=256, lstm_size=256, fc_size=256, scope='SequenceGenerator', reuse=False):
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.lstm_size = lstm_size
		self.fc_size = fc_size

		with tf.variable_scope(scope, reuse=reuse):
			self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))

			self.lstm = BasicLSTMCell(self.lstm_size)

			self.fc1_w = tf.get_variable("fc1/Weight", shape=[self.lstm_size, self.fc_size],
	           initializer=tf.contrib.layers.xavier_initializer())
			self.fc1_b = tf.get_variable("fc1/Bias", shape=[self.fc_size])

			self.fc2_w = tf.get_variable("fc2/Weight", shape=[self.fc_size, self.vocab_size],
	           initializer=tf.contrib.layers.xavier_initializer())
			self.fc2_b = tf.get_variable("fc2/Bias", shape=[self.vocab_size])

	def __call__(self, x):
		# Always assumes everything is max length, for now
		embeddings = tf.nn.embedding_lookup(self.embeddings, x)

		lstm_outputs, lstm_state = tf.nn.dynamic_rnn(self.lstm, embeddings, dtype=tf.float32)
		c, h = lstm_state
		net = c
		net = tf.nn.relu(tf.matmul(net, self.fc1_w) + self.fc1_b)
		net = tf.matmul(net, self.fc2_w) + self.fc2_b

		return net

	def decode(self, x):
		return tf.nn.embedding_lookup(self.embeddings, x)

if __name__ == '__main__':

	generator = SequenceGenerator(388)

	x = tf.placeholder(tf.int64, [None, None])

	out = generator(x)
	out = generator(x)
	print(out)

	word = tf.placeholder(tf.int64, [None])
	output_word = generator.decode(word)
	print(output_word)