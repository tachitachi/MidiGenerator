import tensorflow as tf
import ops

# inputs: 3D Tensor -> (batch_size, input_size, channels)
# filters: 3D Tensor -> (kernel_size, input_channels, output_channels)
def _DilatedCausalConv1d(inputs, filters, dilation_rate=1):
	kernel_size = int(filters.shape[0])
	pad_size = dilation_rate * (kernel_size - 1)
	padded = tf.pad(inputs, [[0, 0], [pad_size, 0], [0, 0]])
	return tf.nn.convolution(padded, filters, padding='VALID', dilation_rate=[dilation_rate])


def DilatedCausalConv1d(inputs, kernel_size, channels, dilation_rate=1, name='', dtype=tf.float32, use_bias=True):
	filters = tf.get_variable(name + '_Kernel', [kernel_size, inputs.shape[-1], channels], 
		initializer=tf.contrib.layers.xavier_initializer(), dtype=dtype)
	conv = _DilatedCausalConv1d(inputs, filters, dilation_rate=dilation_rate)
	if use_bias:
		bias = tf.get_variable(name + '_Bias', [1, 1, channels], initializer=tf.constant_initializer(0.0), dtype=dtype)
		conv = conv + bias
	return conv


# TODO: How to properly add bias from conditioning?
def ResidualDilationLayer(inputs, kernel_size, dilation_channels, skip_channels, dilation_rate=1, name='', dtype=tf.float32, use_bias=True):

	# input -> causal conv -> tanh
	with tf.variable_scope(name + '_filter'):
		filter_conv = DilatedCausalConv1d(inputs, kernel_size, dilation_channels, dilation_rate, name, dtype, use_bias)
		filter_conv = tf.nn.tanh(filter_conv)

	# input -> causal conv -> sigmoid
	with tf.variable_scope(name + '_gate'):
		gated_conv = DilatedCausalConv1d(inputs, kernel_size, dilation_channels, dilation_rate, name, dtype, use_bias)
		gated_conv = tf.nn.sigmoid(filter_conv)


	combined = filter_conv * gated_conv

	# 1x1 convolution
	residual = tf.layers.conv1d(combined, filters=dilation_channels, kernel_size=1, strides=1, padding='SAME')
	dense = (inputs + residual) * 0.7071067811865476


	# 1x1 convolution
	skip = tf.layers.conv1d(combined, filters=skip_channels, kernel_size=1, strides=1, padding='SAME')

	return dense, skip

def ResidualDilationLayerNC(inputs, kernel_size, dilation_channels, skip_channels, dilation_rate=1, name='', dtype=tf.float32, use_bias=True):
	x = tf.nn.relu(inputs)
	with tf.variable_scope(name + '_NC'):
		x = tf.layers.conv1d(x, filters=dilation_channels, kernel_size=kernel_size, strides=1, padding='SAME')
		x = tf.nn.relu(x)

	residual = tf.layers.conv1d(x, filters=dilation_channels, kernel_size=1, strides=1, padding='SAME')
	skip = tf.layers.conv1d(x, filters=skip_channels, kernel_size=1, strides=1, padding='SAME')

	return residual, skip



class WaveNet(object):
	def __init__(self, dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256], filter_width=2, 
		         dilation_channels=32, skip_channels=256, output_channels=128, scope='WaveNet'):

		self.dilations = dilations
		self.filter_width = filter_width
		self.dilation_channels = dilation_channels
		self.skip_channels = skip_channels
		self.output_channels = output_channels
		self.scope = scope

	def __call__(self, x, reuse=False):

		with tf.variable_scope(self.scope, reuse=reuse):
		
			skip_layers = []

			h = DilatedCausalConv1d(x, self.filter_width, channels=self.dilation_channels, dilation_rate=1, name='causal_conv')

			for i in range(len(self.dilations)):
				dilation = self.dilations[i]
				name = 'dilated_conv_{}'.format(i)
				h, skip = ResidualDilationLayer(h, kernel_size=self.filter_width, dilation_channels=self.dilation_channels, 
					skip_channels=self.skip_channels, dilation_rate=dilation, name=name)
				skip_layers.append(skip)


			total = tf.reduce_sum(skip_layers, axis=0)
			total = tf.nn.relu(total)

			total = tf.layers.conv1d(total, filters=self.skip_channels, kernel_size=1, strides=1, padding='SAME')
			total = tf.nn.relu(total)

			total = tf.layers.conv1d(total, filters=self.output_channels, kernel_size=1, strides=1, padding='SAME')

			return total



