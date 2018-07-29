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
		         dilation_channels=32, skip_channels=256, output_channels=128, activation_fn=None, scope='WaveNet'):

		self.dilations = dilations
		self.filter_width = filter_width
		self.dilation_channels = dilation_channels
		self.skip_channels = skip_channels
		self.output_channels = output_channels
		self.activation_fn = activation_fn
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

			output = tf.layers.conv1d(total, filters=self.output_channels, kernel_size=1, strides=1, padding='SAME')
			#note_count = tf.squeeze(tf.layers.conv1d(total, filters=1, kernel_size=1, strides=1, padding='SAME'), 2)

			if self.activation_fn is not None:
				output = self.activation_fn(output)

			return output




class WavePatch(object):
	def __init__(self, dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256], filter_width=2, 
		         dilation_channels=32, skip_channels=256, pool_stride=128, scope='WavePatch'):

		self.dilations = dilations
		self.filter_width = filter_width
		self.dilation_channels = dilation_channels
		self.skip_channels = skip_channels
		self.pool_stride = pool_stride
		self.scope = scope

	def __call__(self, x, reuse=False):

		with tf.variable_scope(self.scope, reuse=reuse):
		
			skip_layers = []

			h, _ = ResidualDilationLayerNC(x, kernel_size=self.filter_width, dilation_channels=self.dilation_channels, 
					skip_channels=self.skip_channels, dilation_rate=1, name='nc_conv')

			for i in range(len(self.dilations)):
				dilation = self.dilations[i]
				name = 'dilated_conv_{}'.format(i)
				h, skip = ResidualDilationLayerNC(h, kernel_size=self.filter_width, dilation_channels=self.dilation_channels, 
					skip_channels=self.skip_channels, dilation_rate=dilation, name=name)
				skip_layers.append(skip)


			total = tf.reduce_sum(skip_layers, axis=0)
			#total = tf.nn.relu(total)

			#total = tf.layers.conv1d(total, filters=self.skip_channels, kernel_size=1, strides=1, padding='SAME')
			#total = tf.nn.relu(total)

			total = tf.layers.conv1d(total, filters=1, kernel_size=1, strides=1, padding='SAME')
			
			logits = tf.nn.pool(total, window_shape=(self.pool_stride,), strides=(self.pool_stride,), pooling_type='AVG', padding='VALID')

			return logits




class MidiAutoencoder(object):
	def __init__(self, ndf=64, pool_stride=16, output_channels=128, activation_fn=tf.nn.relu, scope='MidiAutoencoder'):

		self.ndf = ndf
		self.pool_stride = pool_stride
		self.output_channels = output_channels
		self.activation_fn = activation_fn
		self.scope = scope

	def __call__(self, x, reuse=False):

		with tf.variable_scope(self.scope, reuse=reuse):

			layers = {}

			with tf.variable_scope('encoder'):

				net = x
			
				# 512 -> 512
				net = tf.layers.conv1d(net, filters=self.ndf, kernel_size=7, strides=1, padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)
				layers['conv1'] = net

				# 512 -> 256
				net = tf.layers.conv1d(net, filters=self.ndf * 2, kernel_size=5, strides=2, padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)
				layers['conv2'] = net

				# 256 -> 128
				net = tf.layers.conv1d(net, filters=self.ndf * 4, kernel_size=3, strides=2, padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)
				layers['conv3'] = net

				# 128 -> 64
				net = tf.layers.conv1d(net, filters=self.ndf * 8, kernel_size=3, strides=2, padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)
				layers['conv4'] = net

				# 64 -> 64
				net = tf.layers.conv1d(net, filters=self.ndf * 8, kernel_size=3, strides=1, padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)
				layers['conv5'] = net

				# 64 -> 64
				net = tf.layers.conv1d(net, filters=self.ndf * 8, kernel_size=3, strides=1, padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)
				layers['conv6'] = net


				# 64 -> 64
				#net = tf.layers.conv1d(layers['conv4'] + net, filters=self.ndf * 8, kernel_size=3, strides=1, padding='SAME')
				net = tf.layers.conv1d(net, filters=self.ndf * 8, kernel_size=3, strides=1, padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)
				layers['conv7'] = net

				# 64 -> 64
				net = tf.layers.conv1d(net, filters=self.ndf * 8, kernel_size=3, strides=1, padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)
				layers['conv8'] = net

				net = tf.transpose(net, [0, 2, 1])
				encoding = tf.nn.pool(net, window_shape=(self.pool_stride,), strides=(self.pool_stride,), pooling_type='AVG', padding='VALID')


			with tf.variable_scope('decoder'):


				#net = tf.expand_dims(layers['conv6'] + net, 2)
				expanded_decoding = ops.ResizeEmbeddingNearestNeighbor(encoding, self.pool_stride * tf.shape(encoding)[1])

				expanded_decoding = tf.transpose(expanded_decoding, [0, 2, 1])

				expanded_decoding.set_shape([None, None, self.ndf * 8])

				net = tf.expand_dims(expanded_decoding, 2)

				# 64 -> 128
				net = tf.layers.conv2d_transpose(net, filters=self.ndf * 4, kernel_size=(3,1), strides=(2,1), padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)
				layers['conv9'] = net

				# 128 -> 256
				#net = tf.layers.conv2d_transpose(tf.expand_dims(layers['conv3'], 2) + net, filters=self.ndf * 2, kernel_size=(3,1), strides=(2,1), padding='SAME')
				net = tf.layers.conv2d_transpose(net, filters=self.ndf * 2, kernel_size=(3,1), strides=(2,1), padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)
				layers['conv10'] = net

				# 512 -> 512
				#net = tf.layers.conv2d_transpose(tf.expand_dims(layers['conv2'], 2) + net, filters=self.ndf, kernel_size=(5,1), strides=(2,1), padding='SAME')
				net = tf.layers.conv2d_transpose(net, filters=self.ndf, kernel_size=(5,1), strides=(2,1), padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)
				layers['conv12'] = net

				# 512 -> 128
				net = tf.layers.conv2d_transpose(net, filters=self.output_channels, kernel_size=(7,1), strides=(1,1), padding='SAME')
				layers['conv13'] = net

				output = tf.squeeze(net, 2)


			return output, encoding

	def decode(self, x, reuse=False):
		with tf.variable_scope(self.scope, reuse=reuse):

			with tf.variable_scope('decoder'):

				encoding = x

				#net = tf.expand_dims(layers['conv6'] + net, 2)
				expanded_decoding = ops.ResizeEmbeddingNearestNeighbor(encoding, self.pool_stride * tf.shape(encoding)[1])

				expanded_decoding = tf.transpose(expanded_decoding, [0, 2, 1])

				expanded_decoding.set_shape([None, None, self.ndf * 8])

				net = tf.expand_dims(expanded_decoding, 2)

				# 64 -> 128
				net = tf.layers.conv2d_transpose(net, filters=self.ndf * 4, kernel_size=(3,1), strides=(2,1), padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)

				# 128 -> 256
				#net = tf.layers.conv2d_transpose(tf.expand_dims(layers['conv3'], 2) + net, filters=self.ndf * 2, kernel_size=(3,1), strides=(2,1), padding='SAME')
				net = tf.layers.conv2d_transpose(net, filters=self.ndf * 2, kernel_size=(3,1), strides=(2,1), padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)

				# 512 -> 512
				#net = tf.layers.conv2d_transpose(tf.expand_dims(layers['conv2'], 2) + net, filters=self.ndf, kernel_size=(5,1), strides=(2,1), padding='SAME')
				net = tf.layers.conv2d_transpose(net, filters=self.ndf, kernel_size=(5,1), strides=(2,1), padding='SAME')
				net = tf.contrib.layers.layer_norm(net, activation_fn=self.activation_fn)

				# 512 -> 128
				net = tf.layers.conv2d_transpose(net, filters=self.output_channels, kernel_size=(7,1), strides=(1,1), padding='SAME')

				output = tf.squeeze(net, 2)

				return output