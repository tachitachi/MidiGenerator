import tensorflow as tf

def RightShift(inputs, shift_size=1):
	p = tf.pad(inputs, [[0, 0], [shift_size, 0], [0, 0]])
	return tf.slice(p, [0, 0, 0], [-1, tf.shape(p)[1]-shift_size, inputs.shape[2]])

def RightShift2D(inputs, shift_size=1):
	p = tf.pad(inputs, [[0, 0], [shift_size, 0]])
	return tf.slice(p, [0, 0], [-1, tf.shape(p)[1]-shift_size])


def ResizeEmbeddingNearestNeighbor(inputs, output_size):
	embedding_size = inputs.shape[1]
	embedding_channels = tf.shape(inputs)[2]

	# Add fake channel size of 1
	reshaped = tf.expand_dims(inputs, 3)


	resized = tf.image.resize_nearest_neighbor(reshaped, [output_size, embedding_channels])

	return tf.squeeze(resized, axis=[3])