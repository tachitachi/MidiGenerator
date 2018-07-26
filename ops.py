import tensorflow as tf

def RightShift(inputs, shift_size=1):
	p = tf.pad(inputs, [[0, 0], [shift_size, 0], [0, 0]])
	return tf.slice(p, [0, 0, 0], [-1, tf.shape(p)[1]-shift_size, inputs.shape[2]])

def RightShift2D(inputs, shift_size=1):
	p = tf.pad(inputs, [[0, 0], [shift_size, 0]])
	return tf.slice(p, [0, 0], [-1, tf.shape(p)[1]-shift_size])