import argparse
import tensorflow as tf
from data2 import MidiDataset
import numpy as np
from tqdm import tqdm
import os
import time
from model2 import MidiAutoencoder
import ops
import matplotlib.pyplot as plt

def main(args):

	# get datasets
	dataset = MidiDataset(args.split, args.dataset_dir)

	
	x = tf.cast(dataset.x, tf.float32)
	input_encoding = tf.placeholder(tf.float32, [None, None, None])

	batch_x = tf.expand_dims(x, 0)
	MidiDataset.to_summary('input', batch_x)


	# build models
	model = MidiAutoencoder(scope='model')
	output, encoding = model(batch_x)
	output_from_encoding = model.decode(input_encoding, reuse=True)

	output = tf.nn.sigmoid(output)
	midi = tf.round(output)

	output_from_encoding = tf.round(tf.nn.sigmoid(output_from_encoding))

	MidiDataset.to_summary('output', output)
	MidiDataset.to_summary('midi_output', midi)
	MidiDataset.to_summary('encoding', encoding)


	# set up logging

	# Gather initial summaries.
	summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

	# Merge all summaries together.
	summary_op = tf.summary.merge(list(summaries), name='summary_op') if len(summaries) > 0 else tf.no_op() 


	# create train loop
	saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
	writer = tf.summary.FileWriter(os.path.join(args.checkpoint_dir, 'eval'))

	with tf.Session() as sess:
		# Tensorflow initializations
		sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
		tf.train.start_queue_runners(sess=sess)
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))

		
		for i in tqdm(range(args.num_batches)):

			random_encoding = np.random.normal(0, 1, (1, 32, 64))

			inputs, out, m, e, t = sess.run([batch_x, output, midi, encoding, output_from_encoding], {input_encoding: random_encoding})


			print(e, np.min(e), np.max(e))
			plt.subplot(161)
			plt.imshow(inputs[0])
			plt.subplot(162)
			plt.imshow(out[0])
			plt.subplot(163)
			plt.imshow(m[0])
			plt.subplot(164)
			plt.imshow(e[0])
			plt.subplot(165)
			plt.imshow(random_encoding[0])
			plt.subplot(166)
			plt.imshow(t[0])
			plt.show()
			




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, default='test')
	parser.add_argument('--dataset_dir', type=str, default='data')
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--num_batches', type=int, default=10)
	parser.add_argument('--shuffle', type=bool, default=True)
	parser.add_argument('--checkpoint_dir', type=str, required=True)

	args = parser.parse_args()

	main(args)