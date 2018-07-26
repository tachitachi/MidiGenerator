import argparse
import tensorflow as tf
from data2 import MidiDataset
import numpy as np
from tqdm import tqdm
import os
import time
from model2 import WaveNet
import ops

def main(args):

	# get datasets
	dataset = MidiDataset(args.split, args.dataset_dir)

	x = tf.placeholder(tf.float32, [None, None, MidiDataset.notes])
	MidiDataset.to_summary('input', x)

	#batch_x = data2.create_batch([x], batch_size=args.batch_size)
	batch_x = tf.expand_dims(tf.cast(dataset.x, tf.float32), 0)
	batch_y = tf.expand_dims(tf.cast(dataset.y, tf.float32), 0)
	#MidiDataset.to_summary('input', batch_x)
	#MidiDataset.to_summary('gt_output', batch_y)


	# build models

	wavenet = WaveNet(scope='model')

	generated = wavenet(x)
	midi = tf.round(tf.nn.sigmoid(generated))
	MidiDataset.to_summary('generated', tf.nn.sigmoid(generated))
	MidiDataset.to_summary('generated_midi', tf.round(tf.nn.sigmoid(generated)))


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

		

		output, output_ = sess.run([batch_x, batch_y])
		output[:,args.base_length:,:] = 0

		for i in tqdm(range(args.total_length)):

			if i % args.log_every_n_steps == 0:
				summary, m, g = sess.run([summary_op, midi, generated], {x: output})
				writer.add_summary(summary, i+1)
			else:
				m, g = sess.run([midi, generated], {x: output})

			output[:, args.base_length+i+1,:] = m[:, args.base_length+i,:] 
			




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, default='train')
	parser.add_argument('--dataset_dir', type=str, default='data')
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--total_length', type=int, default=3000)
	parser.add_argument('--base_length', type=int, default=50)
	parser.add_argument('--log_every_n_steps', type=int, default=100)
	parser.add_argument('--shuffle', type=bool, default=True)
	parser.add_argument('--checkpoint_dir', type=str, required=True)

	args = parser.parse_args()

	main(args)