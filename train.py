import argparse
import tensorflow as tf
import data
import numpy as np
from tqdm import tqdm
import os
import time
import model

def main(args):

	# get datasets
	dataset = data.MidiDataset(args.split, args.dataset_dir)

	x, y = dataset.x, dataset.y

	batch_x, batch_y = data.create_batch([x, y], batch_size=args.batch_size)


	# build models

	generator = model.SequenceGenerator(dataset.vec_size, scope='model')

	preds = generator(batch_x)
	words = tf.nn.softmax(preds)

	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, 1), batch_y), tf.float32))
	tf.summary.scalar('accuracy', accuracy)

	# create loss functions

	total_loss = tf.losses.sparse_softmax_cross_entropy(batch_y, preds)
	

	inc_global_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
	tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, inc_global_step)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_tensor = tf.train.AdamOptimizer(args.learning_rate).minimize(total_loss)

		# Set up train op to return loss
		with tf.control_dependencies([train_tensor]):
			train_op = tf.identity(total_loss, name='train_op')




	# set up logging

	# Gather initial summaries.
	summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

	# Add summaries for losses.
	for loss in tf.get_collection(tf.GraphKeys.LOSSES):
		summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

	# Add summaries for variables.
	for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
		summaries.add(tf.summary.histogram(variable.op.name, variable))

	# Merge all summaries together.
	summary_op = tf.summary.merge(list(summaries), name='summary_op') if len(summaries) > 0 else tf.no_op() 


	# create train loop

	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)

	saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
	checkpoint_path = os.path.join(args.output_dir, 'model.ckpt')
	writer = tf.summary.FileWriter(args.output_dir)

	with tf.Session() as sess:
		# Tensorflow initializations
		sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
		tf.train.start_queue_runners(sess=sess)
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		last_log_time = 0
		last_save_time = 0
		for i in tqdm(range(args.num_batches)):

			if last_log_time < time.time() - args.log_every_n_seconds:
				last_log_time = time.time()
				summary, loss_val, global_step = sess.run([summary_op, train_op, tf.train.get_global_step()])
				writer.add_summary(summary, global_step)
				writer.flush()
			else:
				loss_val, global_step = sess.run([train_op, tf.train.get_global_step()])

			if last_save_time < time.time() - args.save_every_n_seconds:
				last_save_time = time.time()
				saver.save(sess, checkpoint_path, global_step=global_step)

		saver.save(sess, checkpoint_path, global_step=args.num_batches)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, default='train')
	parser.add_argument('--dataset_dir', type=str, default='data')
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--num_batches', type=int, default=100000)
	parser.add_argument('--shuffle', type=bool, default=True)
	parser.add_argument('--output_dir', type=str, default='output/%d' % int(time.time() * 1000))
	parser.add_argument('--log_every_n_seconds', type=int, default=30)
	parser.add_argument('--save_every_n_seconds', type=int, default=300)
	parser.add_argument('--learning_rate', type=float, default=1e-4)
	parser.add_argument('--beta1', type=float, default=0.9)
	parser.add_argument('--beta2', type=float, default=0.99)
	parser.add_argument('--epsilon', type=float, default=1e-8)

	args = parser.parse_args()

	main(args)