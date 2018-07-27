import argparse
import tensorflow as tf
from data2 import MidiDataset, create_batch
import numpy as np
from tqdm import tqdm
import os
import time
from model2 import WaveNet, WavePatch

def main(args):

	# get datasets
	dataset = MidiDataset(args.split, args.dataset_dir)

	x = tf.cast(dataset.x, tf.float32)
	y = tf.cast(dataset.y, tf.float32)

	batch_x, batch_y = create_batch([x, y], batch_size=args.batch_size)
	#batch_x = tf.expand_dims(x, 0)
	#batch_y = tf.expand_dims(y, 0)
	MidiDataset.to_summary('input', batch_x)
	MidiDataset.to_summary('gt_output', batch_y)

	noise = tf.random_normal((args.batch_size, int(y.shape[0]), 1), dtype=tf.float32)
	print(noise, y, batch_y)


	# build models

	dilations = [1, 1, 2, 4, 8, 16, 32, 64, 128]

	wavenet = WaveNet(dilations=dilations, activation_fn=tf.nn.sigmoid, scope='model/generator')

	generated = wavenet(noise)
	#MidiDataset.to_summary('generated', tf.nn.softmax(generated))
	MidiDataset.to_summary('generated', generated)
	MidiDataset.to_summary('output_midi', tf.round(generated))

	wavepatch = WavePatch(dilations=dilations, pool_stride=64, scope='model/discriminator')

	preds = wavepatch(tf.concat([batch_y, generated], axis=0))

	labels_real = tf.zeros((args.batch_size, 1, 1), dtype=tf.float32)
	labels_fake = tf.ones((args.batch_size, 1, 1), dtype=tf.float32)

	labels = tf.tile(tf.concat([labels_real, labels_fake], axis=0), [1, tf.shape(preds)[1], tf.shape(preds)[2]])


	# create loss functions

	discriminator_loss = tf.losses.sigmoid_cross_entropy(labels, preds, scope='discriminator_loss')
	generator_loss = tf.losses.sigmoid_cross_entropy(1 - labels, preds, scope='generator_loss')

	total_loss = discriminator_loss + generator_loss

	discriminator_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.round(tf.nn.sigmoid(preds)), tf.int32), tf.cast(labels, tf.int32)), tf.float32))
	tf.summary.scalar('discriminator_accuracy', discriminator_accuracy)


	#distribution_loss = tf.losses.softmax_cross_entropy(probs, generated)
	#distribution_loss = tf.losses.sigmoid_cross_entropy(batch_y, generated)
	#count_loss = tf.losses.mean_squared_error(num_notes, counts)

	#total_loss = distribution_loss + count_loss
	

	inc_global_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
	tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, inc_global_step)

	discriminator_vars = var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/discriminator')
	generator_vars = var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/generator')

	optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=0.5)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_discriminator = optimizer.minimize(discriminator_loss, var_list=discriminator_vars)
		train_generator = optimizer.minimize(generator_loss, var_list=generator_vars)

		train_tensor = tf.group([train_discriminator, train_generator])

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