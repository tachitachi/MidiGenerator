import mido
import numpy as np
import os
import argparse
from tqdm import tqdm
import tensorflow as tf
import ops

from collections import defaultdict

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

class MidiDataset(object):
	splits = {
		'train': 'train.tfrecord',
		'test': 'test.tfrecord',
	}

	notes = 128

	def __init__(self, split, data_dir):
		self.data_dir = data_dir

		self.processed_dir = os.path.join(self.data_dir, 'processed')

		if not os.path.isfile(os.path.join(self.processed_dir, MidiDataset.splits['train'])) or not os.path.isfile(os.path.join(self.processed_dir, MidiDataset.splits['test'])):
			all_files = []

			for root, dirs, files in os.walk(data_dir):
				all_files.extend(list(map(lambda x: os.path.join(root, x), files)))

			# randomly choose 20% for the test set
			train_idx = sorted(np.random.choice(np.arange(len(all_files)), int(len(all_files) * 0.8), replace=False))
			test_idx = sorted(list(set(np.arange(len(all_files))) - set(train_idx)))


			def process(all_files, idx, output_name):
				os.makedirs(self.processed_dir, exist_ok=True)

				writer = tf.python_io.TFRecordWriter(os.path.join(self.processed_dir, '%s.tfrecord' % output_name))

				with open(os.path.join(self.processed_dir, '%s.txt' % output_name), 'w') as f:
					for i in tqdm(idx):
						file = all_files[i]
						f.write('%s\n' % os.path.basename(file))
						data = np.load(file)

						shape = list(data.shape)
						notes = data.shape[1]
						length = data.shape[0]

						feature = {
							'data': _int64_feature(data.reshape([-1])),
							'notes': _int64_feature([notes]),
							'length': _int64_feature([length]),
						}

						example = tf.train.Example(features=tf.train.Features(feature=feature))
						writer.write(example.SerializeToString())

			process(all_files, train_idx, 'train')
			process(all_files, test_idx, 'test')

		self.x, self.y = self._load(split)

	def _load(self, split):
    # sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))

		def _parse_function(example_proto):
			features = {
				'data': tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
				'notes': tf.FixedLenFeature([1], dtype=tf.int64),
				'length': tf.FixedLenFeature([1], dtype=tf.int64),
			}
			parsed_features = tf.parse_single_example(example_proto, features)

			#return tf.reshape(parsed_features['data'], parsed_features['shape'])
			return tf.reshape(parsed_features['data'], [-1, MidiDataset.notes])

		dataset = tf.data.TFRecordDataset(os.path.join(self.processed_dir, MidiDataset.splits[split]))
		dataset = dataset.map(_parse_function)

		dataset = dataset.repeat()

		iterator = dataset.make_initializable_iterator()
		next_y = iterator.get_next()

		next_x = tf.reshape(ops.RightShift2D(next_y), [-1, MidiDataset.notes])

		print(next_x, next_y)

		tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

		return next_x, next_y

	@staticmethod
	def to_summary(name, x):
		tf.summary.image(name, tf.cast(tf.expand_dims(tf.transpose(x, [0, 2, 1]), 3), tf.float32))
						
				
class Note(object):
	note = 0
	on = False
	time = 0


class MidiParse(object):
	note_on = 0
	note_off = 1
	velocity = 2
	time = 3

	def __init__(self, data_dir, notes=128, step_size=16):
		self.data_dir = data_dir
		self.notes = notes
		self.step_size = step_size

		self.ticks_per_beat = 480
		self.tempo = 500000

		self.unit_ticks = int(mido.second2tick(0.01, self.ticks_per_beat, self.tempo))
		self.ticks_per_second = int(mido.second2tick(1, self.ticks_per_beat, self.tempo))

		if not os.path.isdir(self.data_dir):
			os.makedirs(self.data_dir)

	def _parse_vec(self, event):
		shape = self.notes + self.notes + self.velocities + self.times
		vec = np.zeros(shape)

		if event[0] == MidiParse.note_on:
			vec[event[1]] = 1
		if event[0] == MidiParse.note_off:
			vec[self.notes + event[1]] = 1
		if event[0] == MidiParse.velocity:
			vec[self.notes + self.notes + event[1]] = 1
		if event[0] == MidiParse.time:
			vec[self.notes + self.notes + self.velocities + event[1]] = 1

		return vec

	def update_timing(self, midi):

		self.ticks_per_beat = midi.ticks_per_beat

		meta_track = midi.tracks[0]

		for msg in meta_track:
			if hasattr(msg, 'tempo'):
				self.tempo = msg.tempo

		self.unit_ticks = int(mido.second2tick(0.01, self.ticks_per_beat, self.tempo))
		self.ticks_per_second = int(mido.second2tick(1, self.ticks_per_beat, self.tempo))

	def ticksToUnits(self, time):
		ticks_per_unit = self.ticks_per_second / self.step_size

		return np.round(time / ticks_per_unit).astype(np.int32)

	def updateEncoding(self, note, start, end, encoding):
		start_unit = self.ticksToUnits(start)
		end_unit = self.ticksToUnits(end)
		encoding[start_unit:end_unit,note] = 1

	def parse(self, path):
		midi = mido.MidiFile(path)

		self.update_timing(midi)

		#print('ticks per second: %d' % self.ticks_per_second)
		ticks_per_unit = self.ticks_per_second / self.step_size
		#print('ticks per unit: %d' % ticks_per_unit)

		unit = 1 / self.step_size

		song_length = np.ceil(midi.length * self.step_size).astype(np.int32)

		if midi.length < 30:
			return

		notes_in_use = defaultdict(Note)

		# Create music box encoding
		encoding = np.zeros((song_length, self.notes), dtype=np.uint8)

		# note_on velocity=0, and note_off are equivalent
		tracks = mido.merge_tracks(midi.tracks[1:])
		total_time = 0
		for msg in tracks:
			if msg.is_meta or not hasattr(msg, 'note'):
				continue
			if msg.type == 'note_on':
				note = msg.note
				time = msg.time
				velocity = msg.velocity

				total_time += time

				if velocity == 0:
					prev_time = notes_in_use[note].time
					#print('%d from %d - %d' % (note, self.ticksToUnits(prev_time), self.ticksToUnits(total_time)))
					self.updateEncoding(note, prev_time, total_time, encoding)
					
				else:
					notes_in_use[note].note = note
					notes_in_use[note].on = True
					notes_in_use[note].time = total_time


			if msg.type == 'note_off':
				note = msg.note
				time = msg.time

				total_time += time

				prev_time = notes_in_use[note].time
				#print('%d from %d - %d' % (note, self.ticksToUnits(prev_time), self.ticksToUnits(total_time)))
				self.updateEncoding(note, prev_time, total_time, encoding)


		#print(encoding[:,60])
		#print(np.sum(encoding))

		basename, _ = os.path.splitext(os.path.basename(path))
		out_path = os.path.join(self.data_dir, basename)
		np.save(out_path, encoding)

def create_batch(tensors, batch_size=32, shuffle=False, queue_size=10000, min_queue_size=5000, num_threads=1):
    # Must initialize tf.GraphKeys.QUEUE_RUNNERS
    # tf.train.start_queue_runners(sess=sess)
    if shuffle:
        return tf.train.shuffle_batch(tensors, batch_size=batch_size, capacity=queue_size, min_after_dequeue=min_queue_size, num_threads=num_threads)
    else:
        return tf.train.batch(tensors, batch_size=batch_size, capacity=queue_size, num_threads=num_threads)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data')
	parser.add_argument('--output_dir', type=str, default='data/parsed')

	args = parser.parse_args()

	parser = MidiParse(args.output_dir)

	for root, dirs, files in os.walk(args.data_dir):
		for file in tqdm(files):
			if file.endswith('.mid') or file.endswith('.midi'):
				#parser.parse(os.path.join(root, file))
				try:
					parser.parse(os.path.join(root, file))
				except:
					print('failed to parse midi file %s' % file)