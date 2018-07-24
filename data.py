import mido
import numpy as np
import os
import argparse
from tqdm import tqdm
import tensorflow as tf

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

	vec_size = 388

	def __init__(self, split, data_dir, sequence_length=500, shuffle_buffer=10000):
		self.data_dir = data_dir
		self.sequence_length = sequence_length
		self.shuffle_buffer = shuffle_buffer

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
						for idx in range(0, data.shape[0] - self.sequence_length):
							x = data[idx:idx+self.sequence_length]
							y = data[idx+self.sequence_length]

							feature = {
								'x': _int64_feature(x),
								'y': _int64_feature([y])
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
				'x': tf.FixedLenFeature([self.sequence_length], dtype=tf.int64),
				'y': tf.FixedLenFeature([1], dtype=tf.int64),
			}
			parsed_features = tf.parse_single_example(example_proto, features)

			return parsed_features['x'], parsed_features['y']

		dataset = tf.data.TFRecordDataset(os.path.join(self.processed_dir, MidiDataset.splits[split]))
		dataset = dataset.map(_parse_function)
		if self.shuffle_buffer:
			dataset = dataset.shuffle(buffer_size=self.shuffle_buffer)
		dataset = dataset.repeat()

		iterator = dataset.make_initializable_iterator()
		next_x, next_y = iterator.get_next()

		tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

		return next_x, next_y
						
				



class MidiParse(object):
	note_on = 0
	note_off = 1
	velocity = 2
	time = 3

	def __init__(self, data_dir, notes=128, velocities=32, times=100):
		self.data_dir = data_dir
		self.notes = notes
		self.velocities = velocities
		self.times = times

		self.velocity_div = 128 // self.velocities

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

	def _add_time(self, time, events):

		def append(t, events):
			events.append((MidiParse.time, min(t, self.times - 1)))

		while time > 0:
			if time > self.ticks_per_second:
				append(int(1.0 * self.times), events)
				time = time - self.ticks_per_second
			else:
				append(int(round(time / self.ticks_per_second, 2) * self.times), events)
				return

	def parse(self, path):
		midi = mido.MidiFile(path)

		self.update_timing(midi)


		# note_on velocity=0, and note_off are equivalent

		last_velocity = 64

		events = []

		total_time = 0

		tracks = mido.merge_tracks(midi.tracks[1:])
		for msg in tracks:
			if msg.is_meta or not hasattr(msg, 'note'):
				continue
			if msg.type == 'note_on':
				note = msg.note
				velocity = msg.velocity
				time = msg.time

				self._add_time(time, events)

				if velocity != last_velocity and velocity != 0:
					events.append((MidiParse.velocity, velocity // self.velocity_div))
					last_velocity = velocity

				if velocity == 0:
					events.append((MidiParse.note_off, note))
				else:
					events.append((MidiParse.note_on, note))


			if msg.type == 'note_off':
				note = msg.note
				time = msg.time

				self._add_time(time, events)

				events.append((MidiParse.note_off, note))

			total_time += time

		if total_time / self.ticks_per_second < 30:
			return

		out = np.argmax(np.array(list(map(lambda x: self._parse_vec(x), events))), 1).astype(np.uint16)
		basename, _ = os.path.splitext(os.path.basename(path))
		out_path = os.path.join(self.data_dir, basename)
		np.save(out_path, out)

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
				try:
					parser.parse(os.path.join(root, file))
				except:
					print('failed to parse midi file %s' % file)