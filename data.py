import mido
import numpy as np
import os
import argparse
from tqdm import tqdm

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

		out = np.argmax(np.array(list(map(lambda x: self._parse_vec(x), events))), 1).astype(np.uint16)
		basename, _ = os.path.splitext(os.path.basename(path))
		out_path = os.path.join(self.data_dir, basename)
		np.save(out_path, out)

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