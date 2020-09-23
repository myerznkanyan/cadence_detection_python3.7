# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
# create directories
dataset_home = "train_image_14|09|2020/"
subdirs = ["train/", "test/"]
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ["Cadence/", 'noCadence/']
	for labldir in labeldirs:
		newdir = dataset_home + subdir + labldir
		makedirs(newdir, exist_ok=True)

# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.2
# copy training dataset images into subdirectories
src_directory = 'test_images'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'train/'
	if random() < val_ratio:
		dst_dir = 'test/'
	if file.startswith('Cad'):
		dst = dataset_home + dst_dir + 'Cadence/'  + file
		copyfile(src, dst)
	elif file.startswith('no'):
		dst = dataset_home + dst_dir + 'noCadence/'  + file
		copyfile(src, dst)

