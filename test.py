import os
import scipy.misc
import numpy as np
import pdb
import pprint
from model import DCGAN
from utils import pp, visualize, show_all_variables

import tensorflow as tf

#arguments parsers
flags = tf.app.flags

#name, defulat value, description
#below are for train and test
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")


#below are for model construction
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")

flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value with input_height [None]")

flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. ")

flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()
pp.pprint(flags.FLAGS.__flags)


