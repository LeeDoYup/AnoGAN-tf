import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, show_all_variables

import tensorflow as tf

#arguments parsers
flags = tf.app.flags

#name, defulat value, description
#below are for train and test
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_integer("test_epoch", 100, "Epoch for latent mapping in anomaly detection to train [100]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("test_learning_rate", 0.001, "Learning rate for finding latent variable z [0.05]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("anomaly_test", False, "True for anomaly test in test directory, not anomaly test [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")


#below are for model construction
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("test_batch_size", 1, "The size of test batch images in anomaly detection to [1]")

flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value with input_height [None]")

flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. ")

flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("test_dir", "test_data", "Directory name to load the anomaly detstion result [test_data]")
flags.DEFINE_string("test_result_dir", "test_result", "Directory name to save the anomaly test result [test_data/test_result]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")

FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)


  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  #run_config.gpu_options.per_process_gpu_memory_fraction = 0.4


  with tf.Session(config=run_config) as sess:
    if FLAGS.dataset == 'mnist':
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          test_batch_size=FLAGS.test_batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=10,
          z_dim=FLAGS.generate_test_images,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          test_dir = FLAGS.test_dir)
    else:
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          test_batch_size=FLAGS.test_batch_size,
          sample_num=FLAGS.batch_size,
          z_dim=FLAGS.generate_test_images,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          test_dir = FLAGS.test_dir)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
    
    if FLAGS.anomaly_test:
      dcgan.anomaly_detector()
      assert len(dcgan.test_data_names) > 0
      for idx in range(len(dcgan.test_data_names)):
	test_input = np.expand_dims(dcgan.test_data[idx],axis=0)
	test_name = dcgan.test_data_names[idx]
        dcgan.train_anomaly_detector(FLAGS, test_input, test_name)

    # Below is codes for visualization
    #OPTION = 1
    #visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
