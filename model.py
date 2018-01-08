from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', test_batch_size = 1, checkpoint_dir=None, sample_dir=None, test_dir = None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.test_batch_size = test_batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim
    
    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.test_dir = os.path.join('./',test_dir)

    if self.dataset_name == 'mnist':
      self.data_X, self.data_y = self.load_mnist()
      self.c_dim = self.data_X[0].shape[-1]
    else:
      self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
      #Check number of channels
      imreadImg = imread(self.data[0])
      if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
        self.c_dim = imread(self.data[0]).shape[-1]
      else:
        self.c_dim = 1

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    #placeholders
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop: #for training
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else: #for test
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    #Construct Generator and Discriminators
    self.G                  = self.generator(self.z, self.y)
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)

    self.sampler            = self._sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    
    #summary op.
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    #Create Loss Functions
    def sigmoid_cross_entropy_with_logits(x, y):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    #summary op.
    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    
    #total loss
    self.d_loss = self.d_loss_real + self.d_loss_fake


    #summary op.
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)

    tf.global_variables_initializer().run()

    #summary_op: merge summary
    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

    #Create Tensorboard
    self.writer = SummaryWriter("./logs", self.sess.graph)

    #Create Sample Benchmarks for monitoring of train results: use same random noises and real-images
    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
    if config.dataset == 'mnist':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]
    else:
      sample_files = self.data[0:self.sample_num] #name_list
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]

      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.dataset == 'mnist':
      batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size #config.train_size: default is np.inf
      sample_feed_dict = {self.z: sample_z, self.inputs: sample_inputs, self.y:sample_labels}
    else:      
      batch_idxs = min(len(self.data), config.train_size) // config.batch_size
      sample_feed_dict = {self.z: sample_z, self.inputs: sample_inputs}

    for epoch in xrange(config.epoch):
      for idx in xrange(0, batch_idxs):

        #Prepare batch data for learning
        if config.dataset == 'mnist':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [ get_image(batch_file, input_height=self.input_height, input_width=self.input_width, resize_height=self.output_height, resize_width=self.output_width, crop=self.crop, grayscale=self.grayscale) for batch_file in batch_files]
          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)

        #Prepare batch random noises for learning
        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

        #Make feed dictionary
        if config.dataset == 'mnist':
          d_feed_dict = {self.inputs: batch_images, self.z: batch_z, self.y: batch_labels}
          d_fake_feed_dict  = {self.z: batch_z, self.y: batch_labels}
          d_real_feed_dict  = {self.inputs: batch_images,  self.y: batch_labels}
          g_feed_dict = {self.z: batch_z, self.y: batch_labels}

        else:
          d_feed_dict = {self.inputs: batch_images, self.z: batch_z}
          d_fake_feed_dict  = {self.z: batch_z}
          d_real_feed_dict  = {self.inputs: batch_images}
          g_feed_dict = {self.z:batch_z}

        #Run Optimization and Summary Operation of Discriminator
        _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict = d_feed_dict)
        self.writer.add_summary(summary_str, counter)

        #Run Optimization and Summary Operation of Generator
        _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict = g_feed_dict)
        self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict = g_feed_dict)
        self.writer.add_summary(summary_str, counter)

        # Calculate Loss Values of Discriminator and Generator

        errD_fake = self.d_loss_fake.eval(feed_dict = d_fake_feed_dict)
        errD_real = self.d_loss_real.eval(feed_dict = d_real_feed_dict)
        errG = self.g_loss.eval(feed_dict = g_feed_dict)

        counter += 1


        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))

        if np.mod(counter, 100) == 1:
          samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss], feed_dict = sample_feed_dict)
          save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
          print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, y=None, reuse=False, batch_size = None):
    if batch_size == None: batch_size = self.batch_size
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:
        h0 = leak_relu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = leak_relu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = leak_relu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = leak_relu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        #h4 = linear(tf.contrib.layers.flatten(h3),1,'d_h4_lin')
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4
      else:
        yb = tf.reshape(y, [batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)

        h0 = leak_relu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = leak_relu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [batch_size, -1]) #flatten
        h1 = concat([h1, y], 1)
        
        h2 = leak_relu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        h2 = concat([h2, y], 1)

        h3 = linear(h2, 1, 'd_h3_lin')
        
        return tf.nn.sigmoid(h3), h3

  def feature_match_layer(self, image, y=None, reuse=False, batch_size = None):
      if batch_size == None: batch_size = self.batch_size
      with tf.variable_scope("discriminator") as scope:
        if reuse:
          scope.reuse_variables()

        if not self.y_dim:
          h0 = leak_relu(conv2d(image, self.df_dim, name='d_h0_conv'))
          h1 = leak_relu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
          h2 = leak_relu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
          h3 = leak_relu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
          return h3

        else:
          yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
          x = conv_cond_concat(image, yb)

          h0 = leak_relu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
          h0 = conv_cond_concat(h0, yb)

          h1 = leak_relu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
          h1 = tf.reshape(h1, [batch_size, -1]) #flatten
          h1 = concat([h1, y], 1)
          
          h2 = leak_relu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
          return h2

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(
            self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(
            deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def _sampler(self, z, y=None, batch_size = None):
    if batch_size == None:
      batch_size = self.batch_size

    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(
            deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def get_test_data(self):
    self.test_data_names = glob(self.test_dir+'/*.*')
    batch = [get_image(name, input_height=self.input_height, input_width = self.input_width, resize_height = self.output_height, resize_width = self.output_width, crop = self.crop, grayscale=self.grayscale) for name in self.test_data_names]

    if self.grayscale:
      batch_images = np.array(batch).astype(np.float32)[:,:,:,None]
    else:
      batch_images = np.array(batch).astype(np.float32)
    #print np.shape(batch_images)
    self.test_data = batch_images
    print "[*] test data for anomaly detection is loaded"


  def anomaly_detector(self, ano_para=0.1, dis_loss='feature'):
      self.get_test_data()

    #with variable_scope("anomaly_detector"):
      if self.y_dim:
        self.ano_y = tf.placeholder(tf.float32, [self.test_batch_size, self.y_dim], name='y')
      else:
        self.y = None

      if self.crop: 
        image_dims = [self.output_height, self.output_width, self.c_dim]
      else: #for test
        image_dims = [self.input_height, self.input_width, self.c_dim]

      self.test_inputs = tf.placeholder(tf.float32, [1] + image_dims, name='test_images')
      test_inputs = self.test_inputs

      self.ano_z = tf.get_variable('ano_z', shape = [1, self.z_dim], dtype = tf.float32, 
        initializer = tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32))

      self.ano_y = None

      self.ano_G = self._sampler(self.ano_z, self.ano_y, batch_size=1)

      self.res_loss = tf.reduce_mean(
        tf.reduce_sum(tf.abs(tf.subtract(test_inputs, self.ano_G))))

      #Create Anomaly Score 
      if dis_loss == 'feature': # if discrimination loss with feature matching (same with paper)
        dis_f_z, dis_f_input = self.feature_match_layer(self.ano_G, self.ano_y, reuse=True,batch_size=1), self.feature_match_layer(test_inputs, self.ano_y, reuse=True, batch_size=1)
        self.dis_loss = tf.reduce_mean(
          tf.reduce_sum(tf.abs(tf.subtract(dis_f_z, dis_f_input))))
      else: # if dis_loss with original generator's loss in  DCGAN
        test_D, test_D_logits_ = self.discriminator(ano_G, ano_y, reuse=True, batch_size=1)
        self.dis_loss = tf.reduce_mean(
          sigmoid_cross_entropy_with_logits(test_D_logits_, tf.ones_like(test_D)))

      self.anomaly_score = (1. - ano_para)* self.res_loss + ano_para* self.dis_loss

      t_vars = tf.trainable_variables()
      self.z_vars = [var for var in t_vars if 'ano_z' in var.name]
      print test_inputs, self.ano_G, dis_f_z, dis_f_input

  def train_anomaly_detector(self, config, test_data, test_data_name):
    print "Filename: ", test_data_name, "Anomaly is detecting"
    print np.shape(test_data)
    #self.sess.run(self.ano_z.initializer)
    z_optim = tf.train.AdamOptimizer(config.test_learning_rate, beta1=config.beta1) \
          .minimize(self.anomaly_score, var_list = self.z_vars)
    initialize_uninitialized(self.sess)

    for epoch in range(config.test_epoch):
      if not self.y_dim:
        feed_dict = {self.test_inputs: test_data} 
      else:
        print "Not yet prepared anomaly detection model of MNIST dataset"
        feed_dict = {}
      _, ano_score, res_loss = self.sess.run([z_optim, self.anomaly_score, self.res_loss], feed_dict = feed_dict)
      
      
      print("Epoch: [{:02d}], anomaly score: {:.8f}, res loss: {:.8f}"\
        .format(epoch, ano_score, res_loss))
      save_epoch = [0, config.test_epoch/2, config.test_epoch-1]
      if epoch in save_epoch:
        samples = self.sess.run(self.ano_G)
        errors = samples-test_data

        print np.shape(samples)
        samples = np.squeeze(samples)
        samples = (np.array(samples)+1)*127.5
	if not self.grayscale:
          errors = np.mean(np.squeeze(errors),axis=2)
        errors = (np.array(errors)+1)*127.5

        _path = './test_data/'
        path = os.path.join(_path, config.test_result_dir)
        if not os.path.isdir(path):
          os.mkdir(path)
        filename = ['AD_'+str(epoch)+'_'+test_data_name.split('/')[-1], 'AD_error_'+str(epoch)+'_'+test_data_name.split('/')[-1]]

        scipy.misc.imsave(os.path.join(path,filename[0]),samples)
        scipy.misc.imsave(os.path.join(path,filename[1]),errors)
        #np.save('./{}/test_error_{}_{:02d}.png'.format(config.test_dir, test_data_name, epoch), errors)

  def load_mnist(self):
    data_dir = os.path.join("./data", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    #Make one-hot
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    index_offset = np.arange(len(y)) * self.y_dim
    y_vec.flat[index_offset + y.ravel()] = 1
    
    return X/255.,y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
