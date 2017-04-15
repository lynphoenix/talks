import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time
import os
from datetime import datetime

max_steps = 1000000
batch_size = 128
data_dir1 = '/tmp/cifar10_data'
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")



def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def train_eval():
  data_dir = os.path.join(data_dir1, 'cifar-10-batches-bin')
  images_train, labels_train = cifar10_input.distorted_inputs(
                                        data_dir=data_dir,
                                        batch_size=batch_size)
  data_dir = os.path.join(data_dir1, 'cifar-10-batches-bin')
  images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                        data_dir=data_dir,
                                        batch_size=batch_size)
  image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
  label_holder = tf.placeholder(tf.int32, [batch_size])

  logits = cifar10.inference(image_holder)
  loss = cifar10.loss(logits, label_holder)

  train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
  top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  tf.train.start_queue_runners()

  for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss], feed_dict={
        image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
      num_examples_per_step = batch_size
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)

      format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
      print (format_str % (datetime.now(), step, loss_value,
                           examples_per_sec, sec_per_batch))

  num_examples = 10000
  import math
  num_iter = in(math.ceil(num_examples / batch_size))
  true_count = 0
  total_sample_count = num_iter * batch_size
  step = 0
  while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={
        image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1
  prediction = true_count / total_sample_count
  print("precision @ 1 = %.3f" % precision)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train_eval()


if __name__ == '__main__':
  tf.app.run()
