"""
该文件是在整个数据集上进行整体评估, 是以60000张图片作为数据的评估
"""

import os

import tensorflow as tf

import cfg
from train_densenet import CifarData


def input_fn_test(dataset_params):
  data_dir = dataset_params['data_path']
  batch_size = cfg.NUM_IMAGES['validation']

  cifar_dataset = CifarData()
  filenames = cifar_dataset.get_filenames(False, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, cfg.RECORD_BYTES)
  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.apply(tf.contrib.data.map_and_batch(
    lambda value: cifar_dataset.parse_record(value, False),
    batch_size=batch_size,
    num_parallel_batches=1,
    drop_remainder=False))

  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  return dataset


def cifar_dataset_test(dataset_params):
  dataset = input_fn_test(dataset_params)
  return dataset


def main(argv=None):
  model_folder = os.path.join(cfg.dataset_params['model_path'],
                              cfg.common_params['net_name'], 'ckpt')
  checkpoint = tf.train.get_checkpoint_state(model_folder)
  input_checkpoint = checkpoint.model_checkpoint_path
  saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                     clear_devices=True)

  dataset = cifar_dataset_test(cfg.dataset_params)
  test_iterator = dataset.make_one_shot_iterator()
  test_Loader = test_iterator.get_next()
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    saver.restore(sess, input_checkpoint)
    images, labels = sess.run(test_Loader)

    total_count = 0
    count = 1
    # 抽取数据集里的数据来进行评估
    for i in range(0, len(images), 50):
      image = images[i:i + 50, :, :, :]
      predict = sess.run(cfg.graph_node['output'],
                         feed_dict={cfg.graph_node['input']      : image,
                                    cfg.graph_node['is_training']: False,
                                    cfg.graph_node['keep_prob']  : 1.0})
      correct_pred = tf.equal(
        tf.argmax(predict, 1, output_type=tf.int32), labels[i:i + 50])
      acc = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
      total_count += sess.run(acc)
      accuracy = total_count / (count * 50)
      print("test samples:%d,accuracy:%d/%d = %.4f " % (
        count * 50, total_count, count * 50, accuracy))
      count += 1


# 如果你的代码中的入口函数不叫main()，而是一个其他名字的函数，如test()，则你应该这样写入口
# tf.app.run(test())
# 如果你的代码中的入口函数叫main()，则你就可以把入口写成tf.app.run()
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
