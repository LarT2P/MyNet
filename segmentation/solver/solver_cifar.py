from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

from network.vgg import vgg11, vgg13, vgg16, vgg19
from network.resnet import resnet20, resnet32, resnet44, resnet56
from network.xception import XceptionNet
from network.mobileNet import MobileNet
from network.denseNet import DensetNet40_12, DenseNet100_12, DenseNet100_24, \
  DenseNetBC100_12, DenseNetBC250_24, DenseNetBC190_40
from network.resnext import ResNext50, ResNext101
from network.squeezeNet import SqueezeNetA, SqueezeNetB
from network.seNet import SE_Resnet_50, SE_Resnet_101

os.environ['CUDA_VISIBLE_DEVICE'] = '1'


class Solver(object):
  """docstring for Solver"""

  def __init__(self, netname, dataset, common_params, dataset_params):
    super(Solver, self).__init__()
    self.dataset = dataset
    self.learning_rate = common_params['learning_rate']
    self.moment = common_params['moment']
    self.batch_size = common_params['batch_size']
    self.height, self.width = common_params['image_size']

    self.display_step = common_params['display_step']
    self.predict_step = common_params['predict_step']

    self.netname = netname
    model_dir = os.path.join(dataset_params['model_path'], self.netname,
                             'ckpt')
    if not tf.gfile.Exists(model_dir):
      tf.gfile.MakeDirs(model_dir)
    self.model_name = os.path.join(model_dir, 'model.ckpt')

    self.log_dir = os.path.join(dataset_params['model_path'], self.netname,
                                'log')
    if not tf.gfile.Exists(self.log_dir):
      tf.gfile.MakeDirs(self.log_dir)

    self.construct_graph()

  def construct_graph(self):
    # 确定图上的各个关键变量
    self.global_step = tf.Variable(0, trainable=False)
    self.images = tf.placeholder(
      tf.float32, (None, self.height, self.width, 3), name='input')
    self.labels = tf.placeholder(
      tf.int32, None)
    self.is_training = tf.placeholder_with_default(
      False, None, name='is_training')
    self.keep_prob = tf.placeholder(
      tf.float32, None, name='keep_prob')

    # eval() 函数用来执行一个字符串表达式，并返回表达式的值。这里是执行网络
    self.net = eval(self.netname)(
      is_training=self.is_training, keep_prob=self.keep_prob)

    # 前向计算网络
    self.predicts, self.softmax_out = self.net.forward(self.images)
    # 计算网络损失
    self.total_loss = self.net.loss(self.predicts, self.labels)
    # 确定学习率变化, lr = lr * 0.1^(global_step/39062)
    # 如果staircase=True，那就表明每decay_steps次计算学习速率变化，更新原始学习速率，
    # 如果是False，那就是每一步都更新学习速率
    self.learning_rate = tf.train.exponential_decay(
      self.learning_rate, self.global_step, 39062, 0.1, staircase=True)
    # 确定优化器
    optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.moment)

    # 从更新操作的图集合汇总取出全部变量
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # 执行了update_ops后才会执行后面这个操作
    with tf.control_dependencies(update_ops):
      self.train_op = optimizer.minimize(
        self.total_loss, global_step=self.global_step)

    # 计算准确率
    correct_pred = tf.equal(
      tf.argmax(self.softmax_out, 1, output_type=tf.int32), self.labels)
    self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('loss', self.total_loss)
    tf.summary.scalar('accuracy', self.accuracy)

  def solve(self):
    """
    训练
    """
    train_iterator = self.dataset['train'].make_one_shot_iterator()
    train_dataset = train_iterator.get_next()
    test_iterator = self.dataset['test'].make_one_shot_iterator()
    test_dataset = test_iterator.get_next()

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars

    # 存储变量到checkpoint文件中
    saver = tf.train.Saver(var_list=var_list)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()

    summary_op = tf.summary.merge_all()

    sess = tf.Session(config=config)
    # sess.run 并没有计算整个图，只是计算了与想要fetch的值相关的部分
    sess.run(init)
    summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

    step = 0
    acc_count = 0
    total_accuracy = 0
    try:
      while True:
        # 训练迭代一步, 取一步的数据, 训练一步, 计算一步的学习率
        images, labels = sess.run(train_dataset)
        sess.run(self.train_op, feed_dict={
          self.images     : images, self.labels: labels,
          self.is_training: True, self.keep_prob: 0.5})
        lr = sess.run(self.learning_rate)
        # 迭代一步, 考虑是否显示结果, 要显示, 就计算一下准确率
        # ques: 那这里的准确率实际上不是所有的图片的计算结果了呗? 只是定期抽样?
        if step % self.display_step == 0:
          acc = sess.run(self.accuracy,
                         feed_dict={self.images     : images,
                                    self.labels     : labels,
                                    self.is_training: True,
                                    self.keep_prob  : 0.5})
          total_accuracy += acc
          acc_count += 1
          loss = sess.run(self.total_loss,
                          feed_dict={self.images     : images,
                                     self.labels     : labels,
                                     self.is_training: True,
                                     self.keep_prob  : 0.5})
          print('Iter step:%d learning rate:%.4f loss:%.4f accuracy:%.4f' %
                (step, lr, loss, total_accuracy / acc_count))

        # 定期预测一下结果
        if step % self.predict_step == 0:
          summary_str = sess.run(summary_op, feed_dict={self.images     : images,
                                                        self.labels     : labels,
                                                        self.is_training: True,
                                                        self.keep_prob  : 0.5})
          summary_writer.add_summary(summary_str, step)
          # 获取测试集
          test_images, test_labels = sess.run(test_dataset)
          acc = sess.run(self.accuracy, feed_dict={self.images     : test_images,
                                                   self.labels     : test_labels,
                                                   self.is_training: False,
                                                   self.keep_prob  : 1.0})
          print('test loss:%.4f' % (acc))
        # 5000步保存下结果
        if step % 5000 == 0:
          saver.save(sess, self.model_name, global_step=step)
        step += 1
    except tf.errors.OutOfRangeError:
      print("finish training !")
    sess.close()
