import os

import tensorflow as tf

import cfg

os.environ['CUDA_VISIBLE_DEVICE'] = '0'


# 构造网络 ######################################################################
class SqueezeNet(object):
  def __init__(self, squeezename, is_training, keep_prob=0.5, num_classes=10):
    super(SqueezeNet, self).__init__()
    self.squeezename = squeezename
    self.num_classes = num_classes
    self.short_cut_list = cfg.net_layers[squeezename]
    self.regularizer = tf.contrib.layers.l2_regularizer(scale=5e-4)
    self.initializer = tf.contrib.layers.xavier_initializer()
    self.model = squeezename[-1]
    self.is_training = is_training
    self.keep_prob = keep_prob

  def forward(self, inputs):
    with tf.name_scope('squeezenet'):
      height, width = inputs.shape[1:3]
      out = self.conv2d(
        inputs=inputs, out_channel=96, kernel_size=7, strides=2, name='conv1'
      )
      out = tf.layers.average_pooling2d(
        out, pool_size=3, strides=2, name='maxpool1'
      )

      if self.model == 'A':
        out = self.make_layers_A(inputs)
      elif self.model == 'B':
        out = self.make_layers_B(inputs)
      elif self.model == 'C':
        out = self.make_layers_C(inputs)
      else:
        raise Exception('请使用现有的模型...')

      out = self.conv2d(
        inputs=inputs, out_channel=1000, kernel_size=1, strides=1, name='conv10'
      )
      # 缩放为/16
      pool_height, pool_width = height // 16, width // 16
      out = tf.layers.average_pooling2d(
        out, pool_size=(pool_height, pool_width),
        strides=(pool_height, pool_width), name='avepool10'
      )
      out = tf.layers.flatten(out, name='flatten')
      predicts = tf.layers.dense(
        out, units=self.num_classes, kernel_initializer=self.initializer,
        kernel_regularizer=self.regularizer, name='fc'
      )
      softmax_out = tf.nn.softmax(predicts, name='output')
      return predicts, softmax_out

  def conv2d(
      self, inputs, out_channel, kernel_size=3, strides=1, name=None, relu=True
  ):
    with tf.name_scope(name):
      inputs = tf.layers.conv2d(
        inputs, filters=out_channel, kernel_size=kernel_size, strides=strides,
        padding='same', kernel_initializer=self.initializer,
        kernel_regularizer=self.regularizer
      )
      inputs = tf.layers.batch_normalization(
        inputs, training=self.is_training
      )
      inputs = tf.nn.relu(inputs) if relu else inputs
      return inputs

  def make_layers_A(self, inputs):
    s1 = [16, 16, 32, 32, 48, 48, 64, 64]
    e1 = [64, 64, 128, 128, 192, 192, 256, 256]

    for i in range(2, 9):
      inputs = self.fire_block(
        inputs, [s1[i - 2], e1[i - 2], e1[i - 1]], name='fire_{}'.format(i)
      )
      if i == 4 or i == 8:
        inputs = tf.layers.max_pooling2d(
          inputs, 3, 2, padding='same', name='maxpool_{}'.format(i)
        )

    return inputs

  def make_layers_B(self, inputs):
    s1 = [16, 16, 32, 32, 48, 48, 64, 64]
    e1 = [64, 64, 128, 128, 192, 192, 256, 256]

    for i in range(2, 9):
      if i - 1 in self.short_cut_list:
        short_cut = tf.identity(inputs)
      inputs = self.fire_block(
        inputs, [s1[i - 2], e1[i - 2], e1[i - 1]], name='fire_{}'.format(i)
      )
      if i - 1 in self.short_cut_list:
        inputs = tf.add(inputs, short_cut)

      if i == 4 or i == 8:
        inputs = tf.layers.max_pooling2d(
          inputs, 3, 2, padding='same', name='maxpool_{}'.format(i)
        )
    return inputs

  def make_layers_C(self, inputs):
    s1 = [16, 16, 32, 32, 48, 48, 64, 64]
    e1 = [64, 64, 128, 128, 192, 192, 256, 256]

    for i in range(2, 9):
      if i - 1 in self.short_cut_list:
        short_cut = tf.identity(inputs)
      inputs = self.fire_block(
        inputs, [s1[i - 2], e1[i - 2], e1[i - 1]], name='fire_{}'.format(i)
      )
      if i - 1 in self.short_cut_list:
        inputs = tf.add(inputs, short_cut)

      if i == 4 or i == 8:
        inputs = tf.layers.max_pooling2d(
          inputs, 3, 2, padding='same', name='maxpool_{}'.format(i)
        )

    return inputs

  def fire_block(self, inputs, block_size, name=None):
    with tf.name_scope(name):
      s1, e1, s3 = block_size
      inputs = self.conv2d(inputs, s1, 1, 1, name='s1', relu=True)
      inputs_e1 = self.conv2d(inputs, e1, 1, 1, name='e1', relu=True)
      inputs_s3 = self.conv2d(inputs, s3, 3, 1, name='s3', relu=True)
      inputs = tf.concat([inputs_e1, inputs_s3], axis=-1)
      return inputs

  def loss(self, predicts, labels):
    losses = tf.reduce_mean(
      tf.losses.sparse_softmax_cross_entropy(labels, predicts)
    )
    l2_reg = tf.losses.get_regularization_losses()
    losses += tf.add_n(l2_reg)
    return losses


# 构造处理类 #####################################################################
class Solver(object):
  """docstring for Solver"""

  def __init__(self, dataset, common_params, dataset_params):
    super(Solver, self).__init__()

    self.dataset = dataset
    self.learning_rate = common_params['learning_rate']
    self.moment = common_params['moment']
    self.batch_size = common_params['batch_size']
    self.height, self.width = common_params['image_size']
    self.display_step = common_params['display_step']
    self.predict_step = common_params['predict_step']

    self.netname = common_params['net_name']
    model_dir = os.path.join(dataset_params['model_path'], self.netname, 'ckpt')
    if not tf.gfile.Exists(model_dir):
      tf.gfile.MakeDirs(model_dir)
    self.model_name = os.path.join(model_dir, 'model.ckpt')

    self.log_dir = os.path.join(
      dataset_params['model_path'], self.netname, 'log')
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
    # 确定学习率变化, lr = lr * 0.1^(global_step/39062):
    # global_step/39062=50000/128*100 相当于就是100个epoch下降一次
    # 如果staircase=True，那就表明每39062次计算学习速率变化，更新原始学习速率，阶梯
    # 下降如果是False，那就是每一步都更新学习速率
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
    train_iterator = self.dataset['train'].make_initializable_iterator()
    train_dataset = train_iterator.get_next()
    test_iterator = self.dataset['test'].make_initializable_iterator()
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

    with tf.Session(config=config) as sess:
      test_steps = (10000 // cfg.common_params['batch_size']) + 1
      train_steps = (50000
                     * cfg.common_params['num_epochs']
                     // cfg.common_params['batch_size']
                     + 1)

      # sess.run 并没有计算整个图，只是计算了与想要fetch的值相关的部分
      sess.run(init)
      summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
      sess.run(train_iterator.initializer)

      acc_count = 0
      total_accuracy = 0
      # 因为原本的数据集已经根据周期进行了重复, 所以顺着迭代执行即可
      for step in range(train_steps):
        # 训练迭代一步, 取一步的数据, 训练一步, 计算一步的学习率
        if step % 20 == 0:
          print('step{0}/total{1}'.format(step, train_steps))
        images, labels = sess.run(train_dataset)
        sess.run(self.train_op, feed_dict={self.images     : images,
                                           self.labels     : labels,
                                           self.is_training: True,
                                           self.keep_prob  : 0.5})
        lr = sess.run(self.learning_rate)

        # 定期针对这一个batch(step)计算显示一下准确率
        if step % self.display_step == 0:
          # 迭代一步, 就计算一下准确率
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

        if step % self.predict_step == 0:
          # 总体计算测试准确率
          sess.run(test_iterator.initializer)

          test_acc_count = 0
          test_total_accuracy = 0
          for test_step in range(test_steps):
            summary_str = sess.run(summary_op,
                                   feed_dict={self.images     : images,
                                              self.labels     : labels,
                                              self.is_training: True,
                                              self.keep_prob  : 0.5})
            summary_writer.add_summary(summary_str, test_step)
            # 获取测试集
            test_images, test_labels = sess.run(test_dataset)
            test_acc = sess.run(self.accuracy,
                                feed_dict={self.images     : test_images,
                                           self.labels     : test_labels,
                                           self.is_training: False,
                                           self.keep_prob  : 1.0})
            test_total_accuracy += test_acc
            test_acc_count += 1
          print('test acc:%.4f' % (test_total_accuracy / test_acc_count))

        if step % 5000 == 0:
          saver.save(sess, self.model_name, global_step=step)

      print("finish training !")


# 获取数据并处理 #################################################################
class CifarData(object):
  def __init__(self,
               common_params=cfg.common_params,
               dataset_params=cfg.dataset_params):
    super(CifarData, self).__init__()
    self.common_params = common_params
    self.dataset_params = dataset_params

  def get_filenames(self, is_training, data_dir):
    """Returns a list of filenames."""
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

    assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.'
    )

    if is_training:
      return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, cfg.NUM_DATA_FILES + 1)
      ]
    else:
      return [os.path.join(data_dir, 'test_batch.bin')]

  def parse_record(self, raw_record, is_training):
    record_vector = tf.decode_raw(raw_record, tf.uint8)
    label = tf.cast(record_vector[0], tf.int32)
    depth_major = tf.reshape(record_vector[1:cfg.RECORD_BYTES],
                             [cfg.NUM_CHANNELS, cfg.HEIGHT, cfg.WIDTH])

    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
    image = self.preprocess_image(image, is_training)
    return image, label

  def preprocess_image(self, image, is_training):
    """
    图像预处理
    """
    if is_training:
      # 调整图像到固定大小
      # 剪裁的时候, 从图像的中心进行的剪裁
      image = tf.image.resize_image_with_crop_or_pad(
        image, cfg.HEIGHT + 8, cfg.WIDTH + 8)

      # 随机选择位置进行剪裁
      image = tf.random_crop(image, [cfg.HEIGHT, cfg.WIDTH, cfg.NUM_CHANNELS])

      # 1/2的概率随机左右翻转
      image = tf.image.random_flip_left_right(image)

    # 标准化, 零均值, 单位方差, 输出大小和输入一样
    image = tf.image.per_image_standardization(image)
    return image

  def input_fn(self, is_training, common_params, dataset_params):
    """
    获取文件, 读取数据,
    """
    data_dir = dataset_params['data_path']
    batch_size = common_params['batch_size']
    num_epochs = common_params['num_epochs']

    # is_training = True 返回训练集名字, False 返回测试集名字
    filenames = self.get_filenames(is_training, data_dir)

    # 这个函数的输入是一个文件的列表和一个record_bytes，之后dataset的每一个元素就是文件中固
    # 定字节数record_bytes的内容。通常用来读取以二进制形式保存的文件，CIFAR10数据集就是这种
    # 形式这里每条记录中的字节数是图像大小加上一比特
    dataset = tf.data.FixedLengthRecordDataset(filenames, cfg.RECORD_BYTES)

    # 每次执行一次, 都要从数据里去拿出batchsize大小的数据
    # 这样操作能确保下个批次始终对 GPU 可用，减少 GPU 的闲置时间。Buffer_size 是 GPU
    # 预读取的批次数量。
    dataset = dataset.prefetch(buffer_size=batch_size)

    # 训练的时候, 对于拿出来的batchsize的数据进行混淆
    if is_training:
      # 随机混淆数据后抽取buffer_size大小的数据
      dataset = dataset.shuffle(buffer_size=cfg.NUM_IMAGES['train'])
      # 只是重复训练数据集, 重复周期次, 这么多周期都用使用相同的数据
      dataset = dataset.repeat(num_epochs)

    # 把转换函数应用到数据集上
    # map映射函数, 并使用batch操作进行批提取
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
      lambda value: self.parse_record(value, is_training),
      batch_size=batch_size, num_parallel_batches=1, drop_remainder=False))

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

  def cifar_dataset(self):
    """
    获取输入数据, 测试数据, 整理成数据集
    """
    # train是多了一个混淆抽取
    train_dataset = self.input_fn(True, self.common_params, self.dataset_params)
    test_dataset = self.input_fn(False, self.common_params, self.dataset_params)
    dataset = {'train': train_dataset, 'test': test_dataset}
    return dataset


# 网络入口 ######################################################################
def SqueezeNetA(is_training=True, keep_prob=0.5):
  # 没有short_cut
  net = SqueezeNet(
    squeezename='SqueezeNetA', is_training=is_training, keep_prob=keep_prob
  )
  return net


def SqueezeNetB(is_training=True, keep_prob=0.5):
  # 有简单的short_cut
  net = SqueezeNet(
    squeezename='SqueezeNetB', is_training=is_training, keep_prob=keep_prob
  )
  return net


def SqueezeNetC(is_training=True, keep_prob=0.5):
  # 有复杂的short_cut
  net = SqueezeNet(
    squeezename='SqueezeNetC', is_training=is_training, keep_prob=keep_prob
  )
  return net


# 程序入口 ######################################################################
def main(argv=None):
  if cfg.common_params['net_name'] in cfg.net_style:
    print(cfg.common_params)
    get_data = CifarData(cfg.common_params, cfg.dataset_params)
    # 获取数据集
    dataset = get_data.cifar_dataset()
    # 确定各个采纳数, 学习率, 动量, 批大小, 图形大小, 特定的步数, 构建图
    solver = Solver(dataset, cfg.common_params, cfg.dataset_params)
    solver.solve()
  else:
    print('undefined net_name...')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
