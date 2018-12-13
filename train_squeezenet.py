# todo:
#  [v] BN: 显著提高了训练集的准确率, 但是似乎有些过拟合
#  [v] l2正则化
#  [x] 尝试ReLU6
#  [v] 尝试Leaky ReLU
#  [v] 调整Dropout
#  [x] 数据增强
#  [ ] Zero Y启发式方法*
#  [ ] 限制训练时间, 再合适的时间上停下来
#  [ ] 学习率预热*
#  [ ] 无偏衰减*
#  [不会用] FP16
#  [x] 余弦学习率衰减
#  [x] 标签平滑
#  [x] 混合训练

import os

import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import Progbar

import cfg

os.environ['CUDA_VISIBLE_DEVICE'] = '0'


# 构造网络 ######################################################################
class SqueezeNet(object):
  def __init__(self, squeezename, is_training, keep_prob=0.5, num_classes=10):
    super(SqueezeNet, self).__init__()
    self.squeezename = squeezename
    self.num_classes = num_classes
    self.regularizer = tf.contrib.layers.l2_regularizer(scale=5e-4)
    self.initializer = tf.contrib.layers.xavier_initializer()
    self.model = squeezename[-1]
    self.is_training = is_training
    self.keep_prob = keep_prob
    self.conv_num = 1
    self.has_batchnorm = cfg.common_params['has_batchnorm']
    self.relu_style = cfg.common_params['relu_style']
    self.labels_smooth = cfg.common_params['labels_smooth']
    self.labels_smooth_epsilon = cfg.common_params['labels_smooth_epsilon']

    # *******************************
    self.sr = 0.5
    self.base = 128
    self.incre = 128
    self.pct33 = 0.5
    self.freq = 2

  def forward(self, inputs):
    height, width = inputs.shape[1:3]
    out = self.conv2d(
      inputs=inputs, out_channel=96, kernel_size=7, strides=2
    )
    out = tf.layers.max_pooling2d(
      out, pool_size=3, strides=2, padding='same', name='maxpool1'
    )

    if self.model == 'A':
      out = self.make_layers_A(out)
    elif self.model == 'B':
      out = self.make_layers_B(out)
    elif self.model == 'C':
      out = self.make_layers_C(out)
    else:
      raise Exception('请使用现有的模型...')

    out = self.conv2d(
      inputs=out, out_channel=1000, kernel_size=1, strides=1
    )
    # 缩放为/16
    pool_height, pool_width = height // 16, width // 16
    out = tf.layers.average_pooling2d(
      out, pool_size=(pool_height, pool_width),
      strides=(pool_height, pool_width), name='avepool10'
    )
    out = tf.layers.flatten(out, name='flatten')
    out = tf.layers.dropout(out, rate=self.keep_prob, name='dropout')
    predicts = tf.layers.dense(
      out, units=self.num_classes, kernel_initializer=self.initializer,
      kernel_regularizer=self.regularizer,
      name='fc'
    )
    softmax_out = tf.nn.softmax(predicts, name='output')
    return predicts, softmax_out

  def conv2d(
      self, inputs, out_channel, kernel_size=3, strides=1, relu=True
  ):
    inputs = tf.layers.conv2d(
      inputs, filters=out_channel, kernel_size=kernel_size, strides=strides,
      padding='same', kernel_initializer=self.initializer,
      kernel_regularizer=self.regularizer,
      name='conv_' + str(self.conv_num))

    self.conv_num += 1

    if self.has_batchnorm:
      inputs = tf.layers.batch_normalization(
        inputs, training=self.is_training
      )

    if self.relu_style == 'relu6':
      inputs = tf.nn.relu6(inputs)
    elif self.relu_style == 'leaky_relu':
      inputs = tf.nn.leaky_relu(inputs)
    elif self.relu_style == 'relu':
      inputs = tf.nn.relu(inputs)
    else:
      print("the relu_style isn't defined.")

    return inputs

  def make_layers_A(self, inputs):
    max_pool_loc = [4, 8]
    pool_num = 1
    for i in range(2, 10):
      # 这里的括号不可以去掉, 属于一个向下取整, 也就是说out_channel=[128, 128, 128*2,
      # 128*2, 128*3, 128*3, 128*4, 128*4]
      out_channel = self.base + self.incre * ((i - 2) // self.freq)
      inputs = self.fire_block(inputs, out_channel)
      if i in max_pool_loc:
        inputs = tf.layers.max_pooling2d(
          inputs, pool_size=3, strides=2, padding='same',
          name='maxpool_' + str(pool_num))
        pool_num += 1
    return inputs

  # def make_layers_B(self, inputs):
  #   max_pool_loc = [4, 8]
  #   short_cuts = [3, 5, 7, 9]
  #   pool_num = 1
  #   for i in range(2, 10):
  #
  #     if i in short_cuts:
  #       short_cut = tf.identity(inputs)
  #
  #     # 这里的括号不可以去掉, 属于一个向下取整, 也就是说out_channel=[128, 128, 128*2,
  #     # 128*2, 128*3, 128*3, 128*4, 128*4]
  #     out_channel = self.base + self.incre * ((i - 2) // self.freq)
  #     inputs = self.fire_block(inputs, out_channel)
  #
  #     if i in short_cuts:
  #       inputs = tf.add(inputs, short_cut)
  #
  #     if i in max_pool_loc:
  #       inputs = tf.layers.max_pooling2d(
  #         inputs, pool_size=3, strides=2, padding='same',
  #         name='maxpool_' + str(pool_num))
  #       pool_num += 1
  #   return inputs
  def make_layers_B(self, inputs):
    max_pool_loc = [4, 8]
    short_cuts = [2, 3, 4, 5, 6, 7, 8, 9]
    pool_num = 1
    for i in range(2, 10):

      if i in short_cuts:
        short_cut = tf.identity(inputs)

      # 这里的括号不可以去掉, 属于一个向下取整, 也就是说out_channel=[128, 128, 128*2,
      # 128*2, 128*3, 128*3, 128*4, 128*4]
      out_channel = self.base + self.incre * ((i - 2) // self.freq)
      inputs = self.fire_block(inputs, out_channel)

      if i in short_cuts:
        if inputs.shape[-1] != short_cut.shape[-1]:
          short_cut = self.conv2d(short_cut, inputs.shape[-1], 1, 1, False)
        inputs = tf.add(inputs, short_cut)

      if i in max_pool_loc:
        inputs = tf.layers.max_pooling2d(
          inputs, pool_size=3, strides=2, padding='same',
          name='maxpool_' + str(pool_num))
        pool_num += 1
    return inputs

  def make_layers_C(self, inputs):
    max_pool_loc = [4, 8]
    short_cuts = [2, 3, 4, 5, 6, 7, 8, 9]
    pool_num = 1
    for i in range(2, 10):

      if i in short_cuts:
        short_cut = tf.identity(inputs)

      # 这里的括号不可以去掉, 属于一个向下取整, 也就是说out_channel=[128, 128, 128*2,
      # 128*2, 128*3, 128*3, 128*4, 128*4]
      out_channel = self.base + self.incre * ((i - 2) // self.freq)
      inputs = self.fire_block(inputs, out_channel)

      if i in short_cuts:
        if inputs.shape[-1] != short_cut.shape[-1]:
          short_cut = self.conv2d(short_cut, inputs.shape[-1], 1, 1, False)
        inputs = tf.add(inputs, short_cut)

      if i in max_pool_loc:
        inputs = tf.layers.max_pooling2d(
          inputs, pool_size=3, strides=2, padding='same',
          name='maxpool_' + str(pool_num))
        pool_num += 1
    return inputs

  def fire_block(self, inputs, out_channel):
    sfilter1x1 = self.sr * out_channel
    efilter1x1 = (1 - self.pct33) * out_channel
    efilter3x3 = self.pct33 * out_channel
    out = self.conv2d(inputs, sfilter1x1, kernel_size=1, strides=1)
    out_1 = self.conv2d(out, efilter1x1, kernel_size=1, strides=1)
    out_2 = self.conv2d(out, efilter3x3, kernel_size=3, strides=1)
    out = tf.concat([out_1, out_2], axis=-1)
    return out

  def loss(self, predicts, labels):
    if self.labels_smooth and self.labels_smooth_epsilon:
      labels = tf.one_hot(labels, predicts.shape[-1])
      losses = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(
          labels, predicts, label_smoothing=self.labels_smooth_epsilon
        )
      )
    else:
      losses = tf.reduce_mean(
        tf.losses.sparse_softmax_cross_entropy(labels, predicts)
      )
    l2_reg = tf.losses.get_regularization_losses()
    losses += tf.add_n(l2_reg)
    return losses


# 构造处理类 #####################################################################
class Solver(object):
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

    self.restore = cfg.common_params['restore']
    self.test_steps = (10000 // cfg.common_params['batch_size']) + 1
    self.train_steps = int(50000 / cfg.common_params['batch_size'] * \
                           cfg.common_params['num_epochs'])

    self.construct_graph()

  def construct_graph(self):
    ###################################
    ########### Parameters ############
    ###################################

    # 确定图上的各个关键变量
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    # 确定学习率变化, lr = lr * 0.1^(global_step/39062):
    # global_step/39062=50000/128*100 相当于就是100个epoch下降一次
    # 如果staircase=True，那就表明每39062次计算学习速率变化，更新原始学习速率，阶梯
    # 下降如果是False，那就是每一步都更新学习速率
    # self.learning_rate = tf.train.exponential_decay(
    #   self.learning_rate, self.global_step, 39062, 0.1, staircase=True)
    self.learning_rate = tf.train.cosine_decay(
      self.learning_rate, self.global_step, 39062)

    ###############################################
    ########### Defining place holders ############
    ###############################################
    self.images = tf.placeholder(
      tf.float32, (None, self.height, self.width, 3), name='input')
    self.labels = tf.placeholder(tf.int32, None)
    self.is_training = tf.placeholder_with_default(
      False, None, name='is_training')
    self.keep_prob = tf.placeholder(tf.float32, None, name='keep_prob')

    ##################################################
    ########### Model + Loss + Accuracy ##############
    ##################################################

    # MODEL
    # eval() 函数用来执行一个字符串表达式，并返回表达式的值。这里是执行网络
    self.net = eval(self.netname)(
      is_training=self.is_training, keep_prob=self.keep_prob)
    # 前向计算网络
    self.predicts, self.softmax_out = self.net.forward(self.images)

    # Define loss
    with tf.name_scope('loss'):
      self.total_loss = self.net.loss(self.predicts, self.labels)

    # Accuracy
    with tf.name_scope('accuracy'):
      correct_pred = tf.equal(
        tf.argmax(self.softmax_out, 1, output_type=tf.int32), self.labels)
      self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #############################################
    ########### training operation ##############
    #############################################

    # 确定优化器
    optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.moment)

    # gradient update.
    with tf.name_scope('train'):
      # 从更新操作的图集合汇总取出全部变量
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      # 执行了update_ops后才会执行后面这个操作
      with tf.control_dependencies(update_ops):
        self.train_op = optimizer.minimize(
          self.total_loss, global_step=self.global_step)

  def get_acc_test(self, sess, test_iterator, test_dataset):
    # 总体计算测试准确率
    sess.run(test_iterator.initializer)

    # 测试集进度条
    progbar_test = Progbar(target=self.test_steps)

    test_acc_count = 0
    test_total_accuracy = 0

    for test_step in range(self.test_steps):
      # 获取测试集
      test_images, test_labels = sess.run(test_dataset)
      test_acc = sess.run(self.accuracy,
                          feed_dict={self.images     : test_images,
                                     self.labels     : test_labels,
                                     self.is_training: False,
                                     self.keep_prob  : 1.0})
      test_total_accuracy += test_acc
      test_acc_count += 1

      # 更新进度条
      progbar_test.update(test_step + 1)

    test_acc_final = test_total_accuracy / test_acc_count

    return test_acc_final

  def solve(self):
    train_iterator = self.dataset['train'].make_one_shot_iterator()
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

    if self.restore:
      model_folder = os.path.join(cfg.dataset_params['model_path'],
                                  cfg.common_params['net_name'], 'ckpt')
      checkpoint = tf.train.get_checkpoint_state(model_folder)
      input_checkpoint = checkpoint.model_checkpoint_path
      saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                         clear_devices=True)

    ############################################
    ############ Run the Session ###############
    ############################################
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
      step = 0
      acc_count = 0
      total_accuracy = 0
      final_acc = 0
      final_train_acc = 0
      test_acc_final = 0

      # sess.run 并没有计算整个图，只是计算了与想要fetch的值相关的部分
      sess.run(tf.global_variables_initializer())

      # tf.train.saver.save()在保存check - point的同时也会保存Meta Graph。但是在恢复
      # 图时，tf.train.saver.restore()只恢复Variable，如果要从MetaGraph恢复图，需要
      # 使用import_meta_graph。这是其实为了方便用户，有时我们不需要从MetaGraph恢复的图，
      # 而是需要在 python中构建神经网络图，并恢复对应的 Variable。
      # tf.train.Saver()/saver.restore() 则能够完完整整保存和恢复神经网络的训练。
      # Check-point分为两个文件保存Variable的二进制信息。ckpt文件保存了Variable的二进
      # 制信息，index文件用于保存 ckpt 文件中对应 Variable 的偏移量信息。
      if self.restore:
        saver.restore(sess, input_checkpoint)
        step = int(input_checkpoint.split('/')[-1].split('-')[-1])

      # 因为原本的数据集已经根据周期进行了重复, 所以顺着迭代执行即可

      ###################################################
      ############ Training / Evaluation ###############
      ###################################################
      try:
        while True:
          # 训练迭代一步, 取一步的数据, 训练一步, 计算一步的学习率
          if step % 20 == 0:
            print('step{0}/total{1}'.format(step, self.train_steps))

          images, labels = sess.run(train_dataset)
          # labels 为(1x128)的向量
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
            test_acc_final = self.get_acc_test(
              sess=sess, test_iterator=test_iterator, test_dataset=test_dataset
            )
            print('test acc:%.4f' % (test_acc_final))

            if test_acc_final > final_acc:
              # 记录测试集
              final_acc = test_acc_final
              # 记录训练集
              final_train_acc = (total_accuracy / acc_count)
              saver.save(sess, self.model_name, global_step=step)

          step += 1
      except tf.errors.OutOfRangeError:
        test_acc_final = self.get_acc_test(
          sess=sess, test_iterator=test_iterator, test_dataset=test_dataset
        )
        print('test acc:%.4f' % (test_acc_final))

        if test_acc_final > final_acc:
          # 记录测试集
          final_acc = test_acc_final
          # 记录训练集
          final_train_acc = (total_accuracy / acc_count)
          saver.save(sess, self.model_name, global_step=step)

        print("finish training !")
        print("best test_acc:{0}&train_acc:{1}".format(
          final_acc, final_train_acc))


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

      # # 随机设置图片的亮度
      # image = tf.image.random_brightness(image, max_delta=30)
      # # 随机设置图片的色度
      # image = tf.image.random_hue(image, max_delta=0.3)
      # # 随机设置图片的饱和度
      # image = tf.image.random_saturation(image, lower=0.2, upper=1.8)

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
    dataset = dataset.prefetch(buffer_size=batch_size)
    # 训练的时候, 对于拿出来的batchsize的数据进行混淆
    if is_training:
      # 随机混淆数据后抽取buffer_size大小的数据
      dataset = dataset.shuffle(buffer_size=cfg.NUM_IMAGES['train'])
      # 将数据集重复周期次, 这么多周期都用使用相同的数据
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
    print(cfg.common_params)
  else:
    print('undefined net_name...')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
