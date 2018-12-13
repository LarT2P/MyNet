dataset_params = {
  'data_path' : './cifar10_data',
  'model_path': './model/train'
}

# relu_style: relu6, leaky_relu, relu
common_params = {
  'net_name'     : 'SqueezeNetC',
  'batch_size'   : 128,
  'image_size'   : (32, 32),
  'learning_rate': 0.01,
  'moment'       : 0.9,
  'display_step' : 100,
  'num_epochs'   : 800,
  'predict_step' : 782,
  'restore'      : False,
  'has_batchnorm': True,
  'relu_style'   : 'leaky_relu',
  'labels_smooth': True,
  'labels_smooth_epsilon': 0.1
}

graph_node = {
  'input'      : 'input:0',
  'is_training': 'is_training:0',
  'keep_prob'  : 'keep_prob:0',
  'output'     : 'output:0'
}

net_style = ['vgg11', 'vgg13', 'vgg16', 'vgg19',
             'resnet20', 'resnet32', 'resnet44', 'resnet56',
             'XceptionNet',
             'MobileNet',
             'DenseNet40_12', 'DenseNet100_12', 'DenseNet100_24',
             'DenseNetBC100_12', 'DenseNetBC250_24', 'DenseNetBC190_40',
             'ResNext50', 'ResNext101',
             'SqueezeNetA', 'SqueezeNetB', 'SqueezeNetC',
             'SE_Resnet_20', 'SE_Resnet_32', 'SE_Resnet_44', 'SE_Resnet_56']

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
# The record is the image plus a one-byte label
RECORD_BYTES = DEFAULT_IMAGE_BYTES + 1
NUM_CLASSES = 10
NUM_DATA_FILES = 5
NUM_IMAGES = {'train': 50000, 'validation': 10000, }

net_layers = {
  'VGG11'           : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
                       512, 'M'],
  'VGG13'           : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M',
                       512,
                       512, 'M'],
  'VGG16'           : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                       512,
                       'M', 512, 512, 512, 'M'],
  'VGG19'           : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512,
                       512,
                       512, 512, 'M', 512, 512, 512, 512, 'M'],
  # 6n个3x3也就是18, 即n=3
  'ResNet20'        : [
    [[[3, 16, 1, 1], [3, 16, 1, 0]], 6],

    [[[3, 32, 2, 1], [3, 32, 1, 0]], 1],
    [[[3, 32, 1, 1], [3, 32, 1, 0]], 5],

    [[[3, 64, 2, 1], [3, 64, 1, 0]], 1],
    [[[3, 64, 1, 1], [3, 64, 1, 0]], 5],
  ],
  # n=5
  'ResNet32'        : [
    [[[3, 16, 1], [3, 16, 1]], 10],

    [[[3, 32, 2], [3, 32, 1]], 1],
    [[[3, 32, 1], [3, 32, 1]], 9],

    [[[3, 64, 2], [3, 64, 1]], 1],
    [[[3, 64, 1], [3, 64, 1]], 9],
  ],
  # n=7
  'ResNet44'        : [
    [[[3, 16, 1], [3, 16, 1]], 14],

    [[[3, 32, 2], [3, 32, 1]], 1],
    [[[3, 32, 1], [3, 32, 1]], 13],

    [[[3, 64, 2], [3, 64, 1]], 1],
    [[[3, 64, 1], [3, 64, 1]], 13],
  ],
  # n=9
  'ResNet56'        : [
    [[[3, 16, 1], [3, 16, 1]], 18],

    [[[3, 32, 2], [3, 32, 1]], 1],
    [[[3, 32, 1], [3, 32, 1]], 17],

    [[[3, 64, 2], [3, 64, 1]], 1],
    [[[3, 64, 1], [3, 64, 1]], 17],
  ],
  # L=40, k=12
  'DenseNet40_12'   : [40, 12],
  'DenseNet100_12'  : [100, 12],
  'DenseNet100_24'  : [100, 12],
  'DenseNetBC100_12': [100, 12],
  'DenseNetBC250_24': [250, 24],
  'DenseNetBC190_40': [190, 40]
}
